"""Flow matching loss functions."""

import jax
import jax.numpy as jnp

from jax_flow.core.utils import get_batch_size


def flow_loss(network, encoder, interpolant, batch, rng, config, step=None):
    """Standard flow matching loss.

    Learns to match the velocity field v_t(x_t) = dx_t/dt.

    Args:
        network: Flow network (MLP).
        encoder: Observation encoder.
        interpolant: Interpolant for x0 -> x1.
        batch: Batch of data with 'observations' and 'actions'.
        rng: Random key.
        config: Configuration dict.

    Returns:
        Tuple of (loss, info_dict).
    """
    observations = batch["observations"]
    actions = batch["actions"]  # (batch, horizon, action_dim)

    batch_size = get_batch_size(observations)
    horizon = actions.shape[1]
    action_dim = actions.shape[2]

    # Sample random timesteps
    rng, t_rng = jax.random.split(rng)
    t = jax.random.uniform(t_rng, (batch_size,), minval=0.0, maxval=1.0)

    # Sample noise
    rng, noise_rng, crop_rng = jax.random.split(rng, 3)
    x0 = jax.random.normal(noise_rng, (batch_size, horizon, action_dim))

    # Data
    x1 = actions

    # Encode observations
    cond = encoder(
        observations, training=True, rngs={"crop": crop_rng}
    )  # (batch, cond_dim)

    # Interpolate
    x_t = interpolant.interpolate(t, x0, x1)  # (batch, horizon, action_dim)

    # Compute target velocity
    target_velocity = interpolant.velocity(t, x0, x1)  # (batch, horizon, action_dim)

    # Predict velocity (network now handles horizon dimension internally)
    predicted_velocity = network(x_t, t, t, cond, training=True)

    # Compute loss (with optional per-sample weighting for DQC)
    per_sample = jnp.mean((predicted_velocity - target_velocity) ** 2, axis=(1, 2))
    if "sample_weights" in batch:
        loss = jnp.mean(batch["sample_weights"] * per_sample)
    else:
        loss = jnp.mean(per_sample)
    loss = loss * config.get("loss_scale", 0.1)

    info = {
        "loss": loss,
        "flow_loss": loss,
    }

    return loss, info


def mip_loss(network, encoder, interpolant, batch, rng, config, step=None):
    """Minimum Iterative Policy (MIP) loss.

    Two-step prediction with deterministic zero initialization.
    Follows much-ado-about-noising reference implementation:
      Step 1: network(zeros, s=0, s=0, cond) -> velocity, target = actions
      Step 2: network(act_t, s=t_two_step, s=t_two_step, cond) -> velocity, target = actions
    where act_t = actions + (1 - t_two_step) * noise.

    Args:
        network: Flow network with signature (at, s, t, cond, training).
        encoder: Observation encoder.
        interpolant: Interpolant (not used in MIP).
        batch: Batch of data.
        rng: Random key.
        config: Configuration dict with 't_two_step'.

    Returns:
        Tuple of (loss, info_dict).
    """
    observations = batch["observations"]
    actions = batch["actions"]

    batch_size = get_batch_size(observations)
    horizon = actions.shape[1]
    action_dim = actions.shape[2]

    t_two_step = config.get("t_two_step", 0.9)

    # Encode observations
    rng, crop_rng = jax.random.split(rng)
    cond = encoder(observations, training=True, rngs={"crop": crop_rng})

    # Step 1: predict velocity from zeros at s=0
    s0 = jnp.zeros((batch_size,))
    x0 = jnp.zeros((batch_size, horizon, action_dim))
    pred_step1 = network(x0, s0, s0, cond, training=True)

    # Step 2: predict velocity from noisy target at s=t_two_step
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, (batch_size, horizon, action_dim))
    act_t = actions + (1 - t_two_step) * noise

    st = jnp.full((batch_size,), t_two_step)
    pred_step2 = network(act_t, st, st, cond, training=True)

    # Per-sample L2 norm squared, normalized by time scale
    per_sample1 = jnp.sum((pred_step1 - actions) ** 2, axis=(1, 2)) / (t_two_step**2)
    per_sample2 = jnp.sum((pred_step2 - actions) ** 2, axis=(1, 2)) / ((1 - t_two_step) ** 2)

    if "sample_weights" in batch:
        w = batch["sample_weights"]
        loss1 = jnp.mean(w * per_sample1)
        loss2 = jnp.mean(w * per_sample2)
    else:
        loss1 = jnp.mean(per_sample1)
        loss2 = jnp.mean(per_sample2)

    loss = (loss1 + loss2) * config.get("loss_scale", 0.1)

    info = {
        "loss": loss,
        "mip_loss": loss,
        "loss1": loss1,
        "loss2": loss2,
    }

    return loss, info


def _sample_logit_normal_times(rng, batch_size, mu=-0.4, sigma=1.0, ratio_equal=0.25):
    """Sample (s, t) pairs using logit-normal distribution.

    Draws two values from N(mu, sigma), maps through sigmoid to (0,1),
    and assigns the smaller to s and larger to t. A fraction ratio_equal
    of samples have s=t (degenerating to standard flow matching).

    Returns:
        s, t: Arrays of shape (batch_size,) with 0 < s <= t < 1.
    """
    rng_z, rng_mask = jax.random.split(rng)

    # Sample two values per batch element from N(mu, sigma)
    z = jax.random.normal(rng_z, (batch_size, 2)) * sigma + mu
    times = jax.nn.sigmoid(z)  # (batch_size, 2), values in (0, 1)

    # Sort so s <= t
    s = jnp.minimum(times[:, 0], times[:, 1])
    t = jnp.maximum(times[:, 0], times[:, 1])

    # With probability ratio_equal, set s = t (standard FM regime)
    mask = jax.random.uniform(rng_mask, (batch_size,)) < ratio_equal
    s = jnp.where(mask, t, s)

    return s, t


def _compute_delta_t(step, config):
    """Compute delta_t based on progressive schedule.

    Schedule: delta_t = clip((progress - warmup) / rampup, 0, 1) * max_delta_t
    where progress = step / gradient_steps.

    Args:
        step: Current training step (or None for evaluation).
        config: Config dict with 'delta_t_schedule' and 'gradient_steps'.

    Returns:
        Scalar delta_t value.
    """
    schedule = config.get("delta_t_schedule", {})
    warmup_ratio = schedule.get("warmup_ratio", 0.0)
    rampup_ratio = schedule.get("rampup_ratio", 0.5)
    max_delta_t = schedule.get("max_delta_t", 1.0)
    gradient_steps = config.get("gradient_steps", 300000)

    if step is None:
        return max_delta_t

    progress = jnp.asarray(step, dtype=jnp.float32) / gradient_steps
    delta_t = jnp.clip((progress - warmup_ratio) / jnp.maximum(rampup_ratio, 1e-8), 0.0, 1.0)
    return delta_t * max_delta_t


def mf_loss(network, encoder, interpolant, batch, rng, config, step=None):
    """Mean Flow loss: flow matching term + JVP self-distillation term.

    Combines:
    1. Independent flow matching loss (learns accurate instantaneous velocity)
    2. MeanFlow self-distillation with delta_t progressive schedule

    Args:
        network: Flow network with signature (at, s, t, cond, training).
        encoder: Observation encoder.
        interpolant: Interpolant for x0 -> x1.
        batch: Batch of data with 'observations' and 'actions'.
        rng: Random key.
        config: Configuration dict.
        step: Current training step (None for evaluation).

    Returns:
        Tuple of (loss, info_dict).
    """
    observations = batch["observations"]
    actions = batch["actions"]  # (batch, horizon, action_dim)

    batch_size = get_batch_size(observations)
    horizon = actions.shape[1]
    action_dim = actions.shape[2]

    # === Shared: encode observations, sample noise ===
    rng, crop_rng = jax.random.split(rng)
    cond = encoder(observations, training=True, rngs={"crop": crop_rng})

    rng, noise_rng = jax.random.split(rng)
    x0 = jax.random.normal(noise_rng, (batch_size, horizon, action_dim))
    x1 = actions

    # === Term 1: Independent flow matching loss ===
    rng, t_flow_rng = jax.random.split(rng)
    t_flow = jax.random.uniform(t_flow_rng, (batch_size,), minval=0.0, maxval=1.0)

    x_t_flow = interpolant.interpolate(t_flow, x0, x1)
    v_target = interpolant.velocity(t_flow, x0, x1)
    v_pred = network(x_t_flow, t_flow, t_flow, cond, training=True)  # s=t

    fm_per_sample = jnp.mean((v_pred - v_target) ** 2, axis=(1, 2))

    # === Term 2: MeanFlow self-distillation with delta_t schedule ===
    delta_t = _compute_delta_t(step, config)

    # Time sampling: uniform + delta_t clamp
    rng, time_rng = jax.random.split(rng)
    temp1, temp2 = jax.random.uniform(time_rng, (2, batch_size), minval=0.0, maxval=1.0)
    s = jnp.minimum(temp1, temp2)
    t = jnp.maximum(temp1, temp2)
    s = jnp.maximum(s, t - delta_t)

    # Construct x_s and v_s from interpolant
    x_s = interpolant.interpolate(s, x0, x1)
    v_s = interpolant.velocity(s, x0, x1)

    # JVP computation
    def net_fn(x, s_arg, t_arg):
        return network(x, s_arg, t_arg, cond, training=True)

    primals = (x_s, s, t)
    tangents = (v_s, jnp.ones((batch_size,)), jnp.zeros((batch_size,)))

    u, duds = jax.jvp(net_fn, primals, tangents)

    # Target: u_target = v_s + (t - s) * duds
    ts_diff = (t - s)[:, None, None]
    u_target = v_s + ts_diff * duds

    # Self-distillation loss
    error_sq = (u - jax.lax.stop_gradient(u_target)) ** 2

    # Adaptive weighting (only on mf_term)
    aw_cfg = config.get("adaptive_weight", {})
    if aw_cfg.get("enabled", False):
        c = aw_cfg.get("c", 0.01)
        p = aw_cfg.get("p", 1.0)
        per_sample_error = jnp.mean(error_sq, axis=(1, 2))
        w = 1.0 / (per_sample_error + c) ** p
        w = jax.lax.stop_gradient(w)
        mf_per_sample = w * per_sample_error
    else:
        mf_per_sample = jnp.mean(error_sq, axis=(1, 2))

    # === Combine with optional sample weights ===
    loss_scale = config.get("loss_scale", 0.1)
    if "sample_weights" in batch:
        sw = batch["sample_weights"]
        flow_matching_loss = jnp.mean(sw * fm_per_sample)
        mf_term = jnp.mean(sw * mf_per_sample)
    else:
        flow_matching_loss = jnp.mean(fm_per_sample)
        mf_term = jnp.mean(mf_per_sample)
    total_loss = loss_scale * (flow_matching_loss + mf_term)

    info = {
        "loss": total_loss,
        "flow_matching_loss": flow_matching_loss,
        "mf_term": mf_term,
        "delta_t": delta_t,
    }

    return total_loss, info


def imf_loss(network, encoder, interpolant, batch, rng, config, step=None):
    """Improved MeanFlow loss: flow matching term + v-loss via compound function.

    Key difference from mf_loss: instead of u-loss with network-dependent target,
    constructs compound function V = u - (t-s)*sg(duds) and regresses to the
    fixed conditional velocity v_s. This yields a standard regression problem.

    Derivation (对 s 求导版本):
        v_s = u(z_s, s, t) - (t - s) * d/ds u(z_s, s, t)
        => V := u - (t-s) * stopgrad(duds)  should match v_s

    Args:
        network: Flow network with signature (at, s, t, cond, training).
        encoder: Observation encoder.
        interpolant: Interpolant for x0 -> x1.
        batch: Batch of data with 'observations' and 'actions'.
        rng: Random key.
        config: Configuration dict.
        step: Current training step (None for evaluation).

    Returns:
        Tuple of (loss, info_dict).
    """
    observations = batch["observations"]
    actions = batch["actions"]  # (batch, horizon, action_dim)

    batch_size = get_batch_size(observations)
    horizon = actions.shape[1]
    action_dim = actions.shape[2]

    # === Shared: encode observations, sample noise ===
    rng, crop_rng = jax.random.split(rng)
    cond = encoder(observations, training=True, rngs={"crop": crop_rng})

    rng, noise_rng = jax.random.split(rng)
    x0 = jax.random.normal(noise_rng, (batch_size, horizon, action_dim))
    x1 = actions

    # === Term 1: Independent flow matching loss ===
    rng, t_flow_rng = jax.random.split(rng)
    t_flow = jax.random.uniform(t_flow_rng, (batch_size,), minval=0.0, maxval=1.0)

    x_t_flow = interpolant.interpolate(t_flow, x0, x1)
    v_target = interpolant.velocity(t_flow, x0, x1)
    v_pred = network(x_t_flow, t_flow, t_flow, cond, training=True)  # s=t

    fm_per_sample = jnp.mean((v_pred - v_target) ** 2, axis=(1, 2))

    # === Term 2: Improved MeanFlow v-loss with delta_t schedule ===
    delta_t = _compute_delta_t(step, config)

    # Time sampling: uniform + delta_t clamp
    rng, time_rng = jax.random.split(rng)
    temp1, temp2 = jax.random.uniform(time_rng, (2, batch_size), minval=0.0, maxval=1.0)
    s = jnp.minimum(temp1, temp2)
    t = jnp.maximum(temp1, temp2)
    s = jnp.maximum(s, t - delta_t)

    # Construct x_s and v_s from interpolant
    x_s = interpolant.interpolate(s, x0, x1)
    v_s_true = interpolant.velocity(s, x0, x1)
    # JVP computation: d/ds u(x_s, s, t) with tangents (v_s, 1, 0)
    def net_fn(x, s_arg, t_arg):
        return network(x, s_arg, t_arg, cond, training=True)

    v_s = net_fn(x_s, s, s)
    primals = (x_s, s, t)
    tangents = (v_s, jnp.ones((batch_size,)), jnp.zeros((batch_size,)))

    u, duds = jax.jvp(net_fn, primals, tangents)

    # Compound function V: recover instantaneous velocity from mean velocity
    # v_s = u - (t - s) * duds  =>  V = u - (t-s) * sg(duds)
    ts_diff = (t - s)[:, None, None]
    V = u - ts_diff * jax.lax.stop_gradient(duds)

    # v-loss: regress V to fixed conditional velocity v_s
    error_sq = (V - v_s_true) ** 2

    # Adaptive weighting (only on imf_term)
    aw_cfg = config.get("adaptive_weight", {})
    if aw_cfg.get("enabled", False):
        c = aw_cfg.get("c", 0.01)
        p = aw_cfg.get("p", 1.0)
        per_sample_error = jnp.mean(error_sq, axis=(1, 2))
        w = 1.0 / (per_sample_error + c) ** p
        w = jax.lax.stop_gradient(w)
        imf_per_sample = w * per_sample_error
    else:
        imf_per_sample = jnp.mean(error_sq, axis=(1, 2))

    # === Combine with optional sample weights ===
    loss_scale = config.get("loss_scale", 0.1)
    if "sample_weights" in batch:
        sw = batch["sample_weights"]
        flow_matching_loss = jnp.mean(sw * fm_per_sample)
        imf_term = jnp.mean(sw * imf_per_sample)
    else:
        flow_matching_loss = jnp.mean(fm_per_sample)
        imf_term = jnp.mean(imf_per_sample)
    total_loss = loss_scale * (flow_matching_loss + imf_term)

    info = {
        "loss": total_loss,
        "flow_matching_loss": flow_matching_loss,
        "imf_term": imf_term,
        "delta_t": delta_t,
    }

    return total_loss, info


def get_loss_fn(loss_type="flow"):
    """Get loss function by type.

    Args:
        loss_type: Type of loss ('flow', 'mip', 'mf', 'imf').

    Returns:
        Loss function.
    """
    if loss_type == "flow" or loss_type == "flow_matching":
        return flow_loss
    elif loss_type == "mip":
        return mip_loss
    elif loss_type == "mf" or loss_type == "meanflow":
        return mf_loss
    elif loss_type == "imf" or loss_type == "improved_meanflow":
        return imf_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
