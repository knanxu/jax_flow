"""Flow matching loss functions."""

import jax
import jax.numpy as jnp

from jax_flow.core.utils import get_batch_size


def flow_loss(network, encoder, interpolant, batch, rng, config):
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

    # Compute loss
    loss = jnp.mean((predicted_velocity - target_velocity) ** 2)
    loss = loss * config.get("loss_scale", 0.1)

    info = {
        "loss": loss,
        "flow_loss": loss,
    }

    return loss, info


def mip_loss(network, encoder, interpolant, batch, rng, config):
    """Minimum Iterative Policy (MIP) loss.

    Two-step denoising with deterministic initialization.

    Args:
        network: Flow network.
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

    # Step 1: s=0, predict from zeros
    s0 = jnp.zeros((batch_size,))
    x0 = jnp.zeros((batch_size, horizon, action_dim))
    pred_step1 = network(x0, s0, s0, cond, training=True)

    # Step 2: s=t_two_step, predict from noisy target
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, (batch_size, horizon, action_dim))
    x_t = actions + (1 - t_two_step) * noise

    st = jnp.full((batch_size,), t_two_step)
    pred_step2 = network(x_t, st, st, cond, training=True)

    # Compute losses
    loss1 = jnp.mean((pred_step1 - actions) ** 2) / (t_two_step**2)
    loss2 = jnp.mean((pred_step2 - actions) ** 2) / ((1 - t_two_step) ** 2)

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


def mf_loss(network, encoder, interpolant, batch, rng, config):
    """Mean Flow loss with JVP-based Jacobian computation.

    Core equation (t=0 noise, t=1 data, network input x_s):
        u, duds = jvp(u_theta(x, s, t, cond), primals=(x_s, s, t), tangents=(v_s, 1, 0))
        u_target = v_s + (t - s) * duds
        loss = mean((u - stop_gradient(u_target))^2)

    Args:
        network: Flow network with signature (at, s, t, cond, training).
        encoder: Observation encoder.
        interpolant: Interpolant for x0 -> x1.
        batch: Batch of data with 'observations' and 'actions'.
        rng: Random key.
        config: Configuration dict with time_sampling and adaptive_weight params.

    Returns:
        Tuple of (loss, info_dict).
    """
    observations = batch["observations"]
    actions = batch["actions"]  # (batch, horizon, action_dim)

    batch_size = get_batch_size(observations)
    horizon = actions.shape[1]
    action_dim = actions.shape[2]

    # Encode observations
    rng, crop_rng = jax.random.split(rng)
    cond = encoder(observations, training=True, rngs={"crop": crop_rng})

    # Sample noise
    rng, noise_rng = jax.random.split(rng)
    x0 = jax.random.normal(noise_rng, (batch_size, horizon, action_dim))
    x1 = actions

    # Sample (s, t) pairs via logit-normal distribution
    time_cfg = config.get("time_sampling", {})
    mu = time_cfg.get("mu", -0.4)
    sigma = time_cfg.get("sigma", 1.0)
    ratio_equal = time_cfg.get("ratio_equal", 0.25)

    rng, time_rng = jax.random.split(rng)
    s, t = _sample_logit_normal_times(time_rng, batch_size, mu, sigma, ratio_equal)

    # Construct x_s and v_s from interpolant
    x_s = interpolant.interpolate(s, x0, x1)   # (batch, horizon, action_dim)
    v_s = interpolant.velocity(s, x0, x1)       # (batch, horizon, action_dim)

    # ========== JVP computation ==========
    # Wrap network into a pure function of (x, s_arg, t_arg) with cond captured
    def net_fn(x, s_arg, t_arg):
        return network(x, s_arg, t_arg, cond, training=True)

    # Primals: (x_s, s, t)
    # Tangents: (v_s, ones, zeros) — differentiate w.r.t. s only
    primals = (x_s, s, t)
    tangents = (v_s, jnp.ones((batch_size,)), jnp.zeros((batch_size,)))

    u, duds = jax.jvp(net_fn, primals, tangents)

    # ========== Target computation ==========
    # u_target = v_s + (t - s) * duds
    # Broadcast (t - s) from (batch,) to (batch, horizon, action_dim)
    ts_diff = (t - s)[:, None, None]
    u_target = v_s + ts_diff * duds

    # ========== Loss ==========
    error_sq = (u - jax.lax.stop_gradient(u_target)) ** 2

    # Adaptive weighting: w = 1 / (error^2 + c)^p
    aw_cfg = config.get("adaptive_weight", {})
    if aw_cfg.get("enabled", False):
        c = aw_cfg.get("c", 0.01)
        p = aw_cfg.get("p", 1.0)
        per_sample_error = jnp.mean(error_sq, axis=(1, 2))  # (batch,)
        w = 1.0 / (per_sample_error + c) ** p
        w = jax.lax.stop_gradient(w)
        loss = jnp.mean(w * per_sample_error)
    else:
        loss = jnp.mean(error_sq)

    loss = loss * config.get("loss_scale", 0.1)

    info = {
        "loss": loss,
        "mf_loss": loss,
        "mean_error": jnp.mean(error_sq),
    }

    return loss, info


def get_loss_fn(loss_type="flow"):
    """Get loss function by type.

    Args:
        loss_type: Type of loss ('flow', 'mip', 'mf').

    Returns:
        Loss function.
    """
    if loss_type == "flow" or loss_type == "flow_matching":
        return flow_loss
    elif loss_type == "mip":
        return mip_loss
    elif loss_type == "mf" or loss_type == "meanflow":
        return mf_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
