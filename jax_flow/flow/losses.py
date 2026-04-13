"""Flow matching loss functions."""

import jax
import jax.numpy as jnp

from jax_flow.core.utils import get_batch_size


def _per_sample_loss(error_sq):
    """Reduce squared error to per-sample scalar: sum over act_dim, mean over horizon.

    Args:
        error_sq: (batch, horizon, action_dim)

    Returns:
        (batch,) per-sample loss.
    """
    return jnp.mean(jnp.sum(error_sq, axis=-1), axis=-1)


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

    # Compute loss: sum over act_dim, mean over horizon and batch
    per_sample = _per_sample_loss((predicted_velocity - target_velocity) ** 2)
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

    # Per-sample: sum over act_dim, mean over horizon, normalized by time scale
    per_sample1 = _per_sample_loss((pred_step1 - actions) ** 2) / (t_two_step**2)
    per_sample2 = _per_sample_loss((pred_step2 - actions) ** 2) / ((1 - t_two_step) ** 2)

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
def sample_discrete_t(rng, batch_size, time_steps=100):
    """Sample two independent discrete timesteps from a uniform grid.

    Returns:
        t1, t2: Arrays of shape (batch_size,) in (0, 1].
    """
    t_rng, t_con_rng = jax.random.split(rng)
    time_values = jnp.linspace(1 / time_steps, 1.0, time_steps)
    t_indices = jax.random.randint(t_rng, (batch_size,), 0, time_steps)
    t_con_indices = jax.random.randint(t_con_rng, (batch_size,), 0, time_steps)
    t1 = time_values[t_indices]  # (batch_size,)
    t2 = time_values[t_con_indices]  # (batch_size,)
    return t1, t2

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

def _adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """Adaptive L2 loss from MeanFlowQL: down-weights large-error samples.

    Matches MeanFlowQL's reduction: mean(error**2) over all non-batch dims
    to get per-sample delta_sq, then adaptive reweighting.

    Args:
        error: (batch, ...) raw error (g - g_tgt), NOT squared.
        gamma: Controls reweighting strength (0=uniform, 1=full inverse).
        c: Small constant to prevent division by zero.

    Returns:
        Scalar loss.
    """
    # Mean over all non-batch dims (action_dim, horizon) — matches MeanFlowQL
    reduce_axes = tuple(range(1, error.ndim))
    delta_sq = jnp.mean(error ** 2, axis=reduce_axes)
    delta_sq = jnp.maximum(delta_sq, 1e-12)

    p = 1.0 - gamma
    w = 1.0 / jnp.maximum(jnp.power(delta_sq + c, p), 1e-12)
    w = jnp.clip(w, 1e-6, 1e6)
    return jnp.mean(jax.lax.stop_gradient(w) * delta_sq)


def mf_loss_stable(network, encoder, interpolant, batch, rng, config, step=None):
    """MeanFlow loss adapted from MeanFlowQL with stability techniques.

    Convention: x_s = (1-s)*x_0 + s*x_1, x_0=noise, x_1=data.
    Network signature: u(x, s, t, cond) -> mean velocity over [s, t].

    MeanFlow identity (our convention):
        g_tgt = x_s + s*v_s + (1-s)*dg/ds

    where dg/ds is the total derivative along the flow, computed via JVP.
    Here t is fixed to 1 (one-step generation target).

    Stability techniques from MeanFlowQL:
    - Target clipping to [-5, 5]
    - Adaptive L2 loss (inverse-power reweighting)
    - Optional consistency loss (default off, consistency_alpha=0.0)

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

    # === Encode observations, sample noise ===
    rng, crop_rng = jax.random.split(rng)
    cond = encoder(observations, training=True, rngs={"crop": crop_rng})

    rng, noise_rng = jax.random.split(rng)
    x0 = jax.random.normal(noise_rng, (batch_size, horizon, action_dim))
    x1 = actions

    # === Sample time s ~ U(0, 1), fix t = 1 ===
    rng, time_rng = jax.random.split(rng)
    time_steps = config.get("time_steps", 0)
    if time_steps > 0:
        # Discrete grid sampling (MeanFlowQL style)
        time_values = jnp.linspace(1 / time_steps, 1.0, time_steps)
        indices = jax.random.randint(time_rng, (batch_size,), 0, time_steps)
        s = time_values[indices]  # (batch_size,)
    else:
        # Continuous uniform sampling
        s = jax.random.uniform(time_rng, (batch_size,))
    t = jnp.ones((batch_size,))

    # x_s = (1-s)*x0 + s*x1,  v_s = dx_s/ds (= x1 - x0 for linear)
    x_s = interpolant.interpolate(s, x0, x1)
    v_s = interpolant.velocity(s, x0, x1)

    # === MeanFlow JVP ===
    # Total derivative dg/ds = (∂g/∂x)·v_s + ∂g/∂s, with t fixed (tangent=0)
    def net_fn(x, s_arg, t_arg):
        return network(x, s_arg, t_arg, cond, training=True)

    primals = (x_s, s, t)
    tangents = (v_s, jnp.ones((batch_size,)), jnp.zeros((batch_size,)))

    g, dgds = jax.jvp(net_fn, primals, tangents)

    # MeanFlow target (our convention): g_tgt = x_s + s*v_s + (1-s)*dg/ds
    s_bc = s[:, None, None]  # broadcast to (batch, horizon, action_dim)
    g_tgt = x_s + s_bc * v_s + (1 - s_bc) * dgds
    g_tgt = jax.lax.stop_gradient(g_tgt)
    g_tgt = jnp.clip(g_tgt, -5, 5)  # Target clipping (MeanFlowQL)

    err = g - g_tgt  # raw error, (batch, horizon, action_dim)

    # === Adaptive L2 loss or standard MSE ===
    aw_cfg = config.get("adaptive_weight", {})
    if aw_cfg.get("enabled", False):
        gamma = aw_cfg.get("gamma", 0.5)
        c = aw_cfg.get("c", 1e-3)
        mf_loss_val = _adaptive_l2_loss(err, gamma=gamma, c=c)
    else:
        mf_loss_val = jnp.mean(err ** 2)

    # === Optional consistency loss (default off) ===
    consistency_alpha = config.get("consistency_alpha", 0.0)
    consistency_loss_val = 0.0
    if consistency_alpha > 0.0:
        rng, con_time_rng, con_noise_rng = jax.random.split(rng, 3)
        con_time_steps = config.get("time_steps", 50)
        if con_time_steps <= 0:
            con_time_steps = 50
        t1, t2 = sample_discrete_t(con_time_rng, batch_size, time_steps=con_time_steps)
        x0_con = jax.random.normal(con_noise_rng, (batch_size, horizon, action_dim))

        x_t1 = interpolant.interpolate(t1, x0_con, actions)
        x_t2 = interpolant.interpolate(t2, x0_con, actions)

        # One-step prediction of x_1 from x_t: x_1_pred = x_t + (1-t) * u(x_t, t, 1)
        t1_ones = jnp.ones((batch_size,))
        t2_ones = jnp.ones((batch_size,))
        u_t1 = network(x_t1, t1, t1_ones, cond, training=True)
        u_t2 = network(x_t2, t2, t2_ones, cond, training=True)

        t1_bc = t1[:, None, None]
        t2_bc = t2[:, None, None]
        x_1_from_t1 = x_t1 + (1 - t1_bc) * (u_t1 - x_t1)
        x_1_from_t2 = x_t2 + (1 - t2_bc) * (u_t2 - x_t2)
        # Asymmetric gradient: only t1 branch gets gradients
        x_1_from_t2 = jax.lax.stop_gradient(x_1_from_t2)

        consistency_loss_val = jnp.mean(_per_sample_loss(
            jnp.square(x_1_from_t1 - x_1_from_t2)
        ))

    # === Total loss ===
    loss_scale = config.get("loss_scale", 0.1)
    total_loss = loss_scale * (mf_loss_val + consistency_alpha * consistency_loss_val)

    info = {
        "loss": total_loss,
        "mf_loss": mf_loss_val,
        "consistency_loss": consistency_loss_val,
        "consistency_alpha": consistency_alpha,
    }

    return total_loss, info


def mf_loss(network, encoder, interpolant, batch, rng, config, step=None):
    """Mean Flow loss: flow matching term + JVP self-distillation term.

    Combines:
    1. Independent flow matching loss (learns accurate instantaneous velocity)
    2. MeanFlow self-distillation with delta_t progressive schedule

    =========================================================================
    Convention differences from the original MeanFlow paper (arXiv 2505.13447)
    =========================================================================

    Original paper:
        z_t = (1-t)*z_0 + t*z_1,  z_0 = data, z_1 = noise
        t=0 is data, t=1 is noise. ODE flows data -> noise.
        Sampling reverses: start from noise z_1, recover data z_0.

    This implementation:
        x_s = (1-s)*x_0 + s*x_1,  x_0 = noise, x_1 = data
        s=0 is noise, s=1 is data. ODE flows noise -> data.
        Sampling goes forward: start from noise x_0, reach data x_1.

    The paper derives the MeanFlow identity by differentiating w.r.t. t
    (the noise-end variable). We instead differentiate w.r.t. s (the
    noise-end variable in our convention), yielding an equivalent identity.

    =========================================================================
    Mathematical derivation (d/ds version)
    =========================================================================

    Define the mean velocity over interval [s, t]:

        u(z_s, s, t) = 1/(t-s) * ∫_s^t v(z_τ, τ) dτ

    where v is the instantaneous velocity field and z_τ follows the ODE.

    Differentiate u w.r.t. the lower limit s via Leibniz rule. The total
    derivative along the characteristic z_s (where dz_s/ds = v(z_s, s))
    gives:

        d/ds u(z_s, s, t) = (u - v_s) / (t - s)

    Rearranging:

        u(z_s, s, t) = v_s + (t - s) * d/ds u(z_s, s, t)        ... (*)

    This is the self-consistency identity we enforce. The total derivative
    d/ds u(z_s, s, t) is computed via JVP with tangent (dz_s/ds, ds/ds, dt/ds)
    = (v_s, 1, 0):

        d/ds u = (∂u/∂z) · v_s + ∂u/∂s

    =========================================================================
    Sampling (single-step generation)
    =========================================================================

    At inference, the displacement from s=0 to t=1 is:

        x_1 - x_0 = ∫_0^1 v(z_τ, τ) dτ = (1-0) * u(z_0, 0, 1) = u(x_0, 0, 1)

    Therefore:

        x_1 = x_0 + u_θ(x_0, s=0, t=1, cond)

    where x_0 ~ N(0, I) (stochastic) or x_0 = 0 (deterministic).

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

    fm_per_sample = _per_sample_loss((v_pred - v_target) ** 2)

    # === Term 2: MeanFlow self-distillation with delta_t schedule ===
    # Enforce identity (*): u = v_s + (t - s) * d/ds u
    delta_t = _compute_delta_t(step, config)

    # Sample (s, t) with 0 <= s <= t <= 1, gap clamped by delta_t
    rng, time_rng = jax.random.split(rng)
    temp1, temp2 = jax.random.uniform(time_rng, (2, batch_size), minval=0.0, maxval=1.0)
    s = jnp.minimum(temp1, temp2)
    t = jnp.maximum(temp1, temp2)
    s = jnp.maximum(s, t - delta_t)

    # x_s = (1-s)*x0 + s*x1,  v_s = dx_s/ds = x1 - x0 (linear case)
    x_s = interpolant.interpolate(s, x0, x1)
    v_s = interpolant.velocity(s, x0, x1)

    # Compute u and its total derivative d/ds u(z_s, s, t) via JVP.
    # Tangents (v_s, 1, 0) correspond to (dz_s/ds, ds/ds, dt/ds),
    # giving: duds = (∂u/∂z)·v_s + ∂u/∂s
    def net_fn(x, s_arg, t_arg):
        return network(x, s_arg, t_arg, cond, training=True)

    primals = (x_s, s, t)
    tangents = (v_s, jnp.ones((batch_size,)), jnp.zeros((batch_size,)))

    u, duds = jax.jvp(net_fn, primals, tangents)

    # From identity (*): u_target = v_s + (t - s) * duds
    ts_diff = (t - s)[:, None, None]
    u_target = v_s + ts_diff * duds

    # Self-distillation: push u toward sg(u_target)
    error_sq = (u - jax.lax.stop_gradient(u_target)) ** 2

    # Adaptive weighting (only on mf_term)
    aw_cfg = config.get("adaptive_weight", {})
    if aw_cfg.get("enabled", False):
        c = aw_cfg.get("c", 0.01)
        p = aw_cfg.get("p", 1.0)
        per_sample_error = _per_sample_loss(error_sq)
        w = 1.0 / (per_sample_error + c) ** p
        w = jax.lax.stop_gradient(w)
        mf_per_sample = w * per_sample_error
    else:
        mf_per_sample = _per_sample_loss(error_sq)

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

    Combines:
    1. Independent flow matching loss (learns accurate instantaneous velocity)
    2. Improved MeanFlow v-loss with delta_t progressive schedule

    =========================================================================
    Difference from mf_loss
    =========================================================================

    mf_loss enforces the identity directly as a u-loss:

        L = ||u_θ - sg(u_target)||^2,   u_target = v_s + (t-s) * duds

    The target u_target depends on the network itself (through duds), making
    it a moving target (self-distillation). This can be unstable.

    imf_loss instead rearranges the identity (*) from mf_loss:

        u = v_s + (t - s) * d/ds u        ... (*)

    into:

        v_s = u - (t - s) * d/ds u        ... (**)

    and constructs a compound function:

        V := u_θ - (t - s) * sg(d/ds u_θ)

    which is regressed to the fixed, known conditional velocity v_s:

        L = ||V - v_s||^2

    This converts the self-distillation problem into a standard regression
    with a network-independent target, improving training stability.

    =========================================================================
    JVP tangent choice: learned velocity vs ground truth
    =========================================================================

    The total derivative d/ds u(z_s, s, t) requires dz_s/ds as the tangent
    for the z-component. In mf_loss, this is the ground-truth interpolant
    velocity v_s = x_1 - x_0. Here we use the network's own prediction
    v̂_s = u_θ(z_s, s, s) instead, since at convergence u_θ(·, s, s) = v(·, s).
    This avoids requiring access to the ground-truth conditional velocity
    in the JVP tangent.

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

    fm_per_sample = _per_sample_loss((v_pred - v_target) ** 2)

    # === Term 2: Improved MeanFlow v-loss with delta_t schedule ===
    # Enforce identity (**): v_s = u - (t - s) * d/ds u
    delta_t = _compute_delta_t(step, config)

    # Sample (s, t) with 0 <= s <= t <= 1, gap clamped by delta_t
    rng, time_rng = jax.random.split(rng)
    temp1, temp2 = jax.random.uniform(time_rng, (2, batch_size), minval=0.0, maxval=1.0)
    s = jnp.minimum(temp1, temp2)
    t = jnp.maximum(temp1, temp2)
    s = jnp.maximum(s, t - delta_t)

    # x_s = (1-s)*x0 + s*x1,  v_s_true = x1 - x0 (ground-truth conditional velocity)
    x_s = interpolant.interpolate(s, x0, x1)
    v_s_true = interpolant.velocity(s, x0, x1)

    def net_fn(x, s_arg, t_arg):
        return network(x, s_arg, t_arg, cond, training=True)

    # v̂_s = u_θ(z_s, s, s): network's estimate of instantaneous velocity,
    # used as the z-tangent instead of ground-truth v_s (see docstring)
    v_s = net_fn(x_s, s, s)

    # Compute u and d/ds u via JVP with tangent (v̂_s, 1, 0)
    primals = (x_s, s, t)
    tangents = (v_s, jnp.ones((batch_size,)), jnp.zeros((batch_size,)))

    u, duds = jax.jvp(net_fn, primals, tangents)

    # Compound function V from identity (**): V = u - (t-s) * sg(duds)
    # At convergence V = v_s, so regress V to the known v_s_true
    ts_diff = (t - s)[:, None, None]
    V = u - ts_diff * jax.lax.stop_gradient(duds)

    # v-loss: regress V to fixed conditional velocity v_s_true
    error_sq = (V - v_s_true) ** 2

    # Adaptive weighting (only on imf_term)
    aw_cfg = config.get("adaptive_weight", {})
    if aw_cfg.get("enabled", False):
        c = aw_cfg.get("c", 0.01)
        p = aw_cfg.get("p", 1.0)
        per_sample_error = _per_sample_loss(error_sq)
        w = 1.0 / (per_sample_error + c) ** p
        w = jax.lax.stop_gradient(w)
        imf_per_sample = w * per_sample_error
    else:
        imf_per_sample = _per_sample_loss(error_sq)

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


def _cdist(x, y, eps=1e-8):
    """Pairwise L2 distance using dot-product formula.

    Matches the official JAX drifting kernel exactly.

    Args:
        x: (B, N, D)
        y: (B, M, D)

    Returns:
        (B, N, M) pairwise L2 distances.
    """
    xydot = jnp.einsum("bnd,bmd->bnm", x, y)
    xnorms = jnp.einsum("bnd,bnd->bn", x, x)
    ynorms = jnp.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return jnp.sqrt(jnp.clip(sq_dist, a_min=eps))


def _drift_loss_single(gen, fixed_pos, R_list=(0.02, 0.05, 0.2)):
    """Core drift loss on a single timestep slice.

    Faithful port of drifting_util.drift_loss from phamtrongthang123/drifting_policy.
    Goal computation is entirely stop_gradient; gradients flow only through gen.

    Args:
        gen: (B, G, S) generated samples (G = gen_per_label).
        fixed_pos: (B, 1, S) positive (real) samples.
        R_list: Temperature values for multi-scale force computation.

    Returns:
        loss: (B,) per-sample loss.
        info: dict with 'scale' and per-R force norms.
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]

    old_gen = jax.lax.stop_gradient(gen)
    # targets = [old_gen, fixed_pos]  (no negatives for robot tasks)
    targets = jnp.concatenate([old_gen, fixed_pos], axis=1)  # (B, C_g + C_p, S)
    # All weights = 1
    targets_w = jnp.ones((B, C_g + C_p))

    # --- Goal computation (no gradients) ---
    dist = _cdist(old_gen, targets)  # (B, C_g, C_g + C_p)
    weighted_dist = dist * targets_w[:, None, :]
    scale = jnp.mean(weighted_dist) / jnp.mean(targets_w)

    scale_inputs = jnp.clip(scale / jnp.sqrt(S + 0.0), a_min=1e-3)
    old_gen_scaled = old_gen / scale_inputs
    targets_scaled = targets / scale_inputs

    dist_normed = dist / jnp.clip(scale, a_min=1e-3)

    # Mask self-connections for gen block (diagonal in gen-to-gen submatrix)
    mask_val = 100.0
    diag_mask = jnp.eye(C_g)  # (C_g, C_g)
    block_mask = jnp.pad(diag_mask, ((0, 0), (0, C_p)))  # (C_g, C_g + C_p)
    block_mask = block_mask[None, :, :]  # (1, C_g, C_g + C_p)
    dist_normed = dist_normed + block_mask * mask_val

    # Force loop over temperatures
    force_across_R = jnp.zeros_like(old_gen_scaled)
    info = {"scale": scale}

    for R in R_list:
        logits = -dist_normed / R

        affinity = jax.nn.softmax(logits, axis=-1)
        aff_transpose = jax.nn.softmax(logits, axis=-2)
        affinity = jnp.sqrt(jnp.clip(affinity * aff_transpose, a_min=1e-6))

        affinity = affinity * targets_w[:, None, :]

        # Split into neg (gen) and pos (real) blocks
        split_idx = C_g
        aff_neg = affinity[:, :, :split_idx]   # (B, C_g, C_g)
        aff_pos = affinity[:, :, split_idx:]    # (B, C_g, C_p)

        sum_pos = jnp.sum(aff_pos, axis=-1, keepdims=True)   # (B, C_g, 1)
        r_coeff_neg = -aff_neg * sum_pos                       # repulsive
        sum_neg = jnp.sum(aff_neg, axis=-1, keepdims=True)    # (B, C_g, 1)
        r_coeff_pos = aff_pos * sum_neg                        # attractive

        R_coeff = jnp.concatenate([r_coeff_neg, r_coeff_pos], axis=2)  # (B, C_g, C_g+C_p)

        total_force_R = jnp.einsum("biy,byx->bix", R_coeff, targets_scaled)
        total_coeffs = jnp.sum(R_coeff, axis=-1)  # (B, C_g)
        total_force_R = total_force_R - total_coeffs[:, :, None] * old_gen_scaled

        f_norm_val = jnp.mean(total_force_R ** 2)
        info[f"loss_{R}"] = f_norm_val

        force_scale = jnp.sqrt(jnp.clip(f_norm_val, a_min=1e-8))
        force_across_R = force_across_R + total_force_R / force_scale

    goal_scaled = old_gen_scaled + force_across_R
    goal_scaled = jax.lax.stop_gradient(goal_scaled)

    # --- Loss with gradients through gen ---
    gen_scaled = gen / jax.lax.stop_gradient(scale_inputs)
    diff = gen_scaled - goal_scaled
    loss = jnp.mean(diff ** 2, axis=(-1, -2))  # (B,)

    return loss, info


def drift_loss(network, encoder, interpolant, batch, rng, config, step=None):
    """Drift policy loss: particle-based single-step generation.

    Generates gen_per_label samples per observation, computes pairwise drift
    forces between generated and real actions, and regresses network output
    toward force-displaced goals. Per-timestep loss (independent drift loss
    at each horizon step) matching all configs in phamtrongthang123/drifting_policy.

    Args:
        network: Flow network (at, s, t, cond, training).
        encoder: Observation encoder.
        interpolant: Unused (kept for signature compatibility).
        batch: Batch with 'observations' and 'actions'.
        rng: Random key.
        config: Config dict with 'gen_per_label', 'drift_temperatures', 'loss_scale'.
        step: Current training step (unused).

    Returns:
        Tuple of (loss, info_dict).
    """
    observations = batch["observations"]
    actions = batch["actions"]  # (B, horizon, action_dim)

    batch_size = get_batch_size(observations)
    horizon = actions.shape[1]
    action_dim = actions.shape[2]
    G = config.get("gen_per_label", 8)
    R_list = tuple(config.get("drift_temperatures", [0.02, 0.05, 0.2]))

    # Encode observations
    rng, crop_rng = jax.random.split(rng)
    cond = encoder(observations, training=True, rngs={"crop": crop_rng})

    # Repeat cond G times: (B, cond_dim) -> (B*G, cond_dim)
    cond_rep = jnp.repeat(cond, G, axis=0)

    # Sample G noise vectors per obs
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, (batch_size * G, horizon, action_dim))

    # Forward pass with s=t=0 (drift has no time concept)
    s_zeros = jnp.zeros((batch_size * G,))
    pred_all = network(noise, s_zeros, s_zeros, cond_rep, training=True)

    # Reshape: (B*G, horizon, action_dim) -> (B, G, horizon, action_dim)
    pred_actions = pred_all.reshape(batch_size, G, horizon, action_dim)

    # Per-timestep drift loss: vmap over horizon dimension
    def per_timestep_fn(t_idx):
        gen_t = pred_actions[:, :, t_idx, :]            # (B, G, action_dim)
        pos_t = actions[:, t_idx, :][:, None, :]        # (B, 1, action_dim)
        loss_t, info_t = _drift_loss_single(gen_t, pos_t, R_list=R_list)
        return loss_t.mean(), info_t

    total_loss = 0.0
    accumulated_scale = 0.0
    for t in range(horizon):
        loss_t, info_t = per_timestep_fn(t)
        total_loss = total_loss + loss_t
        accumulated_scale = accumulated_scale + info_t["scale"]

    loss = total_loss / horizon
    loss = loss * config.get("loss_scale", 1.0)

    info = {
        "loss": loss,
        "drift_loss": loss,
        "drift_scale": accumulated_scale / horizon,
    }

    return loss, info


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
    elif loss_type == "mf_stable" or loss_type == "meanflow_stable":
        return mf_loss_stable
    elif loss_type == "drift" or loss_type == "drift_policy":
        return drift_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
