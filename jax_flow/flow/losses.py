"""Flow matching loss functions."""

import jax
import jax.numpy as jnp


def _predict_over_horizon(network, x_t, s, t, cond, horizon, training=True):
    """Predict velocity over horizon dimension using vmap.

    Args:
        network: Flow network.
        x_t: Current state. Shape: (batch, horizon, action_dim)
        s: Start time. Shape: (batch,)
        t: End time. Shape: (batch,)
        cond: Condition. Shape: (batch, cond_dim)
        horizon: Horizon length.
        training: Whether in training mode.

    Returns:
        Predicted velocities. Shape: (batch, horizon, action_dim)
    """
    # Vectorize over horizon dimension
    def predict_single(x_single):
        return network(x_single, s, t, cond, training=training)

    # vmap over axis 1 (horizon dimension)
    return jax.vmap(predict_single, in_axes=1, out_axes=1)(x_t)


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
    observations = batch["observations"]  # (batch, obs_steps, obs_dim)
    actions = batch["actions"]  # (batch, horizon, action_dim)

    batch_size = observations.shape[0]
    horizon = actions.shape[1]
    action_dim = actions.shape[2]

    # Sample random timesteps
    rng, t_rng = jax.random.split(rng)
    t = jax.random.uniform(t_rng, (batch_size,), minval=0.0, maxval=1.0)

    # Sample noise
    rng, noise_rng = jax.random.split(rng)
    x0 = jax.random.normal(noise_rng, (batch_size, horizon, action_dim))

    # Data
    x1 = actions

    # Encode observations
    cond = encoder(observations, training=True)  # (batch, cond_dim)

    # Interpolate
    x_t = interpolant.interpolate(t, x0, x1)  # (batch, horizon, action_dim)

    # Compute target velocity
    target_velocity = interpolant.velocity(t, x0, x1)  # (batch, horizon, action_dim)

    # Predict velocity for each action step (vectorized)
    predicted_velocity = _predict_over_horizon(
        network, x_t, t, t, cond, horizon, training=True
    )

    # Compute loss
    loss = jnp.mean((predicted_velocity - target_velocity) ** 2)
    loss = loss * config.get("loss_scale", 1.0)

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

    batch_size = observations.shape[0]
    horizon = actions.shape[1]
    action_dim = actions.shape[2]

    t_two_step = config.get("t_two_step", 0.9)

    # Encode observations
    cond = encoder(observations, training=True)

    # Step 1: s=0, predict from zeros (vectorized)
    s0 = jnp.zeros((batch_size,))
    x0 = jnp.zeros((batch_size, horizon, action_dim))
    pred_step1 = _predict_over_horizon(network, x0, s0, s0, cond, horizon, training=True)

    # Step 2: s=t_two_step, predict from noisy target (vectorized)
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, (batch_size, horizon, action_dim))
    x_t = actions + (1 - t_two_step) * noise

    st = jnp.full((batch_size,), t_two_step)
    pred_step2 = _predict_over_horizon(network, x_t, st, st, cond, horizon, training=True)

    # Compute losses
    loss1 = jnp.mean((pred_step1 - actions) ** 2) / (t_two_step**2)
    loss2 = jnp.mean((pred_step2 - actions) ** 2) / ((1 - t_two_step) ** 2)

    loss = (loss1 + loss2) * config.get("loss_scale", 1.0)

    info = {
        "loss": loss,
        "mip_loss": loss,
        "loss1": loss1,
        "loss2": loss2,
    }

    return loss, info


def mf_loss(network, encoder, interpolant, batch, rng, config):
    """Mean Flow (MF) loss.

    Combines standard flow matching with mean flow regularization.

    Args:
        network: Flow network.
        encoder: Observation encoder.
        interpolant: Interpolant.
        batch: Batch of data.
        rng: Random key.
        config: Configuration dict.

    Returns:
        Tuple of (loss, info_dict).
    """
    observations = batch["observations"]
    actions = batch["actions"]

    batch_size = observations.shape[0]
    horizon = actions.shape[1]
    action_dim = actions.shape[2]

    # Encode observations
    cond = encoder(observations, training=True)

    # ========== Standard flow matching loss ==========
    rng, t_rng = jax.random.split(rng)
    t_flow = jax.random.uniform(t_rng, (batch_size,), minval=0.0, maxval=1.0)

    rng, noise_rng = jax.random.split(rng)
    x0 = jax.random.normal(noise_rng, (batch_size, horizon, action_dim))
    x1 = actions

    x_t = interpolant.interpolate(t_flow, x0, x1)
    target_velocity = interpolant.velocity(t_flow, x0, x1)

    # Predict velocities (vectorized)
    pred_velocity = _predict_over_horizon(
        network, x_t, t_flow, t_flow, cond, horizon, training=True
    )

    flow_matching_loss = jnp.mean((pred_velocity - target_velocity) ** 2)

    # ========== Mean flow term ==========
    # Compute mean flow regularization (simplified version)
    # This is a placeholder - full MF loss requires JVP computation
    # For now, just use flow matching loss
    mf_term = jnp.zeros_like(flow_matching_loss)

    total_loss = (flow_matching_loss + mf_term) * config.get("loss_scale", 1.0)

    info = {
        "loss": total_loss,
        "flow_loss": flow_matching_loss,
        "mf_term": mf_term,
    }

    return total_loss, info


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
