"""Samplers for flow matching inference."""

import jax
import jax.numpy as jnp


def _predict_over_horizon(network, x, s, t, cond, training=False):
    """Predict velocity over horizon dimension using vmap.

    Args:
        network: Flow network.
        x: Current state. Shape: (batch, horizon, action_dim)
        s: Start time. Shape: (batch,)
        t: End time. Shape: (batch,)
        cond: Condition. Shape: (batch, cond_dim)
        training: Whether in training mode.

    Returns:
        Predicted velocities. Shape: (batch, horizon, action_dim)
    """
    # Vectorize over horizon dimension
    def predict_single(x_single):
        return network(x_single, s, t, cond, training=training)

    # vmap over axis 1 (horizon dimension)
    return jax.vmap(predict_single, in_axes=1, out_axes=1)(x)


def euler_sampler(network, encoder, observations, num_steps, rng, config):
    """Euler method ODE sampler.

    Args:
        network: Flow network.
        encoder: Observation encoder.
        observations: Observations. Shape: (batch, obs_steps, obs_dim)
        num_steps: Number of ODE steps.
        rng: Random key.
        config: Configuration dict.

    Returns:
        Sampled actions. Shape: (batch, horizon, action_dim)
    """
    batch_size = observations.shape[0]
    horizon = config.get("horizon", 10)
    action_dim = config.get("action_dim", 7)

    # Encode observations
    cond = encoder(observations, training=False)

    # Initialize from noise
    x = jax.random.normal(rng, (batch_size, horizon, action_dim))

    # Euler integration
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = jnp.full((batch_size,), step * dt)

        # Predict velocity (vectorized over horizon)
        velocity = _predict_over_horizon(network, x, t, t, cond, training=False)

        # Euler step
        x = x + velocity * dt

    return x


def heun_sampler(network, encoder, observations, num_steps, rng, config):
    """Heun's method (2nd order) ODE sampler.

    Args:
        network: Flow network.
        encoder: Observation encoder.
        observations: Observations.
        num_steps: Number of ODE steps.
        rng: Random key.
        config: Configuration dict.

    Returns:
        Sampled actions.
    """
    batch_size = observations.shape[0]
    horizon = config.get("horizon", 10)
    action_dim = config.get("action_dim", 7)

    # Encode observations
    cond = encoder(observations, training=False)

    # Initialize from noise
    x = jax.random.normal(rng, (batch_size, horizon, action_dim))

    # Heun integration
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = jnp.full((batch_size,), step * dt)
        t_next = jnp.full((batch_size,), (step + 1) * dt)

        # First velocity estimate (vectorized)
        v1 = _predict_over_horizon(network, x, t, t, cond, training=False)

        # Predictor step
        x_pred = x + v1 * dt

        # Second velocity estimate (vectorized)
        v2 = _predict_over_horizon(network, x_pred, t_next, t_next, cond, training=False)

        # Corrector step (average of two velocities)
        x = x + (v1 + v2) * 0.5 * dt

    return x


def mip_sampler(network, encoder, observations, rng, config):
    """MIP (two-step) sampler.

    Args:
        network: Flow network.
        encoder: Observation encoder.
        observations: Observations.
        rng: Random key (unused for MIP).
        config: Configuration dict.

    Returns:
        Sampled actions.
    """
    batch_size = observations.shape[0]
    horizon = config.get("horizon", 10)
    action_dim = config.get("action_dim", 7)

    # Encode observations
    cond = encoder(observations, training=False)

    # Step 1: Predict from zeros (vectorized)
    s0 = jnp.zeros((batch_size,))
    x0 = jnp.zeros((batch_size, horizon, action_dim))
    pred_step1 = _predict_over_horizon(network, x0, s0, s0, cond, training=False)

    # Step 2: Predict from step 1 output (vectorized)
    t_two_step = config.get("t_two_step", 0.9)
    st = jnp.full((batch_size,), t_two_step)
    pred_step2 = _predict_over_horizon(network, pred_step1, st, st, cond, training=False)

    return pred_step2


def get_sampler(sampler_type="euler"):
    """Get sampler function by type.

    Args:
        sampler_type: Type of sampler ('euler', 'heun', 'mip').

    Returns:
        Sampler function.
    """
    if sampler_type == "euler":
        return euler_sampler
    elif sampler_type == "heun":
        return heun_sampler
    elif sampler_type == "mip":
        return mip_sampler
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
