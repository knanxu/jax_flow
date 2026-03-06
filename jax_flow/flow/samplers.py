"""Samplers for flow matching inference."""

import jax
import jax.numpy as jnp

from jax_flow.core.utils import get_batch_size


def euler_sampler(network, encoder, observations, num_steps, rng, config):
    """Euler method ODE sampler.

    Args:
        network: Flow network.
        encoder: Observation encoder.
        observations: Observations. Shape: (batch, obs_steps, obs_dim) or dict.
        num_steps: Number of ODE steps.
        rng: Random key.
        config: Configuration dict.

    Returns:
        Sampled actions. Shape: (batch, horizon, action_dim)
    """
    batch_size = get_batch_size(observations)
    horizon = config.get("horizon", 10)
    action_dim = config.get("action_dim", 7)

    # Encode observations (no crop rng needed for inference - uses center crop)
    cond = encoder(observations, training=False, rngs={})

    # Initialize from noise
    x = jax.random.normal(rng, (batch_size, horizon, action_dim))

    # Euler integration
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = jnp.full((batch_size,), step * dt)

        # Predict velocity (network handles horizon dimension)
        velocity = network(x, t, t, cond, training=False)

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
    batch_size = get_batch_size(observations)
    horizon = config.get("horizon", 10)
    action_dim = config.get("action_dim", 7)

    # Encode observations (no crop rng needed for inference - uses center crop)
    cond = encoder(observations, training=False, rngs={})

    # Initialize from noise
    x = jax.random.normal(rng, (batch_size, horizon, action_dim))

    # Heun integration
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = jnp.full((batch_size,), step * dt)
        t_next = jnp.full((batch_size,), (step + 1) * dt)

        # First velocity estimate
        v1 = network(x, t, t, cond, training=False)

        # Predictor step
        x_pred = x + v1 * dt

        # Second velocity estimate
        v2 = network(x_pred, t_next, t_next, cond, training=False)

        # Corrector step (average of two velocities)
        x = x + (v1 + v2) * 0.5 * dt

    return x


def mip_sampler(network, encoder, observations, num_steps, rng, config):
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
    batch_size = get_batch_size(observations)
    horizon = config.get("horizon", 10)
    action_dim = config.get("action_dim", 7)

    # Encode observations (no crop rng needed for inference - uses center crop)
    cond = encoder(observations, training=False, rngs={})

    # Step 1: Predict from zeros
    s0 = jnp.zeros((batch_size,))
    x0 = jnp.zeros((batch_size, horizon, action_dim))
    pred_step1 = network(x0, s0, s0, cond, training=False)

    # Step 2: Predict from step 1 output
    t_two_step = config.get("t_two_step", 0.9)
    st = jnp.full((batch_size,), t_two_step)
    pred_step2 = network(pred_step1, st, st, cond, training=False)

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
