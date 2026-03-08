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
        config: Configuration dict. Supports 'sample_mode': 'stochastic' (default) or 'zero'.

    Returns:
        Sampled actions. Shape: (batch, horizon, action_dim)
    """
    batch_size = get_batch_size(observations)
    horizon = config.get("horizon", 10)
    action_dim = config.get("action_dim", 7)
    sample_mode = config.get("sample_mode", "stochastic")

    # Encode observations
    cond = encoder(observations, training=False, rngs={})

    # Initialize from noise or zeros
    if sample_mode == "zero":
        x = jnp.zeros((batch_size, horizon, action_dim))
    else:
        x = jax.random.normal(rng, (batch_size, horizon, action_dim))

    # Euler integration from t=0 to t=1
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = jnp.full((batch_size,), step * dt)
        velocity = network(x, t, t, cond, training=False)
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
    sample_mode = config.get("sample_mode", "stochastic")

    cond = encoder(observations, training=False, rngs={})

    if sample_mode == "zero":
        x = jnp.zeros((batch_size, horizon, action_dim))
    else:
        x = jax.random.normal(rng, (batch_size, horizon, action_dim))

    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = jnp.full((batch_size,), step * dt)
        t_next = jnp.full((batch_size,), (step + 1) * dt)

        v1 = network(x, t, t, cond, training=False)
        x_pred = x + v1 * dt
        v2 = network(x_pred, t_next, t_next, cond, training=False)
        x = x + (v1 + v2) * 0.5 * dt

    return x


def mip_sampler(network, encoder, observations, num_steps, rng, config):
    """MIP (two-step) sampler.

    Args:
        network: Flow network.
        encoder: Observation encoder.
        observations: Observations.
        num_steps: Unused for MIP.
        rng: Random key (unused).
        config: Configuration dict.

    Returns:
        Sampled actions.
    """
    batch_size = get_batch_size(observations)
    horizon = config.get("horizon", 10)
    action_dim = config.get("action_dim", 7)

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


def meanflow_sampler(network, encoder, observations, num_steps, rng, config):
    """MeanFlow single-step sampler.

    Generates actions in one network evaluation:
        actions = x0 + u_theta(x0, s=0, t=1, cond)

    Args:
        network: Flow network.
        encoder: Observation encoder.
        observations: Observations.
        num_steps: Unused (always single-step).
        rng: Random key.
        config: Configuration dict.

    Returns:
        Sampled actions. Shape: (batch, horizon, action_dim)
    """
    batch_size = get_batch_size(observations)
    horizon = config.get("horizon", 10)
    action_dim = config.get("action_dim", 7)

    cond = encoder(observations, training=False, rngs={})

    x0 = jax.random.normal(rng, (batch_size, horizon, action_dim))
    s = jnp.zeros((batch_size,))
    t = jnp.ones((batch_size,))

    actions = x0 + network(x0, s, t, cond, training=False)
    return actions


def get_sampler(sampler_type="euler"):
    """Get sampler function by type."""
    if sampler_type == "euler":
        return euler_sampler
    elif sampler_type == "heun":
        return heun_sampler
    elif sampler_type == "mip":
        return mip_sampler
    elif sampler_type == "meanflow":
        return meanflow_sampler
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
