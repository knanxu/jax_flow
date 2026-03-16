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
    """MIP (two-step) sampler following much-ado-about-noising.

    Step 1: velocity_0 = network(zeros, s=0, s=0, cond)
    Step 2: velocity_1 = network(velocity_0, s=t_two_step, s=t_two_step, cond)
    Output: velocity_1 (the final action prediction)

    Args:
        network: Flow network.
        encoder: Observation encoder.
        observations: Observations.
        num_steps: Unused for MIP (always 2 steps).
        rng: Random key (unused).
        config: Configuration dict with 't_two_step'.

    Returns:
        Sampled actions. Shape: (batch, horizon, action_dim)
    """
    batch_size = get_batch_size(observations)
    horizon = config.get("horizon", 10)
    action_dim = config.get("action_dim", 7)
    t_two_step = config.get("t_two_step", 0.9)

    cond = encoder(observations, training=False, rngs={})

    # Step 1: predict from zeros at s=0
    s0 = jnp.zeros((batch_size,))
    x0 = jnp.zeros((batch_size, horizon, action_dim))
    act_pred_0 = network(x0, s0, s0, cond, training=False)

    # Step 2: feed step1 output as input at s=t_two_step
    st = jnp.full((batch_size,), t_two_step)
    act_pred_1 = network(act_pred_0, st, st, cond, training=False)

    return act_pred_1


def meanflow_sampler_stable(network, encoder, observations, num_steps, rng, config):
    """MeanFlow single-step sampler.

    The MeanFlow network g_θ(x_s, s, t) directly predicts data (x_1).
    At inference: actions = g_θ(x_0, s=0, t=1, cond).

    Supports two initialization modes:
    - "zero": deterministic, x_0 = 0
    - "stochastic": x_0 ~ N(0, I)

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
    sample_mode = config.get("sample_mode", "stochastic")

    cond = encoder(observations, training=False, rngs={})

    if sample_mode == "zero":
        x0 = jnp.zeros((batch_size, horizon, action_dim))
    else:
        x0 = jax.random.normal(rng, (batch_size, horizon, action_dim))

    s = jnp.zeros((batch_size,))
    t = jnp.ones((batch_size,))

    actions = network(x0, s, t, cond, training=False)
    return actions

def meanflow_sampler(network, encoder, observations, num_steps, rng, config):
    """MeanFlow single-step sampler.

    The MeanFlow network g_θ(x_s, s, t) directly predicts data (x_1).
    At inference: actions = g_θ(x_0, s=0, t=1, cond).

    Supports two initialization modes:
    - "zero": deterministic, x_0 = 0
    - "stochastic": x_0 ~ N(0, I)

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
    sample_mode = config.get("sample_mode", "stochastic")

    cond = encoder(observations, training=False, rngs={})

    if sample_mode == "zero":
        x0 = jnp.zeros((batch_size, horizon, action_dim))
    else:
        x0 = jax.random.normal(rng, (batch_size, horizon, action_dim))

    s = jnp.zeros((batch_size,))
    t = jnp.ones((batch_size,))

    actions = x0 + network(x0, s, t, cond, training=False)
    return actions
# meanflow_stable uses the same single-step sampling as meanflow



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
    elif sampler_type == "meanflow_stable":
        return meanflow_sampler_stable
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
