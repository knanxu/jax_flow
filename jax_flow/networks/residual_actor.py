"""Residual actor network for RL fine-tuning.

Deterministic policy that outputs small residual corrections to a frozen BC policy.
Key design choices from ResFiT:
- Tanh output * action_scale (default 0.1) to limit residual magnitude
- Last layer zero-initialized so initial behavior = BC policy exactly
- TD3-style deterministic policy gradient (not SAC)
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


class ResidualActor(nn.Module):
    """Deterministic residual actor.

    Architecture: Dense → LayerNorm → ReLU × num_layers → Dense → Tanh × action_scale.

    The actor takes encoded features (or raw obs) + base_action as input
    and outputs a small residual action in [-action_scale, action_scale].
    """

    action_dim: int
    hidden_dim: int = 1024
    num_layers: int = 2
    action_scale: float = 0.1
    layer_norm: bool = True

    @nn.compact
    def __call__(self, features, base_action):
        """Forward pass.

        Args:
            features: Encoded observation features. Shape: (batch, feat_dim).
            base_action: Frozen BC policy action. Shape: (batch, action_dim).

        Returns:
            Residual action in [-action_scale, action_scale]. Shape: (batch, action_dim).
        """
        x = jnp.concatenate([features, base_action], axis=-1)

        for i in range(self.num_layers):
            x = nn.Dense(self.hidden_dim, name=f"fc_{i}")(x)
            if self.layer_norm:
                x = nn.LayerNorm(name=f"ln_{i}")(x)
            x = nn.relu(x)

        # Last layer with zero initialization
        x = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="output",
        )(x)
        x = nn.tanh(x)

        return x * self.action_scale


def add_exploration_noise(
    action: jnp.ndarray,
    rng: jax.Array,
    stddev: float,
    clip: float = 0.3,
) -> jnp.ndarray:
    """TD3-style clipped Gaussian exploration noise.

    Args:
        action: Deterministic action (batch, action_dim)
        rng: JAX PRNG key
        stddev: Noise standard deviation (ResFiT default: 0.05)
        clip: Noise clipping range (default: 0.3)

    Returns:
        Noisy action clipped to [-1, 1]
    """
    noise = jax.random.normal(rng, action.shape) * stddev
    noise = jnp.clip(noise, -clip, clip)
    return jnp.clip(action + noise, -1.0, 1.0)
