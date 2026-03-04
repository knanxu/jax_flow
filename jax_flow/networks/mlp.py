"""MLP network for flow matching."""

import flax.linen as nn
import jax.numpy as jnp

from jax_flow.core.utils import get_activation
from jax_flow.networks.embeddings import FourierEmbedding


class MLP(nn.Module):
    """Multi-layer perceptron for flow matching.

    Takes (at, s, t, obs) and outputs velocity field.
    Supports action sequences with horizon dimension.

    Network signature: (at, s, t, obs) -> velocity
    - at: (batch, horizon, action_dim) - action sequence
    - s: (batch,) - start time
    - t: (batch,) - end time
    - obs: (batch, cond_dim) - observation encoding
    - velocity: (batch, horizon, action_dim) - predicted velocity field
    """

    action_dim: int
    hidden_dims: tuple[int, ...] = (512, 512, 512)
    cond_dim: int = 256
    activation: str = "gelu"
    layer_norm: bool = False
    dropout: float = 0.0
    timestep_embed_dim: int = 128

    @nn.compact
    def __call__(self, at, s, t, obs, training=False):
        """Forward pass.

        Args:
            at: Current action state. Shape: (batch, horizon, action_dim)
            s: Start time. Shape: (batch,) or (batch, 1)
            t: End time. Shape: (batch,) or (batch, 1)
            obs: Observation encoding. Shape: (batch, cond_dim)
            training: Whether in training mode.

        Returns:
            Velocity field. Shape: (batch, horizon, action_dim)
        """
        batch_size, horizon, action_dim = at.shape
        act_fn = get_activation(self.activation)

        # Embed timesteps
        s_emb = FourierEmbedding(embed_dim=self.timestep_embed_dim, name='s_embed')(s)
        t_emb = FourierEmbedding(embed_dim=self.timestep_embed_dim, name='t_embed')(t)

        # Flatten action sequence and concatenate all inputs
        at_flat = at.reshape(batch_size, -1)  # (batch, horizon * action_dim)
        h = jnp.concatenate([at_flat, s_emb, t_emb, obs], axis=-1)

        # MLP layers
        for i, dim in enumerate(self.hidden_dims):
            h = nn.Dense(dim, name=f'dense_{i}')(h)
            if self.layer_norm:
                h = nn.LayerNorm(name=f'ln_{i}')(h)
            h = act_fn(h)
            if self.dropout > 0.0 and training:
                h = nn.Dropout(rate=self.dropout, deterministic=not training)(h)

        # Output layer
        output_dim = horizon * action_dim
        velocity_flat = nn.Dense(output_dim, name='output')(h)

        # Reshape to (batch, horizon, action_dim)
        velocity = velocity_flat.reshape(batch_size, horizon, action_dim)

        return velocity


def create_mlp(
    action_dim,
    hidden_dims=(512, 512, 512),
    cond_dim=256,
    activation="gelu",
    layer_norm=False,
    dropout=0.0,
    timestep_embed_dim=128,
):
    """Factory function to create MLP network.

    Args:
        action_dim: Action dimension (per timestep).
        hidden_dims: Hidden layer dimensions.
        cond_dim: Condition dimension (observation encoding).
        activation: Activation function.
        layer_norm: Whether to use layer normalization.
        dropout: Dropout rate.
        timestep_embed_dim: Timestep embedding dimension.

    Returns:
        MLP module.
    """
    return MLP(
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        cond_dim=cond_dim,
        activation=activation,
        layer_norm=layer_norm,
        dropout=dropout,
        timestep_embed_dim=timestep_embed_dim,
    )
