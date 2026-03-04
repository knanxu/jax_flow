"""MLP network for flow matching."""

import flax.linen as nn
import jax.numpy as jnp

from jax_flow.core.utils import get_activation
from jax_flow.networks.embeddings import FourierEmbedding


class MLP(nn.Module):
    """Multi-layer perceptron for flow matching.

    Takes (x, s, t, condition) and outputs velocity field.
    """

    action_dim: int
    hidden_dims: tuple[int, ...] = (512, 512, 512)
    cond_dim: int = 256
    activation: str = "gelu"
    layer_norm: bool = False
    dropout: float = 0.0
    timestep_embed_dim: int = 128

    def setup(self):
        """Setup network layers."""
        # Timestep embeddings
        self.s_embed = FourierEmbedding(embed_dim=self.timestep_embed_dim)
        self.t_embed = FourierEmbedding(embed_dim=self.timestep_embed_dim)

        # Activation function
        self.act_fn = get_activation(self.activation)

        # MLP layers
        self.layers = [nn.Dense(dim) for dim in self.hidden_dims]
        if self.layer_norm:
            self.norms = [nn.LayerNorm() for _ in self.hidden_dims]

        # Output layer
        self.output_layer = nn.Dense(self.action_dim)

    def __call__(self, x, s, t, cond, training=False):
        """Forward pass.

        Args:
            x: Current state (action). Shape: (batch, action_dim)
            s: Start time. Shape: (batch,) or (batch, 1)
            t: End time. Shape: (batch,) or (batch, 1)
            cond: Condition (observation encoding). Shape: (batch, cond_dim)
            training: Whether in training mode.

        Returns:
            Velocity field. Shape: (batch, action_dim)
        """
        # Embed timesteps
        s_emb = self.s_embed(s)  # (batch, embed_dim)
        t_emb = self.t_embed(t)  # (batch, embed_dim)

        # Concatenate inputs
        h = jnp.concatenate([x, s_emb, t_emb, cond], axis=-1)

        # MLP layers
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if self.layer_norm:
                h = self.norms[i](h)
            h = self.act_fn(h)
            if self.dropout > 0.0 and training:
                h = nn.Dropout(rate=self.dropout, deterministic=not training)(h)

        # Output layer
        velocity = self.output_layer(h)

        return velocity


def create_mlp(
    action_dim,
    hidden_dims=(512, 512, 512),
    cond_dim=256,
    activation="gelu",
    layer_norm=False,
    dropout=0.0,
):
    """Factory function to create MLP network.

    Args:
        action_dim: Action dimension.
        hidden_dims: Hidden layer dimensions.
        cond_dim: Condition dimension.
        activation: Activation function.
        layer_norm: Whether to use layer normalization.
        dropout: Dropout rate.

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
    )
