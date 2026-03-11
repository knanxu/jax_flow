"""MLP network for flow matching with residual blocks.

Architecture follows IDQLMlp / much-ado-about-noising pattern:
- Input projection: concat(action_flat, s_emb, t_emb, obs) -> Dense -> emb_dim
- N residual blocks: LayerNorm -> Dense(d, 4d) -> GELU -> Dropout -> Dense(4d, d) -> Dropout + residual
- Output: LayerNorm -> Dense(emb_dim, horizon * action_dim)
"""

import flax.linen as nn
import jax.numpy as jnp

from jax_flow.networks.embeddings import FourierEmbedding


class MLPResidualBlock(nn.Module):
    """Residual block with pre-norm LayerNorm and expansion factor.

    Structure: LayerNorm -> Dense(d -> expansion*d) -> GELU -> Dropout
               -> Dense(expansion*d -> d) -> Dropout -> + residual
    """

    dim: int
    expansion_factor: int = 4
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, training=False):
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.dim * self.expansion_factor)(x)
        x = nn.gelu(x)
        if self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        x = nn.Dense(self.dim)(x)
        if self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        return x + residual


class MLP(nn.Module):
    """MLP for flow matching with residual blocks.

    Network signature: (at, s, t, obs) -> velocity
    - at: (batch, horizon, action_dim)
    - s, t: (batch,) timesteps
    - obs: (batch, cond_dim) observation encoding
    - velocity: (batch, horizon, action_dim)
    """

    action_dim: int
    emb_dim: int = 512
    n_blocks: int = 6
    expansion_factor: int = 4
    dropout: float = 0.1
    timestep_embed_dim: int = 128
    max_freq: float = 100.0
    # Legacy compat: these are ignored but accepted so old configs don't break
    hidden_dims: tuple[int, ...] = ()
    cond_dim: int = 0
    activation: str = "gelu"
    layer_norm: bool = False

    @nn.compact
    def __call__(self, at, s, t, obs, training=False):
        """Forward pass.

        Args:
            at: Current action state. Shape: (batch, horizon, action_dim)
            s: Start time. Shape: (batch,)
            t: End time. Shape: (batch,)
            obs: Observation encoding. Shape: (batch, cond_dim)
            training: Whether in training mode.

        Returns:
            Velocity field. Shape: (batch, horizon, action_dim)
        """
        batch_size, horizon, action_dim = at.shape

        # Embed timesteps with uniform-frequency Fourier features
        s_emb = FourierEmbedding(
            embed_dim=self.timestep_embed_dim, max_freq=self.max_freq, name="s_embed"
        )(s)
        t_emb = FourierEmbedding(
            embed_dim=self.timestep_embed_dim, max_freq=self.max_freq, name="t_embed"
        )(t)

        # Flatten action sequence and concatenate all inputs
        at_flat = at.reshape(batch_size, -1)  # (batch, horizon * action_dim)
        h = jnp.concatenate([at_flat, s_emb, t_emb, obs], axis=-1)

        # Input projection
        h = nn.Dense(self.emb_dim, name="input_proj")(h)
        h = nn.LayerNorm(name="input_ln")(h)
        h = nn.gelu(h)

        # Residual blocks
        for i in range(self.n_blocks):
            h = MLPResidualBlock(
                dim=self.emb_dim,
                expansion_factor=self.expansion_factor,
                dropout=self.dropout,
                name=f"block_{i}",
            )(h, training=training)

        # Output projection
        h = nn.LayerNorm(name="output_ln")(h)
        output_dim = horizon * action_dim
        velocity_flat = nn.Dense(output_dim, name="output_proj")(h)

        velocity = velocity_flat.reshape(batch_size, horizon, action_dim)
        return velocity


class SmallMLP(nn.Module):
    """Lightweight MLP for flow matching, following qc/ACFQL pattern.

    Simple stacked Dense layers with GELU activation — no residual blocks,
    no expansion factor, no dropout. Much faster for testing and small tasks.

    Network signature: (at, s, t, obs) -> velocity
    """

    action_dim: int
    hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    timestep_embed_dim: int = 128
    max_freq: float = 100.0
    layer_norm: bool = True

    @nn.compact
    def __call__(self, at, s, t, obs, training=False):
        batch_size, horizon, action_dim = at.shape

        # Embed timesteps
        s_emb = FourierEmbedding(
            embed_dim=self.timestep_embed_dim, max_freq=self.max_freq, name="s_embed"
        )(s)
        t_emb = FourierEmbedding(
            embed_dim=self.timestep_embed_dim, max_freq=self.max_freq, name="t_embed"
        )(t)

        # Flatten and concatenate all inputs
        at_flat = at.reshape(batch_size, -1)
        h = jnp.concatenate([at_flat, s_emb, t_emb, obs], axis=-1)

        # Stacked Dense + GELU (qc pattern)
        for i, dim in enumerate(self.hidden_dims):
            h = nn.Dense(dim, name=f"fc_{i}")(h)
            h = nn.gelu(h)
            if self.layer_norm:
                h = nn.LayerNorm(name=f"ln_{i}")(h)

        # Output projection (no activation)
        output_dim = horizon * action_dim
        velocity_flat = nn.Dense(output_dim, name="output_proj")(h)

        return velocity_flat.reshape(batch_size, horizon, action_dim)


def create_mlp(
    action_dim,
    emb_dim=512,
    n_blocks=6,
    expansion_factor=4,
    dropout=0.1,
    timestep_embed_dim=128,
    max_freq=100.0,
    **kwargs,
):
    """Factory function to create MLP network."""
    return MLP(
        action_dim=action_dim,
        emb_dim=emb_dim,
        n_blocks=n_blocks,
        expansion_factor=expansion_factor,
        dropout=dropout,
        timestep_embed_dim=timestep_embed_dim,
        max_freq=max_freq,
    )
