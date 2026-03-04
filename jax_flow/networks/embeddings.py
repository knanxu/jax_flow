"""Timestep embeddings for flow matching networks."""

import flax.linen as nn
import jax.numpy as jnp


class FourierEmbedding(nn.Module):
    """Fourier (sinusoidal) timestep embedding.

    Used in Diffusion Policy and other flow matching methods.
    """

    embed_dim: int = 128
    max_freq: float = 1000.0

    @nn.compact
    def __call__(self, timesteps):
        """Compute Fourier embedding.

        Args:
            timesteps: Timesteps in [0, 1]. Shape: (batch,) or (batch, 1)

        Returns:
            Embeddings. Shape: (batch, embed_dim)
        """
        # Ensure timesteps is 1D
        timesteps = jnp.atleast_1d(timesteps)
        if timesteps.ndim > 1:
            timesteps = timesteps.squeeze(-1)

        # Compute frequencies
        half_dim = self.embed_dim // 2
        freqs = jnp.exp(-jnp.log(self.max_freq) * jnp.arange(half_dim) / half_dim)

        # Compute embeddings
        args = timesteps[:, None] * freqs[None, :]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

        return embedding


class LearnedEmbedding(nn.Module):
    """Learned timestep embedding (simple linear projection)."""

    embed_dim: int = 128

    @nn.compact
    def __call__(self, timesteps):
        """Compute learned embedding.

        Args:
            timesteps: Timesteps in [0, 1]. Shape: (batch,) or (batch, 1)

        Returns:
            Embeddings. Shape: (batch, embed_dim)
        """
        # Ensure timesteps is 2D
        timesteps = jnp.atleast_1d(timesteps)
        if timesteps.ndim == 1:
            timesteps = timesteps[:, None]

        # Linear projection
        x = nn.Dense(self.embed_dim)(timesteps)
        x = nn.gelu(x)
        x = nn.Dense(self.embed_dim)(x)

        return x


def create_timestep_embedding(embed_type="fourier", embed_dim=128):
    """Factory function to create timestep embedding.

    Args:
        embed_type: Type of embedding ('fourier' or 'learned').
        embed_dim: Embedding dimension.

    Returns:
        Embedding module.
    """
    if embed_type == "fourier":
        return FourierEmbedding(embed_dim=embed_dim)
    elif embed_type == "learned":
        return LearnedEmbedding(embed_dim=embed_dim)
    else:
        raise ValueError(f"Unknown embedding type: {embed_type}")
