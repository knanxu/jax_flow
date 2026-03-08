"""Timestep embeddings for flow matching networks."""

import flax.linen as nn
import jax.numpy as jnp


class FourierEmbedding(nn.Module):
    """Uniform-frequency Fourier timestep embedding.

    Uses uniformly spaced frequencies from 0 to max_freq (much-ado-about-noising style).
    For timesteps in [0, 1], computes sin(t * freq) and cos(t * freq).
    """

    embed_dim: int = 128
    max_freq: float = 100.0

    @nn.compact
    def __call__(self, timesteps):
        """Compute Fourier embedding.

        Args:
            timesteps: Timesteps in [0, 1]. Shape: (batch,) or (batch, 1)

        Returns:
            Embeddings. Shape: (batch, embed_dim)
        """
        timesteps = jnp.atleast_1d(timesteps)
        if timesteps.ndim > 1:
            timesteps = timesteps.squeeze(-1)

        half_dim = self.embed_dim // 2
        # Uniform frequencies from 0 to max_freq
        freqs = jnp.linspace(0.0, self.max_freq, half_dim)

        angles = timesteps[:, None] * freqs[None, :]
        embedding = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)

        return embedding


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding (Diffusion Policy / Transformer style).

    Uses exponentially decaying frequencies with base 10000.
    Standard for UNet timestep conditioning.
    """

    embed_dim: int = 256

    @nn.compact
    def __call__(self, timesteps):
        timesteps = jnp.atleast_1d(timesteps)
        if timesteps.ndim > 1:
            timesteps = timesteps.squeeze(-1)

        half_dim = self.embed_dim // 2
        emb = jnp.log(10000.0) / (half_dim - 1)
        freqs = jnp.exp(jnp.arange(half_dim) * -emb)

        args = timesteps[:, None] * freqs[None, :]
        embedding = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)

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
