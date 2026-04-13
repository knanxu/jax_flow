"""MinViT image encoder.

Lightweight Vision Transformer for robotic manipulation tasks.
Uses double-conv patch embedding and pre-norm transformer blocks.
Supports both pooled output (for BC pipeline) and patch output.
"""

import flax.linen as nn
import jax.numpy as jnp

from jax_flow.networks.transformer_blocks import MultiHeadAttention


class PatchEmbedding(nn.Module):
    """Double-conv patch embedding.

    Conv(3→embed_dim, k=8, s=4) → GroupNorm → ReLU → Conv(embed_dim→embed_dim, k=3, s=2)
    For 84×84 input with SAME padding: 84→21→11 → 121 patches.
    """

    embed_dim: int = 128

    @nn.compact
    def __call__(self, x):
        """Embed image patches.

        Args:
            x: Image input. Shape: (batch, H, W, C), values in [0, 1].

        Returns:
            Patch embeddings. Shape: (batch, num_patches, embed_dim).
        """
        # Normalize to [-0.5, 0.5]
        x = x - 0.5

        # First conv: downsample 4x
        x = nn.Conv(self.embed_dim, (8, 8), strides=(4, 4), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=min(8, self.embed_dim))(x)
        x = nn.relu(x)

        # Second conv: downsample 2x
        x = nn.Conv(self.embed_dim, (3, 3), strides=(2, 2), padding="SAME")(x)

        # Flatten spatial dims → (batch, num_patches, embed_dim)
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1, self.embed_dim)


class ViTBlock(nn.Module):
    """Pre-norm Transformer encoder block.

    LayerNorm → SelfAttn → residual → LayerNorm → FFN → residual.
    Reuses MultiHeadAttention from transformer_blocks.py.
    """

    num_heads: int = 4
    embed_dim: int = 128
    ffn_mult: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, training=False):
        """Forward pass.

        Args:
            x: Input tokens. Shape: (batch, seq, embed_dim).
            training: Whether in training mode.

        Returns:
            Output tokens. Shape: (batch, seq, embed_dim).
        """
        # Self-attention with pre-norm
        residual = x
        x = nn.LayerNorm()(x)
        x = MultiHeadAttention(
            n_head=self.num_heads,
            n_emb=self.embed_dim,
            dropout=self.dropout,
            name="self_attn",
        )(x, x, x, training=training)
        x = x + residual

        # FFN with pre-norm
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.embed_dim * self.ffn_mult)(x)
        x = nn.gelu(x)
        if self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        x = nn.Dense(self.embed_dim)(x)
        if self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        x = x + residual

        return x


class MinViTEncoder(nn.Module):
    """Lightweight ViT image encoder.

    PatchEmbedding → learnable pos_emb → ViTBlocks → LayerNorm.

    Two output modes:
        - return_patches=False: mean pool → Dense(output_dim) → (batch, output_dim)
        - return_patches=True: (batch, num_patches, embed_dim)
    """

    embed_dim: int = 128
    num_heads: int = 4
    depth: int = 1
    ffn_mult: int = 4
    dropout: float = 0.0
    output_dim: int = 64
    return_patches: bool = False
    pool_type: str = "mean"

    @nn.compact
    def __call__(self, x, training=False):
        """Forward pass.

        Args:
            x: Image input. Shape: (batch, H, W, C), values in [0, 1].
            training: Whether in training mode.

        Returns:
            If return_patches=True: (batch, num_patches, embed_dim)
            If return_patches=False: (batch, output_dim)
        """
        # Patch embedding
        x = PatchEmbedding(embed_dim=self.embed_dim)(x)
        num_patches = x.shape[1]

        # Learnable positional embedding
        pos_emb = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (1, num_patches, self.embed_dim),
        )
        x = x + pos_emb

        # Transformer blocks
        for i in range(self.depth):
            x = ViTBlock(
                num_heads=self.num_heads,
                embed_dim=self.embed_dim,
                ffn_mult=self.ffn_mult,
                dropout=self.dropout,
                name=f"block_{i}",
            )(x, training=training)

        # Final layer norm
        x = nn.LayerNorm()(x)

        if self.return_patches:
            return x  # (batch, num_patches, embed_dim)

        # Pool and project
        x = jnp.mean(x, axis=1)  # (batch, embed_dim)
        x = nn.Dense(self.output_dim)(x)
        return x
