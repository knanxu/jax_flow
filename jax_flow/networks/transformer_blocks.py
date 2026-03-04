"""Transformer building blocks for flow matching.

Pre-norm encoder-decoder blocks with multi-head attention:
- TransformerEncoderBlock: Self-attention + FFN
- TransformerDecoderBlock: Self-attention + Cross-attention + FFN
"""

import flax.linen as nn
import jax.numpy as jnp


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""

    n_head: int
    n_emb: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, q, k, v, mask=None, training=False):
        """Forward pass.

        Args:
            q: Query. Shape: (batch, seq_q, n_emb)
            k: Key. Shape: (batch, seq_k, n_emb)
            v: Value. Shape: (batch, seq_k, n_emb)
            mask: Optional attention mask.
            training: Whether in training mode.

        Returns:
            Output. Shape: (batch, seq_q, n_emb)
        """
        head_dim = self.n_emb // self.n_head
        batch_size = q.shape[0]

        # Project to heads
        q = nn.Dense(self.n_emb, name='q_proj')(q)
        k = nn.Dense(self.n_emb, name='k_proj')(k)
        v = nn.Dense(self.n_emb, name='v_proj')(v)

        # Reshape to (batch, n_head, seq, head_dim)
        q = q.reshape(batch_size, -1, self.n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.n_head, head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = head_dim ** -0.5
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale

        if mask is not None:
            attn = jnp.where(mask, attn, jnp.finfo(attn.dtype).min)

        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.dropout, deterministic=not training)(attn)

        # Combine heads
        out = jnp.matmul(attn, v)  # (batch, n_head, seq_q, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.n_emb)

        # Output projection
        out = nn.Dense(self.n_emb, name='out_proj')(out)
        return out


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer encoder block: LayerNorm → Self-Attention → LayerNorm → FFN."""

    n_head: int = 4
    n_emb: int = 256
    dropout: float = 0.1
    ffn_mult: int = 4

    @nn.compact
    def __call__(self, x, training=False):
        """Forward pass.

        Args:
            x: Input tokens. Shape: (batch, seq, n_emb)
            training: Whether in training mode.

        Returns:
            Output tokens. Shape: (batch, seq, n_emb)
        """
        # Self-attention with pre-norm
        residual = x
        x = nn.LayerNorm()(x)
        x = MultiHeadAttention(
            n_head=self.n_head, n_emb=self.n_emb, dropout=self.dropout,
            name='self_attn',
        )(x, x, x, training=training)
        x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        x = x + residual

        # FFN with pre-norm
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.n_emb * self.ffn_mult)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        x = nn.Dense(self.n_emb)(x)
        x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        x = x + residual

        return x


class TransformerDecoderBlock(nn.Module):
    """Pre-norm transformer decoder block: Self-Attention → Cross-Attention → FFN."""

    n_head: int = 4
    n_emb: int = 256
    dropout: float = 0.1
    ffn_mult: int = 4

    @nn.compact
    def __call__(self, x, encoder_out, training=False):
        """Forward pass.

        Args:
            x: Decoder tokens. Shape: (batch, seq_dec, n_emb)
            encoder_out: Encoder output. Shape: (batch, seq_enc, n_emb)
            training: Whether in training mode.

        Returns:
            Output tokens. Shape: (batch, seq_dec, n_emb)
        """
        # Self-attention with pre-norm
        residual = x
        x = nn.LayerNorm()(x)
        x = MultiHeadAttention(
            n_head=self.n_head, n_emb=self.n_emb, dropout=self.dropout,
            name='self_attn',
        )(x, x, x, training=training)
        x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        x = x + residual

        # Cross-attention with pre-norm
        residual = x
        x = nn.LayerNorm()(x)
        x = MultiHeadAttention(
            n_head=self.n_head, n_emb=self.n_emb, dropout=self.dropout,
            name='cross_attn',
        )(x, encoder_out, encoder_out, training=training)
        x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        x = x + residual

        # FFN with pre-norm
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.n_emb * self.ffn_mult)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        x = nn.Dense(self.n_emb)(x)
        x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        x = x + residual

        return x
