"""Transformer for flow matching with dual time conditioning.

TransformerForFlow uses encoder-decoder architecture with cross-attention:
- Encoder processes [time, obs] as conditioning
- Decoder processes action sequence with cross-attention to encoder
- Dual timestep embeddings (s, t) for flow matching
"""

import flax.linen as nn
import jax.numpy as jnp

from jax_flow.networks.embeddings import FourierEmbedding
from jax_flow.networks.transformer_blocks import (
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)


class TransformerForFlow(nn.Module):
    """Transformer for flow matching with dual time conditioning.

    Network signature: (at, s, t, obs) -> velocity
    - at: (batch, horizon, action_dim)
    - s, t: (batch,) source/target timesteps
    - obs: (batch, cond_dim) observation encoding
    - velocity: (batch, horizon, action_dim)

    Architecture:
    - Encoder: Process [time_emb, obs_emb] as conditioning tokens
    - Decoder: Process action sequence with cross-attention to encoder
    - Learnable positional embeddings for both encoder and decoder
    """

    action_dim: int
    n_layer: int = 4
    n_head: int = 4
    n_emb: int = 256
    cond_dim: int = 256
    dropout: float = 0.1
    timestep_embed_dim: int = 128

    @nn.compact
    def __call__(self, at, s, t, obs, training=False):
        """Forward pass.

        Args:
            at: Current action state. Shape: (batch, horizon, action_dim)
            s: Source time. Shape: (batch,)
            t: Target time. Shape: (batch,)
            obs: Observation encoding. Shape: (batch, cond_dim)
            training: Whether in training mode.

        Returns:
            Velocity field. Shape: (batch, horizon, action_dim)
        """
        batch_size, horizon, action_dim = at.shape

        # Dual timestep embedding for flow matching
        s_emb = FourierEmbedding(embed_dim=self.timestep_embed_dim // 2, name='s_embed')(s)
        t_emb = FourierEmbedding(embed_dim=self.timestep_embed_dim // 2, name='t_embed')(t)
        time_emb = jnp.concatenate([s_emb, t_emb], axis=-1)  # (B, timestep_embed_dim)
        time_emb = time_emb[:, None, :]  # (B, 1, timestep_embed_dim)

        # Observation embedding
        obs_emb = obs[:, None, :]  # (B, 1, cond_dim)

        # Project each token to n_emb before concatenating along seq dim
        time_token = nn.Dense(self.n_emb, name='time_proj')(time_emb)   # (B, 1, n_emb)
        obs_token = nn.Dense(self.n_emb, name='obs_proj')(obs_emb)      # (B, 1, n_emb)

        # Encoder: process [time, obs] as conditioning tokens
        encoder_tokens = jnp.concatenate([time_token, obs_token], axis=1)  # (B, 2, n_emb)

        # Add learnable positional embeddings
        encoder_pos_emb = self.param(
            'encoder_pos_emb',
            nn.initializers.normal(stddev=0.02),
            (1, 2, self.n_emb)
        )
        encoder_tokens = encoder_tokens + encoder_pos_emb

        # Encoder layers
        for i in range(self.n_layer):
            encoder_tokens = TransformerEncoderBlock(
                n_head=self.n_head,
                n_emb=self.n_emb,
                dropout=self.dropout,
                name=f'encoder_block_{i}',
            )(encoder_tokens, training=training)

        # Final encoder layer norm
        encoder_tokens = nn.LayerNorm(name='encoder_ln')(encoder_tokens)

        # Decoder: process action sequence with cross-attention to encoder
        # Project actions to n_emb dimension
        action_tokens = nn.Dense(self.n_emb, name='decoder_input_proj')(at)  # (B, horizon, n_emb)

        # Add learnable positional embeddings
        decoder_pos_emb = self.param(
            'decoder_pos_emb',
            nn.initializers.normal(stddev=0.02),
            (1, horizon, self.n_emb)
        )
        action_tokens = action_tokens + decoder_pos_emb

        # Decoder layers with cross-attention
        for i in range(self.n_layer):
            action_tokens = TransformerDecoderBlock(
                n_head=self.n_head,
                n_emb=self.n_emb,
                dropout=self.dropout,
                name=f'decoder_block_{i}',
            )(action_tokens, encoder_tokens, training=training)

        # Final decoder layer norm
        action_tokens = nn.LayerNorm(name='decoder_ln')(action_tokens)

        # Output projection to action_dim
        velocity = nn.Dense(action_dim, name='output_proj')(action_tokens)

        return velocity
