"""Observation encoders."""

import flax.linen as nn

from jax_flow.core.utils import get_activation


class IdentityEncoder(nn.Module):
    """Identity encoder (pass-through)."""

    @nn.compact
    def __call__(self, obs, training=False):
        """Forward pass.

        Args:
            obs: Observations. Shape: (batch, obs_steps, obs_dim)
            training: Whether in training mode.

        Returns:
            Encoded observations. Shape: (batch, obs_steps * obs_dim)
        """
        # Flatten observation history
        batch_size = obs.shape[0]
        return obs.reshape(batch_size, -1)


class MLPEncoder(nn.Module):
    """MLP encoder for state observations."""

    hidden_dims: tuple[int, ...] = (256, 256)
    output_dim: int = 256
    activation: str = "gelu"
    layer_norm: bool = False
    dropout: float = 0.0

    @nn.compact
    def __call__(self, obs, training=False):
        """Forward pass.

        Args:
            obs: Observations. Shape: (batch, obs_steps, obs_dim)
            training: Whether in training mode.

        Returns:
            Encoded observations. Shape: (batch, output_dim)
        """
        # Flatten observation history
        batch_size = obs.shape[0]
        x = obs.reshape(batch_size, -1)

        # Activation function
        act_fn = get_activation(self.activation)

        # MLP layers
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = act_fn(x)
            if self.dropout > 0.0 and training:
                x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)

        # Output layer
        x = nn.Dense(self.output_dim)(x)

        return x


def create_encoder(
    encoder_type="mlp",
    hidden_dims=(256, 256),
    output_dim=256,
    activation="gelu",
    layer_norm=False,
    dropout=0.0,
):
    """Factory function to create encoder.

    Args:
        encoder_type: Type of encoder ('identity' or 'mlp').
        hidden_dims: Hidden layer dimensions.
        output_dim: Output dimension.
        activation: Activation function.
        layer_norm: Whether to use layer normalization.
        dropout: Dropout rate.

    Returns:
        Encoder module.
    """
    if encoder_type == "identity":
        return IdentityEncoder()
    elif encoder_type == "mlp":
        return MLPEncoder(
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
