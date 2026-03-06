"""Observation encoders.

Encoders process raw observations into fixed-dimensional embeddings
that are used as conditioning for the flow matching network.

All encoders follow the same interface:
    Input: (batch, obs_steps, obs_dim) or (batch, obs_dim)
    Output: (batch, output_dim)
"""

import flax.linen as nn

from jax_flow.core.utils import get_activation


class IdentityEncoder(nn.Module):
    """Identity encoder that flattens observation history.

    This is the simplest encoder - it just flattens the observation
    history into a single vector. Useful for low-dimensional state
    observations where no additional processing is needed.
    """

    @nn.compact
    def __call__(self, obs, training=False):
        """Forward pass.

        Args:
            obs: Observations. Shape: (batch, obs_steps, obs_dim) or (batch, obs_dim)
            training: Whether in training mode (unused).

        Returns:
            Encoded observations. Shape: (batch, obs_steps * obs_dim) or (batch, obs_dim)
        """
        # Flatten observation history
        batch_size = obs.shape[0]
        return obs.reshape(batch_size, -1)


class MLPEncoder(nn.Module):
    """MLP encoder for state observations.

    Flattens observation history and processes through MLP layers
    to produce a fixed-dimensional embedding. This is useful when
    you want to learn a compressed representation of the observation
    history rather than just concatenating it.
    """

    hidden_dims: tuple[int, ...] = (256, 256)
    output_dim: int = 256
    activation: str = "gelu"
    layer_norm: bool = False
    dropout: float = 0.0

    @nn.compact
    def __call__(self, obs, training=False):
        """Forward pass.

        Args:
            obs: Observations. Shape: (batch, obs_steps, obs_dim) or (batch, obs_dim)
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
    # Image encoder kwargs
    image_keys=('agentview_image',),
    lowdim_keys=('robot0_eef_pos', 'robot0_gripper_qpos'),
    share_image_encoder=False,
    use_spatial_softmax=True,
    crop_shape=None,
):
    """Factory function to create encoder.

    Args:
        encoder_type: Type of encoder ('identity', 'mlp', 'resnet', 'multi_image').
        hidden_dims: Hidden layer dimensions (for MLP encoder).
        output_dim: Output dimension.
        activation: Activation function (for MLP encoder).
        layer_norm: Whether to use layer normalization (for MLP encoder).
        dropout: Dropout rate (for MLP encoder).
        image_keys: Image observation keys (for image encoders).
        lowdim_keys: Low-dim observation keys (for multi_image encoder).
        share_image_encoder: Share backbone across cameras (for multi_image).
        use_spatial_softmax: Use spatial softmax pooling (for image encoders).

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
    elif encoder_type == "resnet":
        from jax_flow.networks.encoders.resnet import ResNet18Encoder
        return ResNet18Encoder(
            output_dim=output_dim,
            use_spatial_softmax=use_spatial_softmax,
            pool_type="spatial_softmax" if use_spatial_softmax else "avg_pool",
        )
    elif encoder_type == "multi_image":
        from jax_flow.networks.encoders.multi_image import MultiImageEncoder
        return MultiImageEncoder(
            image_keys=image_keys,
            lowdim_keys=lowdim_keys,
            image_encoder_output_dim=output_dim,
            share_image_encoder=share_image_encoder,
            use_spatial_softmax=use_spatial_softmax,
            crop_shape=crop_shape,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
