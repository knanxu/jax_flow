"""ResNet-18 image encoder in JAX/Flax.

Lightweight CNN backbone for robotic manipulation tasks.
Uses GroupNorm instead of BatchNorm for small batch stability.
Optionally uses SpatialSoftmax for compact spatial features.
"""

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp

from jax_flow.networks.encoders.spatial_softmax import SpatialSoftmax


class ResidualBlock(nn.Module):
    """Basic residual block for ResNet-18.

    Conv -> GroupNorm -> ReLU -> Conv -> GroupNorm + skip connection.
    """

    features: int
    stride: int = 1
    groups: int = 4

    @nn.compact
    def __call__(self, x):
        residual = x

        # First conv
        x = nn.Conv(self.features, (3, 3), strides=(self.stride, self.stride), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=min(self.groups, self.features))(x)
        x = nn.relu(x)

        # Second conv
        x = nn.Conv(self.features, (3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=min(self.groups, self.features))(x)

        # Downsample residual if needed
        if residual.shape != x.shape:
            residual = nn.Conv(self.features, (1, 1), strides=(self.stride, self.stride))(residual)
            residual = nn.GroupNorm(num_groups=min(self.groups, self.features))(residual)

        return nn.relu(x + residual)


class ResNet18Encoder(nn.Module):
    """ResNet-18 image encoder.

    Produces spatial features or pooled features from images.
    Uses GroupNorm instead of BatchNorm for robustness with small batches.

    Output modes:
        - spatial_softmax: (batch, num_channels * 2) keypoint coordinates
        - avg_pool: (batch, 512) global average pooled features
    """

    output_dim: int = 64
    use_spatial_softmax: bool = True
    pool_type: str = "spatial_softmax"  # "spatial_softmax" or "avg_pool"
    groups: int = 4

    @nn.compact
    def __call__(self, x):
        """Forward pass.

        Args:
            x: Image input. Shape: (batch, H, W, C), values in [0, 1].

        Returns:
            features: (batch, output_dim)
        """
        # Layer 1: conv 7x7, stride 2
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=min(self.groups, 64))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

        # Layer 2: 2 residual blocks, 64 channels
        x = ResidualBlock(64, stride=1, groups=self.groups)(x)
        x = ResidualBlock(64, stride=1, groups=self.groups)(x)

        # Layer 3: 2 residual blocks, 128 channels
        x = ResidualBlock(128, stride=2, groups=self.groups)(x)
        x = ResidualBlock(128, stride=1, groups=self.groups)(x)

        # Layer 4: 2 residual blocks, 256 channels
        x = ResidualBlock(256, stride=2, groups=self.groups)(x)
        x = ResidualBlock(256, stride=1, groups=self.groups)(x)

        # Layer 5: 2 residual blocks, 512 channels
        x = ResidualBlock(512, stride=2, groups=self.groups)(x)
        x = ResidualBlock(512, stride=1, groups=self.groups)(x)

        # Pooling
        if self.pool_type == "spatial_softmax":
            x = SpatialSoftmax()(x)  # (batch, 512 * 2)
        else:
            # Global average pooling
            x = jnp.mean(x, axis=(1, 2))  # (batch, 512)

        # Project to output_dim
        x = nn.Dense(self.output_dim)(x)

        return x
