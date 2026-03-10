"""ResNet-18 image encoder in JAX/Flax.

Lightweight CNN backbone for robotic manipulation tasks.
Uses FrozenBatchNorm: BN statistics (mean/var) are frozen in batch_stats collection,
while scale/bias remain trainable in params. This enables ImageNet pretrained weights.
"""

import flax.linen as nn
import jax.numpy as jnp

from jax_flow.networks.encoders.spatial_softmax import SpatialSoftmax


class FrozenBatchNorm(nn.Module):
    """BatchNorm with frozen running statistics.

    mean/var live in 'batch_stats' collection (not updated by optimizer).
    scale/bias live in 'params' collection (trainable).
    """

    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        features = x.shape[-1]
        scale = self.param("scale", nn.initializers.ones, (features,))
        bias = self.param("bias", nn.initializers.zeros, (features,))
        mean = self.variable("batch_stats", "mean", jnp.zeros, (features,))
        var = self.variable("batch_stats", "var", jnp.ones, (features,))
        x = (x - mean.value) / jnp.sqrt(var.value + self.eps)
        return x * scale + bias


class ResidualBlock(nn.Module):
    """Basic residual block for ResNet-18.

    Conv -> FrozenBatchNorm -> ReLU -> Conv -> FrozenBatchNorm + skip connection.
    """

    features: int
    stride: int = 1

    @nn.compact
    def __call__(self, x):
        residual = x

        # First conv
        x = nn.Conv(
            self.features,
            (3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
        )(x)
        x = FrozenBatchNorm()(x)
        x = nn.relu(x)

        # Second conv
        x = nn.Conv(
            self.features,
            (3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = FrozenBatchNorm()(x)

        # Downsample residual if needed
        if residual.shape != x.shape:
            residual = nn.Conv(
                self.features,
                (1, 1),
                strides=(self.stride, self.stride),
                use_bias=False,
            )(residual)
            residual = FrozenBatchNorm()(residual)

        return nn.relu(x + residual)


class ResNet18Encoder(nn.Module):
    """ResNet-18 image encoder with FrozenBatchNorm.

    Architecture: stem → layer1(64) → layer2(128) → layer3(256) → SpatialSoftmax → Dense.
    No layer4(512) — stops at 256 channels for efficiency.

    Flax auto-naming:
        Conv_0 (7x7 stem) + FrozenBatchNorm_0
        ResidualBlock_0, _1  (64ch,  layer1)
        ResidualBlock_2, _3  (128ch, layer2, _2 has downsample)
        ResidualBlock_4, _5  (256ch, layer3, _4 has downsample)
        SpatialSoftmax_0
        Dense_0
    """

    output_dim: int = 64
    use_spatial_softmax: bool = True
    pool_type: str = "spatial_softmax"

    @nn.compact
    def __call__(self, x):
        """Forward pass.

        Args:
            x: Image input. Shape: (batch, H, W, C), values in [0, 1].

        Returns:
            features: (batch, output_dim)
        """
        # Stem: conv 7x7, stride 2
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = FrozenBatchNorm()(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        # Layer 1: 2 residual blocks, 64 channels
        x = ResidualBlock(64, stride=1)(x)
        x = ResidualBlock(64, stride=1)(x)

        # Layer 2: 2 residual blocks, 128 channels
        x = ResidualBlock(128, stride=2)(x)
        x = ResidualBlock(128, stride=1)(x)

        # Layer 3: 2 residual blocks, 256 channels
        x = ResidualBlock(256, stride=2)(x)
        x = ResidualBlock(256, stride=1)(x)

        # Pooling
        if self.pool_type == "spatial_softmax":
            x = SpatialSoftmax()(x)  # (batch, 256 * 2)
        else:
            x = jnp.mean(x, axis=(1, 2))  # (batch, 256)

        # Project to output_dim
        x = nn.Dense(self.output_dim)(x)

        return x
