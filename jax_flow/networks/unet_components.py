"""1D UNet building blocks for action sequence prediction.

Components for ConditionalUnet1D following Diffusion Policy's architecture:
- Conv1dBlock: Conv1d → GroupNorm → Mish
- Downsample1d / Upsample1d: Strided convolutions for resolution changes
- ConditionalResidualBlock1D: FiLM-conditioned residual block
"""

import flax.linen as nn
import jax.numpy as jnp


class Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish activation.

    Operates on (batch, channels, horizon) layout.
    Uses flax Conv with feature_group_count for grouped normalization.
    """

    out_channels: int
    kernel_size: int = 5
    n_groups: int = 8

    @nn.compact
    def __call__(self, x):
        """Forward pass.

        Args:
            x: Input tensor. Shape: (batch, channels, horizon)

        Returns:
            Output tensor. Shape: (batch, out_channels, horizon)
        """
        # Transpose to (batch, horizon, channels) for flax Conv
        x = jnp.transpose(x, (0, 2, 1))
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,),
            padding="SAME",
        )(x)
        # Transpose back to (batch, channels, horizon)
        x = jnp.transpose(x, (0, 2, 1))

        # GroupNorm operates on (batch, ..., channels), transpose for it
        x = jnp.transpose(x, (0, 2, 1))  # (batch, horizon, channels)
        x = nn.GroupNorm(num_groups=self.n_groups)(x)
        x = jnp.transpose(x, (0, 2, 1))  # (batch, channels, horizon)

        # Mish activation: x * tanh(softplus(x))
        x = x * jnp.tanh(nn.softplus(x))
        return x


class Downsample1d(nn.Module):
    """Downsample by factor of 2 using strided convolution."""

    out_channels: int

    @nn.compact
    def __call__(self, x):
        """Forward pass.

        Args:
            x: Input tensor. Shape: (batch, channels, horizon)

        Returns:
            Output tensor. Shape: (batch, out_channels, horizon // 2)
        """
        # (batch, channels, horizon) -> (batch, horizon, channels)
        x = jnp.transpose(x, (0, 2, 1))
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(3,),
            strides=(2,),
            padding="SAME",
        )(x)
        # (batch, horizon//2, channels) -> (batch, channels, horizon//2)
        x = jnp.transpose(x, (0, 2, 1))
        return x


class Upsample1d(nn.Module):
    """Upsample by factor of 2 using transposed convolution."""

    out_channels: int

    @nn.compact
    def __call__(self, x):
        """Forward pass.

        Args:
            x: Input tensor. Shape: (batch, channels, horizon)

        Returns:
            Output tensor. Shape: (batch, out_channels, horizon * 2)
        """
        # (batch, channels, horizon) -> (batch, horizon, channels)
        x = jnp.transpose(x, (0, 2, 1))
        x = nn.ConvTranspose(
            features=self.out_channels,
            kernel_size=(4,),
            strides=(2,),
            padding="SAME",
        )(x)
        # (batch, horizon*2, channels) -> (batch, channels, horizon*2)
        x = jnp.transpose(x, (0, 2, 1))
        return x


class ConditionalResidualBlock1D(nn.Module):
    """Conditional residual block with FiLM modulation.

    Two Conv1dBlocks with a residual connection. Global conditioning
    is injected via FiLM (Feature-wise Linear Modulation) between
    the two conv blocks.

    FiLM: out = scale * out + bias (when cond_predict_scale=True)
          out = out + bias          (when cond_predict_scale=False)
    """

    out_channels: int
    kernel_size: int = 5
    n_groups: int = 8
    cond_dim: int = 256
    cond_predict_scale: bool = True

    @nn.compact
    def __call__(self, x, cond):
        """Forward pass.

        Args:
            x: Input tensor. Shape: (batch, channels, horizon)
            cond: Global conditioning. Shape: (batch, cond_dim)

        Returns:
            Output tensor. Shape: (batch, out_channels, horizon)
        """
        out = Conv1dBlock(
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            name="conv1",
        )(x)

        # FiLM conditioning (Diffusion Policy: Mish -> Linear -> reshape)
        cond_act = cond * jnp.tanh(nn.softplus(cond))  # Mish activation on cond
        if self.cond_predict_scale:
            cond_channels = self.out_channels * 2
            cond_params = nn.Dense(cond_channels, name="cond_dense")(cond_act)
            scale, bias = jnp.split(cond_params, 2, axis=-1)
            scale = scale[:, :, None]
            bias = bias[:, :, None]
            out = scale * out + bias
        else:
            bias = nn.Dense(self.out_channels, name="cond_dense")(cond_act)
            bias = bias[:, :, None]
            out = out + bias

        out = Conv1dBlock(
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            name="conv2",
        )(out)

        # Residual connection (with projection if channels differ)
        if x.shape[1] != self.out_channels:
            # 1x1 conv for channel matching
            residual = jnp.transpose(x, (0, 2, 1))
            residual = nn.Conv(
                features=self.out_channels, kernel_size=(1,), name="residual_conv"
            )(residual)
            residual = jnp.transpose(residual, (0, 2, 1))
        else:
            residual = x

        return out + residual
