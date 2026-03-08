"""1D UNet for action sequence prediction with FiLM conditioning.

ConditionalUnet1D follows Diffusion Policy's architecture adapted for
flow matching with dual timestep embeddings (s, t).

Timestep conditioning: SinusoidalPosEmb -> Linear(d, 4d) -> Mish -> Linear(4d, d)
then concatenated with obs encoding as global_cond for FiLM modulation.
"""

import flax.linen as nn
import jax.numpy as jnp

from jax_flow.networks.embeddings import SinusoidalPosEmb
from jax_flow.networks.unet_components import (
    ConditionalResidualBlock1D,
    Downsample1d,
    Upsample1d,
)


def mish(x):
    """Mish activation: x * tanh(softplus(x))."""
    return x * jnp.tanh(nn.softplus(x))


class ConditionalUnet1D(nn.Module):
    """1D UNet for action sequence prediction with FiLM conditioning.

    Network signature: (at, s, t, obs) -> velocity
    - at: (batch, horizon, action_dim)
    - s, t: (batch,) timesteps
    - obs: (batch, cond_dim) observation encoding
    - velocity: (batch, horizon, action_dim)

    Architecture (Diffusion Policy standard):
    - Timestep: SinusoidalPosEmb -> MLP(d -> 4d -> d) with Mish
    - Input projection to down_dims[0] channels
    - Down path: 2x ResBlocks + Downsample with skip connections
    - Mid: Two ResBlocks
    - Up path: Concat skips + 2x ResBlocks + Upsample
    - Final: Conv1dBlock + Conv1d(1x1) projection to action_dim
    """

    action_dim: int
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    cond_dim: int = 256
    timestep_embed_dim: int = 256
    cond_predict_scale: bool = False

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
        # Reshape: (B, H, A) -> (B, A, H) for 1D conv processing
        x = jnp.transpose(at, (0, 2, 1))

        dsed = self.timestep_embed_dim

        # Timestep embedding: SinusoidalPosEmb -> Linear(d, 4d) -> Mish -> Linear(4d, d)
        s_raw = SinusoidalPosEmb(embed_dim=dsed, name="s_sinusoidal")(s)
        s_emb = nn.Dense(dsed * 4, name="s_mlp1")(s_raw)
        s_emb = mish(s_emb)
        s_emb = nn.Dense(dsed, name="s_mlp2")(s_emb)

        t_raw = SinusoidalPosEmb(embed_dim=dsed, name="t_sinusoidal")(t)
        t_emb = nn.Dense(dsed * 4, name="t_mlp1")(t_raw)
        t_emb = mish(t_emb)
        t_emb = nn.Dense(dsed, name="t_mlp2")(t_emb)

        time_emb = jnp.concatenate([s_emb, t_emb], axis=-1)  # (B, 2*dsed)

        # Global conditioning: time + obs
        global_cond = jnp.concatenate([time_emb, obs], axis=-1)
        global_cond_dim = global_cond.shape[-1]

        # Input projection: (B, action_dim, H) -> (B, down_dims[0], H)
        x = jnp.transpose(x, (0, 2, 1))  # (B, H, action_dim)
        x = nn.Conv(features=self.down_dims[0], kernel_size=(1,), name="input_conv")(x)
        x = jnp.transpose(x, (0, 2, 1))  # (B, down_dims[0], H)

        # Down path
        skips = []
        for i, dim_out in enumerate(self.down_dims):
            x = ConditionalResidualBlock1D(
                out_channels=dim_out,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                cond_dim=global_cond_dim,
                cond_predict_scale=self.cond_predict_scale,
                name=f"down_block1_{i}",
            )(x, global_cond)
            x = ConditionalResidualBlock1D(
                out_channels=dim_out,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                cond_dim=global_cond_dim,
                cond_predict_scale=self.cond_predict_scale,
                name=f"down_block2_{i}",
            )(x, global_cond)
            skips.append(x)
            if i < len(self.down_dims) - 1:
                x = Downsample1d(out_channels=dim_out, name=f"downsample_{i}")(x)

        # Mid blocks
        mid_dim = self.down_dims[-1]
        x = ConditionalResidualBlock1D(
            out_channels=mid_dim,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            cond_dim=global_cond_dim,
            cond_predict_scale=self.cond_predict_scale,
            name="mid_block1",
        )(x, global_cond)
        x = ConditionalResidualBlock1D(
            out_channels=mid_dim,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            cond_dim=global_cond_dim,
            cond_predict_scale=self.cond_predict_scale,
            name="mid_block2",
        )(x, global_cond)

        # Up path (reverse order, skip last since no downsample after it)
        for i, dim_out in enumerate(reversed(self.down_dims)):
            skip = skips.pop()
            x = jnp.concatenate([x, skip], axis=1)  # Concat along channel dim
            x = ConditionalResidualBlock1D(
                out_channels=dim_out,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                cond_dim=global_cond_dim,
                cond_predict_scale=self.cond_predict_scale,
                name=f"up_block1_{i}",
            )(x, global_cond)
            x = ConditionalResidualBlock1D(
                out_channels=dim_out,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                cond_dim=global_cond_dim,
                cond_predict_scale=self.cond_predict_scale,
                name=f"up_block2_{i}",
            )(x, global_cond)
            if i < len(self.down_dims) - 1:
                x = Upsample1d(out_channels=dim_out, name=f"upsample_{i}")(x)

        # Final projection: (B, down_dims[0], H) -> (B, action_dim, H)
        x = jnp.transpose(x, (0, 2, 1))  # (B, H, channels)
        x = nn.Conv(features=self.action_dim, kernel_size=(1,), name="output_conv")(x)
        x = jnp.transpose(x, (0, 2, 1))  # (B, action_dim, H)

        # Reshape: (B, A, H) -> (B, H, A)
        velocity = jnp.transpose(x, (0, 2, 1))
        return velocity
