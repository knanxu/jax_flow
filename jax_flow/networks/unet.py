"""1D UNet for action sequence prediction with FiLM conditioning.

ConditionalUnet1D follows ChiUNet (much-ado-about-noising) architecture adapted
for flow matching with dual timestep embeddings (s, t).

Architecture:
- Timestep: s/t each emb_dim//2 via PositionalEmbedding, concat → shared MLP (GELU)
- Obs: Linear(To*obs_dim → emb_dim), concat with time → global_cond (emb_dim*2)
- Down: resnet1(dim_in→dim_out) + resnet2(dim_out→dim_out) + Downsample (except last)
- Mid: 2x ResBlock
- Up: concat skip + resnet1(dim_out*2→dim_in) + resnet2(dim_in→dim_in) + Upsample (except last)
- Final: Conv1d(model_dim) → GroupNorm → GELU → Conv1d(act_dim)
"""

import flax.linen as nn
import jax.numpy as jnp

from jax_flow.networks.unet_components import (
    ConditionalResidualBlock1D,
    Downsample1d,
    Upsample1d,
)


class PositionalEmbedding(nn.Module):
    """DDPM++/ADM positional embedding matching much-ado PositionalEmbedding.

    freqs = (1/max_positions) ^ (i / (dim//2))
    output = [cos(x * freqs), sin(x * freqs)]
    """

    embed_dim: int = 128
    max_positions: int = 10000

    @nn.compact
    def __call__(self, x):
        half_dim = self.embed_dim // 2
        freqs = jnp.arange(half_dim, dtype=jnp.float32) / half_dim
        freqs = (1.0 / self.max_positions) ** freqs
        args = x[:, None] * freqs[None, :]
        return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


class ConditionalUnet1D(nn.Module):
    """1D UNet for action sequence prediction with FiLM conditioning.

    Aligned with ChiUNet from much-ado-about-noising.

    Network signature: (at, s, t, obs) -> velocity
    - at: (batch, horizon, action_dim)  — horizon must be power of 2
    - s, t: (batch,) timesteps
    - obs: (batch, cond_dim) observation encoding
    - velocity: (batch, horizon, action_dim)
    """

    action_dim: int
    cond_dim: int = 256
    model_dim: int = 256
    emb_dim: int = 256
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    dim_mult: tuple[int, ...] = (1, 2, 2)

    @nn.compact
    def __call__(self, at, s, t, obs, training=False):
        # (B, H, A) -> (B, A, H) for 1D conv processing
        x = jnp.transpose(at, (0, 2, 1))

        # --- Timestep embedding: s/t each emb_dim//2, concat, shared MLP ---
        half_emb = self.emb_dim // 2
        s_emb = PositionalEmbedding(embed_dim=half_emb, name="map_s")(s)
        t_emb = PositionalEmbedding(embed_dim=half_emb, name="map_t")(t)
        emb = jnp.concatenate([s_emb, t_emb], axis=-1)  # (B, emb_dim)

        # Shared MLP: Linear(emb_dim, 4*emb_dim) → GELU → Linear(4*emb_dim, emb_dim)
        emb = nn.Dense(self.emb_dim * 4, name="map_emb1")(emb)
        emb = nn.gelu(emb)
        emb = nn.Dense(self.emb_dim, name="map_emb2")(emb)

        # Global conditioning: concat time emb + obs encoding
        global_cond = jnp.concatenate([emb, obs], axis=-1)  # (B, emb_dim*2)
        global_cond_dim = global_cond.shape[-1]

        # --- Dimension schedule ---
        # dims = [action_dim, model_dim*1, model_dim*2, model_dim*2]
        cum_mult = []
        p = 1
        for m in self.dim_mult:
            p *= m
            cum_mult.append(p)
        dims = [self.action_dim] + [self.model_dim * m for m in cum_mult]
        in_out = list(zip(dims[:-1], dims[1:], strict=False))
        # e.g. [(action_dim, 256), (256, 512), (512, 512)]
        num_resolutions = len(in_out)
        mid_dim = dims[-1]

        # --- Down path ---
        skips = []
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= (num_resolutions - 1)
            x = ConditionalResidualBlock1D(
                out_channels=dim_out,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                cond_dim=global_cond_dim,
                cond_predict_scale=self.cond_predict_scale,
                name=f"down_res1_{i}",
            )(x, global_cond)
            x = ConditionalResidualBlock1D(
                out_channels=dim_out,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                cond_dim=global_cond_dim,
                cond_predict_scale=self.cond_predict_scale,
                name=f"down_res2_{i}",
            )(x, global_cond)
            skips.append(x)
            if not is_last:
                x = Downsample1d(out_channels=dim_out, name=f"downsample_{i}")(x)

        # --- Mid blocks ---
        x = ConditionalResidualBlock1D(
            out_channels=mid_dim,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            cond_dim=global_cond_dim,
            cond_predict_scale=self.cond_predict_scale,
            name="mid_res1",
        )(x, global_cond)
        x = ConditionalResidualBlock1D(
            out_channels=mid_dim,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            cond_dim=global_cond_dim,
            cond_predict_scale=self.cond_predict_scale,
            name="mid_res2",
        )(x, global_cond)

        # --- Up path: reversed(in_out[1:]) ---
        reversed_in_out = list(reversed(in_out[1:]))
        for i, (dim_in, dim_out) in enumerate(reversed_in_out):
            is_last = i >= (num_resolutions - 1)
            skip = skips.pop()
            x = jnp.concatenate([x, skip], axis=1)  # (B, dim_out*2, H)

            # resnet1: dim_out*2 → dim_in (channel reduction)
            x = ConditionalResidualBlock1D(
                out_channels=dim_in,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                cond_dim=global_cond_dim,
                cond_predict_scale=self.cond_predict_scale,
                name=f"up_res1_{i}",
            )(x, global_cond)
            # resnet2: dim_in → dim_in
            x = ConditionalResidualBlock1D(
                out_channels=dim_in,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                cond_dim=global_cond_dim,
                cond_predict_scale=self.cond_predict_scale,
                name=f"up_res2_{i}",
            )(x, global_cond)

            if not is_last:
                x = Upsample1d(out_channels=dim_in, name=f"upsample_{i}")(x)

        # --- Final conv: model_dim → model_dim → action_dim ---
        # x is now (B, model_dim, H) — no skip concat here
        x = jnp.transpose(x, (0, 2, 1))  # (B, H, model_dim)
        x = nn.Conv(
            features=self.model_dim,
            kernel_size=(self.kernel_size,),
            padding="SAME",
            name="final_conv1",
        )(x)
        x = nn.GroupNorm(num_groups=self.n_groups, name="final_norm")(x)
        x = nn.gelu(x)
        x = nn.Conv(features=self.action_dim, kernel_size=(1,), name="final_conv2")(x)
        x = jnp.transpose(x, (0, 2, 1))  # (B, action_dim, H)

        # (B, A, H) -> (B, H, A)
        velocity = jnp.transpose(x, (0, 2, 1))
        return velocity
