"""Multi-modal observation encoder.

Handles mixed observations: multiple camera images + low-dimensional state.
Each image goes through a shared or separate encoder (ResNet or ViT), then all
features are concatenated.
"""

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp

from jax_flow.networks.encoders.crop_randomizer import CropRandomizer
from jax_flow.networks.encoders.resnet import ResNet18Encoder
from jax_flow.networks.encoders.vit import MinViTEncoder


class MultiImageEncoder(nn.Module):
    """Multi-modal encoder for images + lowdim observations.

    Encodes multiple camera views and low-dimensional state into a
    single feature vector (or patch features + prop for SpatialEmbCritic).

    When return_patches=True, returns a tuple:
        (patch_features: (batch, total_patches, embed_dim),
         prop_features: (batch, prop_dim))

    Example observation dict:
        {
            'agentview_image': (batch, obs_steps, H, W, 3),
            'robot0_eye_in_hand_image': (batch, obs_steps, H, W, 3),
            'robot0_eef_pos': (batch, obs_steps, 3),
            'robot0_gripper_qpos': (batch, obs_steps, 2),
        }
    """

    image_keys: Sequence[str] = ("agentview_image",)
    lowdim_keys: Sequence[str] = ("robot0_eef_pos", "robot0_gripper_qpos")
    image_encoder_output_dim: int = 64
    share_image_encoder: bool = False
    use_spatial_softmax: bool = True
    crop_shape: tuple[int, int] | None = None
    image_backbone: str = "resnet"
    vit_embed_dim: int = 128
    vit_num_heads: int = 4
    vit_depth: int = 1
    vit_ffn_mult: int = 4
    return_patches: bool = False

    @nn.compact
    def __call__(self, obs_dict, training=False):
        """Forward pass.

        Args:
            obs_dict: Dict of observations.
                - Image keys: (batch, obs_steps, H, W, C) in [0, 1]
                - Lowdim keys: (batch, obs_steps, dim)
            training: Whether in training mode.

        Returns:
            If return_patches=False: (batch, total_feature_dim)
            If return_patches=True: tuple of
                (patch_features: (batch, total_patches, embed_dim),
                 prop_features: (batch, prop_dim))
        """
        use_vit = self.image_backbone == "vit"
        image_features = []

        # Encode images
        if use_vit:
            # ViT backbone — shared or separate per camera
            if self.share_image_encoder:
                encoder = MinViTEncoder(
                    embed_dim=self.vit_embed_dim,
                    num_heads=self.vit_num_heads,
                    depth=self.vit_depth,
                    ffn_mult=self.vit_ffn_mult,
                    output_dim=self.image_encoder_output_dim,
                    return_patches=self.return_patches,
                )
            for i, key in enumerate(self.image_keys):
                if key not in obs_dict:
                    continue
                imgs = obs_dict[key]  # (batch, obs_steps, H, W, C)
                batch_size = imgs.shape[0]
                imgs_flat = imgs.reshape(-1, *imgs.shape[2:])

                if self.crop_shape is not None:
                    crop = CropRandomizer(
                        crop_h=self.crop_shape[0],
                        crop_w=self.crop_shape[1],
                        name=f"crop_{key}",
                    )
                    imgs_flat = crop(imgs_flat, training=training)

                if not self.share_image_encoder:
                    encoder = MinViTEncoder(
                        embed_dim=self.vit_embed_dim,
                        num_heads=self.vit_num_heads,
                        depth=self.vit_depth,
                        ffn_mult=self.vit_ffn_mult,
                        output_dim=self.image_encoder_output_dim,
                        return_patches=self.return_patches,
                        name=f"image_encoder_{i}",
                    )

                feat = encoder(imgs_flat, training=training)

                if self.return_patches:
                    # feat: (batch*obs_steps, num_patches, embed_dim)
                    obs_steps = imgs.shape[1]
                    feat = feat.reshape(batch_size, obs_steps, *feat.shape[1:])
                    # Flatten obs_steps into patch dim: (batch, obs_steps*num_patches, embed_dim)
                    feat = feat.reshape(batch_size, -1, feat.shape[-1])
                else:
                    feat = feat.reshape(batch_size, -1)
                image_features.append(feat)
        else:
            # ResNet backbone
            if self.share_image_encoder:
                encoder = ResNet18Encoder(
                    output_dim=self.image_encoder_output_dim,
                    use_spatial_softmax=self.use_spatial_softmax,
                    pool_type="spatial_softmax"
                    if self.use_spatial_softmax
                    else "avg_pool",
                )
            for i, key in enumerate(self.image_keys):
                if key not in obs_dict:
                    continue
                imgs = obs_dict[key]  # (batch, obs_steps, H, W, C)
                batch_size = imgs.shape[0]
                imgs_flat = imgs.reshape(-1, *imgs.shape[2:])

                if self.crop_shape is not None:
                    crop = CropRandomizer(
                        crop_h=self.crop_shape[0],
                        crop_w=self.crop_shape[1],
                        name=f"crop_{key}",
                    )
                    imgs_flat = crop(imgs_flat, training=training)

                if not self.share_image_encoder:
                    encoder = ResNet18Encoder(
                        output_dim=self.image_encoder_output_dim,
                        use_spatial_softmax=self.use_spatial_softmax,
                        pool_type="spatial_softmax"
                        if self.use_spatial_softmax
                        else "avg_pool",
                        name=f"image_encoder_{i}",
                    )

                feat = encoder(imgs_flat)
                feat = feat.reshape(batch_size, -1)
                image_features.append(feat)

        # Encode lowdim observations
        lowdim_features = []
        for key in self.lowdim_keys:
            if key in obs_dict:
                lowdim = obs_dict[key]  # (batch, obs_steps, dim)
                batch_size = lowdim.shape[0]
                lowdim_flat = lowdim.reshape(batch_size, -1)
                lowdim_features.append(lowdim_flat)

        if not image_features and not lowdim_features:
            raise ValueError("No valid observation keys found in obs_dict")

        # Return based on mode
        if self.return_patches and use_vit:
            # Concat patches along patch dim across cameras
            patch_features = jnp.concatenate(image_features, axis=1)
            if lowdim_features:
                prop_features = jnp.concatenate(lowdim_features, axis=-1)
            else:
                prop_features = jnp.zeros((patch_features.shape[0], 0))
            return patch_features, prop_features

        # Standard mode: concat everything into single vector
        all_features = image_features + lowdim_features
        return jnp.concatenate(all_features, axis=-1)
