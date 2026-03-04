"""Multi-modal observation encoder.

Handles mixed observations: multiple camera images + low-dimensional state.
Each image goes through a shared or separate ResNet encoder, then all
features are concatenated.
"""

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp

from jax_flow.networks.encoders.resnet import ResNet18Encoder


class MultiImageEncoder(nn.Module):
    """Multi-modal encoder for images + lowdim observations.

    Encodes multiple camera views and low-dimensional state into a
    single feature vector. Each camera can have its own encoder or
    share a single encoder.

    Example observation dict:
        {
            'agentview_image': (batch, obs_steps, H, W, 3),
            'robot0_eye_in_hand_image': (batch, obs_steps, H, W, 3),
            'robot0_eef_pos': (batch, obs_steps, 3),
            'robot0_gripper_qpos': (batch, obs_steps, 2),
        }
    """

    image_keys: Sequence[str] = ('agentview_image',)
    lowdim_keys: Sequence[str] = ('robot0_eef_pos', 'robot0_gripper_qpos')
    image_encoder_output_dim: int = 64
    share_image_encoder: bool = False
    use_spatial_softmax: bool = True

    @nn.compact
    def __call__(self, obs_dict, training=False):
        """Forward pass.

        Args:
            obs_dict: Dict of observations.
                - Image keys: (batch, obs_steps, H, W, C) in [0, 1]
                - Lowdim keys: (batch, obs_steps, dim)
            training: Whether in training mode.

        Returns:
            features: (batch, total_feature_dim)
        """
        features = []

        # Encode images
        if self.share_image_encoder:
            # Single shared encoder for all cameras
            image_encoder = ResNet18Encoder(
                output_dim=self.image_encoder_output_dim,
                use_spatial_softmax=self.use_spatial_softmax,
                pool_type="spatial_softmax" if self.use_spatial_softmax else "avg_pool",
            )

            for key in self.image_keys:
                if key in obs_dict:
                    imgs = obs_dict[key]  # (batch, obs_steps, H, W, C)
                    batch_size, obs_steps = imgs.shape[:2]

                    # Flatten batch and obs_steps
                    imgs_flat = imgs.reshape(-1, *imgs.shape[2:])  # (batch*obs_steps, H, W, C)

                    # Encode
                    feat = image_encoder(imgs_flat)  # (batch*obs_steps, output_dim)

                    # Reshape and flatten
                    feat = feat.reshape(batch_size, -1)  # (batch, obs_steps*output_dim)
                    features.append(feat)
        else:
            # Separate encoder for each camera
            for i, key in enumerate(self.image_keys):
                if key in obs_dict:
                    imgs = obs_dict[key]  # (batch, obs_steps, H, W, C)
                    batch_size, obs_steps = imgs.shape[:2]

                    # Flatten batch and obs_steps
                    imgs_flat = imgs.reshape(-1, *imgs.shape[2:])

                    # Encode with camera-specific encoder
                    image_encoder = ResNet18Encoder(
                        output_dim=self.image_encoder_output_dim,
                        use_spatial_softmax=self.use_spatial_softmax,
                        pool_type="spatial_softmax" if self.use_spatial_softmax else "avg_pool",
                        name=f'image_encoder_{i}',
                    )
                    feat = image_encoder(imgs_flat)

                    # Reshape and flatten
                    feat = feat.reshape(batch_size, -1)
                    features.append(feat)

        # Encode lowdim observations
        for key in self.lowdim_keys:
            if key in obs_dict:
                lowdim = obs_dict[key]  # (batch, obs_steps, dim)
                batch_size = lowdim.shape[0]
                # Flatten obs_steps dimension
                lowdim_flat = lowdim.reshape(batch_size, -1)
                features.append(lowdim_flat)

        # Concatenate all features
        if not features:
            raise ValueError("No valid observation keys found in obs_dict")

        return jnp.concatenate(features, axis=-1)
