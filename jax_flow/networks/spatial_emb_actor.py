"""Spatial embedding residual actor for image-based RL fine-tuning.

Unlike the critic's SpatialEmbTrunk which takes action as input,
the actor's SpatialEmbedding only processes patch_features + prop
(since the actor hasn't produced an action yet).
"""

import flax.linen as nn
import jax.numpy as jnp


class SpatialEmbedding(nn.Module):
    """Spatial embedding for actor feature extraction.

    Processes patch_features + prop via learned patch weights.
    Does NOT take action as input (unlike critic's SpatialEmbTrunk).
    """

    spatial_emb_dim: int = 1024

    @nn.compact
    def __call__(self, patch_features, prop):
        """Forward pass.

        Args:
            patch_features: (batch, num_patches, embed_dim) from ViT encoder.
            prop: (batch, prop_dim) proprioceptive state.

        Returns:
            (batch, spatial_emb_dim + prop_dim).
        """
        batch_size, num_patches, _ = patch_features.shape

        # Broadcast prop to patch dimension
        prop_exp = jnp.broadcast_to(
            prop[:, None, :], (batch_size, num_patches, prop.shape[-1])
        )

        # Concatenate patch + prop
        x = jnp.concatenate([patch_features, prop_exp], axis=-1)

        # Project to spatial_emb_dim
        x = nn.Dense(self.spatial_emb_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # Learned patch weights -> softmax -> weighted sum
        patch_weight = self.param(
            "patch_weight",
            nn.initializers.zeros,
            (1, num_patches, 1),
        )
        weights = nn.softmax(patch_weight, axis=1)
        spatial_emb = jnp.sum(x * weights, axis=1)  # (batch, spatial_emb_dim)

        # Concatenate spatial_emb + prop
        return jnp.concatenate([spatial_emb, prop], axis=-1)


class SpatialEmbResidualActor(nn.Module):
    """Residual actor with independent spatial embedding for image observations.

    Architecture:
    1. SpatialEmbedding(patch_features, prop) -> spatial_feat
    2. Concat(spatial_feat, base_action) -> x
    3. MLP layers -> residual action in [-action_scale, action_scale]

    Last layer is zero-initialized so initial behavior = BC policy exactly.
    """

    action_dim: int
    spatial_emb_dim: int = 1024
    hidden_dim: int = 1024
    num_layers: int = 2
    action_scale: float = 0.1
    layer_norm: bool = True

    @nn.compact
    def __call__(self, patch_features, prop, base_action):
        """Forward pass.

        Args:
            patch_features: (batch, num_patches, embed_dim).
            prop: (batch, prop_dim).
            base_action: (batch, action_dim).

        Returns:
            Residual action in [-action_scale, action_scale]. Shape: (batch, action_dim).
        """
        # 1. Spatial embedding
        spatial_feat = SpatialEmbedding(
            spatial_emb_dim=self.spatial_emb_dim
        )(patch_features, prop)

        # 2. Concat base_action
        x = jnp.concatenate([spatial_feat, base_action], axis=-1)

        # 3. MLP layers
        for i in range(self.num_layers):
            x = nn.Dense(self.hidden_dim, name=f"fc_{i}")(x)
            if self.layer_norm:
                x = nn.LayerNorm(name=f"ln_{i}")(x)
            x = nn.relu(x)

        # 4. Zero-initialized output layer
        x = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="output",
        )(x)
        x = nn.tanh(x)

        return x * self.action_scale
