"""Spatial embedding critic for RL fine-tuning.

Shared spatial embedding trunk + vmap ensemble heads.
Designed to work with patch features from MinViTEncoder.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


class SpatialEmbTrunk(nn.Module):
    """Shared spatial embedding trunk.

    Combines patch features with action and proprioceptive state,
    then uses learned patch weights to produce a spatial embedding.
    """

    spatial_emb_dim: int = 1024

    @nn.compact
    def __call__(self, patch_features, action, prop):
        """Forward pass.

        Args:
            patch_features: (batch, num_patches, embed_dim) from ViT encoder.
            action: (batch, action_dim).
            prop: (batch, prop_dim) proprioceptive state.

        Returns:
            z: (batch, spatial_emb_dim + action_dim + prop_dim).
        """
        batch_size, num_patches, _ = patch_features.shape

        # Broadcast action and prop to patch dimension
        action_exp = jnp.broadcast_to(
            action[:, None, :], (batch_size, num_patches, action.shape[-1])
        )
        prop_exp = jnp.broadcast_to(
            prop[:, None, :], (batch_size, num_patches, prop.shape[-1])
        )

        # Concatenate along feature dim
        x = jnp.concatenate([patch_features, action_exp, prop_exp], axis=-1)

        # Project to spatial embedding dim
        x = nn.Dense(self.spatial_emb_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # Learned patch weights → softmax → weighted sum
        patch_weight = self.param(
            "patch_weight",
            nn.initializers.zeros,
            (1, num_patches, 1),
        )
        weights = nn.softmax(patch_weight, axis=1)  # (1, num_patches, 1)
        spatial_emb = jnp.sum(x * weights, axis=1)  # (batch, spatial_emb_dim)

        # Concat spatial embedding with raw action and prop
        z = jnp.concatenate([spatial_emb, action, prop], axis=-1)
        return z


class _CriticHead(nn.Module):
    """Single critic MLP head."""

    hidden_dim: int
    num_layers: int
    layer_norm: bool

    @nn.compact
    def __call__(self, z):
        x = z
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


class SpatialEmbCritic(nn.Module):
    """Ensemble critic with shared spatial embedding trunk.

    SpatialEmbTrunk processes inputs → z, then K independent MLP heads
    via nn.vmap produce Q-values. Supports RED-Q style ensemble.
    """

    spatial_emb_dim: int = 1024
    hidden_dim: int = 1024
    num_layers: int = 2
    num_q: int = 10
    layer_norm: bool = True

    @nn.compact
    def __call__(self, patch_features, action, prop):
        """Compute ensemble Q-values.

        Args:
            patch_features: (batch, num_patches, embed_dim) from ViT encoder.
            action: (batch, action_dim).
            prop: (batch, prop_dim) proprioceptive state.

        Returns:
            Q-values: (num_q, batch).
        """
        # Shared trunk
        z = SpatialEmbTrunk(spatial_emb_dim=self.spatial_emb_dim)(
            patch_features, action, prop
        )

        # Ensemble heads via vmap
        EnsembleHead = nn.vmap(
            _CriticHead,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_q,
        )
        return EnsembleHead(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            layer_norm=self.layer_norm,
        )(z)


def redq_q_value(q_values, rng, min_q_heads=2):
    """RED-Q: randomly select min_q_heads and take min for target Q.

    Args:
        q_values: (num_q, batch) from SpatialEmbCritic.
        rng: JAX PRNG key.
        min_q_heads: Number of heads to subsample.

    Returns:
        min Q-value: (batch,).
    """
    num_q = q_values.shape[0]
    indices = jax.random.choice(rng, num_q, shape=(min_q_heads,), replace=False)
    selected = q_values[indices]  # (min_q_heads, batch)
    return jnp.min(selected, axis=0)


def ensemble_mean_q(q_values):
    """Mean over all ensemble heads for policy gradient.

    Args:
        q_values: (num_q, batch) from SpatialEmbCritic.

    Returns:
        mean Q-value: (batch,).
    """
    return jnp.mean(q_values, axis=0)
