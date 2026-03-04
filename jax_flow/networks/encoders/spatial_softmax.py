"""Spatial softmax pooling for image features.

Converts spatial feature maps into a compact set of 2D keypoint coordinates.
Standard approach in robotic manipulation (Levine et al., 2016).
"""

import flax.linen as nn
import jax.numpy as jnp


class SpatialSoftmax(nn.Module):
    """Spatial softmax pooling layer.

    Takes a feature map (batch, H, W, C) and produces (batch, C * 2)
    keypoint coordinates by computing the expected spatial position
    for each channel.
    """

    temperature: float = 1.0
    learnable_temperature: bool = False

    @nn.compact
    def __call__(self, feature_map):
        """Forward pass.

        Args:
            feature_map: (batch, H, W, C) spatial feature map.

        Returns:
            keypoints: (batch, C * 2) expected (x, y) coordinates per channel.
        """
        batch_size, h, w, c = feature_map.shape

        # Temperature
        if self.learnable_temperature:
            temperature = self.param(
                'temperature',
                nn.initializers.constant(self.temperature),
                (1,),
            )
        else:
            temperature = self.temperature

        # Create normalized coordinate grids [-1, 1]
        pos_x = jnp.linspace(-1.0, 1.0, w)
        pos_y = jnp.linspace(-1.0, 1.0, h)

        # Reshape feature map for softmax: (batch, H*W, C)
        features_flat = feature_map.reshape(batch_size, h * w, c)

        # Softmax over spatial dimensions
        attention = nn.softmax(features_flat / temperature, axis=1)  # (batch, H*W, C)

        # Reshape attention to (batch, H, W, C)
        attention_map = attention.reshape(batch_size, h, w, c)

        # Expected x coordinate: sum over H, weighted sum over W
        # attention_map: (batch, H, W, C)
        # pos_x: (W,) -> broadcast to (1, 1, W, 1)
        expected_x = jnp.sum(
            attention_map * pos_x[None, None, :, None], axis=(1, 2)
        )  # (batch, C)

        # Expected y coordinate: sum over W, weighted sum over H
        expected_y = jnp.sum(
            attention_map * pos_y[None, :, None, None], axis=(1, 2)
        )  # (batch, C)

        # Concatenate (x, y) coordinates
        keypoints = jnp.concatenate([expected_x, expected_y], axis=-1)  # (batch, C*2)

        return keypoints
