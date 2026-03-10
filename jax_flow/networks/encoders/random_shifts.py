"""DrQ-style random shifts augmentation for image observations.

Training: pad image → random crop back to original size (equivalent to random shift ±pad pixels).
Inference: no-op (identity).
"""

import jax
import jax.numpy as jnp


class RandomShiftsAug:
    """JAX random shifts augmentation (DrQ style).

    Pads images then randomly crops back to original size,
    effectively applying random translation of ±pad pixels.

    Unlike CropRandomizer which changes output resolution,
    this preserves the original image size.

    Args:
        pad: Number of pixels to pad on each side (default: 4)
    """

    def __init__(self, pad: int = 4):
        self.pad = pad

    def __call__(self, images: jnp.ndarray, rng: jax.Array, training: bool = True) -> jnp.ndarray:
        """Apply random shifts augmentation.

        Args:
            images: (batch, H, W, C) float32 [0, 1]
            rng: JAX PRNG key
            training: If True, apply augmentation. If False, return unchanged.

        Returns:
            Augmented images: (batch, H, W, C)
        """
        if not training:
            return images

        batch_size, h, w, c = images.shape
        pad = self.pad

        # Pad with zeros: (batch, H+2*pad, W+2*pad, C)
        padded = jnp.pad(
            images,
            ((0, 0), (pad, pad), (pad, pad), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

        # Sample random crop offsets per image
        rng_h, rng_w = jax.random.split(rng)
        offsets_h = jax.random.randint(rng_h, (batch_size,), 0, 2 * pad + 1)
        offsets_w = jax.random.randint(rng_w, (batch_size,), 0, 2 * pad + 1)

        # Crop each image back to original size
        def crop_single(img, oh, ow):
            return jax.lax.dynamic_slice(img, (oh, ow, 0), (h, w, c))

        return jax.vmap(crop_single)(padded, offsets_h, offsets_w)
