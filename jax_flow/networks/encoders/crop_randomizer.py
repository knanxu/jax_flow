"""Diffusion Policy style random crop augmentation.

Training: random crop from (H, W) to (crop_h, crop_w).
Inference: center crop.
"""

import flax.linen as nn
import jax


class CropRandomizer(nn.Module):
    """Random crop augmentation for image observations.

    Input: (batch, H, W, C) — already flattened from (batch*obs_steps, H, W, C).
    Output: (batch, crop_h, crop_w, C)
    """

    crop_h: int = 76
    crop_w: int = 76

    @nn.compact
    def __call__(self, imgs, training=False):
        """Apply crop augmentation.

        Args:
            imgs: (batch, H, W, C)
            training: If True, random crop. If False, center crop.

        Returns:
            Cropped images: (batch, crop_h, crop_w, C)
        """
        batch_size, h, w, c = imgs.shape
        pad_h = h - self.crop_h
        pad_w = w - self.crop_w

        if not training or pad_h == 0 and pad_w == 0:
            # Center crop
            start_h = pad_h // 2
            start_w = pad_w // 2
            return jax.lax.dynamic_slice(
                imgs,
                (0, start_h, start_w, 0),
                (batch_size, self.crop_h, self.crop_w, c),
            )

        # Random crop: sample one offset per image
        rng = self.make_rng("crop")
        rng_h, rng_w = jax.random.split(rng)
        offsets_h = jax.random.randint(rng_h, (batch_size,), 0, pad_h + 1)
        offsets_w = jax.random.randint(rng_w, (batch_size,), 0, pad_w + 1)

        # Use vmap to crop each image independently
        def crop_single(img, oh, ow):
            return jax.lax.dynamic_slice(img, (oh, ow, 0), (self.crop_h, self.crop_w, c))

        return jax.vmap(crop_single)(imgs, offsets_h, offsets_w)
