"""Data normalization utilities."""

import numpy as np


class MinMaxNormalizer:
    """Min-max normalizer that normalizes data to [-1, 1]."""

    def __init__(self, data=None, min_val=None, max_val=None):
        """Initialize normalizer.

        Args:
            data: Data to compute min/max from. Shape: (N, D)
            min_val: Pre-computed min values. Shape: (D,)
            max_val: Pre-computed max values. Shape: (D,)
        """
        if data is not None:
            self.min_val = np.min(data, axis=0)
            self.max_val = np.max(data, axis=0)
        elif min_val is not None and max_val is not None:
            self.min_val = np.array(min_val)
            self.max_val = np.array(max_val)
        else:
            raise ValueError("Either data or min_val/max_val must be provided")

        # Avoid division by zero
        self.scale = self.max_val - self.min_val
        self.scale = np.where(self.scale < 1e-6, 1.0, self.scale)

    def normalize(self, data):
        """Normalize data to [-1, 1].

        Args:
            data: Data to normalize. Shape: (..., D)

        Returns:
            Normalized data in [-1, 1].
        """
        normalized = 2 * (data - self.min_val) / self.scale - 1
        return np.clip(normalized, -1, 1)

    def unnormalize(self, normalized_data):
        """Unnormalize data from [-1, 1] to original range.

        Args:
            normalized_data: Normalized data in [-1, 1]. Shape: (..., D)

        Returns:
            Unnormalized data.
        """
        return (normalized_data + 1) / 2 * self.scale + self.min_val

    def get_params(self):
        """Get normalization parameters."""
        return {"min": self.min_val, "max": self.max_val}


class ImageNormalizer:
    """Image normalizer that normalizes uint8 images to [0, 1]."""

    def normalize(self, data):
        """Normalize uint8 image to [0, 1].

        Args:
            data: Image data in uint8 [0, 255]. Shape: (..., H, W, C)

        Returns:
            Normalized image in [0, 1].
        """
        return data.astype(np.float32) / 255.0

    def unnormalize(self, normalized_data):
        """Unnormalize image from [0, 1] to uint8 [0, 255].

        Args:
            normalized_data: Normalized image in [0, 1]. Shape: (..., H, W, C)

        Returns:
            Unnormalized image in uint8.
        """
        return (normalized_data * 255).astype(np.uint8)


class IdentityNormalizer:
    """Identity normalizer that does nothing."""

    def normalize(self, data):
        return data

    def unnormalize(self, normalized_data):
        return normalized_data
