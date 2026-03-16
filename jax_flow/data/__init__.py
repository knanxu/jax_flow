"""Data loading utilities."""

from jax_flow.data.dataset_manager import DatasetManager
from jax_flow.data.normalizer import (
    IdentityNormalizer,
    ImageNormalizer,
    MinMaxNormalizer,
)
from jax_flow.data.replay_buffer import OfflineReplayBuffer, ReplayBuffer
from jax_flow.data.robomimic_dataset import RobomimicDataset, make_robomimic_dataset
from jax_flow.data.robomimic_image_dataset import RobomimicImageDataset


def make_dataset(task_source, dataset_path, obs_type="lowdim", mode="train", **kwargs):
    """Dispatch dataset creation based on task_source.

    Args:
        task_source: 'robomimic', 'pusht', or 'kitchen'.
        dataset_path: Path to dataset.
        obs_type: Observation type.
        mode: 'train' or 'val'.
        **kwargs: Additional arguments passed to the dataset constructor.

    Returns:
        Dataset instance.
    """
    if task_source == "pusht":
        from jax_flow.data.pusht_dataset import PushTDataset

        return PushTDataset(
            dataset_path=dataset_path,
            obs_type=obs_type if obs_type != "lowdim" else "state",
            mode=mode,
            **kwargs,
        )
    elif task_source == "kitchen":
        from jax_flow.data.kitchen_dataset import KitchenDataset

        return KitchenDataset(
            dataset_path=dataset_path,
            mode=mode,
            **kwargs,
        )
    else:
        # Default: robomimic
        return make_robomimic_dataset(
            dataset_path=dataset_path,
            obs_type=obs_type,
            mode=mode,
            **kwargs,
        )


__all__ = [
    "MinMaxNormalizer",
    "ImageNormalizer",
    "IdentityNormalizer",
    "RobomimicDataset",
    "RobomimicImageDataset",
    "make_robomimic_dataset",
    "make_dataset",
    "DatasetManager",
    "ReplayBuffer",
    "OfflineReplayBuffer",
]
