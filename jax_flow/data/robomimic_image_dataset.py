"""Robomimic image dataset."""

import h5py
import numpy as np
from tqdm import tqdm

from jax_flow.data.normalizer import ImageNormalizer, MinMaxNormalizer


class RobomimicImageDataset:
    """Robomimic dataset for image observations.

    Loads HDF5 dataset with image observations and provides sequence sampling.
    Supports mixed observations: images + low-dimensional state.

    Data format:
        observations: dict with image and lowdim keys
            - image keys: (obs_steps, H, W, C) in [0, 1]
            - lowdim keys: (obs_steps, dim)
        actions: (horizon, action_dim)
    """

    def __init__(
        self,
        dataset_path,
        horizon=16,
        obs_steps=2,
        act_steps=8,
        image_keys=("agentview_image",),
        lowdim_keys=(
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ),
        image_size=(84, 84),
        val_ratio=0.0,
        mode="train",
        abs_action=False,
    ):
        """Initialize dataset.

        Args:
            dataset_path: Path to HDF5 dataset file.
            horizon: Action prediction horizon.
            obs_steps: Number of observation history steps.
            act_steps: Number of action execution steps.
            image_keys: Image observation keys to load.
            lowdim_keys: Low-dim observation keys to load.
            image_size: Target image size (H, W).
            val_ratio: Validation split ratio.
            mode: 'train' or 'val'.
        """
        self.dataset_path = dataset_path
        self.horizon = horizon
        self.obs_steps = obs_steps
        self.act_steps = act_steps
        self.image_keys = image_keys
        self.lowdim_keys = lowdim_keys
        self.image_size = image_size
        self.mode = mode
        self.abs_action = abs_action

        # Load data from HDF5
        self._load_data(val_ratio)

        # Convert actions to 6D rotation representation if abs_action
        if self.abs_action:
            from jax_flow.data.rotation_utils import transform_action_to_6d

            self.episode_actions = [
                transform_action_to_6d(ep_act) for ep_act in self.episode_actions
            ]
            self.actions = np.concatenate(self.episode_actions, axis=0)
            self.action_dim = self.actions.shape[-1]

        # Create normalizers
        self.image_normalizer = ImageNormalizer()
        self.action_normalizer = MinMaxNormalizer(self.actions)

        # Compute lowdim normalizer if we have lowdim observations
        if self.lowdim_keys and len(self.lowdim_observations) > 0:
            self.lowdim_normalizer = MinMaxNormalizer(self.lowdim_observations)
        else:
            self.lowdim_normalizer = None

    def _load_data(self, val_ratio):
        """Load data from HDF5 file."""
        with h5py.File(self.dataset_path, "r") as f:
            demos = f["data"]

            # Get sorted demo keys
            demo_keys = sorted(demos.keys(), key=lambda k: int(k.split("_")[1]))
            total_demos = len(demo_keys)

            # Split train/val
            if val_ratio > 0.0:
                val_count = max(1, int(total_demos * val_ratio))
                train_count = total_demos - val_count
                if self.mode == "train":
                    demo_keys = demo_keys[:train_count]
                elif self.mode == "val":
                    demo_keys = demo_keys[train_count:]
                else:
                    raise ValueError(f"Invalid mode: {self.mode}")

            # Load all demos into per-episode lists
            self.episode_images = {key: [] for key in self.image_keys}
            self.episode_lowdim = []
            self.episode_actions = []

            all_lowdim = []
            all_actions = []

            for key in tqdm(demo_keys, desc=f"Loading {self.mode} image data"):
                demo = demos[key]

                # Load images for each camera
                for img_key in self.image_keys:
                    imgs = demo["obs"][img_key][:]  # (T, H, W, C) uint8
                    self.episode_images[img_key].append(imgs)

                # Load lowdim observations
                if self.lowdim_keys:
                    lowdim = np.concatenate(
                        [demo["obs"][k][:] for k in self.lowdim_keys], axis=-1
                    ).astype(np.float32)
                    self.episode_lowdim.append(lowdim)
                    all_lowdim.append(lowdim)

                # Load actions
                act = demo["actions"][:].astype(np.float32)
                self.episode_actions.append(act)
                all_actions.append(act)

            # Concatenated arrays for normalizer computation
            self.lowdim_observations = (
                np.concatenate(all_lowdim, axis=0) if all_lowdim else np.array([])
            )
            self.actions = np.concatenate(all_actions, axis=0)

            # Record per-key lowdim dimensions for splitting in __getitem__
            self.lowdim_key_dims = {}
            if self.lowdim_keys and len(self.episode_lowdim) > 0:
                # Infer dims by loading first demo's keys individually
                with h5py.File(self.dataset_path, "r") as f2:
                    first_demo_key = sorted(
                        f2["data"].keys(), key=lambda k: int(k.split("_")[1])
                    )[0]
                    for lk in self.lowdim_keys:
                        dim = f2["data"][first_demo_key]["obs"][lk].shape[-1]
                        self.lowdim_key_dims[lk] = dim

            # Compute obs_dim: sum of all lowdim key dims
            self.obs_dim = (
                sum(self.lowdim_key_dims.values()) if self.lowdim_key_dims else 0
            )

            # Build index
            self.indices = []
            for ep_idx in range(len(self.episode_actions)):
                ep_len = len(self.episode_actions[ep_idx])
                for t in range(ep_len):
                    self.indices.append((ep_idx, t))

            self.size = len(self.indices)
            self.action_dim = self.actions.shape[-1]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """Sample a sequence.

        Returns:
            dict with keys:
                - observations: dict with image and lowdim keys
                - actions: (horizon, action_dim)
        """
        ep_idx, t = self.indices[idx]
        ep_act = self.episode_actions[ep_idx]
        ep_len = len(ep_act)

        observations = {}

        # Sample image observations
        for img_key in self.image_keys:
            ep_imgs = self.episode_images[img_key][ep_idx]
            img_list = []
            for i in range(self.obs_steps):
                img_t = max(t - (self.obs_steps - 1 - i), 0)
                img = ep_imgs[img_t]  # (H, W, C) uint8
                # Normalize to [0, 1]
                img = self.image_normalizer.normalize(img)
                img_list.append(img)
            observations[img_key] = np.stack(img_list, axis=0)  # (obs_steps, H, W, C)

        # Sample lowdim observations
        if self.lowdim_keys and len(self.episode_lowdim) > 0:
            ep_lowdim = self.episode_lowdim[ep_idx]
            lowdim_list = []
            for i in range(self.obs_steps):
                lowdim_t = max(t - (self.obs_steps - 1 - i), 0)
                lowdim_list.append(ep_lowdim[lowdim_t])
            lowdim_obs = np.stack(lowdim_list, axis=0)  # (obs_steps, lowdim_dim)

            # Split concatenated lowdim back into per-key arrays
            offset = 0
            for key in self.lowdim_keys:
                dim = self.lowdim_key_dims[key]
                observations[key] = lowdim_obs[:, offset : offset + dim]
                offset += dim

        # Sample action sequence
        act_list = []
        for i in range(self.horizon):
            act_t = min(t + i, ep_len - 1)
            act_list.append(ep_act[act_t])
        actions = np.stack(act_list, axis=0)  # (horizon, action_dim)

        # Normalize actions to [-1, 1]
        actions = self.action_normalizer.normalize(actions)

        return {
            "observations": observations,
            "actions": actions,
        }

    def sample_batch(self, batch_size, rng=None):
        """Sample a random batch.

        Args:
            batch_size: Number of samples.
            rng: numpy random generator.

        Returns:
            dict with batched observations and actions.
        """
        if rng is None:
            rng = np.random.default_rng()
        indices = rng.integers(0, self.size, size=batch_size)

        # Collect samples
        samples = [self[idx] for idx in indices]

        # Stack observations (dict of arrays)
        obs_dict = {}
        for key in samples[0]["observations"].keys():
            obs_dict[key] = np.stack([s["observations"][key] for s in samples], axis=0)

        # Stack actions
        actions = np.stack([s["actions"] for s in samples], axis=0)

        return {
            "observations": obs_dict,
            "actions": actions,
        }

    def get_normalizer(self):
        """Get normalizers."""
        normalizers = {
            "action": self.action_normalizer,
            "image": self.image_normalizer,
        }
        if self.lowdim_normalizer is not None:
            normalizers["lowdim"] = self.lowdim_normalizer
        return normalizers
