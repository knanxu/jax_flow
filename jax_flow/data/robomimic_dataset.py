"""Robomimic dataset for lowdim observations."""

import h5py
import numpy as np
from tqdm import tqdm

from jax_flow.data.normalizer import MinMaxNormalizer


class RobomimicDataset:
    """Robomimic dataset for state-based (lowdim) observations.

    Loads HDF5 dataset and provides sequence sampling for training.
    All data is loaded into memory for fast access.

    Data format:
        observations: (obs_steps, obs_dim) - observation history
        actions: (horizon, action_dim) - action sequence to predict
    """

    def __init__(
        self,
        dataset_path,
        horizon=16,
        obs_steps=2,
        act_steps=8,
        obs_keys=(
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ),
        val_ratio=0.0,
        mode="train",
    ):
        """Initialize dataset.

        Args:
            dataset_path: Path to HDF5 dataset file.
            horizon: Action prediction horizon.
            obs_steps: Number of observation history steps.
            act_steps: Number of action execution steps.
            obs_keys: Observation keys to load from HDF5.
            val_ratio: Validation split ratio (0.0 = no validation).
            mode: 'train' or 'val'.
        """
        self.dataset_path = dataset_path
        self.horizon = horizon
        self.obs_steps = obs_steps
        self.act_steps = act_steps
        self.obs_keys = obs_keys
        self.mode = mode

        # Load data from HDF5
        self._load_data(val_ratio)

        # Compute normalizers from training data
        self.obs_normalizer = MinMaxNormalizer(self.observations)
        self.action_normalizer = MinMaxNormalizer(self.actions)

    def _load_data(self, val_ratio):
        """Load data from HDF5 file.

        Stores per-episode data and builds an index mapping
        global sample indices to (episode_idx, timestep) pairs.
        """
        with h5py.File(self.dataset_path, "r") as f:
            demos = f["data"]

            # Get sorted demo keys
            demo_keys = sorted(
                demos.keys(), key=lambda k: int(k.split("_")[1])
            )
            total_demos = len(demo_keys)

            # Split train/val by demo
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
            self.episode_obs = []
            self.episode_actions = []

            all_obs = []
            all_actions = []

            for key in tqdm(demo_keys, desc=f"Loading {self.mode} data"):
                demo = demos[key]

                # Load and concatenate observation keys
                obs = np.concatenate(
                    [demo["obs"][k][:] for k in self.obs_keys], axis=-1
                ).astype(np.float32)

                act = demo["actions"][:].astype(np.float32)

                self.episode_obs.append(obs)
                self.episode_actions.append(act)
                all_obs.append(obs)
                all_actions.append(act)

            # Concatenated arrays for normalizer computation
            self.observations = np.concatenate(all_obs, axis=0)
            self.actions = np.concatenate(all_actions, axis=0)

            # Build index: map global idx -> (episode_idx, timestep)
            self.indices = []
            for ep_idx, obs in enumerate(self.episode_obs):
                ep_len = len(obs)
                for t in range(ep_len):
                    self.indices.append((ep_idx, t))

            self.size = len(self.indices)
            self.obs_dim = self.observations.shape[-1]
            self.action_dim = self.actions.shape[-1]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """Sample a sequence.

        Returns normalized data in [-1, 1].

        Returns:
            dict with keys:
                - observations: (obs_steps, obs_dim) normalized to [-1, 1]
                - actions: (horizon, action_dim) normalized to [-1, 1]
        """
        ep_idx, t = self.indices[idx]
        ep_obs = self.episode_obs[ep_idx]
        ep_act = self.episode_actions[ep_idx]
        ep_len = len(ep_obs)

        # Sample observation history with padding at episode start
        obs_list = []
        for i in range(self.obs_steps):
            obs_t = max(t - (self.obs_steps - 1 - i), 0)
            obs_list.append(ep_obs[obs_t])
        observations = np.stack(obs_list, axis=0)  # (obs_steps, obs_dim)

        # Sample action sequence with padding at episode end
        act_list = []
        for i in range(self.horizon):
            act_t = min(t + i, ep_len - 1)
            act_list.append(ep_act[act_t])
        actions = np.stack(act_list, axis=0)  # (horizon, action_dim)

        # Normalize to [-1, 1]
        observations = self.obs_normalizer.normalize(observations)
        actions = self.action_normalizer.normalize(actions)

        return {
            "observations": observations,
            "actions": actions,
        }

    def sample_batch(self, batch_size, rng=None):
        """Sample a random batch.

        Args:
            batch_size: Number of samples.
            rng: numpy random generator (optional).

        Returns:
            dict with batched observations and actions.
        """
        if rng is None:
            rng = np.random.default_rng()
        indices = rng.integers(0, self.size, size=batch_size)

        obs_batch = []
        act_batch = []
        for idx in indices:
            sample = self[idx]
            obs_batch.append(sample["observations"])
            act_batch.append(sample["actions"])

        return {
            "observations": np.stack(obs_batch, axis=0),  # (batch, obs_steps, obs_dim)
            "actions": np.stack(act_batch, axis=0),  # (batch, horizon, action_dim)
        }

    def get_normalizer(self):
        """Get normalizers for observations and actions."""
        return {
            "obs": self.obs_normalizer,
            "action": self.action_normalizer,
        }


def make_robomimic_dataset(
    dataset_path,
    horizon=16,
    obs_steps=2,
    act_steps=8,
    obs_keys=(
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    ),
    obs_type="lowdim",
    val_ratio=0.0,
    mode="train",
    image_keys=None,
    lowdim_keys=None,
):
    """Factory function to create robomimic dataset.

    Args:
        dataset_path: Path to HDF5 dataset.
        horizon: Action prediction horizon.
        obs_steps: Observation history length.
        act_steps: Action execution steps.
        obs_keys: Observation keys to load (lowdim only).
        obs_type: 'lowdim' or 'image'.
        val_ratio: Validation split ratio.
        mode: 'train' or 'val'.
        image_keys: Explicit image keys (image mode).
        lowdim_keys: Explicit lowdim keys (image mode).

    Returns:
        Dataset instance.
    """
    if obs_type == "lowdim":
        return RobomimicDataset(
            dataset_path=dataset_path,
            horizon=horizon,
            obs_steps=obs_steps,
            act_steps=act_steps,
            obs_keys=obs_keys,
            val_ratio=val_ratio,
            mode=mode,
        )
    elif obs_type == "image":
        from jax_flow.data.robomimic_image_dataset import RobomimicImageDataset

        # Use explicit keys if provided, otherwise infer from obs_keys
        if image_keys is None:
            image_keys = tuple(k for k in obs_keys if "image" in k) or ("agentview_image",)
        if lowdim_keys is None:
            lowdim_keys = tuple(k for k in obs_keys if "image" not in k)

        return RobomimicImageDataset(
            dataset_path=dataset_path,
            horizon=horizon,
            obs_steps=obs_steps,
            act_steps=act_steps,
            image_keys=tuple(image_keys),
            lowdim_keys=tuple(lowdim_keys),
            val_ratio=val_ratio,
            mode=mode,
        )
    else:
        raise ValueError(f"Invalid obs_type: {obs_type}")
