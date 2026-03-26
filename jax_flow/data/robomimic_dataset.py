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
        abs_action=False,
        reward_offset=0.0,
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
            reward_offset: Offset added to raw rewards (e.g. -1.0 for DSRL style).
        """
        self.dataset_path = dataset_path
        self.horizon = horizon
        self.obs_steps = obs_steps
        self.act_steps = act_steps
        self.obs_keys = obs_keys
        self.mode = mode
        self.abs_action = abs_action
        self.reward_offset = reward_offset

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
            demo_keys = sorted(demos.keys(), key=lambda k: int(k.split("_")[1]))
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
            self.episode_rewards = []
            self.episode_dones = []
            self.episode_next_obs = []

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

                # Load RL fields (rewards, dones, next_obs)
                if "rewards" in demo:
                    rew = demo["rewards"][:].astype(np.float32) + self.reward_offset
                    self.episode_rewards.append(rew)
                else:
                    # Default: sparse reward at last step
                    rew = np.zeros(len(act), dtype=np.float32)
                    rew[-1] = 1.0 + self.reward_offset
                    self.episode_rewards.append(rew)

                if "dones" in demo:
                    self.episode_dones.append(
                        demo["dones"][:].astype(np.float32)
                    )
                else:
                    dones = np.zeros(len(act), dtype=np.float32)
                    dones[-1] = 1.0
                    self.episode_dones.append(dones)

                if "next_obs" in demo:
                    next_obs = np.concatenate(
                        [demo["next_obs"][k][:] for k in self.obs_keys],
                        axis=-1,
                    ).astype(np.float32)
                    self.episode_next_obs.append(next_obs)
                else:
                    # Shift obs by 1, repeat last obs
                    next_obs = np.concatenate(
                        [obs[1:], obs[-1:]], axis=0
                    )
                    self.episode_next_obs.append(next_obs)

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
            self.action_dim = self.actions.shape[
                -1
            ]  # Updated after 6D transform if abs_action

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
        observations = self._build_obs_history(ep_obs, t)

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

    def _build_obs_history(self, ep_obs, t):
        """Build observation history at timestep t with padding at episode start.

        Args:
            ep_obs: Episode observations array. Shape: (ep_len, obs_dim).
            t: Current timestep.

        Returns:
            obs_history: (obs_steps, obs_dim).
        """
        obs_list = []
        for i in range(self.obs_steps):
            obs_t = max(t - (self.obs_steps - 1 - i), 0)
            obs_list.append(ep_obs[obs_t])
        return np.stack(obs_list, axis=0)

    def sample_sequence(self, batch_size, discount=0.99, rng=None, reward_offset=None):
        """Sample action-chunk sequences for offline RL training.

        Per-episode sampling: randomly selects episodes (weighted by valid
        start positions), then picks a start position ensuring the full
        horizon fits within the episode. Never crosses episode boundaries.

        Args:
            batch_size: Number of sequences to sample.
            discount: Discount factor for cumulative reward.
            rng: NumPy random generator (optional).

        Returns:
            dict with keys:
                observations: (B, obs_steps, obs_dim) — obs history at chunk start
                actions: (B, horizon, action_dim) — full action chunk (for BC loss)
                rewards: (B,) — cumulative discounted reward over chunk
                next_observations: (B, obs_steps, obs_dim) — obs history at chunk end
                masks: (B,) — 0 if terminal within chunk, else 1
        """
        if rng is None:
            rng = np.random.default_rng()

        num_episodes = len(self.episode_obs)

        # Compute valid start counts per episode for weighted sampling
        # A valid start t requires t + horizon <= ep_len
        ep_lengths = np.array([len(ep) for ep in self.episode_obs])
        valid_starts = np.maximum(ep_lengths - self.horizon, 0)
        # Episodes shorter than horizon have 0 valid starts
        weights = valid_starts.astype(np.float64)
        total_weight = weights.sum()
        if total_weight == 0:
            raise ValueError(
                f"No episode long enough for horizon={self.horizon}. "
                f"Max episode length: {ep_lengths.max()}"
            )
        weights /= total_weight

        # Sample episodes weighted by valid start count
        ep_indices = rng.choice(num_episodes, size=batch_size, p=weights)

        obs_batch = []
        act_batch = []
        rew_batch = []
        next_obs_batch = []
        mask_batch = []

        discount_powers = discount ** np.arange(self.horizon)

        for ep_idx in ep_indices:
            ep_obs = self.episode_obs[ep_idx]
            ep_act = self.episode_actions[ep_idx]
            ep_rew = self.episode_rewards[ep_idx]
            ep_done = self.episode_dones[ep_idx]
            ep_next_obs = self.episode_next_obs[ep_idx]
            ep_len = len(ep_obs)

            # Random start position ensuring chunk fits
            max_start = ep_len - self.horizon
            t = rng.integers(0, max_start + 1)

            # Observation history at chunk start
            obs_history = self._build_obs_history(ep_obs, t)

            # Action chunk: [act[t], ..., act[t+horizon-1]]
            actions = ep_act[t : t + self.horizon]

            # Cumulative discounted reward over chunk
            chunk_rewards = ep_rew[t : t + self.horizon]
            cum_reward = np.sum(chunk_rewards * discount_powers)

            # Next observation history at chunk end (after last step)
            # Use next_obs of the last step in the chunk
            last_t = t + self.horizon - 1
            # Build obs history for next state: use next_obs[last_t] as the
            # most recent obs, and ep_obs[last_t], ep_obs[last_t-1], ... for history
            next_obs_list = []
            for i in range(self.obs_steps):
                if i < self.obs_steps - 1:
                    # History steps: use ep_obs shifted by 1
                    hist_t = max(last_t + 1 - (self.obs_steps - 1 - i), 0)
                    next_obs_list.append(ep_obs[min(hist_t, ep_len - 1)])
                else:
                    # Most recent: use next_obs of last step
                    next_obs_list.append(ep_next_obs[last_t])
            next_obs_history = np.stack(next_obs_list, axis=0)

            # Mask: 0 if any terminal within chunk, else 1
            chunk_dones = ep_done[t : t + self.horizon]
            mask = 1.0 - float(np.any(chunk_dones > 0))

            obs_batch.append(obs_history)
            act_batch.append(actions)
            rew_batch.append(cum_reward)
            next_obs_batch.append(next_obs_history)
            mask_batch.append(mask)

        # Normalize
        obs_array = np.stack(obs_batch, axis=0)  # (B, obs_steps, obs_dim)
        act_array = np.stack(act_batch, axis=0)  # (B, horizon, action_dim)
        next_obs_array = np.stack(next_obs_batch, axis=0)

        obs_array = self.obs_normalizer.normalize(obs_array)
        act_array = self.action_normalizer.normalize(act_array)
        next_obs_array = self.obs_normalizer.normalize(next_obs_array)

        return {
            "observations": obs_array,
            "actions": act_array,
            "rewards": np.array(rew_batch, dtype=np.float32),
            "next_observations": next_obs_array,
            "masks": np.array(mask_batch, dtype=np.float32),
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
    abs_action=False,
    reward_offset=0.0,
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
        reward_offset: Offset added to raw rewards.

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
            abs_action=abs_action,
            reward_offset=reward_offset,
        )
    elif obs_type == "image":
        from jax_flow.data.robomimic_image_dataset import RobomimicImageDataset

        # Use explicit keys if provided, otherwise infer from obs_keys
        if image_keys is None:
            image_keys = tuple(k for k in obs_keys if "image" in k) or (
                "agentview_image",
            )
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
            abs_action=abs_action,
            reward_offset=reward_offset,
        )
    else:
        raise ValueError(f"Invalid obs_type: {obs_type}")
