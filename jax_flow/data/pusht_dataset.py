"""Push-T dataset for jax_flow."""

import zipfile
from pathlib import Path

import numpy as np
import zarr
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from jax_flow.data.normalizer import MinMaxNormalizer


def download_pusht_dataset(
    dataset_filename="pusht/pusht_cchi_v7_replay.zarr.zip",
    repo_id="ChaoyiPan/mip-dataset",
):
    """Download Push-T dataset from HuggingFace and extract locally.

    Args:
        dataset_filename: Filename in the HuggingFace dataset repo.
        repo_id: HuggingFace repository ID.

    Returns:
        Path to the extracted zarr dataset.
    """
    print(f"Downloading Push-T dataset from {repo_id}/{dataset_filename}")
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=dataset_filename,
        repo_type="dataset",
    )
    print(f"Downloaded zip file to: {zip_path}")

    zip_path_obj = Path(zip_path)
    extract_dir = zip_path_obj.parent

    print(f"Extracting dataset to {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    zarr_name = zip_path_obj.stem
    if zarr_name.endswith(".zarr"):
        zarr_path = extract_dir / zarr_name
    else:
        zarr_dirs = list(extract_dir.glob("*.zarr"))
        if not zarr_dirs:
            raise FileNotFoundError(f"No .zarr directory found after extracting {zip_path}")
        zarr_path = zarr_dirs[0]

    print(f"Extracted dataset to: {zarr_path}")
    return str(zarr_path)


class PushTDataset:
    """Push-T dataset for state/keypoint/image observations.

    Loads zarr data and converts to episode-list format compatible with jax_flow.
    """

    def __init__(
        self,
        dataset_path,
        horizon=16,
        obs_steps=2,
        act_steps=8,
        obs_type="state",
        val_ratio=0.0,
        mode="train",
    ):
        """Initialize dataset.

        Args:
            dataset_path: Path to zarr dataset directory.
            horizon: Action prediction horizon.
            obs_steps: Number of observation history steps.
            act_steps: Number of action execution steps.
            obs_type: 'state', 'keypoint', or 'image'.
            val_ratio: Validation split ratio.
            mode: 'train' or 'val'.
        """
        self.dataset_path = dataset_path
        self.horizon = horizon
        self.obs_steps = obs_steps
        self.act_steps = act_steps
        self.obs_type = obs_type
        self.mode = mode

        # Load data from zarr
        self._load_data(val_ratio)

        # Compute normalizers
        if obs_type == "state":
            self.obs_normalizer = MinMaxNormalizer(self.observations)
            self.action_normalizer = MinMaxNormalizer(self.actions)
            self.obs_dim = 5
            self.action_dim = 2
        elif obs_type == "keypoint":
            self.obs_normalizer = MinMaxNormalizer(self.observations)
            self.action_normalizer = MinMaxNormalizer(self.actions)
            self.obs_dim = 20
            self.action_dim = 2
        elif obs_type == "image":
            # Image mode: separate normalizers for image and agent_pos
            self.obs_normalizer = None  # Not used
            self.action_normalizer = MinMaxNormalizer(self.actions)
            self.lowdim_normalizer = MinMaxNormalizer(
                np.concatenate([ep[:, :2] for ep in self.episode_agent_pos], axis=0)
            )
            self.obs_dim = None  # Dict obs
            self.action_dim = 2
        else:
            raise ValueError(f"Invalid obs_type: {obs_type}")

    def _load_data(self, val_ratio):
        """Load data from zarr file."""
        root = zarr.open(self.dataset_path, mode="r")

        # Load arrays
        state = np.array(root["data"]["state"])  # (N, 5)
        action = np.array(root["data"]["action"])  # (N, 2)
        episode_ends = np.array(root["meta"]["episode_ends"])  # (num_episodes,)

        if self.obs_type == "keypoint":
            keypoint = np.array(root["data"]["keypoint"])  # (N, 9, 2)
        elif self.obs_type == "image":
            img = np.array(root["data"]["img"])  # (N, H, W, C)

        # Split episodes
        total_episodes = len(episode_ends)
        if val_ratio > 0.0:
            val_count = max(1, int(total_episodes * val_ratio))
            train_count = total_episodes - val_count
            if self.mode == "train":
                episode_indices = list(range(train_count))
            else:
                episode_indices = list(range(train_count, total_episodes))
        else:
            episode_indices = list(range(total_episodes))

        # Convert to per-episode lists
        self.episode_obs = []
        self.episode_actions = []
        if self.obs_type == "image":
            self.episode_images = []
            self.episode_agent_pos = []

        all_obs = []
        all_actions = []

        start = 0
        for ep_idx in tqdm(episode_indices, desc=f"Loading {self.mode} data"):
            end = episode_ends[ep_idx]

            if self.obs_type == "state":
                obs = state[start:end].astype(np.float32)
            elif self.obs_type == "keypoint":
                # Flatten keypoints and concatenate with agent_pos
                kp = keypoint[start:end].reshape(end - start, -1)  # (T, 18)
                agent_pos = state[start:end, :2]  # (T, 2)
                obs = np.concatenate([kp, agent_pos], axis=-1).astype(np.float32)  # (T, 20)
            elif self.obs_type == "image":
                # Store separately
                self.episode_images.append(img[start:end].astype(np.float32))
                self.episode_agent_pos.append(state[start:end, :2].astype(np.float32))
                obs = None  # Handled separately

            act = action[start:end].astype(np.float32)

            if obs is not None:
                self.episode_obs.append(obs)
                all_obs.append(obs)
            self.episode_actions.append(act)
            all_actions.append(act)

            start = end

        # Concatenated arrays for normalizer computation
        if self.obs_type != "image":
            self.observations = np.concatenate(all_obs, axis=0)
        self.actions = np.concatenate(all_actions, axis=0)

        # Build index
        self.indices = []
        if self.obs_type == "image":
            for ep_idx, img_ep in enumerate(self.episode_images):
                ep_len = len(img_ep)
                for t in range(ep_len):
                    self.indices.append((ep_idx, t))
        else:
            for ep_idx, obs in enumerate(self.episode_obs):
                ep_len = len(obs)
                for t in range(ep_len):
                    self.indices.append((ep_idx, t))

        self.size = len(self.indices)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """Sample a sequence.

        Returns:
            dict with keys:
                - observations: (obs_steps, obs_dim) or dict for image mode
                - actions: (horizon, action_dim)
        """
        ep_idx, t = self.indices[idx]

        if self.obs_type == "image":
            ep_img = self.episode_images[ep_idx]
            ep_agent_pos = self.episode_agent_pos[ep_idx]
            ep_act = self.episode_actions[ep_idx]
            ep_len = len(ep_img)

            # Sample observation history
            img_list = []
            agent_pos_list = []
            for i in range(self.obs_steps):
                obs_t = max(t - (self.obs_steps - 1 - i), 0)
                img_list.append(ep_img[obs_t])
                agent_pos_list.append(ep_agent_pos[obs_t])

            images = np.stack(img_list, axis=0)  # (obs_steps, H, W, C)
            agent_pos = np.stack(agent_pos_list, axis=0)  # (obs_steps, 2)

            # Normalize
            images = images / 255.0  # [0, 1]
            if self.lowdim_normalizer is not None:
                agent_pos = self.lowdim_normalizer.normalize(agent_pos)

            observations = {
                "image": images,
                "agent_pos": agent_pos,
            }
        else:
            ep_obs = self.episode_obs[ep_idx]
            ep_act = self.episode_actions[ep_idx]
            ep_len = len(ep_obs)

            # Sample observation history
            obs_list = []
            for i in range(self.obs_steps):
                obs_t = max(t - (self.obs_steps - 1 - i), 0)
                obs_list.append(ep_obs[obs_t])
            observations = np.stack(obs_list, axis=0)  # (obs_steps, obs_dim)

            # Normalize
            if self.obs_normalizer is not None:
                observations = self.obs_normalizer.normalize(observations)

        # Sample action sequence
        act_list = []
        for i in range(self.horizon):
            act_t = min(t + i, ep_len - 1)
            act_list.append(ep_act[act_t])
        actions = np.stack(act_list, axis=0)  # (horizon, action_dim)

        # Normalize actions
        if self.action_normalizer is not None:
            actions = self.action_normalizer.normalize(actions)

        return {
            "observations": observations,
            "actions": actions,
        }

    def sample_batch(self, batch_size, rng=None):
        """Sample a random batch."""
        if rng is None:
            rng = np.random.default_rng()
        indices = rng.integers(0, self.size, size=batch_size)

        obs_batch = []
        act_batch = []
        for idx in indices:
            sample = self[idx]
            obs_batch.append(sample["observations"])
            act_batch.append(sample["actions"])

        if self.obs_type == "image":
            # Dict observations
            observations = {
                "image": np.stack([o["image"] for o in obs_batch], axis=0),
                "agent_pos": np.stack([o["agent_pos"] for o in obs_batch], axis=0),
            }
        else:
            observations = np.stack(obs_batch, axis=0)

        return {
            "observations": observations,
            "actions": np.stack(act_batch, axis=0),
        }

    def get_normalizer(self):
        """Get normalizers for observations and actions."""
        if self.obs_type == "image":
            return {
                "obs": None,
                "action": self.action_normalizer,
                "lowdim": self.lowdim_normalizer,
            }
        else:
            return {
                "obs": self.obs_normalizer,
                "action": self.action_normalizer,
            }
