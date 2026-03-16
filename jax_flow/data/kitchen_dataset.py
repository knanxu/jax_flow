"""Kitchen dataset for jax_flow."""

import pathlib
import struct
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from jax_flow.data.normalizer import MinMaxNormalizer


def parse_mjl_logs(read_filename, skipamount):
    """Parse MuJoCo log binary files."""
    with open(read_filename, mode="rb") as file:
        fileContent = file.read()
    headers = struct.unpack("iiiiiii", fileContent[:28])
    nq = headers[0]
    nv = headers[1]
    nu = headers[2]
    nmocap = headers[3]
    nsensordata = headers[4]
    name_len = headers[6]
    struct.unpack(str(name_len) + "s", fileContent[28 : 28 + name_len])
    rem_size = len(fileContent[28 + name_len :])
    num_floats = int(rem_size / 4)
    dat = np.asarray(struct.unpack(str(num_floats) + "f", fileContent[28 + name_len :]))
    recsz = 1 + nq + nv + nu + 7 * nmocap + nsensordata + headers[5]
    if rem_size % recsz != 0:
        raise ValueError(f"Error parsing {read_filename}")
    dat = np.reshape(dat, (int(len(dat) / recsz), recsz))
    dat = dat.T

    qpos = dat[1 : nq + 1, :].T[::skipamount, :]
    ctrl = dat[nq + nv + 1 : nq + nv + nu + 1, :].T[::skipamount, :]

    return {"qpos": qpos, "ctrl": ctrl}


def download_kitchen_dataset(
    dataset_filename="kitchen/kitchen_demos_multitask.zip",
    repo_id="ChaoyiPan/mip-dataset",
):
    """Download Kitchen dataset from HuggingFace and extract locally."""
    print(f"Downloading Kitchen dataset from {repo_id}/{dataset_filename}")
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

    dataset_name = zip_path_obj.stem
    dataset_path = extract_dir / dataset_name

    if not dataset_path.exists():
        extracted_dirs = [
            d for d in extract_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        if extracted_dirs:
            dataset_path = extracted_dirs[0]
        else:
            raise FileNotFoundError(f"No directory found after extracting {zip_path}")

    # The zip extracts to kitchen_demos_multitask/ which may contain
    # a nested kitchen_demos_multitask/ subdirectory with the actual MJL files.
    # Find the directory that contains */*.mjl files.
    mjl_files = list(dataset_path.glob("*/*.mjl"))
    if not mjl_files:
        # Try one level deeper
        for subdir in dataset_path.iterdir():
            if subdir.is_dir():
                mjl_files = list(subdir.glob("*/*.mjl"))
                if mjl_files:
                    dataset_path = subdir
                    break

    print(f"Extracted dataset to: {dataset_path}")
    return str(dataset_path)


class KitchenDataset:
    """Kitchen dataset for state observations.

    Parses MJL binary logs and converts to episode-list format.
    obs = concat(qpos[:9], qpos[-21:], zeros(30)) = 60D
    action = ctrl = 9D
    """

    def __init__(
        self,
        dataset_path,
        horizon=16,
        obs_steps=2,
        act_steps=8,
        abs_action=True,
        robot_noise_ratio=0.1,
        val_ratio=0.0,
        mode="train",
    ):
        self.dataset_path = dataset_path
        self.horizon = horizon
        self.obs_steps = obs_steps
        self.act_steps = act_steps
        self.mode = mode
        self.obs_dim = 60
        self.action_dim = 9

        self._load_data(val_ratio, robot_noise_ratio)

        self.obs_normalizer = MinMaxNormalizer(self.observations)
        self.action_normalizer = MinMaxNormalizer(self.actions)

    def _load_data(self, val_ratio, robot_noise_ratio):
        """Load data from MJL files."""
        robot_pos_noise_amp = np.array(
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
             0.005, 0.005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
             0.005, 0.005, 0.005,
             0.1, 0.1, 0.1, 0.005, 0.005, 0.005,
             0.1, 0.1, 0.1, 0.005],
            dtype=np.float32,
        )
        rng = np.random.default_rng(seed=42)

        data_directory = pathlib.Path(self.dataset_path)
        mjl_files = sorted(data_directory.glob("*/*.mjl"))

        # Parse all episodes
        all_episodes = []
        for mjl_path in tqdm(mjl_files, desc="Parsing MJL files"):
            try:
                data = parse_mjl_logs(str(mjl_path.absolute()), skipamount=40)
                qpos = data["qpos"].astype(np.float32)
                obs = np.concatenate(
                    [qpos[:, :9], qpos[:, -21:], np.zeros((len(qpos), 30), dtype=np.float32)],
                    axis=-1,
                )
                if robot_noise_ratio > 0:
                    noise = (
                        robot_noise_ratio * robot_pos_noise_amp
                        * rng.uniform(low=-1.0, high=1.0, size=(obs.shape[0], 30))
                    )
                    obs[:, :30] += noise
                act = data["ctrl"].astype(np.float32)
                all_episodes.append((obs, act))
            except Exception as e:
                print(f"Warning: Error parsing {mjl_path}: {e}")

        # Split train/val
        total_episodes = len(all_episodes)
        if val_ratio > 0.0:
            val_count = max(1, int(total_episodes * val_ratio))
            train_count = total_episodes - val_count
            if self.mode == "train":
                episodes = all_episodes[:train_count]
            else:
                episodes = all_episodes[train_count:]
        else:
            episodes = all_episodes

        # Convert to per-episode lists
        self.episode_obs = []
        self.episode_actions = []
        all_obs = []
        all_actions = []

        for obs, act in episodes:
            self.episode_obs.append(obs)
            self.episode_actions.append(act)
            all_obs.append(obs)
            all_actions.append(act)

        self.observations = np.concatenate(all_obs, axis=0)
        self.actions = np.concatenate(all_actions, axis=0)

        # Build index
        self.indices = []
        for ep_idx, obs in enumerate(self.episode_obs):
            ep_len = len(obs)
            for t in range(ep_len):
                self.indices.append((ep_idx, t))

        self.size = len(self.indices)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """Sample a sequence."""
        ep_idx, t = self.indices[idx]
        ep_obs = self.episode_obs[ep_idx]
        ep_act = self.episode_actions[ep_idx]
        ep_len = len(ep_obs)

        # Observation history with padding
        obs_list = []
        for i in range(self.obs_steps):
            obs_t = max(t - (self.obs_steps - 1 - i), 0)
            obs_list.append(ep_obs[obs_t])
        observations = np.stack(obs_list, axis=0)

        # Action sequence with padding
        act_list = []
        for i in range(self.horizon):
            act_t = min(t + i, ep_len - 1)
            act_list.append(ep_act[act_t])
        actions = np.stack(act_list, axis=0)

        # Normalize
        observations = self.obs_normalizer.normalize(observations)
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

        return {
            "observations": np.stack(obs_batch, axis=0),
            "actions": np.stack(act_batch, axis=0),
        }

    def get_normalizer(self):
        """Get normalizers."""
        return {
            "obs": self.obs_normalizer,
            "action": self.action_normalizer,
        }
