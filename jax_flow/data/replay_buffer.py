"""Replay buffer for RL training.

Pure NumPy implementation with support for:
- Dict observations (image + lowdim)
- N-step returns
- Offline buffer filling from BC inference
"""

from collections import deque
from typing import Any

import numpy as np


class ReplayBuffer:
    """NumPy circular replay buffer with dict observation and N-step returns support.

    Stores transitions in a circular buffer and computes N-step returns on-the-fly.
    Supports both array observations and dict observations (e.g., {"image": ..., "lowdim": ...}).

    Args:
        capacity: Maximum buffer size
        obs_shape: Observation shape. Can be:
            - tuple: (obs_dim,) for array observations
            - dict: {"image": (H,W,C), "lowdim": (d,)} for dict observations
        action_dim: Action dimension
        n_step: N-step return horizon (1 = standard TD)
        gamma: Discount factor for N-step returns
    """

    def __init__(
        self,
        capacity: int = 200_000,
        obs_shape: dict[str, tuple] | tuple | None = None,
        action_dim: int | None = None,
        n_step: int = 3,
        gamma: float = 0.99,
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.n_step = n_step
        self.gamma = gamma

        # Main storage (circular buffer)
        self.ptr = 0  # Current write position
        self.size = 0  # Current buffer size

        # Initialize storage arrays
        self._init_storage()

        # N-step buffer (temporary storage for computing N-step returns)
        self.n_step_buffer = deque(maxlen=n_step)

    def _init_storage(self):
        """Initialize storage arrays based on observation shape."""
        if isinstance(self.obs_shape, dict):
            # Dict observation mode
            self.obs = {}
            self.next_obs = {}
            for key, shape in self.obs_shape.items():
                # Store images as uint8 to save memory
                if "image" in key.lower():
                    dtype = np.uint8
                else:
                    dtype = np.float32
                self.obs[key] = np.zeros((self.capacity, *shape), dtype=dtype)
                self.next_obs[key] = np.zeros((self.capacity, *shape), dtype=dtype)
        else:
            # Array observation mode
            self.obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
            self.next_obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)

        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.discounts = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray | dict,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray | dict,
        done: bool,
    ):
        """Add a single transition to the buffer.

        Internally handles N-step return computation via temporary buffer.

        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Episode termination flag
        """
        # Add to N-step buffer
        self.n_step_buffer.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
        })

        # If N-step buffer is full or episode ended, compute N-step return and store
        if len(self.n_step_buffer) == self.n_step or done:
            self._flush_n_step_buffer()

    def _flush_n_step_buffer(self):
        """Compute N-step returns and store transitions from N-step buffer."""
        while len(self.n_step_buffer) > 0:
            # Get the oldest transition
            transition = self.n_step_buffer[0]

            # Compute N-step return
            n_step_reward = 0.0
            discount = 1.0
            done_any = False

            for i, trans in enumerate(self.n_step_buffer):
                n_step_reward += discount * trans["reward"]
                discount *= self.gamma
                if trans["done"]:
                    done_any = True
                    break

            # Get N-step next_obs (last obs in buffer or terminal obs)
            if done_any:
                # Use the terminal next_obs
                n_step_next_obs = self.n_step_buffer[i]["next_obs"]
                n_step_discount = 0.0  # No bootstrap if episode ended
            else:
                # Use the last next_obs in buffer
                n_step_next_obs = self.n_step_buffer[-1]["next_obs"]
                n_step_discount = discount

            # Store in main buffer
            self._store_transition(
                obs=transition["obs"],
                action=transition["action"],
                reward=n_step_reward,
                next_obs=n_step_next_obs,
                done=transition["done"],
                discount=n_step_discount,
            )

            # Remove the oldest transition
            self.n_step_buffer.popleft()

            # If episode ended, clear remaining buffer
            if done_any:
                self.n_step_buffer.clear()
                break

    def _store_transition(
        self,
        obs: np.ndarray | dict,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray | dict,
        done: bool,
        discount: float,
    ):
        """Store a single transition in the main circular buffer."""
        if isinstance(self.obs_shape, dict):
            # Dict observation
            for key in self.obs_shape.keys():
                self.obs[key][self.ptr] = obs[key]
                self.next_obs[key][self.ptr] = next_obs[key]
        else:
            # Array observation
            self.obs[self.ptr] = obs
            self.next_obs[self.ptr] = next_obs

        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.discounts[self.ptr] = discount

        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator = None) -> dict:
        """Uniformly sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample
            rng: NumPy random generator (optional)

        Returns:
            Dictionary with keys: obs, action, reward, next_obs, done, discount
            All values are NumPy arrays ready for JAX conversion.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Sample indices
        indices = rng.integers(0, self.size, size=batch_size)

        # Gather batch
        if isinstance(self.obs_shape, dict):
            # Dict observation: convert images to float32 and normalize
            obs_batch = {}
            next_obs_batch = {}
            for key in self.obs_shape.keys():
                obs_data = self.obs[key][indices]
                next_obs_data = self.next_obs[key][indices]

                # Normalize images from uint8 [0, 255] to float32 [0, 1]
                if "image" in key.lower():
                    obs_data = obs_data.astype(np.float32) / 255.0
                    next_obs_data = next_obs_data.astype(np.float32) / 255.0

                obs_batch[key] = obs_data
                next_obs_batch[key] = next_obs_data
        else:
            # Array observation
            obs_batch = self.obs[indices]
            next_obs_batch = self.next_obs[indices]

        return {
            "obs": obs_batch,
            "action": self.actions[indices],
            "reward": self.rewards[indices],
            "next_obs": next_obs_batch,
            "done": self.dones[indices],
            "discount": self.discounts[indices],
        }

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size


class OfflineReplayBuffer(ReplayBuffer):
    """Offline replay buffer filled from demo dataset + BC inference.

    This buffer is pre-filled with offline data where:
    - obs contains base_action from BC policy
    - action is the ground truth combined action from demos
    - reward is 0 (or sparse reward at success)
    """

    @classmethod
    def from_dataset(
        cls,
        dataset,
        bc_agent,
        action_normalizer,
        capacity: int = 200_000,
        n_step: int = 3,
        gamma: float = 0.99,
        sparse_reward: bool = False,
    ):
        """Create and fill offline buffer from dataset + BC inference.

        Args:
            dataset: RobomimicDataset or RobomimicImageDataset instance
            bc_agent: Frozen BCAgent for computing base_action
            action_normalizer: Action normalizer (same as BC training)
            capacity: Buffer capacity
            n_step: N-step return horizon
            gamma: Discount factor
            sparse_reward: If True, give reward=1 at success, else reward=0

        Returns:
            Filled OfflineReplayBuffer instance
        """
        # Infer observation shape from dataset
        sample_obs = dataset[0]["observations"]
        if isinstance(sample_obs, dict):
            obs_shape = {key: val.shape[1:] for key, val in sample_obs.items()}
            # Add base_action field to obs_shape
            action_dim = dataset[0]["actions"].shape[-1]
            obs_shape["base_action"] = (action_dim,)
        else:
            obs_shape = sample_obs.shape[1:]
            action_dim = dataset[0]["actions"].shape[-1]

        # Create buffer
        buffer = cls(
            capacity=capacity,
            obs_shape=obs_shape,
            action_dim=action_dim,
            n_step=n_step,
            gamma=gamma,
        )

        print(f"Filling offline buffer from {len(dataset)} episodes...")

        # Fill buffer from dataset
        num_transitions = 0
        for ep_idx in range(len(dataset)):
            episode = dataset[ep_idx]
            obs_seq = episode["observations"]  # (T, obs_dim) or dict
            act_seq = episode["actions"]  # (T, action_dim)

            # Get sequence length
            if isinstance(obs_seq, dict):
                seq_len = len(obs_seq[list(obs_seq.keys())[0]])
            else:
                seq_len = len(obs_seq)

            # Process each timestep
            for t in range(seq_len - 1):
                # Extract obs at timestep t
                if isinstance(obs_seq, dict):
                    obs_t = {key: val[t] for key, val in obs_seq.items()}
                    next_obs_t = {key: val[t + 1] for key, val in obs_seq.items()}
                else:
                    obs_t = obs_seq[t]
                    next_obs_t = obs_seq[t + 1]

                # BC inference to get base_action
                # Need to batch the observation
                if isinstance(obs_t, dict):
                    obs_batch = {key: val[None, ...] for key, val in obs_t.items()}
                else:
                    obs_batch = obs_t[None, ...]

                base_actions = bc_agent.eval_actions(obs_batch)  # (1, horizon, action_dim)
                base_action = base_actions[0, 0]  # Take first step of first batch

                # Same for next_obs
                if isinstance(next_obs_t, dict):
                    next_obs_batch = {key: val[None, ...] for key, val in next_obs_t.items()}
                else:
                    next_obs_batch = next_obs_t[None, ...]

                next_base_actions = bc_agent.eval_actions(next_obs_batch)
                next_base_action = next_base_actions[0, 0]

                # Add base_action to observations
                if isinstance(obs_t, dict):
                    obs_with_base = {**obs_t, "base_action": base_action}
                    next_obs_with_base = {**next_obs_t, "base_action": next_base_action}
                else:
                    # For array obs, convert to dict
                    obs_with_base = {"obs": obs_t, "base_action": base_action}
                    next_obs_with_base = {"obs": next_obs_t, "base_action": next_base_action}

                # Get ground truth action (combined action from demo)
                gt_action = act_seq[t]

                # Compute reward
                done = (t == seq_len - 2)
                if sparse_reward and done:
                    reward = 1.0  # Success reward at episode end
                else:
                    reward = 0.0

                # Add to buffer
                buffer.add(
                    obs=obs_with_base,
                    action=gt_action,
                    reward=reward,
                    next_obs=next_obs_with_base,
                    done=done,
                )

                num_transitions += 1

            if (ep_idx + 1) % 100 == 0:
                print(f"  Processed {ep_idx + 1}/{len(dataset)} episodes, "
                      f"{num_transitions} transitions")

        print(f"Offline buffer filled: {len(buffer)} transitions")
        return buffer
