"""ResFiT-specific replay buffer with independent base_action storage.

Extends ReplayBuffer to store base_action and next_base_action as independent
fields (not inside obs dict), making the transition format cleaner for ResFiT training.
"""

from collections import deque

import numpy as np


class ResFiTReplayBuffer:
    """Replay buffer for ResFiT with independent base_action storage.

    Stores transitions with separate fields:
    - obs / next_obs: observation (without base_action)
    - action: combined action
    - base_action / next_base_action: BC policy actions (independent fields)
    - reward, done, discount: standard RL fields

    N-step return computation is inherited from the same logic as ReplayBuffer.

    Args:
        capacity: Maximum buffer size.
        obs_shape: Observation shape (dict or tuple, without base_action).
        action_dim: Action dimension.
        n_step: N-step return horizon (1 = standard TD).
        gamma: Discount factor for N-step returns.
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

        self.ptr = 0
        self.size = 0

        self._init_storage()

        # N-step buffer
        self.n_step_buffer: deque = deque(maxlen=n_step)

    def _init_storage(self):
        """Initialize storage arrays."""
        if isinstance(self.obs_shape, dict):
            self.obs = {}
            self.next_obs = {}
            for key, shape in self.obs_shape.items():
                dtype = np.uint8 if "image" in key.lower() else np.float32
                self.obs[key] = np.zeros((self.capacity, *shape), dtype=dtype)
                self.next_obs[key] = np.zeros((self.capacity, *shape), dtype=dtype)
        else:
            self.obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
            self.next_obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)

        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.base_actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.next_base_actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.discounts = np.zeros((self.capacity, 1), dtype=np.float32)

        # Episode boundary tracking
        self.episode_ids = np.zeros((self.capacity,), dtype=np.int64)
        self._current_episode_id = 0

    def add(
        self,
        obs,
        action: np.ndarray,
        base_action: np.ndarray,
        next_obs,
        next_base_action: np.ndarray,
        reward: float,
        done: bool,
    ):
        """Add a single transition to the buffer.

        Args:
            obs: Current observation (without base_action).
            action: Combined action taken.
            base_action: BC policy action for current step.
            next_obs: Next observation (without base_action).
            next_base_action: BC policy action for next step.
            reward: Reward received.
            done: Episode termination flag.
        """
        self.n_step_buffer.append({
            "obs": obs,
            "action": action,
            "base_action": base_action,
            "next_obs": next_obs,
            "next_base_action": next_base_action,
            "reward": reward,
            "done": done,
        })

        if len(self.n_step_buffer) == self.n_step or done:
            self._flush_n_step_buffer()

    def _flush_n_step_buffer(self):
        """Compute N-step returns and store transitions."""
        while len(self.n_step_buffer) > 0:
            transition = self.n_step_buffer[0]

            # Compute N-step return
            n_step_reward = 0.0
            discount = 1.0
            done_any = False
            last_idx = 0

            for i, trans in enumerate(self.n_step_buffer):
                n_step_reward += discount * trans["reward"]
                discount *= self.gamma
                last_idx = i
                if trans["done"]:
                    done_any = True
                    break

            # Get N-step next_obs and next_base_action
            if done_any:
                n_step_next_obs = self.n_step_buffer[last_idx]["next_obs"]
                n_step_next_base_action = self.n_step_buffer[last_idx]["next_base_action"]
                n_step_discount = 0.0
            else:
                n_step_next_obs = self.n_step_buffer[-1]["next_obs"]
                n_step_next_base_action = self.n_step_buffer[-1]["next_base_action"]
                n_step_discount = discount

            self._store_transition(
                obs=transition["obs"],
                action=transition["action"],
                base_action=transition["base_action"],
                next_obs=n_step_next_obs,
                next_base_action=n_step_next_base_action,
                reward=n_step_reward,
                done=transition["done"],
                discount=n_step_discount,
            )

            self.n_step_buffer.popleft()

            if done_any:
                self.n_step_buffer.clear()
                break

    def _store_transition(
        self,
        obs,
        action: np.ndarray,
        base_action: np.ndarray,
        next_obs,
        next_base_action: np.ndarray,
        reward: float,
        done: bool,
        discount: float,
    ):
        """Store a single transition in the circular buffer."""
        if isinstance(self.obs_shape, dict):
            for key in self.obs_shape.keys():
                self.obs[key][self.ptr] = obs[key]
                self.next_obs[key][self.ptr] = next_obs[key]
        else:
            self.obs[self.ptr] = obs
            self.next_obs[self.ptr] = next_obs

        self.actions[self.ptr] = action
        self.base_actions[self.ptr] = base_action
        self.next_base_actions[self.ptr] = next_base_action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.discounts[self.ptr] = discount
        self.episode_ids[self.ptr] = self._current_episode_id

        if done:
            self._current_episode_id += 1

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator = None) -> dict:
        """Uniformly sample a batch of transitions.

        Returns:
            Dictionary with keys: obs, action, base_action, next_obs,
            next_base_action, reward, done, discount.
        """
        if rng is None:
            rng = np.random.default_rng()

        indices = rng.integers(0, self.size, size=batch_size)

        if isinstance(self.obs_shape, dict):
            obs_batch = {}
            next_obs_batch = {}
            for key in self.obs_shape.keys():
                obs_data = self.obs[key][indices]
                next_obs_data = self.next_obs[key][indices]
                if "image" in key.lower():
                    obs_data = obs_data.astype(np.float32) / 255.0
                    next_obs_data = next_obs_data.astype(np.float32) / 255.0
                obs_batch[key] = obs_data
                next_obs_batch[key] = next_obs_data
        else:
            obs_batch = self.obs[indices]
            next_obs_batch = self.next_obs[indices]

        return {
            "obs": obs_batch,
            "action": self.actions[indices],
            "base_action": self.base_actions[indices],
            "next_obs": next_obs_batch,
            "next_base_action": self.next_base_actions[indices],
            "reward": self.rewards[indices],
            "done": self.dones[indices],
            "discount": self.discounts[indices],
        }

    def __len__(self) -> int:
        return self.size
