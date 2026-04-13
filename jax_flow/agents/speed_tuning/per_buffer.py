"""Prioritized Experience Replay buffer with SumTree."""

import numpy as np


class SumTree:
    """Binary sum-tree for O(log n) proportional sampling.

    Stores priorities in leaf nodes; internal nodes hold partial sums.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_ptr = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, data_idx: int, priority: float):
        """Update priority for a data index."""
        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(self, priority: float) -> int:
        """Add a new entry with given priority. Returns data index."""
        data_idx = self.data_ptr
        self.update(data_idx, priority)
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return data_idx

    def get(self, s: float) -> tuple[int, float]:
        """Sample a leaf by cumulative sum value s.

        Returns (data_idx, priority).
        """
        idx = 0  # root
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left] or right >= len(self.tree):
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return data_idx, self.tree[idx]

    @property
    def total(self) -> float:
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        leaf_start = self.capacity - 1
        if self.size == 0:
            return 1.0
        return max(self.tree[leaf_start : leaf_start + self.size].max(), 1e-6)


class PrioritizedReplayBuffer:
    """PER buffer with proportional prioritization.

    Stores single-step transitions (obs, action, reward, next_obs, done).
    Action is a discrete speed index (int).
    Observations can be arrays or dicts (for image tasks).

    Sampling: P(i) proportional to p_i^alpha.
    IS weights: w_i = (N * P(i))^(-beta) / max(w).
    """

    def __init__(
        self,
        capacity: int = 100_000,
        obs_shape: dict[str, tuple] | tuple | None = None,
        per_alpha: float = 0.6,
        per_epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.per_alpha = per_alpha
        self.per_epsilon = per_epsilon

        self.tree = SumTree(capacity)

        # Storage
        self._dict_obs = isinstance(obs_shape, dict)
        if self._dict_obs:
            self.obs = {}
            self.next_obs = {}
            for k, s in obs_shape.items():
                # Store images as uint8 to save 75% memory
                is_image = "image" in k.lower() and len(s) >= 3
                dtype = np.uint8 if is_image else np.float32
                self.obs[k] = np.zeros((capacity, *s), dtype=dtype)
                self.next_obs[k] = np.zeros((capacity, *s), dtype=dtype)
        else:
            self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
            self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)

        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action: int, reward: float, next_obs, done: bool):
        """Add transition with max priority."""
        priority = self.tree.max_priority**self.per_alpha
        data_idx = self.tree.add(priority)

        if self._dict_obs:
            for k in self.obs_shape:
                val = obs[k]
                nval = next_obs[k]
                # Convert float [0,1] images to uint8 for storage
                if self.obs[k].dtype == np.uint8:
                    if val.dtype != np.uint8:
                        val = (np.clip(val, 0.0, 1.0) * 255).astype(np.uint8)
                    if nval.dtype != np.uint8:
                        nval = (np.clip(nval, 0.0, 1.0) * 255).astype(np.uint8)
                self.obs[k][data_idx] = val
                self.next_obs[k][data_idx] = nval
        else:
            self.obs[data_idx] = obs
            self.next_obs[data_idx] = next_obs

        self.actions[data_idx] = action
        self.rewards[data_idx] = reward
        self.dones[data_idx] = float(done)

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Sample a prioritized batch.

        Args:
            batch_size: Number of transitions.
            beta: IS weight exponent (annealed from beta_start to 1.0).

        Returns:
            (batch_dict, is_weights, tree_indices) where:
              batch_dict has keys: observations, actions, rewards, next_observations, dones
              is_weights: (batch_size,) importance sampling weights
              tree_indices: (batch_size,) for priority updates
        """
        indices = np.empty(batch_size, dtype=np.int32)
        priorities = np.empty(batch_size, dtype=np.float64)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            data_idx, prio = self.tree.get(s)
            # Clamp to valid range
            data_idx = min(data_idx, self.tree.size - 1)
            data_idx = max(data_idx, 0)
            indices[i] = data_idx
            priorities[i] = max(prio, 1e-10)

        # IS weights
        probs = priorities / self.tree.total
        is_weights = (self.tree.size * probs) ** (-beta)
        is_weights /= is_weights.max()
        is_weights = is_weights.astype(np.float32)

        # Gather batch
        if self._dict_obs:
            obs_batch = {}
            next_obs_batch = {}
            for k in self.obs_shape:
                o = self.obs[k][indices]
                no = self.next_obs[k][indices]
                # Convert uint8 images back to float32 [0,1]
                if o.dtype == np.uint8:
                    o = o.astype(np.float32) / 255.0
                    no = no.astype(np.float32) / 255.0
                obs_batch[k] = o
                next_obs_batch[k] = no
        else:
            obs_batch = self.obs[indices]
            next_obs_batch = self.next_obs[indices]

        batch = {
            "observations": obs_batch,
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": next_obs_batch,
            "dones": self.dones[indices],
        }
        return batch, is_weights, indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        priorities = (np.abs(td_errors) + self.per_epsilon) ** self.per_alpha
        for idx, prio in zip(indices, priorities, strict=False):
            self.tree.update(int(idx), float(prio))

    def __len__(self) -> int:
        return self.tree.size
