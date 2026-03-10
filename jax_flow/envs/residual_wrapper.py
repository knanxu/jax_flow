"""Residual environment wrapper for RL fine-tuning.

Embeds a frozen BC policy inside the environment. The RL agent only sees
the residual action space. Internally manages BC action queue for chunking.

Wrapper stack for RL:
  RobomimicWrapper → FrameStackWrapper → ResidualEnvWrapper
  (no ActionChunkingWrapper needed — BC chunking is managed internally)
"""

from collections import deque

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np


class ResidualEnvWrapper(gym.Wrapper):
    """Residual RL environment wrapper.

    Embeds a frozen BC policy. env.step() receives a single-step residual action,
    internally computes combined = base_action + residual and executes it.

    The BC policy's action chunking is managed via an internal action queue:
    - On replan: call BC to get (horizon, action_dim) chunk, enqueue first act_steps
    - Each step: dequeue one base_action, add residual, execute combined
    - When queue is exhausted (after act_steps): trigger replan

    Observation dict includes 'base_action' field for the RL actor.

    Args:
        env: Base environment (RobomimicWrapper + FrameStackWrapper, no ActionChunking)
        bc_agent: Frozen BCAgent instance
        act_steps: Number of steps to execute per BC chunk (default: 8)
        obs_type: 'lowdim' or 'image'
    """

    def __init__(
        self,
        env,
        bc_agent,
        act_steps: int = 8,
        obs_type: str = "lowdim",
    ):
        super().__init__(env)
        self.bc_agent = bc_agent
        self.act_steps = act_steps
        self.obs_type = obs_type

        # BC action queue state
        self._base_action_queue: deque = deque()
        self._current_base_action: np.ndarray | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Call BC to get initial action chunk and populate queue
        self._replan(obs)

        # Augment obs with base_action
        obs = self._augment_obs(obs)
        return obs, info

    def step(self, residual_action):
        """Step with residual action.

        Args:
            residual_action: Single-step residual action (action_dim,).

        Returns:
            Standard Gymnasium step outputs. obs includes 'base_action'.
            info includes 'combined_action' and 'base_action'.
        """
        # Compute combined action
        combined = np.clip(
            self._current_base_action + residual_action, -1.0, 1.0
        )

        # Execute in environment
        obs, reward, terminated, truncated, info = self.env.step(combined)

        # Record actions for replay buffer
        info["combined_action"] = combined.copy()
        info["base_action"] = self._current_base_action.copy()

        # Update base action: dequeue next, or replan if queue empty
        if len(self._base_action_queue) == 0:
            self._replan(obs)
        else:
            self._current_base_action = self._base_action_queue.popleft()

        # Augment obs with base_action
        obs = self._augment_obs(obs)
        return obs, reward, terminated, truncated, info

    def _replan(self, obs):
        """Call frozen BC policy to get action chunk and populate queue."""
        # Batch the observation for BC inference
        obs_batch = self._batch_obs(obs)

        # BC inference: (1, horizon, action_dim)
        bc_actions = self.bc_agent.eval_actions(obs_batch)

        # Convert from JAX to numpy
        bc_actions = np.asarray(bc_actions)

        # Take first act_steps actions, put into queue
        chunk = bc_actions[0, :self.act_steps]  # (act_steps, action_dim)
        self._base_action_queue = deque(chunk[1:])  # Rest goes to queue
        self._current_base_action = chunk[0].copy()  # First one is current

    def _batch_obs(self, obs):
        """Add batch dimension to observation for BC inference."""
        if isinstance(obs, dict):
            # Filter out base_action (not part of BC input)
            return {
                key: jnp.array(val[np.newaxis, ...])
                for key, val in obs.items()
                if key != "base_action"
            }
        else:
            return jnp.array(obs[np.newaxis, ...])

    def _augment_obs(self, obs):
        """Add base_action field to observation."""
        if isinstance(obs, dict):
            obs["base_action"] = self._current_base_action.copy()
        else:
            # Lowdim: wrap in dict
            obs = {"obs": obs, "base_action": self._current_base_action.copy()}
        return obs

    def needs_replan(self):
        """Check if BC needs to replan (queue exhausted)."""
        return len(self._base_action_queue) == 0

    def seed(self, seed=None):
        """Set random seed."""
        return self.env.seed(seed)

    def render(self, **kwargs):
        """Render the environment."""
        return self.env.render(**kwargs)
