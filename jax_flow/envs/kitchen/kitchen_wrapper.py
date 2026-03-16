"""Kitchen environment wrapper adapted for jax_flow."""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from jax_flow.envs.robomimic_env import ActionChunkingWrapper, FrameStackWrapper


class KitchenWrapper(gym.Env):
    """Bridges old-gym Kitchen env to Gymnasium API with normalization.

    Tracks completed_tasks for p1-p7 metrics.
    """

    def __init__(self, env, obs_normalizer=None, action_normalizer=None,
                 max_episode_steps=280):
        self.env = env
        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer
        self.max_episode_steps = max_episode_steps
        self.t = 0

        # Normalized spaces [-1, 1]
        self.observation_space = Box(low=-1, high=1, shape=(60,), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(9,), dtype=np.float32)

    def _make_obs(self, raw_obs):
        """Construct 60D observation: qpos[:9] + qpos[-21:] + zeros(30)."""
        obs = raw_obs.astype(np.float32)
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize(obs)
        return obs

    def _unnormalize_action(self, action):
        if self.action_normalizer is not None:
            return self.action_normalizer.unnormalize(action)
        return action

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
        raw_obs = self.env.reset()  # old gym: returns obs only
        self.t = 0
        return self._make_obs(raw_obs), {}

    def step(self, action):
        raw_action = self._unnormalize_action(action)
        # old gym: returns (obs, reward, done, info)
        raw_obs, reward, done, info = self.env.step(raw_action)
        self.t += 1

        # Extract completed tasks
        completed = info.get("completed_tasks", set())
        num_completed = len(completed)
        info["completed_tasks"] = completed
        info["num_completed"] = num_completed
        info["success"] = float(num_completed >= 4)

        terminated = done
        truncated = self.t >= self.max_episode_steps if not done else False

        return self._make_obs(raw_obs), reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        return self.env.render(mode=mode)


def make_kitchen_env(
    env_name="kitchen-microwave-kettle-burner-light-v0",
    obs_normalizer=None,
    action_normalizer=None,
    max_episode_steps=280,
    frame_stack=None,
    act_exec_steps=None,
    seed=None,
    render_offscreen=None,
    **kwargs,
):
    """Factory function to create Kitchen environment.

    Args:
        env_name: Kitchen env ID (e.g. 'kitchen-all-v0').
        obs_normalizer: Observation normalizer.
        action_normalizer: Action normalizer.
        max_episode_steps: Maximum episode length.
        frame_stack: Number of frames to stack.
        act_exec_steps: Action execution steps for chunking.
        seed: Random seed.

    Returns:
        Wrapped Kitchen environment.
    """
    import gym as old_gym

    # Trigger gym registration
    import jax_flow.envs.kitchen.thirdparty  # noqa: F401
    from jax_flow.envs.kitchen.thirdparty.kitchen_lowdim_wrapper import (
        KitchenLowdimWrapper,
    )

    base_env = old_gym.make(env_name, use_abs_action=True)
    base_env = KitchenLowdimWrapper(env=base_env, init_qpos=None, init_qvel=None)

    env = KitchenWrapper(
        base_env,
        obs_normalizer=obs_normalizer,
        action_normalizer=action_normalizer,
        max_episode_steps=max_episode_steps,
    )

    if frame_stack is not None and frame_stack > 1:
        env = FrameStackWrapper(env, num_stack=frame_stack)

    if act_exec_steps is not None:
        env = ActionChunkingWrapper(env, act_exec_steps=act_exec_steps)

    if seed is not None:
        env.seed(seed)

    return env
