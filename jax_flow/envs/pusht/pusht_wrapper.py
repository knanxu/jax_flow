"""Push-T environment wrapper adapted for jax_flow."""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from jax_flow.envs.robomimic_env import ActionChunkingWrapper, FrameStackWrapper


class PushTStateWrapper(gym.Env):
    """Wrapper for PushT state observations with normalization."""

    def __init__(self, env, obs_normalizer=None, action_normalizer=None, max_episode_steps=300):
        self.env = env
        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer
        self.max_episode_steps = max_episode_steps
        self.t = 0

        # Normalized spaces [-1, 1]
        self.observation_space = Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def _normalize_obs(self, obs):
        obs = obs.astype(np.float32)
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize(obs)
        return obs

    def _unnormalize_action(self, action):
        if self.action_normalizer is not None:
            return self.action_normalizer.unnormalize(action)
        return action

    def seed(self, seed=None):
        self.env.seed(seed)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self.t = 0
        return self._normalize_obs(obs), info

    def step(self, action):
        raw_action = self._unnormalize_action(action)
        obs, reward, terminated, truncated, info = self.env.step(raw_action)
        self.t += 1
        if self.t >= self.max_episode_steps:
            truncated = True
        info["success"] = float(reward >= 0.95)
        return self._normalize_obs(obs), reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        return self.env.render(mode)


class PushTKeypointWrapper(gym.Env):
    """Wrapper for PushT keypoint observations with normalization."""

    def __init__(self, env, obs_normalizer=None, action_normalizer=None, max_episode_steps=300):
        self.env = env
        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer
        self.max_episode_steps = max_episode_steps
        self.t = 0

        # Keypoint obs: (block_kps 18 + agent_pos 2 + mask 20) = 40D from PushTKeypointsEnv
        # We only use the first 20D (kps + agent_pos), drop the mask
        obs_dim = 20
        self.obs_dim = obs_dim
        self.observation_space = Box(low=-1, high=1, shape=(obs_dim,), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def _normalize_obs(self, obs):
        # PushTKeypointsEnv returns (kps_flat + agent_pos + mask), take first half
        half = len(obs) // 2
        obs = obs[:half].astype(np.float32)
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize(obs)
        return obs

    def _unnormalize_action(self, action):
        if self.action_normalizer is not None:
            return self.action_normalizer.unnormalize(action)
        return action

    def seed(self, seed=None):
        self.env.seed(seed)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self.t = 0
        return self._normalize_obs(obs), info

    def step(self, action):
        raw_action = self._unnormalize_action(action)
        obs, reward, terminated, truncated, info = self.env.step(raw_action)
        self.t += 1
        if self.t >= self.max_episode_steps:
            truncated = True
        info["success"] = float(reward >= 0.95)
        return self._normalize_obs(obs), reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        return self.env.render(mode)


class PushTImageWrapper(gym.Env):
    """Wrapper for PushT image observations with normalization."""

    def __init__(self, env, obs_normalizer=None, action_normalizer=None,
                 lowdim_normalizer=None, max_episode_steps=300):
        self.env = env
        self.obs_normalizer = obs_normalizer  # unused for image
        self.action_normalizer = action_normalizer
        self.lowdim_normalizer = lowdim_normalizer  # for agent_pos
        self.max_episode_steps = max_episode_steps
        self.t = 0

        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = None  # Dict space

    def _process_obs(self, obs):
        """Process dict obs from PushTImageEnv."""
        result = {}
        # image: (C, H, W) float [0,1] -> (H, W, C) float [0,1]
        img = obs["image"]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        result["image"] = img.astype(np.float32)

        # agent_pos: (2,) -> normalized
        agent_pos = obs["agent_pos"].astype(np.float32)
        if self.lowdim_normalizer is not None:
            agent_pos = self.lowdim_normalizer.normalize(agent_pos)
        result["agent_pos"] = agent_pos
        return result

    def _unnormalize_action(self, action):
        if self.action_normalizer is not None:
            return self.action_normalizer.unnormalize(action)
        return action

    def seed(self, seed=None):
        self.env.seed(seed)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self.t = 0
        return self._process_obs(obs), info

    def step(self, action):
        raw_action = self._unnormalize_action(action)
        obs, reward, terminated, truncated, info = self.env.step(raw_action)
        self.t += 1
        if self.t >= self.max_episode_steps:
            truncated = True
        info["success"] = float(reward >= 0.95)
        return self._process_obs(obs), reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        return self.env.render(mode)


def make_pusht_env(
    obs_type="state",
    obs_normalizer=None,
    action_normalizer=None,
    lowdim_normalizer=None,
    max_episode_steps=300,
    frame_stack=None,
    act_exec_steps=None,
    seed=None,
    render_offscreen=None,
    **kwargs,
):
    """Factory function to create Push-T environment.

    Args:
        obs_type: 'state', 'keypoint', or 'image'.
        obs_normalizer: Observation normalizer (state/keypoint mode).
        action_normalizer: Action normalizer.
        lowdim_normalizer: Lowdim normalizer (image mode, for agent_pos).
        max_episode_steps: Maximum episode length.
        frame_stack: Number of frames to stack.
        act_exec_steps: Action execution steps for chunking.
        seed: Random seed.

    Returns:
        Wrapped Push-T environment.
    """
    if obs_type == "state":
        from jax_flow.envs.pusht.pusht_env import PushTEnv
        base_env = PushTEnv(legacy=False, render_size=96, render_action=True)
        env = PushTStateWrapper(
            base_env,
            obs_normalizer=obs_normalizer,
            action_normalizer=action_normalizer,
            max_episode_steps=max_episode_steps,
        )
    elif obs_type == "keypoint":
        from jax_flow.envs.pusht.pusht_keypoints_env import PushTKeypointsEnv
        base_env = PushTKeypointsEnv(
            legacy=False, render_size=96, render_action=True,
            keypoint_visible_rate=1.0, agent_keypoints=False,
        )
        env = PushTKeypointWrapper(
            base_env,
            obs_normalizer=obs_normalizer,
            action_normalizer=action_normalizer,
            max_episode_steps=max_episode_steps,
        )
    elif obs_type == "image":
        from jax_flow.envs.pusht.pusht_image_env import PushTImageEnv
        base_env = PushTImageEnv(legacy=False, render_size=96)
        env = PushTImageWrapper(
            base_env,
            obs_normalizer=obs_normalizer,
            action_normalizer=action_normalizer,
            lowdim_normalizer=lowdim_normalizer,
            max_episode_steps=max_episode_steps,
        )
    else:
        raise ValueError(f"Invalid obs_type: {obs_type}")

    # Frame stacking
    if frame_stack is not None and frame_stack > 1:
        env = FrameStackWrapper(env, num_stack=frame_stack)

    # Action chunking
    if act_exec_steps is not None:
        env = ActionChunkingWrapper(env, act_exec_steps=act_exec_steps)

    if seed is not None:
        env.seed(seed)

    return env
