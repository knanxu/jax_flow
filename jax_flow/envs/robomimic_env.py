"""Robomimic environment wrapper."""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class RobomimicWrapper(gym.Env):
    """Robomimic environment wrapper with normalization.

    Follows qc project's design:
    - Normalizes observations and actions to [-1, 1]
    - Uses Gymnasium interface
    - Supports both lowdim and image observations
    """

    def __init__(
        self,
        env,
        obs_keys=(
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ),
        obs_normalizer=None,
        action_normalizer=None,
        max_episode_steps=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
    ):
        """Initialize wrapper.

        Args:
            env: Robomimic environment.
            obs_keys: Observation keys to extract.
            obs_normalizer: MinMaxNormalizer for observations.
            action_normalizer: MinMaxNormalizer for actions.
            max_episode_steps: Maximum episode length.
            render_hw: Render resolution (height, width).
            render_camera_name: Camera name for rendering.
        """
        self.env = env
        self.obs_keys = obs_keys
        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer
        self.max_episode_steps = max_episode_steps
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name

        self._seed = None
        self.t = 0

        # Setup action space: [-1, 1]
        low = np.full(env.action_dimension, fill_value=-1.0)
        high = np.full(env.action_dimension, fill_value=1.0)
        self.action_space = Box(low=low, high=high, dtype=np.float32)

        # Setup observation space: [-1, 1]
        obs_example = self._get_observation()
        low = np.full_like(obs_example, fill_value=-1.0)
        high = np.full_like(obs_example, fill_value=1.0)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def _get_observation(self):
        """Get observation from environment."""
        raw_obs = self.env.get_observation()
        obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)

        # Normalize if normalizer is provided
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize(obs)

        return obs.astype(np.float32)

    def seed(self, seed=None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
            self._seed = seed

    def reset(self, seed=None, options=None):
        """Reset environment.

        Args:
            seed: Random seed.
            options: Additional options (unused).

        Returns:
            observation: Initial observation.
            info: Additional info dict.
        """
        if seed is not None:
            self.seed(seed)

        self.env.reset()
        self.t = 0

        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action):
        """Step environment.

        Args:
            action: Normalized action in [-1, 1].

        Returns:
            observation: Next observation.
            reward: Reward.
            terminated: Whether episode terminated (success).
            truncated: Whether episode truncated (timeout).
            info: Additional info dict.
        """
        # Unnormalize action
        if self.action_normalizer is not None:
            raw_action = self.action_normalizer.unnormalize(action)
        else:
            raw_action = action

        # Step environment
        raw_obs, reward, done, info = self.env.step(raw_action)

        # Get normalized observation
        obs = self._get_observation()

        # Update timestep
        self.t += 1

        # Determine termination
        terminated = reward > 0.0  # Success
        truncated = (
            self.t >= self.max_episode_steps if self.max_episode_steps else False
        )

        # Add success flag to info
        info["success"] = 1 if terminated else 0

        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        """Render environment.

        Args:
            mode: Render mode (only 'rgb_array' supported).

        Returns:
            RGB image array.
        """
        h, w = self.render_hw
        return self.env.render(
            mode=mode, height=h, width=w, camera_name=self.render_camera_name
        )


class FrameStackWrapper(gym.Wrapper):
    """Frame stacking wrapper.

    Stacks the last n observations along a new axis.
    """

    def __init__(self, env, num_stack=2):
        """Initialize wrapper.

        Args:
            env: Environment to wrap.
            num_stack: Number of frames to stack.
        """
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = None

        # Update observation space
        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], num_stack, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def reset(self, **kwargs):
        """Reset environment and initialize frame stack."""
        obs, info = self.env.reset(**kwargs)

        # Initialize frame stack with repeated first observation
        self.frames = np.repeat(obs[np.newaxis, ...], self.num_stack, axis=0)

        return self.frames.copy(), info

    def step(self, action):
        """Step environment and update frame stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Shift frames and add new observation
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = obs

        return self.frames.copy(), reward, terminated, truncated, info


def make_robomimic_env(
    env_name,
    dataset_path,
    obs_keys=(
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    ),
    obs_normalizer=None,
    action_normalizer=None,
    max_episode_steps=400,
    frame_stack=None,
    seed=None,
):
    """Factory function to create robomimic environment.

    Args:
        env_name: Environment name (lift, can, square, etc.).
        dataset_path: Path to dataset (for loading env metadata).
        obs_keys: Observation keys to extract.
        obs_normalizer: Observation normalizer.
        action_normalizer: Action normalizer.
        max_episode_steps: Maximum episode length.
        frame_stack: Number of frames to stack (None = no stacking).
        seed: Random seed.

    Returns:
        Wrapped environment.
    """
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    # Initialize observation modality mapping
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": obs_keys})

    # Load environment metadata from dataset
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

    # Create environment
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
    )

    # Wrap environment
    env = RobomimicWrapper(
        env=env,
        obs_keys=obs_keys,
        obs_normalizer=obs_normalizer,
        action_normalizer=action_normalizer,
        max_episode_steps=max_episode_steps,
    )

    # Add frame stacking if requested
    if frame_stack is not None and frame_stack > 1:
        env = FrameStackWrapper(env, num_stack=frame_stack)

    # Set seed
    if seed is not None:
        env.seed(seed)

    return env
