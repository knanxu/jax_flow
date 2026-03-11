"""Robomimic environment wrapper."""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict

# DexMimicGen environment registry: env_name -> default config
_DUAL_ARM_OBS_KEYS = (
    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
    "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos",
)

DEXMIMICGEN_ENVS = {
    "TwoArmThreading": {
        "obs_keys": _DUAL_ARM_OBS_KEYS,
        "lowdim_keys": _DUAL_ARM_OBS_KEYS,
        "image_keys": ("agentview_image", "robot0_eye_in_hand_image", "robot1_eye_in_hand_image"),
        "max_episode_steps": 400,
    },
    "TwoArmThreePieceAssembly": {
        "obs_keys": _DUAL_ARM_OBS_KEYS,
        "lowdim_keys": _DUAL_ARM_OBS_KEYS,
        "image_keys": ("agentview_image", "robot0_eye_in_hand_image", "robot1_eye_in_hand_image"),
        "max_episode_steps": 300,
    },
    "TwoArmTransport": {
        "obs_keys": _DUAL_ARM_OBS_KEYS,
        "lowdim_keys": _DUAL_ARM_OBS_KEYS,
        "image_keys": (
            "agentview_image", "robot0_eye_in_hand_image", "robot1_eye_in_hand_image",
            "shouldercamera0_image", "shouldercamera1_image",
        ),
        "max_episode_steps": 1200,
    },
}


class RobomimicWrapper(gym.Env):
    """Robomimic environment wrapper with normalization.

    Supports both lowdim and image observations.
    Uses Gymnasium interface with normalized actions [-1, 1].
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

        # Setup observation space
        obs_example = self._get_observation()
        low = np.full_like(obs_example, fill_value=-1.0)
        high = np.full_like(obs_example, fill_value=1.0)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def _get_observation(self):
        """Get concatenated lowdim observation from environment."""
        try:
            raw_obs = self.env.get_observation()
        except KeyError:
            # DexMimicGen envs lack 'object-state'; get raw obs directly from robosuite
            di = self.env.env._get_observations(force_update=True)
            raw_obs = {}
            for robot in self.env.env.robots:
                pf = robot.robot_model.naming_prefix
                for k in di:
                    if k.startswith(pf) and not k.endswith("proprio-state"):
                        raw_obs[k] = np.array(di[k])
        obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)

        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize(obs)

        return obs.astype(np.float32)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            self._seed = seed

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.env.reset()
        self.t = 0

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        # Unnormalize action
        if self.action_normalizer is not None:
            raw_action = self.action_normalizer.unnormalize(action)
        else:
            raw_action = action

        raw_obs, reward, done, info = self.env.step(raw_action)
        obs = self._get_observation()
        self.t += 1

        # Check success via env.is_success() (robomimic standard)
        success = self.env.is_success()
        is_success = success.get("task", False) if isinstance(success, dict) else bool(success)

        terminated = is_success
        truncated = (
            self.t >= self.max_episode_steps if self.max_episode_steps else False
        )

        info["success"] = float(is_success)

        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode, height=h, width=w, camera_name=self.render_camera_name
        )


class RobomimicImageWrapper(gym.Env):
    """Robomimic environment wrapper for image observations.

    Returns dict observations with image and lowdim keys.
    """

    def __init__(
        self,
        env,
        image_keys=("agentview_image",),
        lowdim_keys=("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
        action_normalizer=None,
        lowdim_normalizer=None,
        max_episode_steps=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
    ):
        self.env = env
        self.image_keys = image_keys
        self.lowdim_keys = lowdim_keys
        self.action_normalizer = action_normalizer
        self.lowdim_normalizer = lowdim_normalizer
        self.max_episode_steps = max_episode_steps
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name

        self._seed = None
        self.t = 0

        # Action space
        low = np.full(env.action_dimension, fill_value=-1.0)
        high = np.full(env.action_dimension, fill_value=1.0)
        self.action_space = Box(low=low, high=high, dtype=np.float32)

        # Observation space (simplified)
        self.observation_space = None  # Dict space, set after first obs

    def _get_observation(self):
        """Get dict observation from environment."""
        raw_obs = self.env.get_observation()
        obs = {}

        # Image observations: robomimic returns (C, H, W) float [0, 1], convert to (H, W, C)
        for key in self.image_keys:
            if key in raw_obs:
                img = raw_obs[key]
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                else:
                    img = img.astype(np.float32)
                obs[key] = img

        # Lowdim observations — keep separate keys to match dataset/encoder
        if self.lowdim_keys:
            lowdim_concat = np.concatenate(
                [raw_obs[key] for key in self.lowdim_keys], axis=0
            ).astype(np.float32)
            if self.lowdim_normalizer is not None:
                lowdim_concat = self.lowdim_normalizer.normalize(lowdim_concat)
            # Split back into per-key arrays
            offset = 0
            for key in self.lowdim_keys:
                dim = raw_obs[key].shape[0]
                obs[key] = lowdim_concat[offset:offset + dim]
                offset += dim

        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            self._seed = seed

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.env.reset()
        self.t = 0

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        if self.action_normalizer is not None:
            raw_action = self.action_normalizer.unnormalize(action)
        else:
            raw_action = action

        raw_obs, reward, done, info = self.env.step(raw_action)
        obs = self._get_observation()
        self.t += 1

        success = self.env.is_success()
        is_success = success.get("task", False) if isinstance(success, dict) else bool(success)

        terminated = is_success
        truncated = (
            self.t >= self.max_episode_steps if self.max_episode_steps else False
        )

        info["success"] = float(is_success)

        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode, height=h, width=w, camera_name=self.render_camera_name
        )


class FrameStackWrapper(gym.Wrapper):
    """Frame stacking wrapper.

    Stacks the last n observations along a new axis.
    Works with both array and dict observations.
    """

    def __init__(self, env, num_stack=2):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = None

        # Update observation space for array obs
        if isinstance(self.observation_space, Box):
            low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
            high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
            self.observation_space = Box(
                low=low, high=high, dtype=self.observation_space.dtype
            )

    def _stack_obs(self, obs):
        """Stack observation into frame buffer."""
        if isinstance(obs, dict):
            # Dict observations: stack each key separately
            if self.frames is None:
                self.frames = {
                    key: np.repeat(val[np.newaxis, ...], self.num_stack, axis=0)
                    for key, val in obs.items()
                }
            else:
                for key in obs:
                    self.frames[key] = np.roll(self.frames[key], shift=-1, axis=0)
                    self.frames[key][-1] = obs[key]

            return {key: val.copy() for key, val in self.frames.items()}
        else:
            # Array observations
            if self.frames is None:
                self.frames = np.repeat(obs[np.newaxis, ...], self.num_stack, axis=0)
            else:
                self.frames = np.roll(self.frames, shift=-1, axis=0)
                self.frames[-1] = obs

            return self.frames.copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = None  # Reset frame buffer
        stacked = self._stack_obs(obs)
        return stacked, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        stacked = self._stack_obs(obs)
        return stacked, reward, terminated, truncated, info

    def seed(self, seed=None):
        """Set random seed."""
        return self.env.seed(seed)


class ActionChunkingWrapper(gym.Wrapper):
    """Action chunking wrapper for executing action sequences.

    The policy predicts a sequence of actions (horizon steps),
    but only the first act_exec_steps are executed before re-planning.
    """

    def __init__(self, env, act_exec_steps=8):
        super().__init__(env)
        self.act_exec_steps = act_exec_steps
        self._action_buffer = None
        self._buffer_idx = 0

    def reset(self, **kwargs):
        self._action_buffer = None
        self._buffer_idx = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Step with action chunking.

        Args:
            action: Either a single action (action_dim,) or
                    an action sequence (horizon, action_dim), or None
                    to use buffered actions.

        Returns:
            Standard Gymnasium step outputs.
        """
        if action is not None and action.ndim == 2:
            # Action sequence: store in buffer
            self._action_buffer = action
            self._buffer_idx = 0

        if self._action_buffer is not None and self._buffer_idx < len(self._action_buffer):
            current_action = self._action_buffer[self._buffer_idx]
            self._buffer_idx += 1
        else:
            current_action = action if action is not None and action.ndim == 1 else action[0]

        return self.env.step(current_action)

    def needs_replan(self):
        """Check if the policy needs to re-plan."""
        return (
            self._action_buffer is None
            or self._buffer_idx >= self.act_exec_steps
        )

    def seed(self, seed=None):
        """Set random seed."""
        return self.env.seed(seed)


def make_robomimic_env(
    env_name,
    dataset_path,
    obs_keys=(
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    ),
    obs_type="lowdim",
    obs_normalizer=None,
    action_normalizer=None,
    lowdim_normalizer=None,
    max_episode_steps=400,
    frame_stack=None,
    act_exec_steps=None,
    image_keys=None,
    lowdim_keys=None,
    seed=None,
    render_offscreen=None,
):
    """Factory function to create robomimic environment.

    Args:
        env_name: Environment name (lift, can, square, etc.).
        dataset_path: Path to dataset (for loading env metadata).
        obs_keys: Observation keys for lowdim mode.
        obs_type: 'lowdim' or 'image'.
        obs_normalizer: Observation normalizer (lowdim mode).
        action_normalizer: Action normalizer.
        lowdim_normalizer: Lowdim normalizer (image mode).
        max_episode_steps: Maximum episode length.
        frame_stack: Number of frames to stack (None = no stacking).
        act_exec_steps: Action execution steps for chunking (None = no chunking).
        image_keys: Image observation keys (image mode).
        lowdim_keys: Lowdim observation keys (image mode).
        seed: Random seed.

    Returns:
        Wrapped environment.
    """
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    # Auto-fill defaults from DexMimicGen registry if env_name matches
    if env_name in DEXMIMICGEN_ENVS:
        defaults = DEXMIMICGEN_ENVS[env_name]
        # Use registry defaults for unspecified keys
        if obs_keys == ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"):
            obs_keys = defaults["obs_keys"]
        if image_keys is None:
            image_keys = defaults["image_keys"]
        if lowdim_keys is None:
            lowdim_keys = defaults["lowdim_keys"]
        if max_episode_steps == 400:
            max_episode_steps = defaults["max_episode_steps"]

        # Register DexMimicGen environments
        try:
            import dexmimicgen  # noqa: F401
        except ImportError:
            print("Warning: dexmimicgen not installed. Install with: pip install -e /path/to/dexmimicgen")

    # Initialize observation modality mapping
    if obs_type == "image":
        obs_modality_mapping = {
            "low_dim": list(lowdim_keys or []),
            "rgb": list(image_keys or ["agentview_image"]),
        }
    else:
        obs_modality_mapping = {"low_dim": list(obs_keys)}

    ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_mapping)

    # Load environment metadata from dataset
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

    # Fix MimicGen env names: Strip _D0/_D1/_D2 suffix (e.g. Stack_D0 -> Stack)
    import re
    raw_env_name = env_meta.get("env_name", "")
    stripped = re.sub(r"_D\d+$", "", raw_env_name)
    if stripped != raw_env_name:
        env_meta["env_name"] = stripped

    # Fix old controller config format for robosuite >= 1.5
    # MimicGen datasets use flat OSC_POSE config; robosuite 1.5+ needs BASIC composite wrapper
    # Skip if controller_configs is a list (DexMimicGen dual-arm composite configs)
    env_kwargs = env_meta.get("env_kwargs", {})
    ctrl_cfg = env_kwargs.get("controller_configs", {})
    if isinstance(ctrl_cfg, dict) and ctrl_cfg.get("type") == "OSC_POSE":
        arm_cfg = {k: v for k, v in ctrl_cfg.items() if k != "type"}
        arm_cfg["type"] = "OSC_POSE"
        arm_cfg["input_ref_frame"] = arm_cfg.get("input_ref_frame", "world")
        arm_cfg["gripper"] = arm_cfg.get("gripper", {"type": "GRIP"})
        env_kwargs["controller_configs"] = {
            "type": "BASIC",
            "body_parts": {"right": arm_cfg},
        }
        # Remove deprecated keys
        env_kwargs.pop("render_gpu_device_id", None)
        env_meta["env_kwargs"] = env_kwargs

    # Remove unsupported kwargs for DexMimicGen environments
    if env_name in DEXMIMICGEN_ENVS:
        for unsupported_key in ("env_lang", "render_gpu_device_id"):
            env_kwargs.pop(unsupported_key, None)
        env_meta["env_kwargs"] = env_kwargs

    # Create environment
    use_image = obs_type == "image"
    need_offscreen = render_offscreen if render_offscreen is not None else use_image
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=need_offscreen,
        use_image_obs=use_image,
    )

    # Patch get_observation for DexMimicGen envs (no 'object-state' key)
    if env_name in DEXMIMICGEN_ENVS:
        import robomimic.utils.obs_utils as ObsUtils
        _robomimic_env = env  # this is the EnvRobosuite before any wrapping

        def _patched_get_obs(di=None):
            if di is None:
                di = _robomimic_env.env._get_observations(force_update=True)
            ret = {}
            for k in di:
                if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb"):
                    ret[k] = di[k][::-1]
                    if _robomimic_env.postprocess_visual_obs:
                        ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)
            # Skip 'object-state' — DexMimicGen dual-arm envs don't have it
            if "object-state" in di:
                ret["object"] = np.array(di["object-state"])
            for robot in _robomimic_env.env.robots:
                pf = robot.robot_model.naming_prefix
                for k in di:
                    if k.startswith(pf) and (k not in ret) and (not k.endswith("proprio-state")):
                        ret[k] = np.array(di[k])
            return ret

        _robomimic_env.get_observation = _patched_get_obs

    # Wrap environment
    if obs_type == "image":
        env = RobomimicImageWrapper(
            env=env,
            image_keys=image_keys or ("agentview_image",),
            lowdim_keys=lowdim_keys or (),
            action_normalizer=action_normalizer,
            lowdim_normalizer=lowdim_normalizer,
            max_episode_steps=max_episode_steps,
        )
    else:
        env = RobomimicWrapper(
            env=env,
            obs_keys=obs_keys,
            obs_normalizer=obs_normalizer,
            action_normalizer=action_normalizer,
            max_episode_steps=max_episode_steps,
        )

    # Add frame stacking
    if frame_stack is not None and frame_stack > 1:
        env = FrameStackWrapper(env, num_stack=frame_stack)

    # Add action chunking
    if act_exec_steps is not None:
        env = ActionChunkingWrapper(env, act_exec_steps=act_exec_steps)

    # Set seed
    if seed is not None:
        env.seed(seed)

    return env
