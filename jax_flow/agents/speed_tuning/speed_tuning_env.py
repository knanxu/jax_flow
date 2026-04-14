"""SpeedTuning environment wrapper.

Wraps a base environment with a frozen BC policy. The RL agent controls
a discrete speed multiplier; the wrapper handles BC inference, temporal
interpolation, and multi-step execution.
"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete

from jax_flow.agents.speed_tuning.interpolation import temporal_interpolate

# rot6d column slices for different action formats
# Single-arm 10D: [pos(3), rot6d(6), gripper(1)]
_SINGLE_ARM_ROT6D = (3, 9)
# Dual-arm 20D: [pos(3), rot6d(6), gripper(1), pos(3), rot6d(6), gripper(1)]
_DUAL_ARM_ROT6D_0 = (3, 9)
_DUAL_ARM_ROT6D_1 = (13, 19)


def _infer_rot6d_slices(action_dim: int, abs_action: bool) -> list[tuple[int, int]]:
    """Infer rot6d column slices from action dimension.

    Returns list of (start, end) tuples for rot6d columns, or empty list
    if abs_action is False or action_dim is unrecognized.
    """
    if not abs_action:
        return []
    if action_dim == 10:  # single-arm
        return [_SINGLE_ARM_ROT6D]
    elif action_dim == 20:  # dual-arm
        return [_DUAL_ARM_ROT6D_0, _DUAL_ARM_ROT6D_1]
    return []


class SpeedTuningEnvWrapper(gym.Wrapper):
    """Environment wrapper for SpeedTuning.

    The RL agent outputs a discrete speed index each macro-step.
    Internally, the wrapper:
      1. Calls the frozen BC policy to get an action chunk
      2. Interpolates the chunk based on the selected speed
         (with Gram-Schmidt re-projection for rot6d columns)
      3. Executes up to min(k_skip, len(interpolated)) actions in the inner env
         (dynamic truncation: stops when interpolated actions are exhausted)
      4. Computes r_ST = alpha * v^beta + r_task per step

    The inner env should have FrameStackWrapper but NOT ActionChunkingWrapper,
    since this wrapper manages action execution itself.

    Args:
        env: Base environment (with FrameStackWrapper applied).
        bc_agent: Frozen BCAgent for action chunk prediction.
        speed_options: List of speed multipliers (>= 1.0).
        alpha: Speed reward weight.
        beta: Speed reward exponent.
        k_skip: Number of env steps between speed decisions.
        abs_action: Whether actions use rot6d representation.
    """

    def __init__(
        self,
        env,
        bc_agent,
        speed_options: list[float],
        alpha: float = 0.1,
        beta: float = 2.0,
        k_skip: int = 4,
        abs_action: bool = True,
    ):
        super().__init__(env)
        self.bc_agent = bc_agent
        self.speed_options = speed_options
        self.alpha = alpha
        self.beta = beta
        self.k_skip = k_skip

        # Override action space to discrete speed selection
        self.action_space = Discrete(len(speed_options))

        # Infer rot6d slices from action space
        act_dim = env.action_space.shape[0]
        self._rot6d_slices = _infer_rot6d_slices(act_dim, abs_action)

        # Internal state
        self._current_speed = 1.0
        self._last_obs = None

    def _interpolate(self, action_chunk: np.ndarray) -> np.ndarray:
        """Interpolate action chunk with rotation-aware handling."""
        interp = temporal_interpolate(action_chunk, self._current_speed)
        # Apply Gram-Schmidt to each rot6d slice
        for s, e in self._rot6d_slices:
            if e <= interp.shape[-1]:
                from jax_flow.agents.speed_tuning.interpolation import _gram_schmidt_6d

                interp[:, s:e] = _gram_schmidt_6d(interp[:, s:e])
        return interp

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_speed = 1.0
        self._last_obs = obs
        return obs, info

    def step(self, speed_idx):
        """Execute one macro-step of the SpeedTuning loop.

        Executes min(k_skip, len(interpolated)) steps — never pads with
        repeated actions when the interpolated sequence runs out.

        Args:
            speed_idx: Integer index into speed_options.

        Returns:
            (obs, total_reward, terminated, truncated, info)
        """
        if speed_idx is not None:
            self._current_speed = self.speed_options[int(speed_idx)]

        # Get BC action chunk from frozen policy
        obs = self._last_obs
        obs_batch = self._batchify_obs(obs)
        actions = self.bc_agent.eval_actions(obs_batch)
        action_chunk = np.array(actions[0])  # (horizon, action_dim)

        # Interpolate (with rot6d Gram-Schmidt)
        interpolated = self._interpolate(action_chunk)

        # Dynamic truncation: execute min(k_skip, len(interpolated)) steps
        exec_steps = min(self.k_skip, len(interpolated))
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        step_count = 0

        for i in range(exec_steps):
            obs, r_task, terminated, truncated, info = self.env.step(interpolated[i])
            r_speed = self.alpha * (self._current_speed**self.beta)
            r_st = r_speed + r_task
            total_reward += r_st
            step_count += 1

            if terminated or truncated:
                break

        self._last_obs = obs
        info["speed"] = self._current_speed
        info["speed_idx"] = int(speed_idx) if speed_idx is not None else 0
        info["interpolated_len"] = len(interpolated)
        info["steps_executed"] = step_count

        return obs, total_reward, terminated, truncated, info

    def _batchify_obs(self, obs):
        """Add batch dimension to observation."""
        if isinstance(obs, dict):
            return {k: np.array(v[np.newaxis, ...]) for k, v in obs.items()}
        return obs[np.newaxis, ...]

    def _copy_obs(self, obs):
        """Deep copy observation (array or dict)."""
        if isinstance(obs, dict):
            return {k: v.copy() for k, v in obs.items()}
        return obs.copy()

    def needs_replan(self):
        """Always returns True — speed decision needed every macro-step."""
        return True
