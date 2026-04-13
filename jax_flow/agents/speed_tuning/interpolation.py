"""Linear temporal interpolation for action chunk speed adjustment.

Supports rotation-aware interpolation for abs_action (rot6d) tasks:
  - Position and gripper: standard linear interpolation
  - Rotation 6D: linear interpolation + Gram-Schmidt orthogonalization
"""

import numpy as np

from jax_flow.data.rotation_utils import matrix_to_rotation_6d, rotation_6d_to_matrix


def _gram_schmidt_6d(d6: np.ndarray) -> np.ndarray:
    """Project 6D vector back to valid rotation via Gram-Schmidt.

    Args:
        d6: (..., 6) interpolated 6D rotation vectors (may not be valid SO(3)).

    Returns:
        (..., 6) valid 6D rotation vectors.
    """
    mat = rotation_6d_to_matrix(d6)  # (..., 3, 3)
    return matrix_to_rotation_6d(mat)  # (..., 6)


def temporal_interpolate(
    action_chunk: np.ndarray,
    speed: float,
    rot6d_slice: tuple[int, int] | None = None,
) -> np.ndarray:
    """Interpolate action chunk based on speed multiplier.

    Given an action chunk of length k, produces a shorter sequence when speed > 1.0
    using linear interpolation between adjacent actions.

    For rot6d actions, pass rot6d_slice=(start, end) to indicate which columns
    contain the 6D rotation. After linear interpolation, those columns are
    re-projected to valid SO(3) via Gram-Schmidt.

    Formula: f(t) = f(floor(vt)) + (vt - floor(vt)) * (f(floor(vt)+1) - f(floor(vt)))

    Args:
        action_chunk: (k, action_dim) original action sequence.
        speed: Speed multiplier v >= 1.0. Higher = faster = shorter output.
        rot6d_slice: (start, end) column indices of the rot6d part, e.g. (3, 9)
            for single-arm 10D actions [pos(3), rot6d(6), gripper(1)].
            None = no rotation correction (pure linear interpolation).

    Returns:
        (new_len, action_dim) interpolated action sequence.
        new_len = ceil(k / speed).
    """
    k = len(action_chunk)
    if k == 0:
        return action_chunk

    speed = max(speed, 1.0)

    new_len = int(np.ceil(k / speed))
    new_len = max(new_len, 1)

    out = np.empty((new_len, action_chunk.shape[-1]), dtype=action_chunk.dtype)

    for i in range(new_len):
        t_src = speed * i
        idx_lo = int(np.floor(t_src))
        # Clamp to valid range
        if idx_lo >= k - 1:
            out[i] = action_chunk[-1]
            continue
        idx_hi = idx_lo + 1
        frac = t_src - idx_lo
        out[i] = action_chunk[idx_lo] + frac * (
            action_chunk[idx_hi] - action_chunk[idx_lo]
        )

    # Re-project rotation columns to valid SO(3)
    if rot6d_slice is not None and new_len > 0:
        s, e = rot6d_slice
        out[:, s:e] = _gram_schmidt_6d(out[:, s:e])

    return out


def make_speed_options(max_speed: float = 2.0, granularity: float = 0.1) -> list[float]:
    """Generate discrete speed options from 1.0 to max_speed.

    Args:
        max_speed: Maximum speed multiplier (inclusive).
        granularity: Step size between speed options.

    Returns:
        List of speed values, e.g. [1.0, 1.1, 1.2, ..., 2.0].
    """
    n = int(round((max_speed - 1.0) / granularity)) + 1
    options = [round(1.0 + i * granularity, 4) for i in range(n)]
    # Ensure max_speed is included
    if abs(options[-1] - max_speed) > 1e-6:
        options.append(max_speed)
    return options
