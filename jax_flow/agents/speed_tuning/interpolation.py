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
    k_skip: int,
    rot6d_slice: tuple[int, int] | None = None,
) -> np.ndarray:
    """Interpolate action chunk to produce k_skip accelerated actions (Eq. 9-11).

    Paper formula:
        interp_{f,v}(t) = f(⌊vt⌋) + (vt - ⌊vt⌋)/v * (f(⌊vt+1⌋) - f(⌊vt⌋))

    where ⌊vt+1⌋ means floor(vt + 1), NOT floor(vt) + 1.

    Given action chunk A = a_0:k (k+1 points), produces exactly k_skip actions.
    At speed=1.0, output is the first k_skip actions of the original chunk.

    Args:
        action_chunk: (k+1, action_dim) original action sequence.
        speed: Speed multiplier v >= 1.0.
        k_skip: Number of output actions to produce (fixed length).
        rot6d_slice: (start, end) column indices of rot6d part. None = no correction.

    Returns:
        (k_skip, action_dim) interpolated action sequence.
    """
    n_points = len(action_chunk)  # k+1
    if n_points == 0:
        return action_chunk

    speed = max(speed, 1.0)

    out = np.empty((k_skip, action_chunk.shape[-1]), dtype=action_chunk.dtype)

    for i in range(k_skip):
        vt = speed * i
        idx_lo = int(np.floor(vt))          # ⌊vt⌋
        idx_hi = int(np.floor(vt + 1))      # ⌊vt+1⌋  (NOT ⌊vt⌋+1)

        # Clamp indices to valid range
        idx_lo = min(idx_lo, n_points - 1)
        idx_hi = min(idx_hi, n_points - 1)

        if idx_lo == idx_hi:
            out[i] = action_chunk[idx_lo]
        else:
            frac = (vt - np.floor(vt)) / speed  # (vt - ⌊vt⌋) / v
            out[i] = action_chunk[idx_lo] + frac * (
                action_chunk[idx_hi] - action_chunk[idx_lo]
            )

    # Re-project rotation columns to valid SO(3)
    if rot6d_slice is not None and k_skip > 0:
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
