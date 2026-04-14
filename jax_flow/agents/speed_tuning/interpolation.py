"""Delta-aware temporal interpolation for action chunk speed adjustment.

Actions are delta commands (delta_pos, delta_rotation, gripper). To accelerate:
  1. Accumulate deltas to cumulative space (cumsum for linear, matrix product for rotation)
  2. Interpolate in cumulative space at accelerated time points
  3. Diff back to deltas

Rotation columns use SLERP interpolation via quaternions to handle SO(3) correctly.
"""

import numpy as np

from jax_flow.data.rotation_utils import (
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)


def _gram_schmidt_6d(d6: np.ndarray) -> np.ndarray:
    """Project 6D vector back to valid rotation via Gram-Schmidt.

    Args:
        d6: (..., 6) interpolated 6D rotation vectors (may not be valid SO(3)).

    Returns:
        (..., 6) valid 6D rotation vectors.
    """
    mat = rotation_6d_to_matrix(d6)  # (..., 3, 3)
    return matrix_to_rotation_6d(mat)  # (..., 6)


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions.

    Args:
        q0: (..., 4) quaternion in (w, x, y, z) format.
        q1: (..., 4) quaternion in (w, x, y, z) format.
        t: Interpolation parameter in [0, 1].

    Returns:
        (..., 4) interpolated quaternion.
    """
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)

    # Take short path: flip q1 if dot < 0
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)

    # Near-identity: fall back to NLERP to avoid sin(~0) division
    threshold = 0.9995
    is_close = dot > threshold

    # SLERP path
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    # Guard against division by zero (will be masked by is_close anyway)
    safe_sin = np.where(sin_theta < 1e-10, np.ones_like(sin_theta), sin_theta)
    w0 = np.sin((1.0 - t) * theta) / safe_sin
    w1 = np.sin(t * theta) / safe_sin

    # NLERP path (for near-identity)
    w0_nlerp = 1.0 - t
    w1_nlerp = t

    w0 = np.where(is_close, w0_nlerp, w0)
    w1 = np.where(is_close, w1_nlerp, w1)

    result = w0 * q0 + w1 * q1
    # Normalize
    result = result / (np.linalg.norm(result, axis=-1, keepdims=True) + 1e-12)
    return result


def _interpolate_linear_deltas(
    deltas: np.ndarray,
    speed: float,
    k_skip: int,
) -> np.ndarray:
    """Interpolate linear (additive) delta columns via cumsum → lerp → diff.

    Args:
        deltas: (n, d) delta values (position, gripper, etc.).
        speed: Speed multiplier >= 1.0.
        k_skip: Number of output deltas.

    Returns:
        (k_skip, d) accelerated delta values.
    """
    n = len(deltas)
    d = deltas.shape[-1]

    # Cumulative positions: P[0] = 0, P[k] = P[k-1] + deltas[k-1]
    P = np.zeros((n + 1, d), dtype=deltas.dtype)
    np.cumsum(deltas, axis=0, out=P[1:])

    # Interpolate P at k_skip+1 points: t = v*0, v*1, ..., v*k_skip
    P_interp = np.empty((k_skip + 1, d), dtype=deltas.dtype)
    for i in range(k_skip + 1):
        t = speed * i
        t = min(t, float(n))  # clamp to [0, n]
        idx_lo = int(np.floor(t))
        idx_lo = min(idx_lo, n)  # P has indices 0..n
        idx_hi = min(idx_lo + 1, n)
        frac = t - np.floor(t)
        P_interp[i] = P[idx_lo] + frac * (P[idx_hi] - P[idx_lo])

    # Diff back to deltas
    return np.diff(P_interp, axis=0)  # (k_skip, d)


def _interpolate_rotation_deltas(
    rot6d_deltas: np.ndarray,
    speed: float,
    k_skip: int,
) -> np.ndarray:
    """Interpolate rotation delta columns via cumulative product → SLERP → inverse product.

    Args:
        rot6d_deltas: (n, 6) delta rot6d values.
        speed: Speed multiplier >= 1.0.
        k_skip: Number of output deltas.

    Returns:
        (k_skip, 6) accelerated delta rot6d values.
    """
    n = len(rot6d_deltas)

    # Convert deltas to rotation matrices
    dR = rotation_6d_to_matrix(rot6d_deltas)  # (n, 3, 3)

    # Cumulative product: R[0] = I, R[k] = R[k-1] @ dR[k-1]
    R = np.empty((n + 1, 3, 3), dtype=dR.dtype)
    R[0] = np.eye(3)
    for k in range(n):
        R[k + 1] = R[k] @ dR[k]

    # Convert to quaternions
    Q = matrix_to_quaternion(R)  # (n+1, 4)

    # Fix quaternion continuity: ensure consecutive dot >= 0
    for k in range(1, n + 1):
        if np.dot(Q[k - 1], Q[k]) < 0:
            Q[k] = -Q[k]

    # SLERP interpolation at k_skip+1 points
    Q_interp = np.empty((k_skip + 1, 4), dtype=Q.dtype)
    for i in range(k_skip + 1):
        t = speed * i
        t = min(t, float(n))  # clamp to [0, n]
        idx_lo = int(np.floor(t))
        idx_lo = min(idx_lo, n)  # Q has indices 0..n
        idx_hi = min(idx_lo + 1, n)
        frac = t - np.floor(t)
        Q_interp[i] = quaternion_slerp(Q[idx_lo], Q[idx_hi], frac)

    # Convert back to matrices
    R_interp = quaternion_to_matrix(Q_interp)  # (k_skip+1, 3, 3)

    # Diff: dR'[i] = R'[i]^T @ R'[i+1]
    out_rot6d = np.empty((k_skip, 6), dtype=rot6d_deltas.dtype)
    for i in range(k_skip):
        delta_mat = R_interp[i].T @ R_interp[i + 1]
        out_rot6d[i] = matrix_to_rotation_6d(delta_mat)

    return out_rot6d


def temporal_interpolate(
    action_chunk: np.ndarray,
    speed: float,
    k_skip: int,
    rot6d_slices: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Interpolate delta action chunk to produce k_skip accelerated delta actions.

    For linear columns (position, gripper): cumsum → lerp → diff.
    For rotation columns (rot6d): cumulative product → SLERP → inverse product.

    At speed=1.0, output equals the first k_skip deltas of the original chunk.

    Args:
        action_chunk: (n, action_dim) original delta action sequence.
        speed: Speed multiplier v >= 1.0.
        k_skip: Number of output actions to produce.
        rot6d_slices: List of (start, end) column indices for rot6d parts.
            E.g. [(3, 9)] for single-arm, [(3, 9), (13, 19)] for dual-arm.
            None or [] means all columns are linear.

    Returns:
        (k_skip, action_dim) interpolated delta action sequence.
    """
    n = len(action_chunk)
    if n == 0:
        return action_chunk

    speed = max(speed, 1.0)
    if rot6d_slices is None:
        rot6d_slices = []

    action_dim = action_chunk.shape[-1]
    out = np.empty((k_skip, action_dim), dtype=action_chunk.dtype)

    # Process all columns as linear first
    linear_result = _interpolate_linear_deltas(action_chunk, speed, k_skip)
    out[:] = linear_result

    # Overwrite rotation slices with proper SLERP-based interpolation
    for s, e in rot6d_slices:
        if e <= action_dim:
            out[:, s:e] = _interpolate_rotation_deltas(
                action_chunk[:, s:e], speed, k_skip
            )

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
