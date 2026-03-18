"""Rotation conversion utilities for 6D rotation representation.

Converts between axis-angle (3D) and continuous 6D rotation representation.
6D representation avoids the topological discontinuity of axis-angle at ±π,
which improves MSE-based learning for rotation-heavy tasks (e.g. square/nut assembly).

Reference: "On the Continuity of Rotation Representations in Neural Networks"
           (Zhou et al., CVPR 2019)

Ported from much-ado-about-noising mip/datasets/rotation_conversion.py.
"""

import numpy as np


def axis_angle_to_quaternion(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle to quaternion (w, x, y, z).

    Args:
        aa: (..., 3) axis-angle vectors.

    Returns:
        (..., 4) quaternions in (w, x, y, z) format.
    """
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)  # (..., 1)
    # Avoid division by zero
    safe_angle = np.where(angle > 1e-8, angle, np.ones_like(angle))
    axis = aa / safe_angle  # (..., 3)

    half_angle = angle / 2.0
    w = np.cos(half_angle)  # (..., 1)
    xyz = axis * np.sin(half_angle)  # (..., 3)

    # For near-zero angles, quaternion is (1, 0, 0, 0)
    quat = np.concatenate([w, xyz], axis=-1)  # (..., 4)
    return quat


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to rotation matrix.

    Args:
        quat: (..., 4) quaternions.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    # Normalize
    quat = quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-12)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Build rotation matrix
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z

    mat = np.stack(
        [
            1.0 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1.0 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1.0 - (txx + tyy),
        ],
        axis=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle to rotation matrix.

    Args:
        aa: (..., 3) axis-angle vectors.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(aa))


def matrix_to_rotation_6d(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to 6D representation (first two columns).

    Args:
        mat: (..., 3, 3) rotation matrices.

    Returns:
        (..., 6) 6D rotation vectors.
    """
    return mat[..., :2, :].reshape(mat.shape[:-2] + (6,))


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to rotation matrix via Gram-Schmidt.

    Args:
        d6: (..., 6) 6D rotation vectors.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:6]

    # Normalize first vector
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)

    # Gram-Schmidt: make second vector orthogonal to first
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)

    # Third vector via cross product
    b3 = np.cross(b1, b2, axis=-1)

    return np.stack([b1, b2, b3], axis=-2)  # (..., 3, 3)


def matrix_to_quaternion(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (w, x, y, z).

    Uses Shepperd's method for numerical stability.

    Args:
        mat: (..., 3, 3) rotation matrices.

    Returns:
        (..., 4) quaternions in (w, x, y, z) format.
    """
    batch_shape = mat.shape[:-2]
    mat_flat = mat.reshape(-1, 3, 3)
    n = mat_flat.shape[0]
    quat = np.zeros((n, 4), dtype=mat.dtype)

    for i in range(n):
        m = mat_flat[i]
        trace = m[0, 0] + m[1, 1] + m[2, 2]

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            quat[i, 0] = 0.25 / s
            quat[i, 1] = (m[2, 1] - m[1, 2]) * s
            quat[i, 2] = (m[0, 2] - m[2, 0]) * s
            quat[i, 3] = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            quat[i, 0] = (m[2, 1] - m[1, 2]) / s
            quat[i, 1] = 0.25 * s
            quat[i, 2] = (m[0, 1] + m[1, 0]) / s
            quat[i, 3] = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            quat[i, 0] = (m[0, 2] - m[2, 0]) / s
            quat[i, 1] = (m[0, 1] + m[1, 0]) / s
            quat[i, 2] = 0.25 * s
            quat[i, 3] = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            quat[i, 0] = (m[1, 0] - m[0, 1]) / s
            quat[i, 1] = (m[0, 2] + m[2, 0]) / s
            quat[i, 2] = (m[1, 2] + m[2, 1]) / s
            quat[i, 3] = 0.25 * s

    return quat.reshape(batch_shape + (4,))


def quaternion_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to axis-angle.

    Args:
        quat: (..., 4) quaternions.

    Returns:
        (..., 3) axis-angle vectors.
    """
    # Ensure w >= 0 (canonical form)
    quat = np.where(quat[..., :1] < 0, -quat, quat)

    # Normalize
    quat = quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-12)

    w = np.clip(quat[..., :1], -1.0, 1.0)
    xyz = quat[..., 1:]

    half_angle = np.arccos(w)  # (..., 1)
    angle = 2.0 * half_angle

    # For small angles, axis-angle ≈ 2 * xyz
    sin_half = np.sin(half_angle)
    safe_sin = np.where(sin_half > 1e-8, sin_half, np.ones_like(sin_half))
    axis = xyz / safe_sin

    aa = axis * angle
    # For near-zero rotations, return zero
    aa = np.where(angle > 1e-8, aa, np.zeros_like(aa))
    return aa


def matrix_to_axis_angle(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to axis-angle.

    Args:
        mat: (..., 3, 3) rotation matrices.

    Returns:
        (..., 3) axis-angle vectors.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(mat))


def axis_angle_to_rotation_6d(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle to 6D rotation representation.

    Args:
        aa: (..., 3) axis-angle vectors.

    Returns:
        (..., 6) 6D rotation vectors.
    """
    return matrix_to_rotation_6d(axis_angle_to_matrix(aa))


def rotation_6d_to_axis_angle(d6: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to axis-angle.

    Args:
        d6: (..., 6) 6D rotation vectors.

    Returns:
        (..., 3) axis-angle vectors.
    """
    return matrix_to_axis_angle(rotation_6d_to_matrix(d6))


def _transform_single_arm_to_6d(action: np.ndarray) -> np.ndarray:
    """Transform single-arm 7D action to 10D."""
    pos = action[..., :3]
    aa = action[..., 3:6]
    gripper = action[..., 6:7]
    rot6d = axis_angle_to_rotation_6d(aa)
    return np.concatenate([pos, rot6d, gripper], axis=-1)


def _undo_single_arm_from_6d(action: np.ndarray) -> np.ndarray:
    """Transform single-arm 10D action back to 7D."""
    pos = action[..., :3]
    rot6d = action[..., 3:9]
    gripper = action[..., 9:10]
    aa = rotation_6d_to_axis_angle(rot6d)
    return np.concatenate([pos, aa, gripper], axis=-1)


def transform_action_to_6d(action: np.ndarray) -> np.ndarray:
    """Transform action from axis-angle to 6D rotation representation.

    Supports single-arm (7D → 10D) and dual-arm (14D → 20D).

    Args:
        action: (..., 7) or (..., 14) actions with axis-angle rotation.

    Returns:
        (..., 10) or (..., 20) actions with 6D rotation representation.
    """
    act_dim = action.shape[-1]
    if act_dim == 7:
        return _transform_single_arm_to_6d(action)
    elif act_dim == 14:
        # Dual-arm: two 7D chunks
        arm0 = _transform_single_arm_to_6d(action[..., :7])
        arm1 = _transform_single_arm_to_6d(action[..., 7:14])
        return np.concatenate([arm0, arm1], axis=-1)
    else:
        raise ValueError(
            f"Unsupported action dim {act_dim} for 6D transform (expected 7 or 14)"
        )


def undo_transform_action(action: np.ndarray) -> np.ndarray:
    """Transform action from 6D rotation back to axis-angle representation.

    Supports single-arm (10D → 7D) and dual-arm (20D → 14D).

    Args:
        action: (..., 10) or (..., 20) actions with 6D rotation representation.

    Returns:
        (..., 7) or (..., 14) actions with axis-angle rotation.
    """
    act_dim = action.shape[-1]
    if act_dim == 10:
        return _undo_single_arm_from_6d(action)
    elif act_dim == 20:
        # Dual-arm: two 10D chunks
        arm0 = _undo_single_arm_from_6d(action[..., :10])
        arm1 = _undo_single_arm_from_6d(action[..., 10:20])
        return np.concatenate([arm0, arm1], axis=-1)
    else:
        raise ValueError(
            f"Unsupported action dim {act_dim} for 6D undo (expected 10 or 20)"
        )
