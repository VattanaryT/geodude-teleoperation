"""SE(3) helpers for tag_0 / vention / RTDE base frames.

Naming (crop_meta / ROS calib):
  ``T_tag0_vention`` (aka ``T_tag0_base`` in JSON) = pose of tag_0 in
  ``vention_frame``: ``p_vention = T @ p_tag0``.

RTDE ``getActualTCPPose`` is in the right UR controller base (``right_base``),
not vention. Compose with ``T_rtde_base_vention`` (pose of RTDE base in
vention): ``p_vention = T_rtde_base_vention @ p_rtde``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_T_RTDE_BASE_VENTION = (
    _PACKAGE_ROOT / "assets" / "frames" / "T_right_base_vention.npy"
)
DEFAULT_T_LEFT_BASE_VENTION = (
    _PACKAGE_ROOT / "assets" / "frames" / "T_left_base_vention.npy"
)


def load_SE3(path: Path | None, *, json_key: str | None = None) -> np.ndarray:
    """Load a 4x4 from ``.npy`` / ``.npz`` / JSON (optional key or raw matrix)."""
    if path is None:
        return np.eye(4, dtype=float)
    path = Path(path)
    if path.suffix == ".json":
        with path.open() as f:
            meta = json.load(f)
        if json_key and isinstance(meta, dict) and json_key in meta:
            T = np.asarray(meta[json_key], dtype=float)
        elif isinstance(meta, dict) and "matrix" in meta:
            T = np.asarray(meta["matrix"], dtype=float)
        elif isinstance(meta, dict) and json_key is None:
            for key in (
                "T_tag0_base",
                "T_tag0_vention",
                "T_rtde_base_vention",
                "T_left_base_vention",
                "matrix",
            ):
                if key in meta:
                    T = np.asarray(meta[key], dtype=float)
                    break
            else:
                T = np.asarray(meta, dtype=float)
        else:
            T = np.asarray(meta, dtype=float)
    else:
        T = np.load(path)
        if isinstance(T, np.lib.npyio.NpzFile):
            key = json_key if json_key and json_key in T.files else T.files[0]
            T = T[key]
        T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"SE3 must be 4x4, got {T.shape} from {path}")
    return T


def load_T_tag0_base(path: Path | None) -> np.ndarray:
    """Load crop_meta ``T_tag0_base`` (tag_0 pose in vention_frame)."""
    return load_SE3(path, json_key="T_tag0_base")


def load_T_rtde_base_vention(path: Path | None = None) -> np.ndarray:
    """Load pose of RTDE ``right_base`` in ``vention_frame``."""
    if path is None:
        path = DEFAULT_T_RTDE_BASE_VENTION
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"T_rtde_base_vention not found: {path}. "
            "Pass --T-rtde-base-vention or install assets/frames/T_right_base_vention.npy"
        )
    return load_SE3(path, json_key="T_rtde_base_vention")


def load_T_left_base_vention(path: Path | None = None) -> np.ndarray:
    """Load pose of left UR RTDE ``left_base`` in ``vention_frame``."""
    if path is None:
        path = DEFAULT_T_LEFT_BASE_VENTION
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"T_left_base_vention not found: {path}. "
            "Install assets/frames/T_left_base_vention.npy"
        )
    return load_SE3(path, json_key="T_left_base_vention")


def compose_T_tag0_rtde(
    T_tag0_vention: np.ndarray, T_rtde_base_vention: np.ndarray
) -> np.ndarray:
    """Pose of tag_0 in RTDE base: ``p_rtde = T @ p_tag0``."""
    T_tag0_vention = np.asarray(T_tag0_vention, dtype=float).reshape(4, 4)
    T_rtde_base_vention = np.asarray(T_rtde_base_vention, dtype=float).reshape(4, 4)
    return np.linalg.inv(T_rtde_base_vention) @ T_tag0_vention


def left_base_to_vention_points(
    points_left_base: np.ndarray, T_left_base_vention: np.ndarray
) -> np.ndarray:
    """``p_vention = T_left_base_vention @ p_left_base``."""
    return transform_points(T_left_base_vention, points_left_base)


def vention_to_tag0_points(
    points_vention: np.ndarray, T_tag0_vention: np.ndarray
) -> np.ndarray:
    """``p_tag0 = inv(T_tag0_vention) @ p_vention``."""
    T = np.asarray(T_tag0_vention, dtype=float).reshape(4, 4)
    return transform_points(np.linalg.inv(T), points_vention)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float).reshape(4, 4)
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    return (T[:3, :3] @ P.T).T + T[:3, 3]


def transform_pose_rotvec(T: np.ndarray, pose6: np.ndarray) -> np.ndarray:
    """Apply SE3 to a UR-style pose ``[x,y,z,rx,ry,rz]`` (rotvec)."""
    from scipy.spatial.transform import Rotation

    pose6 = np.asarray(pose6, dtype=float).reshape(6)
    T = np.asarray(T, dtype=float).reshape(4, 4)
    T_pose = np.eye(4)
    T_pose[:3, 3] = pose6[:3]
    T_pose[:3, :3] = Rotation.from_rotvec(pose6[3:6]).as_matrix()
    T_out = T @ T_pose
    return np.concatenate(
        [T_out[:3, 3], Rotation.from_matrix(T_out[:3, :3]).as_rotvec()]
    )


def base_to_tag0_point(p_base: np.ndarray, T_tag0_base: np.ndarray) -> np.ndarray:
    R = T_tag0_base[:3, :3]
    t = T_tag0_base[:3, 3]
    return R.T @ (np.asarray(p_base, dtype=float).reshape(3) - t)


def tag0_rot_to_base(R_tag0: np.ndarray, T_tag0_base: np.ndarray) -> np.ndarray:
    return T_tag0_base[:3, :3] @ np.asarray(R_tag0, dtype=float).reshape(3, 3)


def tip_from_tcp(p_tcp, R_tcp, tip_offset) -> np.ndarray:
    return np.asarray(p_tcp, dtype=float).reshape(3) + (
        np.asarray(R_tcp, dtype=float).reshape(3, 3)
        @ np.asarray(tip_offset, dtype=float).reshape(3)
    )


def warn_tip_offset_xy(tip_offset: np.ndarray) -> str | None:
    tip = np.asarray(tip_offset, dtype=float).reshape(3)
    if abs(tip[0]) > 1e-9 or abs(tip[1]) > 1e-9:
        return (
            "WARNING: tip-offset TX/TY non-zero; prefer TX=TY=0 and only TZ "
            f"(got {tip.tolist()})"
        )
    return None


def transform_msg_to_matrix(transform) -> np.ndarray:
    """Build 4x4 rigid transform from a geometry_msgs/TransformStamped-like message."""
    tr = transform.transform.translation
    q = transform.transform.rotation
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [float(tr.x), float(tr.y), float(tr.z)]
    return T


def lookup_T_tag0_base_tf(
    buffer,
    base_frame: str,
    tag0_frame: str,
    max_age_s: float,
) -> tuple[np.ndarray | None, float | None]:
    """Look up ``T_tag0_base`` from TF. Returns ``(T, age_ms)`` or ``(None, None)``."""
    if buffer is None or not base_frame or not tag0_frame:
        return None, None
    try:
        import rclpy
    except ImportError:
        return None, None

    try:
        when = rclpy.time.Time()
        timeout = rclpy.duration.Duration(seconds=0.1)
        t = buffer.lookup_transform(str(base_frame), str(tag0_frame), when, timeout=timeout)
        T = transform_msg_to_matrix(t)
        stamp = t.header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        now_ns = int(rclpy.clock.Clock().now().nanoseconds)
        age_s = max(0.0, (now_ns - stamp_ns) * 1e-9)
        if age_s > float(max_age_s):
            return None, None
        return T, age_s * 1000.0
    except Exception:
        return None, None
