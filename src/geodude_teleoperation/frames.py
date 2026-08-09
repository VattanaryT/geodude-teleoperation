"""SE(3) helpers. T_tag0_base = pose of tag_0 expressed in base (crop_meta convention)."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np


def load_T_tag0_base(path: Path | None) -> np.ndarray:
    if path is None:
        return np.eye(4, dtype=float)
    path = Path(path)
    if path.suffix == ".json":
        with path.open() as f:
            meta = json.load(f)
        T = np.asarray(meta["T_tag0_base"] if "T_tag0_base" in meta else meta, dtype=float)
    else:
        T = np.load(path)
        if isinstance(T, np.lib.npyio.NpzFile):
            T = T["T_tag0_base"] if "T_tag0_base" in T.files else T[T.files[0]]
        T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T_tag0_base must be 4x4, got {T.shape}")
    return T


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
