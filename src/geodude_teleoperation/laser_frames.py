"""Place the Keyence ``merged.ply`` (left UR base) into vention / tag_0.

Do not PCA-rest or rewrite the cloud into right_base. Both arms share
``vention_frame``; the right TCP queries the same cloud via
``T_tag0_rtde = inv(T_right_base_vention) @ T_tag0_vention``.

Chain (p_left is merged.ply vertices)::

    p_vention = T_left_base_vention @ p_left + t_laser
    T_tag0_vention = T_tag0_base with translation += t_tag0_offset
    p_tag0 = inv(T_tag0_vention) @ p_vention
    T_tag0_rtde = inv(T_right_base_vention) @ T_tag0_vention

``T_tag0_base`` in crop_meta is tag_0 in vention, not in the right controller.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_T_LEFT = _REPO_ROOT / "assets" / "frames" / "T_left_base_vention.json"
DEFAULT_T_RIGHT = _REPO_ROOT / "assets" / "frames" / "T_right_base_vention.json"
DEFAULT_CROP_META = _REPO_ROOT / "assets" / "laser_profilometer" / "crop_meta.json"
DEFAULT_TAG0_OFFSET = (
    _REPO_ROOT / "assets" / "laser_profilometer" / "crop_meta.object_offset.json")
DEFAULT_LASER_OFFSET = (
    _REPO_ROOT / "assets" / "laser_profilometer" / "merged.laser_offset.json")


def is_laser_left_base_object(path) -> bool:
    """True for the Keyence merge (points already in left UR RTDE base)."""
    if path is None:
        return False
    p = Path(path)
    parts = {part.lower() for part in p.parts}
    name = p.name.lower()
    return "laser_profilometer" in parts and name.startswith("merged") and name.endswith(".ply")


def load_SE3(path: Path | str) -> np.ndarray:
    path = Path(path)
    with path.open() as f:
        meta = json.load(f)
    if isinstance(meta, dict):
        for key in (
            "T_tag0_base",
            "T_tag0_vention",
            "T_left_base_vention",
            "T_right_base_vention",
            "matrix",
        ):
            if key in meta:
                T = np.asarray(meta[key], dtype=float)
                break
        else:
            T = np.asarray(meta, dtype=float)
    else:
        T = np.asarray(meta, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"SE3 must be 4x4, got {T.shape} from {path}")
    return T


def load_t_vention(path: Path | str | None) -> np.ndarray:
    if path is None or not Path(path).is_file():
        return np.zeros(3, dtype=float)
    with Path(path).open() as f:
        meta = json.load(f)
    raw = meta.get("t_vention_m", meta.get("t", [0.0, 0.0, 0.0]))
    return np.asarray(raw, dtype=float).reshape(3)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float).reshape(4, 4)
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    return (T[:3, :3] @ P.T).T + T[:3, 3]


def apply_t_vention(T: np.ndarray, t_vention_m: np.ndarray) -> np.ndarray:
    out = np.asarray(T, dtype=float).reshape(4, 4).copy()
    out[:3, 3] = out[:3, 3] + np.asarray(t_vention_m, dtype=float).reshape(3)
    return out


def compose_T_tag0_rtde(T_tag0_vention: np.ndarray, T_rtde_base_vention: np.ndarray) -> np.ndarray:
    """Pose of tag_0 in an RTDE base: ``p_rtde = T @ p_tag0``."""
    return np.linalg.inv(np.asarray(T_rtde_base_vention, dtype=float).reshape(4, 4)) @ (
        np.asarray(T_tag0_vention, dtype=float).reshape(4, 4))


@dataclass(frozen=True)
class LaserPlacement:
    points_vention: np.ndarray
    points_tag0: np.ndarray
    T_left_base_vention: np.ndarray
    T_right_base_vention: np.ndarray
    T_tag0_vention: np.ndarray
    T_tag0_rtde: np.ndarray


def place_laser_points(
    points_left_base: np.ndarray,
    *,
    T_left_base_vention: np.ndarray | None = None,
    T_right_base_vention: np.ndarray | None = None,
    T_tag0_base: np.ndarray | None = None,
    t_laser_vention_m: np.ndarray | None = None,
    t_tag0_vention_m: np.ndarray | None = None,
) -> LaserPlacement:
    """Map left_base points through vention into tag_0. Overlay uses ``points_vention``."""
    T_left = (
        np.asarray(T_left_base_vention, dtype=float)
        if T_left_base_vention is not None else load_SE3(DEFAULT_T_LEFT))
    T_right = (
        np.asarray(T_right_base_vention, dtype=float)
        if T_right_base_vention is not None else load_SE3(DEFAULT_T_RIGHT))
    T_tag0 = (
        np.asarray(T_tag0_base, dtype=float)
        if T_tag0_base is not None else load_SE3(DEFAULT_CROP_META))
    t_laser = (
        np.zeros(3) if t_laser_vention_m is None
        else np.asarray(t_laser_vention_m, dtype=float).reshape(3))
    t_tag0 = (
        np.zeros(3) if t_tag0_vention_m is None
        else np.asarray(t_tag0_vention_m, dtype=float).reshape(3))

    points_vention = transform_points(T_left, points_left_base) + t_laser.reshape(1, 3)
    T_tag0_vention = apply_t_vention(T_tag0, t_tag0)
    points_tag0 = transform_points(np.linalg.inv(T_tag0_vention), points_vention)
    T_tag0_rtde = compose_T_tag0_rtde(T_tag0_vention, T_right)
    return LaserPlacement(
        points_vention=points_vention,
        points_tag0=points_tag0,
        T_left_base_vention=T_left,
        T_right_base_vention=T_right,
        T_tag0_vention=T_tag0_vention,
        T_tag0_rtde=T_tag0_rtde,
    )


def place_laser_object(
    ply_path,
    *,
    T_left_path: Path | str | None = None,
    T_right_path: Path | str | None = None,
    crop_meta_path: Path | str | None = None,
    laser_offset_path: Path | str | None = None,
    tag0_offset_path: Path | str | None = None,
) -> LaserPlacement:
    from geodude_teleoperation.mesh_fixture import load_points

    ply = Path(ply_path)
    sidecar = ply.with_name("merged.laser_offset.json")
    laser_offset = laser_offset_path or (sidecar if sidecar.is_file() else DEFAULT_LASER_OFFSET)
    return place_laser_points(
        load_points(ply),
        T_left_base_vention=load_SE3(T_left_path or DEFAULT_T_LEFT),
        T_right_base_vention=load_SE3(T_right_path or DEFAULT_T_RIGHT),
        T_tag0_base=load_SE3(crop_meta_path or DEFAULT_CROP_META),
        t_laser_vention_m=load_t_vention(laser_offset),
        t_tag0_vention_m=load_t_vention(tag0_offset_path or DEFAULT_TAG0_OFFSET),
    )
