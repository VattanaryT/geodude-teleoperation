"""Load Keyence layup scan clouds (left RTDE base) into vention / tag_0."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from geodude_teleoperation.frames import (
    left_base_to_vention_points,
    load_T_left_base_vention,
    transform_points,
    vention_to_tag0_points,
)
from geodude_teleoperation.paths import data_processed_dir, resolve_repo_path

DEFAULT_LASER_PLY = data_processed_dir() / "laser_profilometer" / "merged.ply"
DEFAULT_LASER_VOXEL_M = 0.002


def resolve_laser_ply(path: Path | str | None) -> Path:
    """Explicit path, else worktree default ``data/processed/laser_profilometer/merged.ply``."""
    if path is not None:
        resolved = resolve_repo_path(path)
        if resolved is not None:
            return resolved
    return Path(DEFAULT_LASER_PLY)


def is_laser_left_base_ply(path: Path | str | None) -> bool:
    """True when ``path`` is the Keyence merge (points already in left UR base).

    ``paths.ply`` is otherwise assumed tag_0; pointing it at ``merged.ply`` must
    go through left_base→vention→tag0 or the layup appears rotated ~90°.
    """
    if path is None:
        return False
    p = resolve_repo_path(path) or Path(path)
    try:
        if p.resolve() == Path(DEFAULT_LASER_PLY).resolve():
            return True
    except OSError:
        pass
    parts = {part.lower() for part in p.parts}
    return "laser_profilometer" in parts and p.name.lower() == "merged.ply"


def load_points_xyz(ply: Path | str) -> np.ndarray:
    """Read PLY XYZ (no downsample)."""
    ply = Path(ply).resolve()
    if not ply.is_file():
        raise FileNotFoundError(ply)
    raw = o3d.io.read_point_cloud(str(ply))
    pts = np.asarray(raw.points, dtype=float)
    if pts.size == 0:
        raise RuntimeError(f"Empty point cloud: {ply}")
    return pts.reshape(-1, 3)


def voxel_downsample_points(points: np.ndarray, voxel_m: float) -> np.ndarray:
    """In-memory voxel downsample; ``voxel_m <= 0`` returns a copy unchanged."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if voxel_m is None or float(voxel_m) <= 0.0:
        return pts.copy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    ds = pcd.voxel_down_sample(float(voxel_m))
    return np.asarray(ds.points, dtype=float).reshape(-1, 3)


def load_laser_points_left_base(
    ply: Path | str | None = None,
    *,
    voxel_m: float = DEFAULT_LASER_VOXEL_M,
) -> np.ndarray:
    """Load ``merged.ply`` (already fused in left UR base) and voxel-downsample."""
    path = resolve_laser_ply(ply)
    pts = load_points_xyz(path)
    return voxel_downsample_points(pts, voxel_m)


def laser_left_base_to_vention(
    points_left_base: np.ndarray,
    T_left_base_vention: np.ndarray | None = None,
) -> np.ndarray:
    """Map left-base points into ``vention_frame``."""
    if T_left_base_vention is None:
        T_left_base_vention = load_T_left_base_vention()
    return left_base_to_vention_points(points_left_base, T_left_base_vention)


def laser_vention_to_tag0(
    points_vention: np.ndarray, T_tag0_vention: np.ndarray
) -> np.ndarray:
    """Map vention points into tag_0 using offset-aware ``T_tag0_vention``."""
    return vention_to_tag0_points(points_vention, T_tag0_vention)


def load_laser_points_vention(
    ply: Path | str | None = None,
    *,
    voxel_m: float = DEFAULT_LASER_VOXEL_M,
    T_left_base_vention: np.ndarray | None = None,
    t_vention_m: np.ndarray | None = None,
) -> np.ndarray:
    """Load + voxel + left_base→vention (+ optional vention translation)."""
    left = load_laser_points_left_base(ply, voxel_m=voxel_m)
    vent = laser_left_base_to_vention(left, T_left_base_vention)
    if t_vention_m is not None:
        t = np.asarray(t_vention_m, dtype=float).reshape(3)
        if float(np.linalg.norm(t)) > 0.0:
            vent = vent + t.reshape(1, 3)
    return vent


def load_laser_points_tag0(
    ply: Path | str | None,
    T_tag0_vention: np.ndarray,
    *,
    voxel_m: float = DEFAULT_LASER_VOXEL_M,
    T_left_base_vention: np.ndarray | None = None,
    t_vention_m: np.ndarray | None = None,
) -> np.ndarray:
    """Load laser cloud expressed in tag_0 (uses effective ``T_tag0_vention``)."""
    vent = load_laser_points_vention(
        ply,
        voxel_m=voxel_m,
        T_left_base_vention=T_left_base_vention,
        t_vention_m=t_vention_m,
    )
    return laser_vention_to_tag0(vent, T_tag0_vention)


def points_vention_in_viz_frame(
    points_vention: np.ndarray,
    *,
    T_tag0_vention: np.ndarray,
    T_rtde_base_vention: np.ndarray | None,
    viz_frame: str,
) -> np.ndarray:
    """Map points already in vention into the --viz frame."""
    V = np.asarray(points_vention, dtype=float).reshape(-1, 3)
    if viz_frame in ("vention", "base"):
        return V.copy()
    if viz_frame == "tag0":
        return vention_to_tag0_points(V, T_tag0_vention)
    if viz_frame == "rtde":
        if T_rtde_base_vention is None:
            raise ValueError("viz_frame=rtde requires T_rtde_base_vention")
        T_rv = np.asarray(T_rtde_base_vention, dtype=float).reshape(4, 4)
        return transform_points(np.linalg.inv(T_rv), V)
    raise ValueError(f"viz_frame must be tag0|vention|rtde|base, got {viz_frame!r}")
