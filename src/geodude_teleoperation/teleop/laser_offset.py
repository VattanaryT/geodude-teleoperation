"""Per-session laser cloud translation in ``vention_frame``."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from geodude_teleoperation.teleop.object_offset import (
    DEFAULT_STEP_M,
    load_object_offset,
    save_object_offset,
)


def sidecar_path_for_laser(laser_ply: Path | str | None) -> Path | None:
    """``merged.ply`` → ``merged.laser_offset.json``."""
    if laser_ply is None:
        return None
    p = Path(laser_ply)
    return p.with_name(f"{p.stem}.laser_offset.json")


def resolve_laser_offset_path(args: Any) -> Path | None:
    """Explicit ``--laser-offset``, else sidecar next to ``--laser-ply`` / default."""
    if bool(getattr(args, "no_laser_offset", False)):
        return None
    explicit = getattr(args, "laser_offset", None)
    if explicit is not None:
        return Path(explicit)
    from geodude_teleoperation.teleop.laser_cloud import resolve_laser_ply

    return sidecar_path_for_laser(resolve_laser_ply(getattr(args, "laser_ply", None)))


def load_laser_offset(path: Path | str | None) -> tuple[np.ndarray, float]:
    """Same JSON schema as object offset: ``t_vention_m``, ``step_m``."""
    return load_object_offset(path)


def save_laser_offset(
    path: Path | str,
    t_vention_m: np.ndarray,
    *,
    step_m: float = DEFAULT_STEP_M,
) -> Path:
    return save_object_offset(path, t_vention_m, step_m=step_m)


def apply_laser_t_offset(
    points_vention: np.ndarray, t_vention_m: np.ndarray
) -> np.ndarray:
    """Translate points already in vention by ``t_vention_m``."""
    P = np.asarray(points_vention, dtype=float).reshape(-1, 3)
    t = np.asarray(t_vention_m, dtype=float).reshape(3)
    return P + t.reshape(1, 3)


def bind_laser_offset_to_args(args: Any) -> tuple[np.ndarray, float, Path | None]:
    """Load offset into mutable ``args.laser_offset_t`` (shared with live viz)."""
    path = resolve_laser_offset_path(args)
    t, step = load_laser_offset(path)
    args.laser_offset_t = np.asarray(t, dtype=float).reshape(3).copy()
    args.laser_offset_step_m = float(step)
    args.laser_offset_path = path
    if path is not None and path.is_file():
        print(
            f"laser offset: loaded t_vention_m={args.laser_offset_t.tolist()} "
            f"from {path}",
            flush=True,
        )
    elif float(np.linalg.norm(args.laser_offset_t)) > 0:
        print(
            f"laser offset: t_vention_m={args.laser_offset_t.tolist()}",
            flush=True,
        )
    return args.laser_offset_t, float(step), path
