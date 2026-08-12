"""Per-session object translation offset on ``T_tag0`` (vention parent)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_STEP_M = 0.001


def sidecar_path_for(T_tag0_base: Path | str | None) -> Path | None:
    """``crop_meta.json`` → ``crop_meta.object_offset.json``."""
    if T_tag0_base is None:
        return None
    p = Path(T_tag0_base)
    return p.with_name(f"{p.stem}.object_offset.json")


def resolve_object_offset_path(args: Any) -> Path | None:
    """Explicit ``--object-offset``, else sidecar next to ``--T-tag0-base``."""
    if bool(getattr(args, "no_object_offset", False)):
        return None
    explicit = getattr(args, "object_offset", None)
    if explicit is not None:
        return Path(explicit)
    return sidecar_path_for(getattr(args, "T_tag0_base", None))


def load_object_offset(
    path: Path | str | None,
) -> tuple[np.ndarray, float]:
    """Return ``(t_vention_m, step_m)``. Missing file → zeros / default step."""
    t = np.zeros(3, dtype=float)
    step = float(DEFAULT_STEP_M)
    if path is None:
        return t, step
    path = Path(path)
    if not path.is_file():
        return t, step
    with path.open() as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError(f"object offset JSON must be an object: {path}")
    raw = meta.get("t_vention_m", meta.get("t", [0.0, 0.0, 0.0]))
    t = np.asarray(raw, dtype=float).reshape(3).copy()
    step = float(meta.get("step_m", step))
    if step <= 0.0:
        step = float(DEFAULT_STEP_M)
    return t, step


def save_object_offset(
    path: Path | str,
    t_vention_m: np.ndarray,
    *,
    step_m: float = DEFAULT_STEP_M,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "t_vention_m": [float(x) for x in np.asarray(t_vention_m, dtype=float).reshape(3)],
        "step_m": float(step_m),
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def apply_object_t_offset(T_tag0_vention: np.ndarray, t_vention_m: np.ndarray) -> np.ndarray:
    """Return a copy of ``T`` with translation ``+= t_vention_m``."""
    T = np.asarray(T_tag0_vention, dtype=float).reshape(4, 4).copy()
    T[:3, 3] = T[:3, 3] + np.asarray(t_vention_m, dtype=float).reshape(3)
    return T


def delta_viz_to_vention(
    delta_viz: np.ndarray,
    *,
    viz_frame: str,
    T_tag0_vention: np.ndarray,
    T_rtde_base_vention: np.ndarray | None,
) -> np.ndarray:
    """Map a translation in viz-frame axes into vention XYZ for ``T_tag0``."""
    d = np.asarray(delta_viz, dtype=float).reshape(3)
    if viz_frame in ("vention", "base"):
        return d.copy()
    if viz_frame == "rtde":
        if T_rtde_base_vention is None:
            return d.copy()
        R = np.asarray(T_rtde_base_vention, dtype=float).reshape(4, 4)[:3, :3]
        return R @ d
    if viz_frame == "tag0":
        R = np.asarray(T_tag0_vention, dtype=float).reshape(4, 4)[:3, :3]
        return R @ d
    raise ValueError(f"viz_frame must be tag0|vention|rtde|base, got {viz_frame!r}")


def bind_object_offset_to_args(args: Any) -> tuple[np.ndarray, float, Path | None]:
    """Load offset into a mutable ``args.object_offset_t`` shared with the RTDE loop."""
    path = resolve_object_offset_path(args)
    t, step = load_object_offset(path)
    args.object_offset_t = np.asarray(t, dtype=float).reshape(3).copy()
    args.object_offset_step_m = float(step)
    args.object_offset_path = path
    if path is not None and path.is_file():
        print(
            f"object offset: loaded t_vention_m={args.object_offset_t.tolist()} "
            f"from {path}",
            flush=True,
        )
    elif np.linalg.norm(args.object_offset_t) > 0:
        print(
            f"object offset: t_vention_m={args.object_offset_t.tolist()}",
            flush=True,
        )
    return args.object_offset_t, float(step), path


def apply_args_object_offset(T_tag0_in_parent: np.ndarray, args: Any) -> np.ndarray:
    """Apply ``args.object_offset_t`` (vention) to a tag0 pose in parent frame.

    When ``--base-frame`` is already an RTDE base link, ``T`` is treated as
    tag0-in-RTDE and the vention translation is mapped through the mount.
    """
    t = getattr(args, "object_offset_t", None)
    if t is None or bool(getattr(args, "no_object_offset", False)):
        return np.asarray(T_tag0_in_parent, dtype=float).reshape(4, 4)
    try:
        t = np.asarray(t, dtype=float).reshape(3)
    except (TypeError, ValueError):
        return np.asarray(T_tag0_in_parent, dtype=float).reshape(4, 4)
    if float(np.linalg.norm(t)) < 1e-15:
        return np.asarray(T_tag0_in_parent, dtype=float).reshape(4, 4)

    parent = str(getattr(args, "base_frame", None) or "")
    in_rtde_parent = parent in (
        "right_base",
        "right_base_link",
        "right_base_link_inertia",
    ) or bool(getattr(args, "no_rtde_base_vention", False))

    T = np.asarray(T_tag0_in_parent, dtype=float).reshape(4, 4)
    if not in_rtde_parent:
        return apply_object_t_offset(T, t)

    # T is tag0 in RTDE: bump vention translation then convert back.
    from geodude_teleoperation.frames import (
        compose_T_tag0_rtde,
        load_T_rtde_base_vention,
    )

    try:
        T_rv = load_T_rtde_base_vention(getattr(args, "T_rtde_base_vention", None))
    except FileNotFoundError:
        return apply_object_t_offset(T, t)
    T_vention = np.asarray(T_rv, dtype=float).reshape(4, 4) @ T
    T_vention = apply_object_t_offset(T_vention, t)
    return compose_T_tag0_rtde(T_vention, T_rv)
