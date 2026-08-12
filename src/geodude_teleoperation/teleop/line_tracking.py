"""Phase 1.5: line tracking (free along surface normal) + geometric soft stand-off.

Pose tracking along the surface normal fights contact forces. Line tracking zeros
the **normal component of position error** so the tip is free to slide along the
``R_field`` approach axis while orientation (and tangent position) still track.

Soft stand-off injects a feedforward linear velocity
``v = -k (d - d*) n_out`` from signed distance — software compliance before F/T.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class LineTrackingConfig:
    """Runtime knobs for Phase 1.5 (and Phase 3 free-normal)."""

    enabled: bool = False
    standoff_enabled: bool = False
    standoff_distance_m: float = 0.002
    standoff_gain: float = 1.0
    # Cap |v_n| from stand-off (m/s); 0 disables.
    standoff_max_speed_m_s: float = 0.05
    # Keyboard field-z becomes feedforward along approach (not a stiff pose offset).
    keyboard_normal_ff: bool = True


def config_from_args(args: Any) -> LineTrackingConfig:
    def _flag(name: str, default: bool = False) -> bool:
        v = getattr(args, name, default)
        return v if isinstance(v, bool) else bool(v) if isinstance(v, (int, float)) else default

    def _float(name: str, default: float) -> float:
        v = getattr(args, name, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    enabled = _flag("line_tracking", False)
    standoff = _flag("soft_standoff", False)
    if standoff and not enabled:
        # Soft stand-off requires free normal; enable line tracking implicitly.
        enabled = True
    return LineTrackingConfig(
        enabled=enabled,
        standoff_enabled=standoff,
        standoff_distance_m=_float("standoff_distance", 0.002),
        standoff_gain=_float("standoff_gain", 1.0),
        standoff_max_speed_m_s=_float("standoff_max_speed", 0.05),
        keyboard_normal_ff=not _flag("no_keyboard_normal_ff", False),
    )


def outward_normal_from_R_field(R_field: np.ndarray) -> np.ndarray:
    """Unit outward normal.

    After ``_orient_approach_into_surface``, ``R_field[:, 2]`` points **into** the
    surface; outward is the negation.
    """
    n = -np.asarray(R_field, dtype=float).reshape(3, 3)[:, 2]
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def approach_axis_from_R_field(R_field: np.ndarray) -> np.ndarray:
    """Unit approach axis (into surface) = ``R_field[:, 2]``."""
    a = np.asarray(R_field, dtype=float).reshape(3, 3)[:, 2]
    norm = float(np.linalg.norm(a))
    if norm < 1e-12:
        return np.array([0.0, 0.0, -1.0])
    return a / norm


def kill_normal_linear_error(err6: np.ndarray, n_outward: np.ndarray) -> np.ndarray:
    """Zero the component of translational error along ``n_outward``."""
    e = np.asarray(err6, dtype=float).reshape(-1).copy()
    n = np.asarray(n_outward, dtype=float).reshape(3)
    n = n / (float(np.linalg.norm(n)) + 1e-12)
    e[:3] = e[:3] - n * float(np.dot(e[:3], n))
    return e


def snap_ref_position_to_line(
    p_cur: np.ndarray, p_ref: np.ndarray, n_outward: np.ndarray
) -> np.ndarray:
    """Move ``p_ref`` along the normal so ``(p_ref - p_cur) · n = 0``."""
    p_cur = np.asarray(p_cur, dtype=float).reshape(3)
    p_ref = np.asarray(p_ref, dtype=float).reshape(3)
    n = np.asarray(n_outward, dtype=float).reshape(3)
    n = n / (float(np.linalg.norm(n)) + 1e-12)
    return p_ref - n * float(np.dot(p_ref - p_cur, n))


def soft_standoff_velocity(
    signed_d: float,
    *,
    d_star: float,
    gain: float,
    n_outward: np.ndarray,
    max_speed: float = 0.0,
) -> np.ndarray:
    """Geometric stand-off: ``v = -k (d - d*) n_out`` (m/s, world frame)."""
    n = np.asarray(n_outward, dtype=float).reshape(3)
    n = n / (float(np.linalg.norm(n)) + 1e-12)
    v_n = -float(gain) * (float(signed_d) - float(d_star))
    if max_speed > 0.0:
        v_n = float(np.clip(v_n, -max_speed, max_speed))
    return v_n * n


def split_field_linear_tangent_normal(
    lin_field: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Split field-frame linear twist into tangent ``[lx,ly,0]`` and normal ``lz``."""
    lin = np.asarray(lin_field, dtype=float).reshape(3)
    tangent = lin.copy()
    tangent[2] = 0.0
    return tangent, float(lin[2])


def field_normal_ff_world(
    lin_n_field: float, R_field: np.ndarray, *, scale: float = 1.0
) -> np.ndarray:
    """Map field-frame approach speed to world (``R[:,2]`` = into surface)."""
    return float(scale) * float(lin_n_field) * approach_axis_from_R_field(R_field)


def apply_line_tracking_pose_error(
    err6: np.ndarray,
    R_field: np.ndarray,
    *,
    enabled: bool,
) -> np.ndarray:
    if not enabled:
        return np.asarray(err6, dtype=float).reshape(-1)
    return kill_normal_linear_error(err6, outward_normal_from_R_field(R_field))


def combine_normal_feedforward(
    *,
    cfg: LineTrackingConfig,
    signed_d: float,
    R_field: np.ndarray,
    keyboard_v_world: np.ndarray | None = None,
) -> np.ndarray:
    """World-frame linear velocity along the free normal (stand-off + optional keys)."""
    v = np.zeros(3, dtype=float)
    if not cfg.enabled:
        return v
    n_out = outward_normal_from_R_field(R_field)
    if cfg.standoff_enabled:
        v = v + soft_standoff_velocity(
            signed_d,
            d_star=cfg.standoff_distance_m,
            gain=cfg.standoff_gain,
            n_outward=n_out,
            max_speed=cfg.standoff_max_speed_m_s,
        )
    if keyboard_v_world is not None:
        v = v + np.asarray(keyboard_v_world, dtype=float).reshape(3)
    return v


def qdot_from_task_linear(
    v_world: np.ndarray,
    jacobian: np.ndarray,
    *,
    damping: float = 0.05,
    max_joint_velocity: float | None = None,
) -> np.ndarray:
    """Map a pure translational task velocity to ``q̇`` via damped least squares."""
    v = np.asarray(v_world, dtype=float).reshape(3)
    if float(np.linalg.norm(v)) < 1e-12:
        return np.zeros(jacobian.shape[1], dtype=float)
    J = np.asarray(jacobian, dtype=float)
    # Geometric Jacobian rows [v; ω] — use linear block.
    Jv = J[:3, :]
    j_inv = np.linalg.inv(Jv.T @ Jv + damping**2 * np.eye(Jv.shape[1]))
    qdot = j_inv @ Jv.T @ v
    if max_joint_velocity is not None:
        speed = float(np.linalg.norm(qdot))
        if speed > float(max_joint_velocity):
            qdot = qdot * (float(max_joint_velocity) / speed)
    return qdot


def vf_R_field(vf) -> np.ndarray | None:
    if vf is None:
        return None
    if hasattr(vf, "motion_basis"):
        return np.asarray(vf.motion_basis(), dtype=float).reshape(3, 3)
    status = getattr(vf, "status", None)
    if isinstance(status, Mapping) and "R_field_base" in status:
        return np.asarray(status["R_field_base"], dtype=float).reshape(3, 3)
    return None


def vf_signed_d(vf) -> float:
    status = getattr(vf, "status", None) or {}
    if isinstance(status, Mapping):
        return float(status.get("signed_d", 0.0))
    return 0.0
