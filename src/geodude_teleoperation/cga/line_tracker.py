"""Line-to-line motor tracking: ``V = log(M_{L_ee}^{L_{a,⊥}})``."""
from __future__ import annotations

from typing import Any

import numpy as np

from geodude_teleoperation.cga.primitives import (
    line_from_point_direction,
    motor_generator_to_twist_se3,
)
from geodude_teleoperation.cga.pygafro_backend import require_pygafro


def ee_line_from_tip(p_tcp: np.ndarray, R_tcp: np.ndarray) -> Any:
    """End-effector approach line: tip + tool ``+Z`` (column 2)."""
    R = np.asarray(R_tcp, dtype=float).reshape(3, 3)
    return line_from_point_direction(p_tcp, R[:, 2])


def target_line_from_approach(x_q: np.ndarray, R_field: np.ndarray) -> Any:
    """Target contact line ``L_{a,⊥}``: desired point + surface approach ``e3``."""
    R = np.asarray(R_field, dtype=float).reshape(3, 3)
    return line_from_point_direction(x_q, R[:, 2])


def line_to_line_motor(L_ee: Any, L_target: Any) -> Any:
    """Motor ``M`` taking ``L_ee`` onto ``L_target`` (pygafro ``Line.getMotor``)."""
    return L_ee.getMotor(L_target)


def alignment_twist_se3(
    p_tcp: np.ndarray,
    R_tcp: np.ndarray,
    x_q: np.ndarray,
    R_field: np.ndarray,
) -> np.ndarray:
    """SE(3) twist ``[v; ω]`` from ``log(M)`` of the line-to-line motor.

    Free along the line: pure slide along the shared direction yields near-zero cost.

    This is a **spatial screw**. When the TCP is not at the origin it generally
    includes translation even if ``x_q == p_tcp`` (pygafro ``getMotor`` need not
    rotate about the tip). In-contact Align should use
    :func:`alignment_twist_orientation_se3` so the tip is not yanked.
    """
    require_pygafro()
    L_ee = ee_line_from_tip(p_tcp, R_tcp)
    L_a = target_line_from_approach(x_q, R_field)
    M = line_to_line_motor(L_ee, L_a)
    gen = M.log().evaluate()
    return motor_generator_to_twist_se3(np.asarray(gen.vector(), dtype=float))


def _z_align_e_R(R_tcp: np.ndarray, R_field: np.ndarray) -> np.ndarray:
    """Axis-angle taking tool +Z onto field n* (DualLoop ``n_ee × n*``)."""
    n_ee = np.asarray(R_tcp, dtype=float).reshape(3, 3)[:, 2]
    n_star = np.asarray(R_field, dtype=float).reshape(3, 3)[:, 2]
    ne = float(np.linalg.norm(n_ee))
    ns = float(np.linalg.norm(n_star))
    if ne < 1e-12 or ns < 1e-12:
        return np.zeros(3, dtype=float)
    a = n_ee / ne
    b = n_star / ns
    c = np.cross(a, b)
    s = float(np.linalg.norm(c))
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-10:
        if d > 0.0:
            return np.zeros(3, dtype=float)
        tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, tmp)
        an = float(np.linalg.norm(axis))
        return (axis / an) * np.pi if an > 1e-12 else np.zeros(3, dtype=float)
    return (c / s) * float(np.arctan2(s, d))


def alignment_twist_orientation_se3(
    R_tcp: np.ndarray,
    R_field: np.ndarray,
) -> np.ndarray:
    """Tip-fixed Align twist: v = 0, omega = -e_R (wrist moment only)."""
    e_R = _z_align_e_R(R_tcp, R_field)
    return np.concatenate([np.zeros(3, dtype=float), -e_R])


def alignment_twist_contact_se3(
    R_tcp: np.ndarray,
    R_field: np.ndarray,
    *,
    lever_m: float = 0.08,
    v_cap_m: float = 0.04,
) -> np.ndarray:
    """Align twist at the pendant TCP: ω = −e_R, v = ω × (lever z).

    A TCP-only moment (``v=0``) maps through ``Jᵀ`` onto the wrist. The
    linear term is a rotation about a point ``lever_m`` behind the pendant
    TCP (toward the flange along −tool Z) so shoulder/elbow share the
    reorient. ``v`` does **not** depend on how far the TCP is from the
    origin (that was the pygafro spatial-screw yank).
    """
    e_R = _z_align_e_R(R_tcp, R_field)
    omega = -e_R
    z = np.asarray(R_tcp, dtype=float).reshape(3, 3)[:, 2]
    zn = float(np.linalg.norm(z))
    if zn < 1e-12:
        return np.concatenate([np.zeros(3, dtype=float), omega])
    z = z / zn
    lever = float(max(lever_m, 0.0))
    v = np.cross(omega, lever * z)
    vn = float(np.linalg.norm(v))
    cap = float(max(v_cap_m, 0.0))
    if cap > 1e-12 and vn > cap:
        v = v * (cap / vn)
    return np.concatenate([v, omega])


def alignment_wrench_from_twist(twist_se3: np.ndarray) -> np.ndarray:
    """``W_{L} = I_c V`` in matching 6-vector coordinates.

    With MotorGenerator dual identification, the SE(3) ``[v; ω]`` layout already
    pairs with geometric Jacobian rows ``[v; ω]`` / wrench ``[f; m]``.
    """
    return np.asarray(twist_se3, dtype=float).reshape(6).copy()
