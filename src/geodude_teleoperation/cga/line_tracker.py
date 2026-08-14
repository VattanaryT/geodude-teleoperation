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
    """
    require_pygafro()
    L_ee = ee_line_from_tip(p_tcp, R_tcp)
    L_a = target_line_from_approach(x_q, R_field)
    M = line_to_line_motor(L_ee, L_a)
    gen = M.log().evaluate()
    return motor_generator_to_twist_se3(np.asarray(gen.vector(), dtype=float))


def alignment_wrench_from_twist(twist_se3: np.ndarray) -> np.ndarray:
    """``W_{L} = I_c V`` in matching 6-vector coordinates.

    With MotorGenerator dual identification, the SE(3) ``[v; ω]`` layout already
    pairs with geometric Jacobian rows ``[v; ω]`` / wrench ``[f; m]``.
    """
    return np.asarray(twist_se3, dtype=float).reshape(6).copy()
