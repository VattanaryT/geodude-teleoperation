"""CGA line primitives via pygafro."""
from __future__ import annotations

from typing import Any

import numpy as np

from geodude_teleoperation.cga.pygafro_backend import require_pygafro


def _unit(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return a / n


def line_from_point_direction(point: np.ndarray, direction: np.ndarray) -> Any:
    """Algebraic line through ``point`` along unit ``direction`` (base frame)."""
    ga = require_pygafro()
    p = np.asarray(point, dtype=float).reshape(3)
    d = _unit(direction)
    p0 = ga.Point(float(p[0]), float(p[1]), float(p[2]))
    p1 = ga.Point(float(p[0] + d[0]), float(p[1] + d[1]), float(p[2] + d[2]))
    return ga.Line(p0, p1)


def motor_generator_to_twist_se3(gen_vec: np.ndarray) -> np.ndarray:
    """Map pygafro MotorGenerator ``[ω; v]`` → SE(3) twist ``[v; ω]`` (matches UR J)."""
    g = np.asarray(gen_vec, dtype=float).reshape(6)
    return np.concatenate([g[3:6], g[0:3]])


def twist_se3_to_motor_generator(twist_se3: np.ndarray) -> np.ndarray:
    """Inverse of :func:`motor_generator_to_twist_se3`."""
    t = np.asarray(twist_se3, dtype=float).reshape(6)
    return np.concatenate([t[3:6], t[0:3]])
