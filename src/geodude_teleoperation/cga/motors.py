"""CGA motors / rotors (pygafro) for local-frame teleop sandwiches."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from geodude_teleoperation.cga.pygafro_backend import require_pygafro


def rotor_from_rotation_matrix(R: np.ndarray) -> Any:
    """Build a pygafro ``Rotor`` from a 3×3 base←local rotation matrix.

    Quaternion order for ``Rotor.fromQuaternion`` is ``[w, x, y, z]``.
    """
    ga = require_pygafro()
    Rm = np.asarray(R, dtype=float).reshape(3, 3)
    q = Rotation.from_matrix(Rm).as_quat()  # scipy: x, y, z, w
    return ga.Rotor.fromQuaternion(
        [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
    )


def motor_from_rotor(R: np.ndarray) -> Any:
    """Orientation-only ``Motor`` (identity translator) from ``R``."""
    ga = require_pygafro()
    return ga.Motor(rotor_from_rotation_matrix(R))


def sandwich_direction(
    R_local: np.ndarray,
    v_local: np.ndarray,
    *,
    motor: Any | None = None,
) -> np.ndarray:
    """Map a local Euclidean direction into the base/world frame via CGA sandwich.

    Uses pygafro ``Rotor.apply(DirectionVector)`` ≡ ``R v ~R`` (translations leave
    directions invariant). ``motor`` may be a ``Rotor`` or ``Motor``; if omitted,
    a rotor is built from ``R_local`` (VF / teleop motion frame).
    """
    ga = require_pygafro()
    v = np.asarray(v_local, dtype=float).reshape(3)
    if motor is None:
        motor = rotor_from_rotation_matrix(R_local)
    dv = ga.DirectionVector(float(v[0]), float(v[1]), float(v[2]))
    apply = getattr(motor, "apply", None)
    if not callable(apply) and hasattr(motor, "getRotor"):
        apply = motor.getRotor().apply
    out = apply(dv)
    return np.asarray(out.vector(), dtype=float).reshape(3)
