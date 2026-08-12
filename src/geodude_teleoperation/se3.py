"""Minimal SE(3) helpers for gafropy-free RTDE teleop."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


class NumpyTwist:
    """Duck-compatible with gafropy.Twist for angular/linear accessors.

    Blade order matches keyboard / gafropy: ``[e12, e13, e23, e1i, e2i, e3i]``
    = ``[angular(3), linear(3)]``.
    """

    def __init__(self, angular: np.ndarray, linear: np.ndarray):
        self._angular = np.asarray(angular, dtype=float).reshape(3)
        self._linear = np.asarray(linear, dtype=float).reshape(3)

    @classmethod
    def from_array(cls, payload: Sequence[float]) -> "NumpyTwist":
        arr = np.asarray(payload, dtype=float).reshape(6)
        return cls(arr[:3], arr[3:])

    def get_angular(self) -> np.ndarray:
        return self._angular.copy()

    def get_linear(self) -> np.ndarray:
        return self._linear.copy()

    def to_array(self) -> np.ndarray:
        return np.concatenate([self._angular, self._linear])


def tcp_pose_to_matrix(tcp_pose: Sequence[float]) -> np.ndarray:
    """UR ``getActualTCPPose`` [x,y,z,rx,ry,rz] axis-angle → 4x4."""
    pose = np.asarray(tcp_pose, dtype=float).reshape(6)
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
    return T


def matrix_to_tcp_pose(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float).reshape(4, 4)
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()])


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def map_jacobian_to_point(
    J_at_a: np.ndarray, p_a: np.ndarray, p_b: np.ndarray
) -> np.ndarray:
    """Map spatial geometric Jacobian from point A to point B (same ω)."""
    J = np.asarray(J_at_a, dtype=float).reshape(6, 6)
    r = np.asarray(p_b, dtype=float).reshape(3) - np.asarray(p_a, dtype=float).reshape(3)
    J_out = J.copy()
    J_out[:3, :] = J[:3, :] - skew(r) @ J[3:, :]
    return J_out


def body_twist_increment(T: np.ndarray, twist: NumpyTwist) -> np.ndarray:
    """Right-multiply ``T`` by ``exp([ω, v])`` (body-frame twist)."""
    T = np.asarray(T, dtype=float).reshape(4, 4)
    w = twist.get_angular()
    v = twist.get_linear()
    dR = Rotation.from_rotvec(w).as_matrix()
    dT = np.eye(4)
    dT[:3, :3] = dR
    dT[:3, 3] = v
    return T @ dT


def pose_error_current_minus_ref(T_cur: np.ndarray, T_ref: np.ndarray) -> np.ndarray:
    """6D error ``[p - p_ref; rotvec(R R_ref^T)]`` for ``qdot = -k J^+ e``."""
    T_cur = np.asarray(T_cur, dtype=float).reshape(4, 4)
    T_ref = np.asarray(T_ref, dtype=float).reshape(4, 4)
    e_pos = T_cur[:3, 3] - T_ref[:3, 3]
    R_err = T_cur[:3, :3] @ T_ref[:3, :3].T
    e_rot = Rotation.from_matrix(R_err).as_rotvec()
    return np.concatenate([e_pos, e_rot])


class _DuckMotor:
    """Minimal motor duck for VF project / numpy bridge (no gafropy)."""

    def __init__(self, translator, R: np.ndarray):
        self._translator = translator
        self._R = np.asarray(R, dtype=float).reshape(3, 3)

    def get_translator(self):
        return self._translator

    def get_rotor(self):
        return Rotation.from_matrix(self._R)


def motor_from_matrix(T: np.ndarray):
    """Duck motor compatible with ``ContactWosOrientationField`` / VF project."""
    T = np.asarray(T, dtype=float).reshape(4, 4)
    p = T[:3, 3]
    R = T[:3, :3]

    class _Translator:
        def x(self):
            return float(p[0])

        def y(self):
            return float(p[1])

        def z(self):
            return float(p[2])

        def to_translation_vector(self):
            return np.asarray(p, dtype=float).copy()

    return _DuckMotor(_Translator(), R)


def matrix_from_motor(ee_motor) -> np.ndarray:
    translator = ee_motor.get_translator()
    if hasattr(translator, "to_translation_vector"):
        p = np.asarray(translator.to_translation_vector(), dtype=float).reshape(3)
    else:
        p = np.array([translator.x(), translator.y(), translator.z()], dtype=float)
    rotor = ee_motor.get_rotor()
    if hasattr(rotor, "as_matrix"):
        R = np.asarray(rotor.as_matrix(), dtype=float).reshape(3, 3)
    else:
        from diffused_fields.manifold._rotor_interop import as_scipy_rotation

        R = as_scipy_rotation(rotor).as_matrix()
    T = np.eye(4)
    T[:3, 3] = p
    T[:3, :3] = R
    return T
