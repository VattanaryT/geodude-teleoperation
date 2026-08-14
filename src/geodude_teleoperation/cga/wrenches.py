"""Wrench frame change + contact PID (numpy 6-vectors ``[f; m]``)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geodude_teleoperation.teleop.tactile_impedance import WrenchPID


def wrench_to_ee_frame(
    wrench_base_or_tool: np.ndarray,
    R_tcp: np.ndarray,
    *,
    wrench_in_tool_frame: bool,
) -> np.ndarray:
    """Return wrench in EE/tool frame as ``[f; m]``.

    UR ``actual_TCP_force`` is typically tool-frame already; when it is base-frame,
    rotate with ``Rᵀ``.
    """
    W = np.asarray(wrench_base_or_tool, dtype=float).reshape(6)
    if wrench_in_tool_frame:
        return W.copy()
    R = np.asarray(R_tcp, dtype=float).reshape(3, 3)
    f = R.T @ W[:3]
    m = R.T @ W[3:]
    return np.concatenate([f, m])


def desired_contact_wrench_ee(
    *,
    desired_force_n: float,
    force_sign: float = 1.0,
) -> np.ndarray:
    """``W_d`` in EE frame: force along tool ``+Z`` (contact / compression axis)."""
    W = np.zeros(6, dtype=float)
    W[2] = float(force_sign) * float(desired_force_n)
    return W


def wrench_ee_to_base(W_ee: np.ndarray, R_tcp: np.ndarray) -> np.ndarray:
    """Rotate EE-frame wrench ``[f; m]`` into the robot base frame (matches base ``J``)."""
    R = np.asarray(R_tcp, dtype=float).reshape(3, 3)
    W = np.asarray(W_ee, dtype=float).reshape(6)
    return np.concatenate([R @ W[:3], R @ W[3:]])


def tangent_wrench_from_world_force(
    f_world: np.ndarray,
    R_tcp: np.ndarray | None = None,
) -> np.ndarray:
    """World/base-frame tangent force FF as wrench ``[f; 0]`` (same frame as ``J``).

    ``R_tcp`` is accepted for API compatibility but unused — do not rotate into EE
    before ``Jᵀ`` (that was the seek frame bug).
    """
    _ = R_tcp
    W = np.zeros(6, dtype=float)
    W[:3] = np.asarray(f_world, dtype=float).reshape(3)
    return W


def damping_wrench_from_qdot(
    jacobian_ee: np.ndarray,
    qdot: np.ndarray,
) -> np.ndarray:
    """``W_V ≈ J_{ee} q̇`` expressed as a task twist used as damping wrench coords."""
    J = np.asarray(jacobian_ee, dtype=float)
    qd = np.asarray(qdot, dtype=float).reshape(-1)
    return (J @ qd).reshape(6)


@dataclass
class ContactWrenchPID:
    """Wrench PID on contact normal (tool ``+Z``) only — not lateral axes.

    Lateral ``Fx/Fy → 0`` with high ``kp`` fights friction in hard contact, blows
    up ``‖W_C‖``, saturates ``τ``, and prevents force unload toward ``Fd``.
    """

    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    _pid: WrenchPID = field(init=False, repr=False)

    def __post_init__(self) -> None:
        k = float(self.kp)
        # Only tool-Z force channel (index 2).
        self._pid = WrenchPID(
            kp=np.array([0.0, 0.0, k, 0.0, 0.0, 0.0]),
            ki=np.array([0.0, 0.0, float(self.ki), 0.0, 0.0, 0.0]),
            kd=np.array([0.0, 0.0, float(self.kd), 0.0, 0.0, 0.0]),
        )

    def reset(self) -> None:
        self._pid.reset()

    def step(self, W_d: np.ndarray, W_m: np.ndarray, dt: float) -> np.ndarray:
        return self._pid.step(W_d, W_m, dt)
