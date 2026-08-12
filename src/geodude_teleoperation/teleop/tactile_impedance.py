"""Phase 3: tactile wrench feedback + task-space impedance torques.

Builds on Phase 1.5 line tracking: tangent position / orientation tracking stays
on the ``speedJ`` (or later DTC) path; the **normal** free axis carries force.

Control law (Franka / tactile ergodic sources): ``τ = −Jᵀ W``.
``W`` comes from a wrench PID on ``actual_TCP_force`` (payload-compensated on UR)
versus a desired contact wrench ``W_d`` (typically force along outward normal).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from geodude_teleoperation.teleop.line_tracking import outward_normal_from_R_field


@dataclass
class WrenchPID:
    """Diagonal PI(D) on wrench error ``W_d - W_meas`` (SI: N, N·m)."""

    kp: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    ki: np.ndarray = field(default_factory=lambda: np.zeros(6))
    kd: np.ndarray = field(default_factory=lambda: np.zeros(6))
    i_clamp: float = 50.0
    _integral: np.ndarray = field(default_factory=lambda: np.zeros(6), repr=False)
    _prev_err: np.ndarray = field(default_factory=lambda: np.zeros(6), repr=False)
    _has_prev: bool = False

    def reset(self) -> None:
        self._integral[:] = 0.0
        self._prev_err[:] = 0.0
        self._has_prev = False

    def step(self, W_d: Sequence[float], W_meas: Sequence[float], dt: float) -> np.ndarray:
        err = np.asarray(W_d, dtype=float).reshape(6) - np.asarray(W_meas, dtype=float).reshape(6)
        self._integral = self._integral + err * float(dt)
        self._integral = np.clip(self._integral, -self.i_clamp, self.i_clamp)
        deriv = np.zeros(6) if not self._has_prev else (err - self._prev_err) / max(float(dt), 1e-6)
        self._prev_err = err.copy()
        self._has_prev = True
        kp = np.asarray(self.kp, dtype=float).reshape(6)
        ki = np.asarray(self.ki, dtype=float).reshape(6)
        kd = np.asarray(self.kd, dtype=float).reshape(6)
        return kp * err + ki * self._integral + kd * deriv


@dataclass
class TactileConfig:
    enabled: bool = False
    desired_normal_force_n: float = 5.0
    # Wrench PID gains (force axes); torque axes default 0.
    kp_force: float = 0.02
    ki_force: float = 0.0
    kd_force: float = 0.0
    # Map wrench command to normal velocity when still on speedJ (Phase 3a).
    wrench_to_vn_gain: float = 0.01
    max_vn_m_s: float = 0.03
    bias_wrench: np.ndarray | None = None


def _flag(args: Any, name: str, default: bool = False) -> bool:
    v = getattr(args, name, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(v)
    return default


def _float(args: Any, name: str, default: float) -> float:
    v = getattr(args, name, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def tactile_config_from_args(args: Any) -> TactileConfig:
    bias = getattr(args, "ft_bias", None)
    bias_arr = None
    if bias is not None:
        try:
            arr = np.asarray(bias, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            arr = np.array([])
        if arr.size == 6:
            bias_arr = arr
    return TactileConfig(
        enabled=_flag(args, "tactile", False),
        desired_normal_force_n=_float(args, "desired_normal_force", 5.0),
        kp_force=_float(args, "wrench_kp", 0.02),
        ki_force=_float(args, "wrench_ki", 0.0),
        kd_force=_float(args, "wrench_kd", 0.0),
        wrench_to_vn_gain=_float(args, "wrench_to_vn", 0.01),
        max_vn_m_s=_float(args, "tactile_max_vn", 0.03),
        bias_wrench=bias_arr,
    )


def desired_wrench_along_normal(
    R_field: np.ndarray, force_n: float, *, compress: bool = True
) -> np.ndarray:
    """``W_d`` with force along outward normal (compress ⇒ into surface = −n_out)."""
    n_out = outward_normal_from_R_field(R_field)
    direction = -n_out if compress else n_out
    W = np.zeros(6, dtype=float)
    W[:3] = float(force_n) * direction
    return W


def compensate_bias(wrench: Sequence[float], bias: Sequence[float] | None) -> np.ndarray:
    w = np.asarray(wrench, dtype=float).reshape(6)
    if bias is None:
        return w
    return w - np.asarray(bias, dtype=float).reshape(6)


def joint_torques_from_wrench(jacobian: np.ndarray, wrench: Sequence[float]) -> np.ndarray:
    """``τ = −Jᵀ W`` (geometric Jacobian, task wrench in same frame as J)."""
    J = np.asarray(jacobian, dtype=float)
    W = np.asarray(wrench, dtype=float).reshape(6)
    return -J.T @ W


class TactileAgent:
    """Phase 3 agent: F/T → wrench PID → normal FF (speedJ) or τ (DTC)."""

    def __init__(self, cfg: TactileConfig):
        self.cfg = cfg
        kp = np.zeros(6)
        kp[:3] = cfg.kp_force
        ki = np.zeros(6)
        ki[:3] = cfg.ki_force
        kd = np.zeros(6)
        kd[:3] = cfg.kd_force
        self.pid = WrenchPID(kp=kp, ki=ki, kd=kd)
        self.last_W_meas = np.zeros(6)
        self.last_W_cmd = np.zeros(6)
        self.last_tau = np.zeros(6)
        self.last_vn_world = np.zeros(3)

    def reset(self) -> None:
        self.pid.reset()

    def update(
        self,
        *,
        wrench_meas: Sequence[float],
        R_field: np.ndarray,
        jacobian: np.ndarray | None,
        dt: float,
        mode: str = "speedj",
    ) -> dict:
        """Compute tactile command.

        ``mode='speedj'``: map normal force error to ``v_n`` along outward normal.
        ``mode='dtc'``: return ``τ = −Jᵀ W_c``.
        """
        if not self.cfg.enabled:
            return {
                "vn_world": np.zeros(3),
                "tau": np.zeros(6),
                "W_cmd": np.zeros(6),
                "W_meas": np.zeros(6),
            }

        W_meas = compensate_bias(wrench_meas, self.cfg.bias_wrench)
        W_d = desired_wrench_along_normal(R_field, self.cfg.desired_normal_force_n)
        W_c = self.pid.step(W_d, W_meas, dt)
        self.last_W_meas = W_meas
        self.last_W_cmd = W_c

        n_out = outward_normal_from_R_field(R_field)
        # Force error along compress direction (−n_out).
        f_err = float(np.dot(W_d[:3] - W_meas[:3], -n_out))
        vn = float(self.cfg.wrench_to_vn_gain) * f_err
        vn = float(np.clip(vn, -self.cfg.max_vn_m_s, self.cfg.max_vn_m_s))
        # Move into surface when we want more compression force.
        vn_world = vn * (-n_out)
        self.last_vn_world = vn_world

        tau = np.zeros(6)
        if mode == "dtc" and jacobian is not None:
            tau = joint_torques_from_wrench(jacobian, W_c)
            self.last_tau = tau

        return {
            "vn_world": vn_world,
            "tau": tau,
            "W_cmd": W_c,
            "W_meas": W_meas,
            "W_d": W_d,
            "f_err": f_err,
        }
