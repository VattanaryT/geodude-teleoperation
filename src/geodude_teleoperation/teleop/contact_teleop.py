"""In-contact teleop bridge: DualLoop + keyboard tangent + VF n* freeze.

Dedicated ``--in-contact`` path (not free-space pose-ref teleop).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from geodude_teleoperation.teleop.controller import qdot_from_error
from geodude_teleoperation.in_contact_control.dual_loop import (
    DualLoopConfig,
    DualLoopController,
)
from geodude_teleoperation.in_contact_control.types import PoseSO3


def _unit(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return a / n


def orthonormal_R_from_approach(n_approach: np.ndarray, R_hint: np.ndarray | None = None) -> np.ndarray:
    """Build ``R`` with column 2 = unit approach (DualLoop ``n* = R[:,2]``)."""
    z = _unit(n_approach)
    hint = np.eye(3) if R_hint is None else np.asarray(R_hint, dtype=float).reshape(3, 3)
    x = hint[:, 0] - z * float(np.dot(hint[:, 0], z))
    if float(np.linalg.norm(x)) < 1e-8:
        x = hint[:, 1] - z * float(np.dot(hint[:, 1], z))
    if float(np.linalg.norm(x)) < 1e-8:
        tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(z, tmp)
    x = _unit(x)
    y = _unit(np.cross(z, x))
    x = _unit(np.cross(y, z))
    return np.column_stack([x, y, z])


def field_tangent_world(
    lin_field: Sequence[float],
    R_local: np.ndarray,
    *,
    scale: float = 1.0,
    n_approach: np.ndarray | None = None,
) -> np.ndarray:
    """Map field/local-frame linear ``[lx, ly, lz]`` to world tangent.

    ``R_local`` is the teleop / VF motion frame (same as free-space WASD:
    columns 0/1 = local x/y). Motion along ``n_approach`` (default ``R_local``
    column 2) is removed so Force owns the normal.
    """
    lin = np.asarray(lin_field, dtype=float).reshape(3)
    R = np.asarray(R_local, dtype=float).reshape(3, 3)
    n = _unit(R[:, 2] if n_approach is None else np.asarray(n_approach, dtype=float))
    v = float(scale) * (float(lin[0]) * R[:, 0] + float(lin[1]) * R[:, 1])
    return v - n * float(np.dot(v, n))


def twist_to_qdot(
    v_world: np.ndarray,
    w_rad: np.ndarray,
    jacobian: np.ndarray,
    *,
    damping: float = 0.05,
    max_joint_velocity: float | None = None,
) -> np.ndarray:
    """Map base-frame twist ``[v; ω]`` (m/s, rad/s) to ``q̇`` via damped LS."""
    twist6 = np.concatenate(
        [
            np.asarray(v_world, dtype=float).reshape(3),
            np.asarray(w_rad, dtype=float).reshape(3),
        ]
    )
    if float(np.linalg.norm(twist6)) < 1e-12:
        return np.zeros(jacobian.shape[1], dtype=float)
    # Reuse pose-error solver with error = -twist (qdot = gain * J+ * (-e) style).
    # Here we want qdot ≈ J+ twist, so pass e = -twist / gain with gain=1.
    return qdot_from_error(
        -twist6,
        jacobian,
        gain=1.0,
        damping=damping,
        max_joint_velocity=max_joint_velocity
        if max_joint_velocity is not None
        else 1e6,
    )


@dataclass
class ContactTeleopConfig:
    desired_force_n: float = 5.0
    # Keyboard step (default 0.01) × scale → tip tangent speed (m/s).
    tangent_scale: float = 8.0
    tangent_hold_s: float = 0.12
    max_linear_m_s: float = 0.04
    max_angular_deg_s: float = 12.0
    damping: float = 0.05
    max_joint_velocity: float = 1.0
    # UR ActualTCPForce (tool frame → base via R_ee): live contact logs showed
    # compression increasing F·n* (into-surface), so force_sign=+1 makes Fs>0 under
    # load. Use -1 only if Enter seeks *away* from the surface.
    wrench_in_tool_frame: bool = True
    force_sign: float = 1.0
    # Soft approach caps (m/s) — keep far below free-space teleop speeds.
    approach_v_max_m_s: float = 0.006
    max_v_along_m_s: float = 0.010
    # Freeze VF n* on Enter (press f to unfreeze). Stops live-n* lateral wander.
    freeze_n_on_arm: bool = True
    # Skip F/T bias if already loaded (avoids zeroing real contact force).
    bias_skip_load_N: float = 2.0
    # Align tool-Z → n* before Force seeks (prevents pressing at an angle).
    align_first: bool = True
    align_enter_tol_deg: float = 5.0  # tool-Z within this of n* → enable Force
    align_exit_tol_deg: float = 12.0  # if misalignment grows past this → pause Force


def config_from_args(args: Any) -> ContactTeleopConfig:
    def _float(name: str, default: float) -> float:
        v = getattr(args, name, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    def _bool(name: str, default: bool) -> bool:
        v = getattr(args, name, default)
        return bool(v) if v is not None else default

    return ContactTeleopConfig(
        desired_force_n=_float("desired_normal_force", 5.0),
        tangent_scale=_float("tangent_scale", 8.0),
        tangent_hold_s=_float("tangent_hold_s", 0.12),
        max_joint_velocity=_float("max_joint_velocity", 1.0),
        force_sign=_float("force_sign", 1.0),
        approach_v_max_m_s=_float("approach_v_max", 0.006),
        max_v_along_m_s=_float("force_max_v", 0.010),
        freeze_n_on_arm=_bool("freeze_n_on_arm", True),
        align_first=_bool("align_first", True),
        align_enter_tol_deg=_float("align_enter_tol_deg", 5.0),
        align_exit_tol_deg=_float("align_exit_tol_deg", 12.0),
    )


class ContactTeleopBridge:
    """Tip-tracked DualLoop + optional frozen n* + one-shot tangent keyboard FF.

    With ``align_first`` (default): after Enter, Align runs at full ``k_ori`` until
    tool-Z is within ``align_enter_tol_deg`` of ``n*``; Force seek/hold stays off
    until then (and pauses again if tilt exceeds ``align_exit_tol_deg``).
    """

    PHASE_ALIGN = "align"
    PHASE_SEEK = "seek"

    def __init__(self, cfg: ContactTeleopConfig | None = None):
        self.cfg = cfg or ContactTeleopConfig()
        dual_cfg = DualLoopConfig()
        dual_cfg.force.Fd_N = float(self.cfg.desired_force_n)
        dual_cfg.force.wrench_in_tool_frame = bool(self.cfg.wrench_in_tool_frame)
        dual_cfg.force.force_sign = float(self.cfg.force_sign)
        dual_cfg.force.approach_v_max_m_s = float(self.cfg.approach_v_max_m_s)
        dual_cfg.force.max_v_along_m_s = float(self.cfg.max_v_along_m_s)
        # High Fd needs gentler gain or the loop chatters (see contact_motion bounce).
        if float(self.cfg.desired_force_n) >= 8.0:
            dual_cfg.force.kp = min(float(dual_cfg.force.kp), 0.0005)
        dual_cfg.max_linear_m_s = float(self.cfg.max_linear_m_s)
        dual_cfg.max_angular_deg_s = float(self.cfg.max_angular_deg_s)
        self.dual = DualLoopController(dual_cfg)
        self._k_ori_full = float(self.dual.config.align.k_ori)
        self.n_frozen = False
        self._R_star: np.ndarray | None = None
        self._pending_v_tan = np.zeros(3, dtype=float)
        self._tan_hold_s_left = 0.0
        self.last_tangent_field = np.zeros(3, dtype=float)
        self.armed = False
        self.phase = self.PHASE_ALIGN if self.cfg.align_first else self.PHASE_SEEK
        self.last_diag: dict = {}

    def toggle_freeze(self) -> bool:
        self.n_frozen = not self.n_frozen
        return self.n_frozen

    def _refresh_R_star(self, R_field: np.ndarray, *, force: bool = False) -> np.ndarray:
        R_field = np.asarray(R_field, dtype=float).reshape(3, 3)
        if force or self._R_star is None or not self.n_frozen:
            self._R_star = orthonormal_R_from_approach(R_field[:, 2], R_hint=R_field)
        assert self._R_star is not None
        return self._R_star

    @staticmethod
    def _tool_z_error_deg(R_tcp: np.ndarray, R_star: np.ndarray) -> float:
        n_ee = _unit(np.asarray(R_tcp, dtype=float).reshape(3, 3)[:, 2])
        n_star = _unit(np.asarray(R_star, dtype=float).reshape(3, 3)[:, 2])
        d = float(np.clip(np.dot(n_ee, n_star), -1.0, 1.0))
        return float(np.degrees(np.arccos(d)))

    def _update_phase(self, ang_z_deg: float) -> None:
        if not self.cfg.align_first:
            self.phase = self.PHASE_SEEK
            self.dual.config.force.enable = True
            self.dual.config.align.k_ori = self._k_ori_full
            return
        # Full Align gain while correcting tilt; Force gated by phase.
        self.dual.config.align.k_ori = self._k_ori_full
        enter = float(self.cfg.align_enter_tol_deg)
        exit_ = float(max(self.cfg.align_exit_tol_deg, enter))
        if self.phase == self.PHASE_ALIGN:
            if ang_z_deg <= enter:
                self.phase = self.PHASE_SEEK
        else:
            if ang_z_deg >= exit_:
                self.phase = self.PHASE_ALIGN
        self.dual.config.force.enable = self.phase == self.PHASE_SEEK

    def arm(
        self,
        p_tcp: Sequence[float],
        R_tcp: np.ndarray,
        R_field: np.ndarray,
        *,
        force_bias_samples: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """Enter / re-arm: seed tip target and optionally set F/T bias."""
        R_star = self._refresh_R_star(R_field, force=True)
        p = np.asarray(p_tcp, dtype=float).reshape(3)
        self.dual.capture(PoseSO3(p=p, R=R_star))
        if force_bias_samples:
            arr = np.asarray(force_bias_samples, dtype=float).reshape(-1, 3)
            load = float(np.linalg.norm(arr.mean(axis=0)))
            if load >= float(self.cfg.bias_skip_load_N):
                # Already loaded — biasing would zero real contact and over-push.
                self.dual.force._f_bias[:] = 0.0
            else:
                self.dual.set_force_bias_from_samples(force_bias_samples)
        if self.cfg.freeze_n_on_arm:
            self.n_frozen = True
        ang0 = self._tool_z_error_deg(R_tcp, R_star)
        self.phase = (
            self.PHASE_SEEK
            if (not self.cfg.align_first or ang0 <= float(self.cfg.align_enter_tol_deg))
            else self.PHASE_ALIGN
        )
        self.dual.config.force.enable = self.phase == self.PHASE_SEEK
        self.dual.config.align.k_ori = self._k_ori_full
        self.armed = True
        self._pending_v_tan[:] = 0.0
        self._tan_hold_s_left = 0.0

    def push_tangent_field_linear(
        self,
        lin_field: Sequence[float],
        R_local: np.ndarray,
        *,
        n_approach: np.ndarray | None = None,
    ) -> None:
        """Queue world tangent velocity from **local/teleop** frame WASD.

        ``R_local`` must be the VF motion / field frame (``motion_basis``), not
        ``R_tcp``. ``n_approach`` kills the DualLoop normal (defaults to frozen
        ``R*`` column 2 when armed).
        """
        lin = np.asarray(lin_field, dtype=float).reshape(3)
        self.last_tangent_field = lin.copy()
        if n_approach is None and self._R_star is not None:
            n_approach = self._R_star[:, 2]
        self._pending_v_tan = field_tangent_world(
            lin,
            R_local,
            scale=float(self.cfg.tangent_scale),
            n_approach=n_approach,
        )
        self._tan_hold_s_left = float(self.cfg.tangent_hold_s)

    def step(
        self,
        *,
        p_tcp: Sequence[float],
        R_tcp: np.ndarray,
        R_field: np.ndarray,
        force_base: Sequence[float],
        jacobian: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Run DualLoop + pending tangent; return ``q̇``."""
        if not self.armed:
            return np.zeros(jacobian.shape[1], dtype=float)

        R_star = self._refresh_R_star(R_field)
        p = np.asarray(p_tcp, dtype=float).reshape(3)
        # Tip-track p* so Align owns orientation (e⊥≈0), not lateral servo.
        self.dual.update_tracked_target(p, R_star)

        ang_z_deg = self._tool_z_error_deg(R_tcp, R_star)
        prev_phase = self.phase
        self._update_phase(ang_z_deg)

        # Once loaded, cut Align ω — residual orientation rates at impact were
        # ~0.25 rad/s joint and kept driving into the surface after seek enable.
        fs_prev = float(self.last_diag.get("Fs_N", 0.0)) if self.last_diag else 0.0
        if abs(fs_prev) >= 0.8:
            self.dual.config.align.k_ori = self._k_ori_full * 0.15
        else:
            self.dual.config.align.k_ori = self._k_ori_full

        measured = PoseSO3(p=p, R=np.asarray(R_tcp, dtype=float).reshape(3, 3))
        cmd, diag = self.dual.step(measured, force_base=force_base, dt=dt)

        # Overload / wrong-sign: never push into the surface, but keep WASD
        # tangent (old path zeroed the full twist → dead keys while |Fs| high).
        fault = float(diag.get("force_fault_fs", 0.0)) > 0.5
        wrong = float(diag.get("force_wrong_sign", 0.0)) > 0.5
        aligning = self.phase == self.PHASE_ALIGN

        v_tan = np.zeros(3, dtype=float)
        if self._tan_hold_s_left > 0.0:
            v_tan = self._pending_v_tan.copy()
            self._tan_hold_s_left = max(0.0, self._tan_hold_s_left - float(dt))
            if self._tan_hold_s_left <= 0.0:
                self._pending_v_tan[:] = 0.0
        v = cmd.as_linear() + v_tan
        w_rad = np.deg2rad(cmd.as_angular_deg())
        if fault or wrong:
            n = _unit(R_star[:, 2])
            v_n = float(np.dot(v, n))
            v_perp = v - n * v_n
            # Retract-only along n* (Fd≥0 → v∥ ≤ 0); preserve field tangent.
            v = v_perp + n * min(0.0, v_n)
            w_rad[:] = 0.0

        vn = float(np.linalg.norm(v))
        if vn > self.cfg.max_linear_m_s and vn > 1e-12:
            v = v * (self.cfg.max_linear_m_s / vn)
        wn = float(np.linalg.norm(np.rad2deg(w_rad)))
        if wn > self.cfg.max_angular_deg_s and wn > 1e-12:
            w_rad = w_rad * (self.cfg.max_angular_deg_s / wn)

        qdot = twist_to_qdot(
            v,
            w_rad,
            jacobian,
            damping=self.cfg.damping,
            max_joint_velocity=self.cfg.max_joint_velocity,
        )
        self.last_diag = {
            **{
                str(k): float(val) if isinstance(val, (int, float)) else val
                for k, val in diag.items()
            },
            "n_freeze": 1.0 if self.n_frozen else 0.0,
            "v_tan_norm": float(np.linalg.norm(v_tan)),
            "Fd_N": float(self.cfg.desired_force_n),
            "force_sign": float(self.cfg.force_sign),
            "k_ori": float(self.dual.config.align.k_ori),
            "force_fault": 1.0 if fault else 0.0,
            "force_wrong_sign": 1.0 if wrong else 0.0,
            "phase_align": 1.0 if aligning else 0.0,
            "ang_z_deg": float(ang_z_deg),
            "force_enable": 1.0 if self.dual.config.force.enable else 0.0,
            "phase_changed": 1.0 if prev_phase != self.phase else 0.0,
        }
        return qdot


def vf_R_field_or_none(vf) -> np.ndarray | None:
    """Teleop / twist frame (``R_motion``) — not the live surface approach."""
    if vf is None:
        return None
    if hasattr(vf, "motion_basis"):
        return np.asarray(vf.motion_basis(), dtype=float).reshape(3, 3)
    status = getattr(vf, "status", None)
    if isinstance(status, Mapping) and "R_field_base" in status:
        return np.asarray(status["R_field_base"], dtype=float).reshape(3, 3)
    return None


def vf_approach_R(vf) -> np.ndarray | None:
    """Surface approach frame with column 2 = into-surface normal.

    Prefer ``R_field_raw_smoothed`` (WoS/EMA field). Do **not** use teleop
    ``R_motion``: that frame tracks the EE and makes align-first see ang_z≈0
    even when the tool is tilted relative to the object.
    """
    if vf is None:
        return None
    status = getattr(vf, "status", None)
    if isinstance(status, Mapping):
        raw = status.get("R_field_raw_smoothed")
        if raw is not None:
            return np.asarray(raw, dtype=float).reshape(3, 3)
    last = getattr(vf, "_last_R_field_base", None)
    if last is not None:
        return np.asarray(last, dtype=float).reshape(3, 3)
    return vf_R_field_or_none(vf)
