"""In-contact teleop bridge: DualLoop (speedJ) or Eq. 56 (DTC) + tangent + VF n* freeze.

Dedicated ``--in-contact`` path (not free-space pose-ref teleop).

``actuation``:
  - ``speedj`` — DualLoop tip twist → q̇ → RTDE ``speedJ`` (legacy).
  - ``dtc`` — pygafro line tracking + Eq. 56 → τ → ``directTorque`` (research path).

Align phase (speedj): ω only (tip holds). Keyboard tangent is one-shot hold;
SpaceMouse uses ``continuous=True`` (cleared on deadzone / age-out).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import time

import numpy as np

from geodude_teleoperation.teleop.controller import qdot_from_error
from geodude_teleoperation.in_contact_control.dual_loop import (
    DualLoopConfig,
    DualLoopController,
)
from geodude_teleoperation.in_contact_control.types import PoseSO3

_YELLOW = "\033[93m"
_RESET = "\033[0m"


def print_actuation_banner(actuation: str, *, desired_force_n: float | None = None) -> None:
    """Yellow terminal warning so speedJ vs DTC cannot be ambiguous."""
    mode = str(actuation).strip().lower()
    if mode == "dtc":
        msg = (
            "WARNING: in-contact ACTUATION=DTC (Eq.56 / pygafro line motors → "
            "direct_torque). Not DualLoop speedJ."
        )
    else:
        msg = (
            "WARNING: in-contact ACTUATION=speedJ (DualLoop tip twist → q̇). "
            "Not UR direct_torque / Eq.56."
        )
    if desired_force_n is not None:
        msg += f" Fd={float(desired_force_n):g} N"
    print(f"{_YELLOW}{msg}{_RESET}", flush=True)


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
    use_cga_motor: bool = True,
) -> np.ndarray:
    """Map field/local-frame linear ``[lx, ly, lz]`` to world tangent.

    ``R_local`` is the teleop / VF motion frame (same as free-space WASD:
    columns 0/1 = local x/y). Motion along ``n_approach`` (default ``R_local``
    column 2) is removed so Force owns the normal.

    When ``use_cga_motor`` (default), local axes are sandwich-mapped with a
    pygafro ``Rotor`` (true CGA motor generator for orientations). Falls back to
    ``R @ v`` only if pygafro is unavailable.
    """
    lin = np.asarray(lin_field, dtype=float).reshape(3)
    R = np.asarray(R_local, dtype=float).reshape(3, 3)
    n = _unit(R[:, 2] if n_approach is None else np.asarray(n_approach, dtype=float))
    # Local xy only (lz ignored for contact tangent); build local vector then sandwich.
    v_local = np.array([float(lin[0]), float(lin[1]), 0.0], dtype=float)
    if use_cga_motor:
        try:
            from geodude_teleoperation.cga.motors import sandwich_direction

            v = float(scale) * sandwich_direction(R, v_local)
        except Exception:
            v = float(scale) * (v_local[0] * R[:, 0] + v_local[1] * R[:, 1])
    else:
        v = float(scale) * (v_local[0] * R[:, 0] + v_local[1] * R[:, 1])
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
    # ``speedj`` = DualLoop→q̇; ``dtc`` = Eq.56→τ (pygafro).
    actuation: str = "speedj"
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
    # Force PID (speedj: v∥ along n*; dtc: wrench PID gains below).
    force_kp: float = 0.001
    force_ki: float = 0.0
    force_kd: float = 0.0
    force_deadband_N: float = 0.35
    force_lpf_hz: float = 5.0
    force_contact_threshold_N: float = 0.6
    # DTC wrench PID (N / N·m command per N error) — separate from speedj force_kp.
    wrench_kp: float = 0.25
    wrench_ki: float = 0.0
    wrench_kd: float = 0.0
    tangent_force_gain: float = 50.0
    max_tau_nm: float = 20.0
    max_wrench_n: float = 12.0
    # Soft approach caps (m/s) — keep far below free-space teleop speeds.
    approach_v_max_m_s: float = 0.006
    max_v_along_m_s: float = 0.010
    # DTC hybrid: DualLoop speedJ until *real* compression along n*, then Eq.56.
    # signed_d alone is unreliable (Polyscope / tip mismatch → air-gap DTC freeze).
    dtc_contact_dist_m: float = 0.008  # optional proximity confirm (looser than load)
    dtc_require_force: bool = True  # require Fs ≥ force_contact_threshold_N
    # Freeze VF n* on Enter (press f to unfreeze). Stops live-n* lateral wander.
    freeze_n_on_arm: bool = True
    # Skip F/T bias if already loaded (avoids zeroing real contact force).
    bias_skip_load_N: float = 2.0
    # Align tool-Z → n* before Force seeks (prevents pressing at an angle).
    align_first: bool = True
    align_enter_tol_deg: float = 5.0  # tool-Z within this of n* → enable Force
    align_exit_tol_deg: float = 12.0  # if misalignment grows past this → pause Force
    # Align spring–damper (geometric admittance → tip twist, not τ).
    align_k_perp: float = 100.0
    align_k_ori: float = 5.0
    align_m_lin: float = 1.0
    align_m_ang: float = 0.05
    align_b_lin: float = 45.0
    align_b_ang: float = 2.5
    # Eq. 56 diagonals (None → research defaults).
    K_line: Sequence[float] | None = None
    D_twist: Sequence[float] | None = None


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

    # Prefer in_contact.desired_force; fall back to tactile.desired_normal_force.
    fd = getattr(args, "contact_desired_force", None)
    if fd is None:
        fd = getattr(args, "desired_normal_force", 5.0)
    try:
        fd_f = float(fd)
    except (TypeError, ValueError):
        fd_f = 5.0

    act = str(getattr(args, "contact_actuation", "speedj") or "speedj").strip().lower()
    if act not in ("speedj", "dtc"):
        act = "speedj"

    def _seq6(name: str) -> Sequence[float] | None:
        v = getattr(args, name, None)
        if v is None:
            return None
        try:
            arr = [float(x) for x in v]
        except (TypeError, ValueError):
            return None
        return arr if len(arr) == 6 else None

    return ContactTeleopConfig(
        desired_force_n=fd_f,
        actuation=act,
        tangent_scale=_float("tangent_scale", 8.0),
        tangent_hold_s=_float("tangent_hold_s", 0.12),
        max_linear_m_s=_float("contact_max_linear_m_s", 0.04),
        max_angular_deg_s=_float("contact_max_angular_deg_s", 12.0),
        damping=_float("contact_damping", 0.05),
        max_joint_velocity=_float("max_joint_velocity", 1.0),
        wrench_in_tool_frame=_bool("wrench_in_tool_frame", True),
        force_sign=_float("force_sign", 1.0),
        force_kp=_float("force_kp", 0.001),
        force_ki=_float("force_ki", 0.0),
        force_kd=_float("force_kd", 0.0),
        force_deadband_N=_float("force_deadband_N", 0.35),
        force_lpf_hz=_float("force_lpf_hz", 5.0),
        force_contact_threshold_N=_float("force_contact_threshold_N", 0.6),
        wrench_kp=_float("contact_wrench_kp", 0.25),
        wrench_ki=_float("contact_wrench_ki", 0.0),
        wrench_kd=_float("contact_wrench_kd", 0.0),
        tangent_force_gain=_float("tangent_force_gain", 50.0),
        max_tau_nm=_float("max_tau_nm", 20.0),
        max_wrench_n=_float("max_wrench_n", 12.0),
        approach_v_max_m_s=_float("approach_v_max", 0.006),
        max_v_along_m_s=_float("force_max_v", 0.010),
        dtc_contact_dist_m=_float("dtc_contact_dist_m", 0.008),
        dtc_require_force=_bool("dtc_require_force", True),
        freeze_n_on_arm=_bool("freeze_n_on_arm", True),
        bias_skip_load_N=_float("bias_skip_load_N", 2.0),
        align_first=_bool("align_first", True),
        align_enter_tol_deg=_float("align_enter_tol_deg", 5.0),
        align_exit_tol_deg=_float("align_exit_tol_deg", 12.0),
        align_k_perp=_float("align_k_perp", 100.0),
        align_k_ori=_float("align_k_ori", 5.0),
        align_m_lin=_float("align_m_lin", 1.0),
        align_m_ang=_float("align_m_ang", 0.05),
        align_b_lin=_float("align_b_lin", 45.0),
        align_b_ang=_float("align_b_ang", 2.5),
        K_line=_seq6("contact_K_line"),
        D_twist=_seq6("contact_D_twist"),
    )


class ContactTeleopBridge:
    """Tip-tracked DualLoop (``speedj``) or Eq. 56 DTC (``dtc``) + tangent FF.

    ``actuation=speedj``: tip twist → ``q̇`` for RTDE ``speedJ``.
    ``actuation=dtc``: pygafro line motors + Eq. 56 → joint τ for ``directTorque``.

    With ``align_first`` (default): after Enter, Align runs until tool-Z is within
    ``align_enter_tol_deg`` of ``n*``; Force seek/hold stays off until then.
    """

    PHASE_ALIGN = "align"
    PHASE_SEEK = "seek"

    def __init__(self, cfg: ContactTeleopConfig | None = None):
        self.cfg = cfg or ContactTeleopConfig()
        self.actuation = str(self.cfg.actuation).strip().lower()
        if self.actuation not in ("speedj", "dtc"):
            self.actuation = "speedj"

        dual_cfg = DualLoopConfig()
        dual_cfg.force.Fd_N = float(self.cfg.desired_force_n)
        dual_cfg.force.wrench_in_tool_frame = bool(self.cfg.wrench_in_tool_frame)
        dual_cfg.force.force_sign = float(self.cfg.force_sign)
        dual_cfg.force.kp = float(self.cfg.force_kp)
        dual_cfg.force.ki = float(self.cfg.force_ki)
        dual_cfg.force.kd = float(self.cfg.force_kd)
        dual_cfg.force.force_deadband_N = float(self.cfg.force_deadband_N)
        dual_cfg.force.force_lpf_hz = float(self.cfg.force_lpf_hz)
        dual_cfg.force.contact_threshold_N = float(self.cfg.force_contact_threshold_N)
        dual_cfg.force.approach_v_max_m_s = float(self.cfg.approach_v_max_m_s)
        dual_cfg.force.max_v_along_m_s = float(self.cfg.max_v_along_m_s)
        # High Fd needs gentler gain or the loop chatters (see contact_motion bounce).
        if float(self.cfg.desired_force_n) >= 8.0:
            dual_cfg.force.kp = min(float(dual_cfg.force.kp), 0.0005)
        dual_cfg.align.k_perp = float(self.cfg.align_k_perp)
        dual_cfg.align.k_ori = float(self.cfg.align_k_ori)
        dual_cfg.align.m_lin = float(self.cfg.align_m_lin)
        dual_cfg.align.m_ang = float(self.cfg.align_m_ang)
        dual_cfg.align.b_lin = float(self.cfg.align_b_lin)
        dual_cfg.align.b_ang = float(self.cfg.align_b_ang)
        dual_cfg.max_linear_m_s = float(self.cfg.max_linear_m_s)
        dual_cfg.max_angular_deg_s = float(self.cfg.max_angular_deg_s)
        self.dual = DualLoopController(dual_cfg)
        self._k_ori_full = float(self.dual.config.align.k_ori)

        self._eq56 = None
        if self.actuation == "dtc":
            from geodude_teleoperation.cga.pygafro_backend import require_pygafro
            from geodude_teleoperation.cga.tactile_law import (
                Eq56Controller,
                Eq56Gains,
                default_eq56_gains,
            )

            require_pygafro()
            gains = default_eq56_gains()
            if self.cfg.K_line is not None:
                gains = Eq56Gains(
                    K_line=np.asarray(self.cfg.K_line, dtype=float).reshape(6),
                    D_twist=gains.D_twist,
                )
            if self.cfg.D_twist is not None:
                gains = Eq56Gains(
                    K_line=gains.K_line,
                    D_twist=np.asarray(self.cfg.D_twist, dtype=float).reshape(6),
                )
            # Soft approach ≥ |Fd| so free-space seek is not weaker than force PID.
            approach_f = max(
                float(self.cfg.desired_force_n),
                float(self.cfg.approach_v_max_m_s) * 1000.0,  # 6mm/s → 6 N floor
                8.0,
            )
            self._eq56 = Eq56Controller(
                gains=gains,
                desired_force_n=float(self.cfg.desired_force_n),
                force_sign=float(self.cfg.force_sign),
                wrench_in_tool_frame=bool(self.cfg.wrench_in_tool_frame),
                wrench_kp=float(self.cfg.wrench_kp),
                wrench_ki=float(self.cfg.wrench_ki),
                wrench_kd=float(self.cfg.wrench_kd),
                tangent_force_gain=float(self.cfg.tangent_force_gain),
                max_tau=float(self.cfg.max_tau_nm),
                contact_threshold_n=float(self.cfg.force_contact_threshold_N),
                approach_force_n=approach_f,
                max_wrench_n=float(self.cfg.max_wrench_n),
            )

        self.n_frozen = False
        self._R_star: np.ndarray | None = None
        self._pending_v_tan = np.zeros(3, dtype=float)
        self._tan_hold_s_left = 0.0
        self._tan_continuous = False
        self.last_tangent_field = np.zeros(3, dtype=float)
        self.armed = False
        self.phase = self.PHASE_ALIGN if self.cfg.align_first else self.PHASE_SEEK
        self.last_diag: dict = {}
        self.last_tau = np.zeros(6, dtype=float)
        self.armed_mono_t: float | None = None
        self._k_perp_full = float(self.dual.config.align.k_perp)
        # DTC: subtract Enter-time F/T (DualLoop bias is 3-vector only).
        self._wrench_bias6 = np.zeros(6, dtype=float)

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
        wrench_bias_samples: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """Enter / re-arm: seed tip target and optionally set F/T bias."""
        R_star = self._refresh_R_star(R_field, force=True)
        p = np.asarray(p_tcp, dtype=float).reshape(3)
        self.dual.capture(PoseSO3(p=p, R=R_star))
        self._wrench_bias6[:] = 0.0
        if force_bias_samples:
            arr = np.asarray(force_bias_samples, dtype=float).reshape(-1, 3)
            load = float(np.linalg.norm(arr.mean(axis=0)))
            if load >= float(self.cfg.bias_skip_load_N):
                # Already loaded — biasing would zero real contact and over-push.
                self.dual.force._f_bias[:] = 0.0
            else:
                self.dual.set_force_bias_from_samples(force_bias_samples)
                # Mirror 3-force bias into DTC wrench (moments stay 0 unless given).
                self._wrench_bias6[:3] = arr.mean(axis=0)
        if wrench_bias_samples:
            warr = np.asarray(wrench_bias_samples, dtype=float).reshape(-1, 6)
            wmean = warr.mean(axis=0)
            load6 = float(np.linalg.norm(wmean[:3]))
            if load6 < float(self.cfg.bias_skip_load_N):
                self._wrench_bias6 = wmean.copy()
                if force_bias_samples is None:
                    self.dual.set_force_bias_from_samples([wmean[:3]])
            elif force_bias_samples is None:
                self.dual.force._f_bias[:] = 0.0
                self._wrench_bias6[:] = 0.0
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
        self.dual.config.align.k_perp = self._k_perp_full
        if self._eq56 is not None:
            self._eq56.reset()
        self.armed = True
        self.armed_mono_t = time.monotonic()
        self._pending_v_tan[:] = 0.0
        self._tan_hold_s_left = 0.0
        self._tan_continuous = False
        self._logged_seek_speedj = False
        self._logged_seek_dtc = False
        print_actuation_banner(self.actuation, desired_force_n=self.cfg.desired_force_n)

    def clear_tangent(self) -> None:
        """Drop pending keyboard/SpaceMouse tangent (e.g. SpaceMouse deadzone)."""
        self._pending_v_tan[:] = 0.0
        self._tan_hold_s_left = 0.0
        self._tan_continuous = False
        self.last_tangent_field[:] = 0.0

    def push_tangent_field_linear(
        self,
        lin_field: Sequence[float],
        R_local: np.ndarray,
        *,
        n_approach: np.ndarray | None = None,
        continuous: bool = False,
    ) -> None:
        """Queue world tangent velocity from **local/teleop** frame WASD/SM.

        ``R_local`` must be the VF motion / field frame (``motion_basis``), not
        ``R_tcp``. ``n_approach`` kills the DualLoop normal (defaults to frozen
        ``R*`` column 2 when armed).

        ``continuous=False`` (keyboard): one-shot hold for ``tangent_hold_s``.
        ``continuous=True`` (SpaceMouse): keep rate until ``clear_tangent`` or
        another push.
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
        self._tan_continuous = bool(continuous)
        if continuous:
            self._tan_hold_s_left = 0.0
        else:
            self._tan_hold_s_left = float(self.cfg.tangent_hold_s)

    def _consume_tangent(self, dt: float, *, aligning: bool) -> np.ndarray:
        v_tan = np.zeros(3, dtype=float)
        if aligning:
            return v_tan
        if self._tan_continuous:
            return self._pending_v_tan.copy()
        if self._tan_hold_s_left > 0.0:
            v_tan = self._pending_v_tan.copy()
            self._tan_hold_s_left = max(0.0, self._tan_hold_s_left - float(dt))
            if self._tan_hold_s_left <= 0.0:
                self._pending_v_tan[:] = 0.0
        return v_tan

    def _estimate_fs_along_n(
        self,
        force_base: Sequence[float],
        R_tcp: np.ndarray,
        R_star: np.ndarray,
    ) -> float:
        """Bias-subtracted Fs = force_sign · F·n* (DualLoop convention)."""
        f = np.asarray(force_base, dtype=float).reshape(3) - self._wrench_bias6[:3]
        if self.cfg.wrench_in_tool_frame:
            R = np.asarray(R_tcp, dtype=float).reshape(3, 3)
            f = R @ f
        n = _unit(np.asarray(R_star, dtype=float).reshape(3, 3)[:, 2])
        return float(np.dot(f, n)) * float(self.cfg.force_sign)

    def _dtc_contact_ready(
        self,
        *,
        signed_d: float | None,
        fs_n: float,
    ) -> bool:
        """True when seek may hand off to Eq.56 DTC.

        Geometric ``signed_d`` alone is not enough (VF/Polyscope tip mismatch leaves
        an air gap while ``d`` is already small). Require compression along n* when
        ``dtc_require_force`` (default).
        """
        d_gate = float(self.cfg.dtc_contact_dist_m)
        near_geom = True
        if signed_d is not None and np.isfinite(float(signed_d)):
            # Allow a bit of mismatch slack beyond the nominal gate.
            near_geom = float(signed_d) <= max(d_gate, 0.015)
        if bool(self.cfg.dtc_require_force):
            loaded = float(fs_n) >= float(self.cfg.force_contact_threshold_N)
            return bool(loaded and near_geom)
        return bool(
            signed_d is not None
            and np.isfinite(float(signed_d))
            and float(signed_d) <= d_gate
        )

    def step(
        self,
        *,
        p_tcp: Sequence[float],
        R_tcp: np.ndarray,
        R_field: np.ndarray,
        force_base: Sequence[float],
        jacobian: np.ndarray,
        dt: float,
        wrench6: Sequence[float] | None = None,
        qdot_meas: Sequence[float] | None = None,
        signed_d: float | None = None,
    ) -> np.ndarray:
        """Run contact law; return ``q̇`` (speedj) or zeros when ``actuation=dtc`` loaded.

        For loaded-surface DTC, call :meth:`step_tau` (or this method then read ``last_tau``).
        """
        if self.actuation == "dtc":
            # Hybrid (hardware bring-up evidence):
            #   ALIGN            → DualLoop speedJ (ω-only)
            #   SEEK unloaded    → DualLoop speedJ (approach + teleop)
            #   SEEK compressed  → Eq.56 directTorque (Fs ≥ contact threshold)
            if not self.armed:
                return np.zeros(jacobian.shape[1], dtype=float)
            R_star = self._refresh_R_star(R_field)
            ang_z_deg = self._tool_z_error_deg(R_tcp, R_star)
            prev_phase = self.phase
            self._update_phase(ang_z_deg)
            if self.phase == self.PHASE_ALIGN:
                if prev_phase != self.PHASE_ALIGN:
                    print(
                        "\033[93mWARNING: DTC path — ALIGN via speedJ (ω-only); "
                        "SEEK via speedJ until Fs shows compression, "
                        "then Eq.56 directTorque.\033[0m",
                        flush=True,
                    )
                self.last_tau = np.zeros(jacobian.shape[1], dtype=float)
                # Fall through to DualLoop velocity align (force off).
            else:
                fs_n = self._estimate_fs_along_n(force_base, R_tcp, R_star)
                ready = self._dtc_contact_ready(signed_d=signed_d, fs_n=fs_n)
                if not ready:
                    if prev_phase == self.PHASE_ALIGN or not getattr(
                        self, "_logged_seek_speedj", False
                    ):
                        self._logged_seek_speedj = True
                        d_s = (
                            f"{float(signed_d):.4f}"
                            if signed_d is not None and np.isfinite(float(signed_d))
                            else "nan"
                        )
                        print(
                            "\033[93mWARNING: DTC path — SEEK approach via DualLoop "
                            f"speedJ (d={d_s}, Fs={fs_n:.2f} N < "
                            f"{self.cfg.force_contact_threshold_N:g} N); "
                            "teleop live; Eq.56 after compression.\033[0m",
                            flush=True,
                        )
                    # Left DTC mid-seek (air gap / lost contact) — allow re-announce.
                    if getattr(self, "_logged_seek_dtc", False):
                        self._logged_seek_dtc = False
                        print(
                            "\033[93mWARNING: DTC path — back to DualLoop speedJ "
                            f"(lost compression Fs={fs_n:.2f} N).\033[0m",
                            flush=True,
                        )
                    self.last_tau = np.zeros(jacobian.shape[1], dtype=float)
                    # Fall through to DualLoop seek (force on + WASD/SM).
                else:
                    if not getattr(self, "_logged_seek_dtc", False):
                        self._logged_seek_dtc = True
                        d_s = (
                            f"{float(signed_d):.4f}"
                            if signed_d is not None and np.isfinite(float(signed_d))
                            else "nan"
                        )
                        print(
                            "\033[93mWARNING: DTC path — SEEK compressed → "
                            f"Eq.56 directTorque (Fs={fs_n:.2f} N, d={d_s} m, "
                            f"ang_z={ang_z_deg:.1f}deg).\033[0m",
                            flush=True,
                        )
                    self.step_tau(
                        p_tcp=p_tcp,
                        R_tcp=R_tcp,
                        R_field=R_field,
                        wrench6=wrench6
                        if wrench6 is not None
                        else np.concatenate(
                            [
                                np.asarray(force_base, dtype=float).reshape(3),
                                np.zeros(3),
                            ]
                        ),
                        jacobian=jacobian,
                        dt=dt,
                        qdot_meas=qdot_meas,
                    )
                    if self.last_diag is not None:
                        self.last_diag["dtc_use_speedj"] = 0.0
                        self.last_diag["actuation"] = "dtc"
                        self.last_diag["Fs_N"] = float(fs_n)
                    return np.zeros(jacobian.shape[1], dtype=float)

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
        aligning = self.phase == self.PHASE_ALIGN
        # Align-only: force off + zero lateral stiffness (in-place ω, no tip chase).
        if aligning:
            self.dual.config.force.enable = False
            self.dual.config.align.k_perp = 0.0
        else:
            self.dual.config.align.k_perp = self._k_perp_full
        cmd, diag = self.dual.step(measured, force_base=force_base, dt=dt)

        # Overload / wrong-sign: never push into the surface, but keep WASD
        # tangent (old path zeroed the full twist → dead keys while |Fs| high).
        fault = float(diag.get("force_fault_fs", 0.0)) > 0.5
        wrong = float(diag.get("force_wrong_sign", 0.0)) > 0.5

        v_tan = self._consume_tangent(dt, aligning=aligning)
        if aligning:
            # Rotate in place: DualLoop Align ω only (no tip slide / no FF).
            v = np.zeros(3, dtype=float)
            w_rad = np.deg2rad(cmd.as_angular_deg())
            align_w_max = min(float(self.cfg.max_angular_deg_s), 8.0)
            wn = float(np.linalg.norm(np.rad2deg(w_rad)))
            if wn > align_w_max and wn > 1e-12:
                w_rad = w_rad * (align_w_max / wn)
        else:
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

        # Align: keep ‖q̇‖ modest (free-space Enter-lock used ~0.03 rad/s).
        qdot_cap = float(self.cfg.max_joint_velocity)
        if aligning:
            qdot_cap = min(qdot_cap, 0.08)

        qdot = twist_to_qdot(
            v,
            w_rad,
            jacobian,
            damping=self.cfg.damping,
            max_joint_velocity=qdot_cap,
        )
        act_tag = self.actuation
        dtc_speedj = 0.0
        if self.actuation == "dtc":
            dtc_speedj = 1.0
            act_tag = "dtc_align_speedj" if aligning else "dtc_seek_speedj"
        self.last_diag = {
            **{
                str(k): float(val) if isinstance(val, (int, float)) else val
                for k, val in diag.items()
            },
            "actuation": act_tag,
            "dtc_use_speedj": float(dtc_speedj),
            "n_freeze": 1.0 if self.n_frozen else 0.0,
            "v_tan_norm": float(np.linalg.norm(v_tan)),
            "v_cmd_norm": float(np.linalg.norm(v)),
            "w_cmd_norm": float(np.linalg.norm(w_rad)),
            "Fd_N": float(self.cfg.desired_force_n),
            "force_sign": float(self.cfg.force_sign),
            "k_ori": float(self.dual.config.align.k_ori),
            "force_fault": 1.0 if fault else 0.0,
            "force_wrong_sign": 1.0 if wrong else 0.0,
            "phase_align": 1.0 if aligning else 0.0,
            "ang_z_deg": float(ang_z_deg),
            "force_enable": 1.0 if self.dual.config.force.enable else 0.0,
            "phase_changed": 1.0 if prev_phase != self.phase else 0.0,
            "tau_norm": float(np.linalg.norm(self.last_tau)),
            "qdot_norm": float(np.linalg.norm(qdot)),
        }
        return qdot

    def step_tau(
        self,
        *,
        p_tcp: Sequence[float],
        R_tcp: np.ndarray,
        R_field: np.ndarray,
        wrench6: Sequence[float],
        jacobian: np.ndarray,
        dt: float,
        qdot_meas: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Eq. 56 joint torque command (DTC). Requires ``actuation=dtc``."""
        n_j = int(jacobian.shape[1])
        if not self.armed or self._eq56 is None:
            self.last_tau = np.zeros(n_j, dtype=float)
            return self.last_tau

        R_star = self._refresh_R_star(R_field)
        p = np.asarray(p_tcp, dtype=float).reshape(3)
        # Tip-track contact point x_q on the target line (line tracking, not point hold).
        x_q = p.copy()
        ang_z_deg = self._tool_z_error_deg(R_tcp, R_star)
        prev_phase = self.phase
        self._update_phase(ang_z_deg)
        aligning = self.phase == self.PHASE_ALIGN
        enable_force = (not aligning) and (self.phase == self.PHASE_SEEK)

        v_tan = self._consume_tangent(dt, aligning=aligning)
        qd = (
            np.zeros(n_j, dtype=float)
            if qdot_meas is None
            else np.asarray(qdot_meas, dtype=float).reshape(n_j)
        )
        # Bias-subtract like DualLoop force path (raw TCP force includes gravity).
        W_raw = np.asarray(wrench6, dtype=float).reshape(6)
        W_meas = W_raw - self._wrench_bias6
        tau = self._eq56.step(
            p_tcp=p,
            R_tcp=np.asarray(R_tcp, dtype=float).reshape(3, 3),
            x_q=x_q,
            R_field=R_star,
            wrench_meas6=W_meas,
            jacobian_ee=jacobian,
            qdot=qd,
            v_tan_world=None if aligning else v_tan,
            dt=dt,
            enable_force=enable_force,
        )
        self.last_tau = np.asarray(tau, dtype=float).reshape(n_j)
        self.last_diag = {
            **dict(self._eq56.last_diag),
            "actuation": "dtc",
            "n_freeze": 1.0 if self.n_frozen else 0.0,
            "v_tan_norm": float(np.linalg.norm(v_tan)),
            "Fd_N": float(self.cfg.desired_force_n),
            "force_sign": float(self.cfg.force_sign),
            "phase_align": 1.0 if aligning else 0.0,
            "ang_z_deg": float(ang_z_deg),
            "force_enable": 1.0 if enable_force else 0.0,
            "phase_changed": 1.0 if prev_phase != self.phase else 0.0,
            "tau_norm": float(np.linalg.norm(self.last_tau)),
        }
        return self.last_tau


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
