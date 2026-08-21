"""In-contact teleop bridge: DualLoop (speedJ) or Eq. 56 (DTC) + tangent + VF n* freeze.

Dedicated ``--in-contact`` path (not free-space pose-ref teleop).

``actuation``:
  - ``speedj`` — DualLoop tip twist → q̇ → RTDE ``speedJ`` (legacy).
  - ``dtc`` — hybrid: DualLoop speedJ for ALIGN + unloaded SEEK; Eq. 56 after
    compression (hardware bring-up default).
  - ``dtc_pure`` — Eq. 56 / pygafro → τ → ``directTorque`` every tick.
    ALIGN: ``W_L`` only (tool-Z → n*). SEEK: ``W_L``+``W_C``.
    ``control.max_joint_velocity`` / ``rtde_accel`` are speedJ-only; DTC uses
    ``dtc_startup_*`` and ``dtc_max_dtau_nm_s`` (τ slew).

Align phase: ω / orientation only (tip holds). Keyboard tangent is
one-shot hold; SpaceMouse uses ``continuous=True`` (cleared on deadzone / age-out).
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


def _print_dtc_dbg(msg: str) -> None:
    """Low-rate operator debug (bright/light yellow)."""
    print(f"{_YELLOW}[dtc-dbg] {msg}{_RESET}", flush=True)


def _fmt3(v: np.ndarray) -> str:
    a = np.asarray(v, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{float(x):+.3f}" for x in a) + "]"


def _rpy_deg(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(np.asarray(R, dtype=float).reshape(3, 3)).as_euler(
        "xyz", degrees=True
    )


def print_pendant_tcp_dbg(
    p_tcp: Sequence[float],
    p_flange: Sequence[float],
    tip_offset_yaml: Sequence[float] | None = None,
) -> None:
    """Confirm CGA / Eq. 56 use RTDE pendant TCP (tool offset included)."""
    p = np.asarray(p_tcp, dtype=float).reshape(3)
    fl = np.asarray(p_flange, dtype=float).reshape(3)
    off = p - fl
    yaml_off = (
        np.zeros(3)
        if tip_offset_yaml is None
        else np.asarray(tip_offset_yaml, dtype=float).reshape(3)
    )
    _print_dtc_dbg(
        "pendant TCP  "
        f"rtde={_fmt3(p)}  "
        f"flange={_fmt3(fl)}  "
        f"flange→tcp={_fmt3(off)}  "
        f"|offset|={float(np.linalg.norm(off)) * 1000.0:.1f}mm  "
        f"yaml_tip={_fmt3(yaml_off)}  "
        "CGA line + J at getActualTCPPose (pendant tool), not flange"
    )

_ACTUATION_ALIASES = {
    "speedj": "speedj",
    "speed_j": "speedj",
    "dtc": "dtc",
    "dtc_hybrid": "dtc",
    "hybrid": "dtc",
    "dtc_pure": "dtc_pure",
    "pure": "dtc_pure",
    "pure_dtc": "dtc_pure",
    "dtc_full": "dtc_pure",
}


def normalize_contact_actuation(actuation: str | None) -> str:
    """Map YAML/CLI aliases onto ``speedj`` | ``dtc`` | ``dtc_pure``."""
    raw = str(actuation or "speedj").strip().lower().replace("-", "_")
    return _ACTUATION_ALIASES.get(raw, "speedj")


def is_dtc_actuation(actuation: str | None) -> bool:
    return normalize_contact_actuation(actuation) in ("dtc", "dtc_pure")


def is_dtc_pure_actuation(actuation: str | None) -> bool:
    return normalize_contact_actuation(actuation) == "dtc_pure"


def print_actuation_banner(actuation: str, *, desired_force_n: float | None = None) -> None:
    """Yellow terminal warning so speedJ vs DTC cannot be ambiguous."""
    mode = normalize_contact_actuation(actuation)
    if mode == "dtc_pure":
        msg = (
            "WARNING: in-contact ACTUATION=DTC_PURE (Eq.56 → direct_torque). "
            "ALIGN=W_L only; SEEK=W_L+W_C. No DualLoop speedJ "
            "(control.* vel/accel unused — see dtc_max_dtau_nm_s)."
        )
    elif mode == "dtc":
        msg = (
            "WARNING: in-contact ACTUATION=DTC hybrid (DualLoop speedJ for "
            "ALIGN + unloaded SEEK; Eq.56 after compression). Not DTC_PURE."
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
) -> tuple[np.ndarray, bool]:
    """Map field/local-frame linear ``[lx, ly, lz]`` to world tangent.

    ``R_local`` is the teleop / VF motion frame (same as free-space WASD:
    columns 0/1 = local x/y). Motion along ``n_approach`` (default ``R_local``
    column 2) is removed so Force owns the normal.

    When ``use_cga_motor`` (default), local axes are sandwich-mapped with a
    pygafro ``Rotor`` (true CGA). Returns ``(v_world, used_cga)``.
    """
    lin = np.asarray(lin_field, dtype=float).reshape(3)
    R = np.asarray(R_local, dtype=float).reshape(3, 3)
    n = _unit(R[:, 2] if n_approach is None else np.asarray(n_approach, dtype=float))
    v_local = np.array([float(lin[0]), float(lin[1]), 0.0], dtype=float)
    used_cga = False
    if use_cga_motor:
        try:
            from geodude_teleoperation.cga.motors import sandwich_direction

            v = float(scale) * sandwich_direction(R, v_local)
            used_cga = True
        except Exception:
            v = float(scale) * (v_local[0] * R[:, 0] + v_local[1] * R[:, 1])
    else:
        v = float(scale) * (v_local[0] * R[:, 0] + v_local[1] * R[:, 1])
    v = v - n * float(np.dot(v, n))
    return v, used_cga


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
    # ``speedj`` | ``dtc`` (hybrid) | ``dtc_pure`` (Eq.56 from ALIGN through SEEK).
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
    # Air SEEK: spring f = max(Fd, k·signed_d) along n* (N/m). 0 → Fd only.
    approach_stiffness_n_per_m: float = 400.0
    # DTC hybrid: DualLoop speedJ until *real* compression along n*, then Eq.56.
    # signed_d alone is unreliable (Polyscope / tip mismatch → air-gap DTC freeze).
    dtc_contact_dist_m: float = 0.008  # optional proximity confirm (looser than load)
    dtc_require_force: bool = True  # require Fs ≥ force_contact_threshold_N
    # Hybrid: also require Fs ≥ this fraction of |Fd| before DTC (else DualLoop
    # builds load + teleop). Avoids DTC at Fs≪Fd with weak P-only W_C.
    dtc_min_fs_frac_of_fd: float = 0.25
    # Exit DTC only below this fraction of the *enter* threshold (hysteresis).
    # 1.0 = no hysteresis (bounce). ~0.35 keeps DTC through soft-ramp dips.
    dtc_exit_fs_frac_of_enter: float = 0.35
    # Once latched, ignore low-|Fs| exit for this long (mode-switch bounce).
    dtc_min_hold_s: float = 1.5
    # |Fs| must stay below exit need this long before DualLoop hand-back.
    dtc_exit_hold_s: float = 0.35
    # dtc_pure: ease W_L / W_C and ‖τ‖ from 0 after first τ tick (no field snap).
    dtc_startup_ramp_s: float = 3.0
    dtc_startup_tau_nm: float = 2.0
    # DTC analog of control.rtde_accel: max ‖Δτ‖/dt [Nm/s]. 0 = disabled.
    dtc_max_dtau_nm_s: float = 8.0
    # Lock n* when SEEK starts (press f anytime). ALIGN always follows live field.
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

    act = normalize_contact_actuation(getattr(args, "contact_actuation", "speedj"))

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
        approach_stiffness_n_per_m=_float("approach_stiffness_n_per_m", 400.0),
        dtc_contact_dist_m=_float("dtc_contact_dist_m", 0.008),
        dtc_require_force=_bool("dtc_require_force", True),
        dtc_min_fs_frac_of_fd=_float("dtc_min_fs_frac_of_fd", 0.25),
        dtc_exit_fs_frac_of_enter=_float("dtc_exit_fs_frac_of_enter", 0.35),
        dtc_min_hold_s=_float("dtc_min_hold_s", 1.5),
        dtc_exit_hold_s=_float("dtc_exit_hold_s", 0.35),
        dtc_startup_ramp_s=_float(
            "dtc_startup_ramp_s", _float("startup_ramp_s", 3.0)
        ),
        dtc_startup_tau_nm=_float("dtc_startup_tau_nm", 2.0),
        dtc_max_dtau_nm_s=_float("dtc_max_dtau_nm_s", 8.0),
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
    """Tip-tracked DualLoop (``speedj``) or Eq. 56 DTC (``dtc`` / ``dtc_pure``).

    ``actuation=speedj``: tip twist → ``q̇`` for RTDE ``speedJ``.
    ``actuation=dtc``: DualLoop speedJ until compression, then Eq. 56 τ.
    ``actuation=dtc_pure``: pygafro line motors + Eq. 56 τ from ALIGN through SEEK.

    With ``align_first`` (default): after Enter, Align runs until tool-Z is within
    ``align_enter_tol_deg`` of ``n*``; Force seek/hold stays off until then.
    """

    PHASE_ALIGN = "align"
    PHASE_SEEK = "seek"

    def __init__(self, cfg: ContactTeleopConfig | None = None):
        self.cfg = cfg or ContactTeleopConfig()
        self.actuation = normalize_contact_actuation(self.cfg.actuation)
        self.cfg.actuation = self.actuation

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
        if is_dtc_actuation(self.actuation):
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
                force_lpf_hz=float(self.cfg.force_lpf_hz),
                approach_stiffness_n_per_m=float(
                    getattr(self.cfg, "approach_stiffness_n_per_m", 400.0)
                ),
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
        # dtc_pure ‖τ‖ ramp: starts on first step_tau (not Enter — WoS can stall).
        self._dtc_ramp_mono_t: float | None = None
        self._tau_cmd_prev: np.ndarray | None = None
        self._k_perp_full = float(self.dual.config.align.k_perp)
        # DTC: subtract Enter-time F/T (DualLoop bias is 3-vector only).
        self._wrench_bias6 = np.zeros(6, dtype=float)
        self._logged_first_wrench = False
        self._logged_first_seek = False
        self._n_star_enter: np.ndarray | None = None
        self._last_live_dbg_t: float | None = None
        self._n_star_logged: np.ndarray | None = None
        self._last_cga_tan = False
        self._logged_force_sign_hint = False
        self._dtc_latched = False
        self._dtc_latch_mono_t: float | None = None
        self._dtc_exit_mono_t: float | None = None
        # DualLoop must not trip/retract below Fd while hybrid waits for DTC.
        fd_abs = abs(float(self.cfg.desired_force_n))
        self.dual.config.force.fault_abs_fs_N = max(
            float(self.dual.config.force.fault_abs_fs_N),
            fd_abs * 1.25,
        )
        # fault_fd_mult=6 made |Fs| trips impossible at Fd=70 (thresh=420 N).
        self.dual.config.force.fault_fd_mult = min(
            float(self.dual.config.force.fault_fd_mult),
            1.35,
        )

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
        prev = self.phase
        if not self.cfg.align_first:
            self.phase = self.PHASE_SEEK
            self.dual.config.force.enable = True
            self.dual.config.align.k_ori = self._k_ori_full
            self._apply_phase_field_lock(prev, ang_z_deg)
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
        self._apply_phase_field_lock(prev, ang_z_deg)

    def _apply_phase_field_lock(self, prev_phase: str, ang_z_deg: float) -> None:
        """ALIGN follows live WoS/field; SEEK may lock that live n* for force."""
        if not self.cfg.freeze_n_on_arm:
            return
        if self.phase == self.PHASE_ALIGN and prev_phase != self.PHASE_ALIGN:
            self.n_frozen = False
            _print_dtc_dbg(
                f"ALIGN live field  ang_z={ang_z_deg:.1f}deg  "
                "(n* tracks tip query; press f to lock)"
            )
        elif self.phase == self.PHASE_SEEK and prev_phase != self.PHASE_SEEK:
            self.n_frozen = True
            n = (
                _unit(self._R_star[:, 2])
                if self._R_star is not None
                else np.zeros(3)
            )
            t1 = (
                _unit(self._R_star[:, 0])
                if self._R_star is not None
                else np.zeros(3)
            )
            _print_dtc_dbg(
                "SEEK freeze n*  "
                f"n*={_fmt3(n)}  line_t1={_fmt3(t1)}  "
                f"ang_z={ang_z_deg:.1f}deg"
            )
        # Do not re-freeze every tick — that fights the user's `f` toggle.

    def _maybe_log_live_n(self, R_star: np.ndarray, ang_z_deg: float) -> None:
        """Yellow live-n* line at most every 2 s, and only if the field moved."""
        if self.n_frozen:
            return
        now = time.monotonic()
        if self._last_live_dbg_t is not None and (now - self._last_live_dbg_t) < 2.0:
            return
        n = _unit(np.asarray(R_star, dtype=float).reshape(3, 3)[:, 2])
        t1 = _unit(np.asarray(R_star, dtype=float).reshape(3, 3)[:, 0])
        n_ref = self._n_star_logged if self._n_star_logged is not None else self._n_star_enter
        drift = 0.0
        if n_ref is not None:
            drift = float(
                np.degrees(np.arccos(float(np.clip(np.dot(n, _unit(n_ref)), -1.0, 1.0))))
            )
        if self._last_live_dbg_t is not None and drift < 8.0:
            return
        self._last_live_dbg_t = now
        self._n_star_logged = n.copy()
        enter_drift = 0.0
        if self._n_star_enter is not None:
            enter_drift = float(
                np.degrees(
                    np.arccos(
                        float(np.clip(np.dot(n, _unit(self._n_star_enter)), -1.0, 1.0))
                    )
                )
            )
        _print_dtc_dbg(
            "live n*   "
            f"n*={_fmt3(n)}  line_t1={_fmt3(t1)}  "
            f"d_enter={enter_drift:.1f}deg  ang_z={ang_z_deg:.1f}deg"
        )

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
        ang0 = self._tool_z_error_deg(R_tcp, R_star)
        n_star = _unit(R_star[:, 2])
        t1 = _unit(R_star[:, 0])
        R_ee = np.asarray(R_tcp, dtype=float).reshape(3, 3)
        n_ee = _unit(R_ee[:, 2])
        self.n_frozen = False
        self._n_star_enter = n_star.copy()
        self._n_star_logged = n_star.copy()
        self._last_live_dbg_t = None
        self._logged_first_wrench = False
        self._logged_first_seek = False
        self.phase = (
            self.PHASE_SEEK
            if (not self.cfg.align_first or ang0 <= float(self.cfg.align_enter_tol_deg))
            else self.PHASE_ALIGN
        )
        if self.cfg.freeze_n_on_arm and self.phase == self.PHASE_SEEK:
            self.n_frozen = True
        _print_dtc_dbg(
            "start TCP  "
            f"p_m={_fmt3(p)}  "
            f"rpy_deg={_fmt3(_rpy_deg(R_ee))}  "
            f"tool_z={_fmt3(n_ee)}"
        )
        _print_dtc_dbg(
            "field n*   "
            f"n*={_fmt3(n_star)}  "
            f"line_t1={_fmt3(t1)}  "
            f"R*_rpy_deg={_fmt3(_rpy_deg(R_star))}  "
            f"ang_z={ang0:.1f}deg  "
            f"{'SEEK freeze' if self.n_frozen else 'ALIGN live query'}"
        )
        self.dual.config.force.enable = self.phase == self.PHASE_SEEK
        self.dual.config.align.k_ori = self._k_ori_full
        self.dual.config.align.k_perp = self._k_perp_full
        if self._eq56 is not None:
            self._eq56.reset()
        self.armed = True
        self.armed_mono_t = time.monotonic()
        self._dtc_ramp_mono_t = None  # start on first step_tau
        self._tau_cmd_prev = None
        self._pending_v_tan[:] = 0.0
        self._tan_hold_s_left = 0.0
        self._tan_continuous = False
        self._logged_seek_speedj = False
        self._logged_seek_dtc = False
        self._dtc_latched = False
        self._dtc_latch_mono_t = None
        self._dtc_exit_mono_t = None
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
        self._pending_v_tan, used_cga = field_tangent_world(
            lin,
            R_local,
            scale=float(self.cfg.tangent_scale),
            n_approach=n_approach,
        )
        self._last_cga_tan = bool(used_cga)
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

    def _dtc_fs_need_enter(self) -> float:
        thr = float(self.cfg.force_contact_threshold_N)
        fd = abs(float(self.cfg.desired_force_n))
        frac = float(self.cfg.dtc_min_fs_frac_of_fd)
        return max(thr, frac * fd) if frac > 0.0 else thr

    def _dtc_fs_need_exit(self) -> float:
        """Lower threshold once latched — soft ramp dips must not bounce to DualLoop."""
        enter = self._dtc_fs_need_enter()
        thr = float(self.cfg.force_contact_threshold_N)
        frac = float(getattr(self.cfg, "dtc_exit_fs_frac_of_enter", 0.35))
        frac = float(np.clip(frac, 0.05, 1.0))
        return max(thr, frac * enter)

    def _dtc_contact_ready(
        self,
        *,
        signed_d: float | None,
        fs_n: float,
        latched: bool = False,
    ) -> bool:
        """True when seek may hand off to / stay on Eq.56 DTC.

        Enter requires |Fs| ≥ enter need. Once latched, stay until |Fs| < exit
        need (hysteresis), subject to min-hold / exit-debounce in ``step``.
        """
        d_gate = float(self.cfg.dtc_contact_dist_m)
        near_geom = True
        if signed_d is not None and np.isfinite(float(signed_d)):
            near_geom = float(signed_d) <= max(d_gate, 0.015)
        if bool(self.cfg.dtc_require_force):
            need = self._dtc_fs_need_exit() if latched else self._dtc_fs_need_enter()
            return bool(abs(float(fs_n)) >= need)
        return bool(
            signed_d is not None
            and np.isfinite(float(signed_d))
            and float(signed_d) <= d_gate
            and bool(near_geom)
        )

    def _dtc_should_stay_latched(self, fs_n: float, *, force_ready: bool) -> bool:
        """Apply min-hold and exit debounce so mode-switch Fs dips don't bounce."""
        now = time.monotonic()
        if not getattr(self, "_dtc_latched", False):
            self._dtc_exit_mono_t = None
            return bool(force_ready)
        # Min hold after handoff (speedJ→DTC often dips Fs for ~100–300 ms).
        t0 = self._dtc_latch_mono_t
        min_hold = float(getattr(self.cfg, "dtc_min_hold_s", 1.5))
        if t0 is not None and (now - float(t0)) < min_hold:
            self._dtc_exit_mono_t = None
            return True
        if force_ready:
            self._dtc_exit_mono_t = None
            return True
        # Debounce exit: |Fs| must stay low continuously.
        exit_hold = float(getattr(self.cfg, "dtc_exit_hold_s", 0.35))
        if exit_hold <= 1e-9:
            return False
        if self._dtc_exit_mono_t is None:
            self._dtc_exit_mono_t = now
            return True
        if (now - float(self._dtc_exit_mono_t)) < exit_hold:
            return True
        return False

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
        tip_nn: float | None = None,
    ) -> np.ndarray:
        """Run contact law; return ``q̇`` (speedj) or zeros when DTC owns the tick.

        For loaded-surface hybrid DTC or any ``dtc_pure`` tick, call
        :meth:`step_tau` (or this method then read ``last_tau``).
        """
        if is_dtc_actuation(self.actuation):
            if not self.armed:
                return np.zeros(jacobian.shape[1], dtype=float)
            if is_dtc_pure_actuation(self.actuation):
                # ALIGN: Eq.56 W_L only. SEEK air gap: speedJ along n* (open-loop
                # DTC cannot close a free gap). SEEK near/on contact: Eq.56.
                R_star = self._refresh_R_star(R_field)
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
                    signed_d=signed_d,
                )
                aligning = self.phase == self.PHASE_ALIGN
                in_air = (
                    self.last_diag is not None
                    and float(self.last_diag.get("in_air", 0.0)) > 0.5
                )
                if self.last_diag is not None:
                    fs_n = self._estimate_fs_along_n(force_base, R_tcp, R_star)
                    self.last_diag["Fs_N"] = float(fs_n)
                if (
                    not aligning
                    and self.phase == self.PHASE_SEEK
                    and in_air
                ):
                    n = _unit(R_star[:, 2])
                    v_app = float(self.cfg.approach_v_max_m_s) * float(
                        self.cfg.force_sign
                    )
                    qdot_air = twist_to_qdot(
                        v_app * n,
                        np.zeros(3),
                        jacobian,
                        damping=float(self.cfg.damping),
                        max_joint_velocity=float(self.cfg.max_joint_velocity),
                    )
                    self.last_tau = np.zeros(jacobian.shape[1], dtype=float)
                    if self.last_diag is not None:
                        self.last_diag["dtc_use_speedj"] = 1.0
                        self.last_diag["actuation"] = "dtc_pure_air"
                        self.last_diag["v_approach_m_s"] = float(v_app)
                    if not getattr(self, "_logged_dtc_pure_air", False):
                        self._logged_dtc_pure_air = True
                        d_s = (
                            f"{float(signed_d):.4f}"
                            if signed_d is not None and np.isfinite(float(signed_d))
                            else "nan"
                        )
                        print(
                            "\033[93mWARNING: dtc_pure — SEEK air gap via speedJ "
                            f"along n* (d={d_s}, v={v_app * 1e3:.1f} mm/s); "
                            "Eq.56 after d ≤ air_gap.\033[0m",
                            flush=True,
                        )
                    return qdot_air
                if getattr(self, "_logged_dtc_pure_air", False) and not in_air:
                    self._logged_dtc_pure_air = False
                    print(
                        "\033[93mWARNING: dtc_pure — air closed → Eq.56 "
                        "directTorque.\033[0m",
                        flush=True,
                    )
                if self.last_diag is not None:
                    self.last_diag["dtc_use_speedj"] = 0.0
                    self.last_diag["actuation"] = (
                        "dtc_pure_align" if aligning else "dtc_pure"
                    )
                return np.zeros(jacobian.shape[1], dtype=float)
            # Hybrid (hardware bring-up evidence):
            #   ALIGN            → DualLoop speedJ (ω-only)
            #   SEEK unloaded    → DualLoop speedJ (approach + teleop)
            #   SEEK compressed  → Eq.56 directTorque (Fs ≥ contact threshold)
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
                latched = bool(getattr(self, "_dtc_latched", False))
                force_ready = self._dtc_contact_ready(
                    signed_d=signed_d, fs_n=fs_n, latched=latched
                )
                ready = self._dtc_should_stay_latched(fs_n, force_ready=force_ready)
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
                            f"speedJ (d={d_s}, Fs={fs_n:.2f} N); need Fs≥"
                            f"{self._dtc_fs_need_enter():.1f} N "
                            f"({self.cfg.dtc_min_fs_frac_of_fd:.0%}·Fd); "
                            "teleop live.\033[0m",
                            flush=True,
                        )
                    # Left DTC mid-seek (truly unloaded past exit hysteresis).
                    if getattr(self, "_dtc_latched", False):
                        self._dtc_latched = False
                        self._dtc_latch_mono_t = None
                        self._dtc_exit_mono_t = None
                        self._logged_seek_dtc = False
                        print(
                            "\033[93mWARNING: DTC path — back to DualLoop speedJ "
                            f"(lost compression Fs={fs_n:.2f} N; "
                            f"exit<|{self._dtc_fs_need_exit():.1f}| N).\033[0m",
                            flush=True,
                        )
                    self.last_tau = np.zeros(jacobian.shape[1], dtype=float)
                    # Fall through to DualLoop seek (force on + WASD/SM).
                else:
                    was_latched = bool(getattr(self, "_dtc_latched", False))
                    self._dtc_latched = True
                    if not was_latched:
                        self._dtc_latch_mono_t = time.monotonic()
                        self._dtc_exit_mono_t = None
                        self._logged_seek_dtc = True
                        d_s = (
                            f"{float(signed_d):.4f}"
                            if signed_d is not None and np.isfinite(float(signed_d))
                            else "nan"
                        )
                        print(
                            "\033[93mWARNING: DTC path — SEEK compressed → "
                            f"Eq.56 directTorque (Fs={fs_n:.2f} N, d={d_s} m, "
                            f"ang_z={ang_z_deg:.1f}deg); "
                            "teleop tangent via CGA Rotor sandwich.\033[0m",
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
                        signed_d=signed_d,
                    )
                    if self.last_diag is not None:
                        self.last_diag["dtc_use_speedj"] = 0.0
                        self.last_diag["actuation"] = "dtc"
                        self.last_diag["Fs_N"] = float(fs_n)
                        self.last_diag["dtc_latched"] = 1.0
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
        self._maybe_log_live_n(R_star, ang_z_deg)

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

        # Overload / wrong-sign: never push into the surface. Under high |Fs|,
        # also kill field tangent — lateral speedJ into a jammed contact killed
        # the RTDE script (Fs≈−115 with WARN=force_sign?).
        fault = float(diag.get("force_fault_fs", 0.0)) > 0.5
        wrong = float(diag.get("force_wrong_sign", 0.0)) > 0.5
        fs_now = float(diag.get("Fs_N", 0.0))
        need_load = max(
            float(self.cfg.force_contact_threshold_N),
            float(self.cfg.dtc_min_fs_frac_of_fd) * abs(float(self.cfg.desired_force_n)),
        )
        jammed = abs(fs_now) >= need_load and (fault or wrong)

        if wrong and abs(fs_now) >= need_load and not self._logged_force_sign_hint:
            self._logged_force_sign_hint = True
            hint = -float(self.cfg.force_sign)
            print(
                "\033[93mWARNING: Fs has wrong sign vs force_sign="
                f"{self.cfg.force_sign:g} (Fs≈{fs_now:.1f} N). "
                f"Set in_contact.force_sign: {hint:g} and re-Enter.\033[0m",
                flush=True,
            )

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
            if jammed:
                v_tan = np.zeros(3, dtype=float)
            v = cmd.as_linear() + v_tan
            w_rad = np.deg2rad(cmd.as_angular_deg())
            if fault or wrong:
                n = _unit(R_star[:, 2])
                v_n = float(np.dot(v, n))
                v_perp = v - n * v_n
                # Retract-only along n* (Fd≥0 → v∥ ≤ 0).
                if jammed:
                    v = n * min(0.0, v_n)
                else:
                    v = v_perp + n * min(0.0, v_n)
                w_rad[:] = 0.0
            vn = float(np.linalg.norm(v))
            if vn > self.cfg.max_linear_m_s and vn > 1e-12:
                v = v * (self.cfg.max_linear_m_s / vn)
            wn = float(np.linalg.norm(np.rad2deg(w_rad)))
            if wn > self.cfg.max_angular_deg_s and wn > 1e-12:
                w_rad = w_rad * (self.cfg.max_angular_deg_s / wn)

        # Near surface: crawl. Prefer tip_nn (signed_d often stale: d=18 mm while
        # tip_nn=7 mm). Also crawl once any real load shows.
        near_surf = False
        if (
            signed_d is not None
            and np.isfinite(float(signed_d))
            and float(signed_d) < 0.025
        ):
            near_surf = True
        if tip_nn is not None and np.isfinite(float(tip_nn)) and float(tip_nn) < 0.015:
            near_surf = True
        if abs(float(diag.get("Fs_N", 0.0))) >= float(
            self.cfg.force_contact_threshold_N
        ):
            near_surf = True
        if not aligning and near_surf:
            n = _unit(R_star[:, 2])
            v_n = float(np.dot(v, n))
            crawl = 0.002  # 2 mm/s
            if v_n > crawl:
                v = v - n * v_n + n * crawl
            fs_hold = float(diag.get("Fs_N", 0.0))
            if abs(fs_hold) >= 0.5 * abs(float(self.cfg.desired_force_n)) and v_n > 0.0:
                v = v - n * v_n  # hold — already near Fd

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
            "cga_tan": 1.0 if self._last_cga_tan else 0.0,
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
        signed_d: float | None = None,
    ) -> np.ndarray:
        """Eq. 56 joint torque command (DTC). Requires ``dtc`` or ``dtc_pure``."""
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
        self._maybe_log_live_n(R_star, ang_z_deg)
        aligning = self.phase == self.PHASE_ALIGN
        # ALIGN: orientation W_L only (tool-Z → n*). SEEK: full Eq.56 with W_C.
        # Enabling Fd approach during ALIGN snapped the live UR off-axis.
        if is_dtc_pure_actuation(self.actuation):
            enable_force = not aligning
            v_tan_use = self._consume_tangent(dt, aligning=aligning)
        else:
            enable_force = (not aligning) and (self.phase == self.PHASE_SEEK)
            v_tan_use = self._consume_tangent(dt, aligning=aligning)

        qd = (
            np.zeros(n_j, dtype=float)
            if qdot_meas is None
            else np.asarray(qdot_meas, dtype=float).reshape(n_j)
        )
        # Bias-subtract like DualLoop force path (raw TCP force includes gravity).
        W_raw = np.asarray(wrench6, dtype=float).reshape(6)
        W_meas = W_raw - self._wrench_bias6
        command_scale = 1.0
        max_tau_limit = float(self.cfg.max_tau_nm)
        ramp_t_s = 0.0
        # Soft τ ramp for hybrid and pure — first hybrid handoff previously dumped
        # scale=1 / |τ|=max into near-air and killed the RTDE script.
        if is_dtc_actuation(self.actuation):
            ramp = float(self.cfg.dtc_startup_ramp_s)
            if self._dtc_ramp_mono_t is None:
                self._dtc_ramp_mono_t = time.monotonic()
            if ramp > 1e-9:
                ramp_t_s = time.monotonic() - float(self._dtc_ramp_mono_t)
                command_scale = float(np.clip(ramp_t_s / ramp, 0.0, 1.0))
            else:
                ramp_t_s = 0.0
                command_scale = 1.0
            tau_lo = float(self.cfg.dtc_startup_tau_nm)
            tau_hi = float(self.cfg.max_tau_nm)
            max_tau_limit = tau_lo + command_scale * (tau_hi - tau_lo)
            # Under existing load, never open DTC at scale=0 / W_C=0 (logs:
            # Fs≈22 N, scale=0, |τ|≈0.4 Nm → UR path-deviation stop).
            if self.cfg.wrench_in_tool_frame:
                fs_est = float(self.cfg.force_sign) * float(W_meas[2])
            else:
                fs_est = float(self.cfg.force_sign) * float(
                    np.dot(W_meas[:3], _unit(R_star[:, 2]))
                )
            fd = abs(float(self.cfg.desired_force_n))
            enter = self._dtc_fs_need_enter()
            if abs(fs_est) >= 0.5 * enter and fd > 1e-6:
                floor = float(np.clip(abs(fs_est) / fd, 0.3, 0.95))
                if command_scale < floor:
                    command_scale = floor
                    max_tau_limit = tau_lo + command_scale * (tau_hi - tau_lo)
        tau = self._eq56.step(
            p_tcp=p,
            R_tcp=np.asarray(R_tcp, dtype=float).reshape(3, 3),
            x_q=x_q,
            R_field=R_star,
            wrench_meas6=W_meas,
            jacobian_ee=jacobian,
            qdot=qd,
            v_tan_world=v_tan_use,
            dt=dt,
            enable_force=enable_force,
            command_scale=command_scale,
            max_tau_limit=max_tau_limit,
            signed_d=signed_d,
        )
        tau = np.asarray(tau, dtype=float).reshape(n_j)
        tau_slew = 0.0
        dtau_lim = float(self.cfg.dtc_max_dtau_nm_s)
        if dtau_lim > 1e-9 and is_dtc_actuation(self.actuation):
            # First DTC tick: accept τ as-is. Slew-from-zero capped |τ| at
            # dtau_lim/hz (~0.4 Nm @ 20 Nm/s × 50 Hz) under Fs≈22 N → path
            # deviation protective stop on speedJ→DTC handoff.
            if self._tau_cmd_prev is None:
                self._tau_cmd_prev = tau.copy()
            else:
                prev = np.asarray(self._tau_cmd_prev, dtype=float).reshape(n_j)
                dtau = tau - prev
                step_max = dtau_lim * float(max(dt, 1e-6))
                dn = float(np.linalg.norm(dtau))
                if dn > step_max and dn > 1e-12:
                    tau = prev + dtau * (step_max / dn)
                    tau_slew = 1.0
                self._tau_cmd_prev = tau.copy()
        else:
            self._tau_cmd_prev = tau.copy()
        self.last_tau = tau
        if is_dtc_pure_actuation(self.actuation):
            act_tag = "dtc_pure_align" if aligning else "dtc_pure"
        else:
            act_tag = "dtc"
        self.last_diag = {
            **dict(self._eq56.last_diag),
            "actuation": act_tag,
            "dtc_use_speedj": 0.0,
            "n_freeze": 1.0 if self.n_frozen else 0.0,
            "v_tan_norm": float(np.linalg.norm(v_tan_use)),
            "Fd_N": float(self.cfg.desired_force_n),
            "force_sign": float(self.cfg.force_sign),
            "phase_align": 1.0 if aligning else 0.0,
            "ang_z_deg": float(ang_z_deg),
            "force_enable": 1.0 if enable_force else 0.0,
            "phase_changed": 1.0 if prev_phase != self.phase else 0.0,
            "tau_norm": float(np.linalg.norm(self.last_tau)),
            "force_fault": float(self._eq56.last_diag.get("fault_fs", 0.0)),
            "force_wrong_sign": float(self._eq56.last_diag.get("wrong_sign", 0.0)),
            "command_scale": float(command_scale),
            "max_tau_limit": float(max_tau_limit),
            "ramp_t_s": float(ramp_t_s),
            "tau_slew": float(tau_slew),
            "dtau_lim_nm_s": float(dtau_lim),
            "signed_d": (
                float(signed_d)
                if signed_d is not None and np.isfinite(float(signed_d))
                else float("nan")
            ),
            "cga_tan": 1.0 if self._last_cga_tan else 0.0,
        }
        if not self._logged_first_wrench:
            self._logged_first_wrench = True
            d = self.last_diag
            w_l = np.array(
                [
                    d.get("W_L_fx", 0.0),
                    d.get("W_L_fy", 0.0),
                    d.get("W_L_fz", 0.0),
                    d.get("W_L_mx", 0.0),
                    d.get("W_L_my", 0.0),
                    d.get("W_L_mz", 0.0),
                ],
                dtype=float,
            )
            w_c = np.array(
                [
                    d.get("f_cmd_base_x", 0.0),
                    d.get("f_cmd_base_y", 0.0),
                    d.get("f_cmd_base_z", 0.0),
                    d.get("m_cmd_base_x", 0.0),
                    d.get("m_cmd_base_y", 0.0),
                    d.get("m_cmd_base_z", 0.0),
                ],
                dtype=float,
            )
            sc = float(max(command_scale, 1e-9))
            _print_dtc_dbg(
                "first wrench  "
                f"phase={'align' if aligning else 'seek'}  "
                f"ang_z={ang_z_deg:.1f}deg  "
                f"scale={command_scale:.3f}  "
                f"W_L[f|m]={_fmt3(w_l[:3])}|{_fmt3(w_l[3:])}  "
                f"W_L_unscaled[m]={_fmt3(w_l[3:] / sc)}  "
                f"W_C[f|m]={_fmt3(w_c[:3])}|{_fmt3(w_c[3:])}  "
                f"tau={_fmt3(self.last_tau)}  "
                f"|τ|={float(np.linalg.norm(self.last_tau)):.2f}Nm"
            )
        if (
            enable_force
            and not aligning
            and not self._logged_first_seek
        ):
            self._logged_first_seek = True
            d = self.last_diag
            tau = self.last_tau
            _print_dtc_dbg(
                "SEEK approach  "
                f"ang_z={ang_z_deg:.1f}deg  "
                f"signed_d={d.get('signed_d', float('nan')):.4f}m  "
                f"Fs={float(d.get('Fs_raw_N', d.get('Fs_comp', 0.0))):.2f}N  "
                f"W_C_N={float(d.get('W_C_norm', 0.0)):.2f}  "
                f"f_base={_fmt3([d.get('f_cmd_base_x', 0.0), d.get('f_cmd_base_y', 0.0), d.get('f_cmd_base_z', 0.0)])}  "
                f"wrong_sign={int(d.get('force_wrong_sign', 0.0))}  "
                f"τ_lin={_fmt3(tau[:3])}  τ_ang={_fmt3(tau[3:])}  "
                f"slew={int(tau_slew)}"
            )
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
