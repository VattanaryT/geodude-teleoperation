"""Eq. 56 contact torque law (coverage citation).

τ = −J_{ee,G}^⊤ · (K_{L} W_L + D_V W_V − W_C − W_tan)
# D W_V in the sum ⇒ τ opposes Jq̇ (paper's −D inside −Jᵀ would anti-damp).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geodude_teleoperation.cga.line_tracker import (
    alignment_twist_contact_se3,
    alignment_wrench_from_twist,
)
from geodude_teleoperation.cga.wrenches import (
    ContactWrenchPID,
    damping_wrench_from_qdot,
    desired_contact_wrench_ee,
    tangent_wrench_from_world_force,
    wrench_ee_to_base,
    wrench_to_ee_frame,
)
from geodude_teleoperation.teleop.tactile_impedance import joint_torques_from_wrench


@dataclass(frozen=True)
class Eq56Gains:
    """Diagonal stiffness / damping in wrench coordinates ``[f; m]``."""

    K_line: np.ndarray  # (6,)
    D_twist: np.ndarray  # (6,)


def default_eq56_gains() -> Eq56Gains:
    """Research defaults for stable kitchenware-style interaction."""
    return Eq56Gains(
        K_line=np.array([30.0, 30.0, 30.0, 750.0, 750.0, 300.0]),
        D_twist=np.array([10.0, 10.0, 10.0, 150.0, 150.0, 50.0]),
    )


def assemble_eq56_tau(
    jacobian_ee: np.ndarray,
    *,
    W_L: np.ndarray,
    W_V: np.ndarray,
    W_C: np.ndarray,
    W_tan: np.ndarray | None = None,
    gains: Eq56Gains | None = None,
) -> np.ndarray:
    """Assemble joint torques from Eq. 56 wrench sum (no gravity — DTC compensates).

    All wrenches must share the Jacobian frame (base/spatial at TCP).

    ``W_L`` is geometric error. ``W_V`` is measured twist ``J q̇``.
    ``W_C`` / ``W_tan`` are wrenches the EE should **apply to the environment**,
    so they enter with opposite sign inside the sum.

    Damping must oppose ``ẋ``: ``τ`` includes ``−Jᵀ D W_V``. (Writing
    ``− D W_V`` in the paper paren with ``τ = −Jᵀ (⋯)`` would *add* energy.)
    """
    g = gains or default_eq56_gains()
    K = np.asarray(g.K_line, dtype=float).reshape(6)
    D = np.asarray(g.D_twist, dtype=float).reshape(6)
    W_sum = (
        K * np.asarray(W_L, dtype=float).reshape(6)
        + D * np.asarray(W_V, dtype=float).reshape(6)
        - np.asarray(W_C, dtype=float).reshape(6)
    )
    if W_tan is not None:
        W_sum = W_sum - np.asarray(W_tan, dtype=float).reshape(6)
    return joint_torques_from_wrench(jacobian_ee, W_sum)


@dataclass
class Eq56Controller:
    """Stateful Eq. 56 tick: line align + normal-force PID + damping + tangent FF."""

    gains: Eq56Gains
    force_pid: ContactWrenchPID
    desired_force_n: float = 5.0
    force_sign: float = 1.0
    wrench_in_tool_frame: bool = True
    tangent_force_gain: float = 50.0
    max_tau: float = 20.0
    contact_threshold_n: float = 0.6
    approach_force_n: float = 8.0
    max_wrench_n: float = 12.0
    overshoot_margin_n: float = 2.0
    retract_force_n: float = 8.0

    def __init__(
        self,
        *,
        gains: Eq56Gains | None = None,
        desired_force_n: float = 5.0,
        force_sign: float = 1.0,
        wrench_in_tool_frame: bool = True,
        wrench_kp: float = 0.25,
        wrench_ki: float = 0.0,
        wrench_kd: float = 0.0,
        tangent_force_gain: float = 50.0,
        max_tau: float = 20.0,
        contact_threshold_n: float = 0.6,
        approach_force_n: float = 8.0,
        max_wrench_n: float = 12.0,
        overshoot_margin_n: float = 2.0,
        retract_force_n: float = 8.0,
        force_lpf_hz: float = 5.0,
        spike_dF_dt_N_s: float = 40.0,
        spike_hold_s: float = 0.15,
        wrong_sign_fs_N: float = -2.0,
        wrong_sign_fd_frac: float = 0.5,
        fault_abs_fs_N: float = 40.0,
        fault_fd_mult: float = 1.35,
        force_rate_damp_n_s: float = 0.02,
        approach_stiffness_n_per_m: float = 400.0,
        air_gap_m: float = 0.008,
    ):
        self.gains = gains or default_eq56_gains()
        self.force_pid = ContactWrenchPID(
            kp=float(wrench_kp), ki=float(wrench_ki), kd=float(wrench_kd)
        )
        self.desired_force_n = float(desired_force_n)
        self.force_sign = float(force_sign)
        self.wrench_in_tool_frame = bool(wrench_in_tool_frame)
        self.tangent_force_gain = float(tangent_force_gain)
        self.max_tau = float(max_tau)
        self.contact_threshold_n = float(contact_threshold_n)
        self.approach_force_n = float(approach_force_n)
        self.max_wrench_n = float(max_wrench_n)
        self.overshoot_margin_n = float(overshoot_margin_n)
        self.retract_force_n = float(retract_force_n)
        self.force_lpf_hz = float(force_lpf_hz)
        self.spike_dF_dt_N_s = float(spike_dF_dt_N_s)
        self.spike_hold_s = float(spike_hold_s)
        self.wrong_sign_fs_N = float(wrong_sign_fs_N)
        self.wrong_sign_fd_frac = float(wrong_sign_fd_frac)
        self.fault_abs_fs_N = float(fault_abs_fs_N)
        self.fault_fd_mult = float(fault_fd_mult)
        self.force_rate_damp_n_s = float(force_rate_damp_n_s)
        self.approach_stiffness_n_per_m = float(approach_stiffness_n_per_m)
        self.air_gap_m = float(air_gap_m)
        self.align_lever_m = 0.08
        self.align_v_cap_m = 0.04
        # SEEK in contact: hold orientation lightly. Air SEEK zeros W_L entirely.
        self.seek_w_l_hold = 0.2
        self.last_diag: dict = {}
        self._fs_raw_prev: float | None = None
        self._f_lpf: float | None = None
        self._spike_timer = 0.0

    def reset(self) -> None:
        self.force_pid.reset()
        self.last_diag = {}
        self._fs_raw_prev = None
        self._f_lpf = None
        self._spike_timer = 0.0

    def step(
        self,
        *,
        p_tcp: np.ndarray,
        R_tcp: np.ndarray,
        x_q: np.ndarray,
        R_field: np.ndarray,
        wrench_meas6: np.ndarray,
        jacobian_ee: np.ndarray,
        qdot: np.ndarray,
        v_tan_world: np.ndarray | None,
        dt: float,
        enable_force: bool = True,
        command_scale: float = 1.0,
        max_tau_limit: float | None = None,
        signed_d: float | None = None,
    ) -> np.ndarray:
        V = alignment_twist_contact_se3(
            R_tcp,
            R_field,
            lever_m=float(self.align_lever_m),
            v_cap_m=float(self.align_v_cap_m),
        )
        W_L = alignment_wrench_from_twist(V)
        W_V = damping_wrench_from_qdot(jacobian_ee, qdot)
        scale = float(np.clip(command_scale, 0.0, 1.0))
        W_L = W_L * scale

        W_m = wrench_to_ee_frame(
            wrench_meas6,
            R_tcp,
            wrench_in_tool_frame=self.wrench_in_tool_frame,
        )
        W_d = desired_contact_wrench_ee(
            desired_force_n=self.desired_force_n,
            force_sign=self.force_sign,
        )
        dt = float(max(dt, 1e-6))
        fs_raw = float(self.force_sign) * float(W_m[2])
        # Stale VF signed_d often says "air" while tip is loaded (logs: d=0.02,
        # tip_nn=0.004, Fs≈9, air=1). Clear air only when geometry is *near*
        # enough that measured load is plausible — far signed_d + gravity bias
        # (Fs≈8 at d=5 cm) must still use the air spring.
        geom_air = (
            signed_d is not None
            and np.isfinite(float(signed_d))
            and float(signed_d) > float(self.air_gap_m)
        )
        near_enough = signed_d is None or (
            np.isfinite(float(signed_d))
            and float(signed_d) <= max(2.5 * float(self.air_gap_m), 0.025)
        )
        load_clears_air = abs(fs_raw) >= float(self.contact_threshold_n) and near_enough
        in_air = bool(geom_air and not load_clears_air)
        # Air SEEK: do not spend τ on residual Align (W_L ate the live approach).
        if enable_force and in_air:
            W_L = np.zeros(6, dtype=float)
        elif enable_force:
            W_L = W_L * float(self.seek_w_l_hold)

        hz = float(self.force_lpf_hz)
        if hz > 0.0:
            tau_f = 1.0 / (2.0 * np.pi * hz)
            alpha = dt / (tau_f + dt)
            if self._f_lpf is None:
                self._f_lpf = fs_raw
            else:
                self._f_lpf = (1.0 - alpha) * self._f_lpf + alpha * fs_raw
            fs_comp = float(self._f_lpf)
        else:
            fs_comp = fs_raw
        if self._fs_raw_prev is None:
            d_fs_raw = 0.0
        else:
            d_fs_raw = (fs_raw - self._fs_raw_prev) / dt
        self._fs_raw_prev = fs_raw
        # PID on filtered compression; trips below use raw.
        W_m_pid = W_m.copy()
        W_m_pid[2] = fs_comp / float(self.force_sign) if abs(self.force_sign) > 1e-12 else fs_comp

        fd = abs(float(self.desired_force_n))
        W_approach = np.zeros(6, dtype=float)
        overshoot = 0.0
        spike = 0.0
        wrong_sign = 0.0
        fault_fs = 0.0
        real_contact = False
        if enable_force:
            if d_fs_raw >= float(self.spike_dF_dt_N_s):
                spike = 1.0
                self._spike_timer = max(self._spike_timer, float(self.spike_hold_s))
                self.force_pid.reset()
            if self._spike_timer > 0.0:
                self._spike_timer = max(0.0, self._spike_timer - dt)
                spike = 1.0

            overshoot_hit = fs_raw > (fd + float(self.overshoot_margin_n))
            mag_overload = abs(fs_raw) > (fd + float(self.overshoot_margin_n))
            ws_thresh = min(
                float(self.wrong_sign_fs_N),
                -abs(float(self.wrong_sign_fd_frac) * fd),
            )
            wrong = fd >= 0.0 and fs_raw <= ws_thresh
            fault_thresh = max(
                float(self.fault_abs_fs_N),
                abs(float(self.fault_fd_mult) * fd),
            )
            fault = abs(fs_raw) >= fault_thresh or mag_overload
            real_contact = (not in_air) and abs(fs_raw) > float(self.contact_threshold_n)
            if overshoot_hit or mag_overload:
                overshoot = 1.0
            if wrong:
                wrong_sign = 1.0
            if fault:
                fault_fs = 1.0

            retract_cap = min(float(self.retract_force_n), 3.0)
            if (
                overshoot_hit
                or mag_overload
                or (spike > 0.5 and fd >= 0.0 and abs(fs_raw) > fd)
                or fault
                or (wrong and abs(fs_raw) >= float(self.contact_threshold_n))
            ):
                # Soft unload / hold — never reverse-yank. Live log: overshoot
                # retract at |τ|≈18 Nm flung TCP +24 cm and killed RTDE.
                W_C = np.zeros(6, dtype=float)
                if spike > 0.5 and not overshoot_hit and not mag_overload and not fault:
                    # Impact rising edge: hold (zero contact wrench) briefly.
                    W_C[2] = 0.0
                elif wrong and abs(fs_raw) >= float(self.contact_threshold_n):
                    # Wrong-sign load: small unload only.
                    W_C[2] = -float(self.force_sign) * retract_cap
                else:
                    # Mild/severe overshoot: command toward Fd; bleed only if ≫Fd.
                    W_C[2] = float(self.force_sign) * fd
                    if abs(fs_raw) > 1.5 * max(fd, 1e-6):
                        bleed = min(retract_cap, 0.25 * (abs(fs_raw) - fd))
                        W_C[2] = float(self.force_sign) * max(0.0, fd - bleed)
                W_C = W_C * scale
                self.force_pid.reset()
            else:
                # Feedforward Fd + P(I) on error. Pure kp*(Fd−Fs) stalls when
                # kp·error ≪ Fd (e.g. Fd=70, Fs=5, kp=0.25 → W_C≈16 N forever).
                W_corr = self.force_pid.step(W_d, W_m_pid, dt)
                W_C = np.asarray(W_d, dtype=float).reshape(6) + W_corr
                if (
                    (in_air or fs_raw < float(self.contact_threshold_n))
                    and wrong_sign < 0.5
                    and spike < 0.5
                ):
                    f_app = max(
                        float(self.approach_force_n), abs(float(self.desired_force_n))
                    )
                    W_approach[2] = float(self.force_sign) * f_app
                    # Approach FF already covered by W_d when Fd set; only boost if
                    # still below contact and W_d alone is tiny.
                    if abs(float(self.desired_force_n)) < float(self.approach_force_n):
                        W_C = W_C + W_approach
                if d_fs_raw > 0.0:
                    W_C[2] -= float(self.force_sign) * float(self.force_rate_damp_n_s) * d_fs_raw
                if (not in_air) and wrong and float(self.force_sign) * float(W_C[2]) > 0.0:
                    W_C[2] = 0.0
                W_C = W_C * scale
            wn = float(np.linalg.norm(W_C[:3]))
            # Allow commanding up to Fd (feedforward); max_wrench_n is a floor.
            w_cap = max(float(self.max_wrench_n), fd)
            if wn > w_cap and wn > 1e-12:
                W_C[:3] = W_C[:3] * (w_cap / wn)
        else:
            W_C = np.zeros(6, dtype=float)
            self._spike_timer = 0.0

        W_tan_ee_or_base = np.zeros(6, dtype=float)
        if v_tan_world is not None and overshoot < 0.5 and fault_fs < 0.5:
            f_world = float(self.tangent_force_gain) * np.asarray(
                v_tan_world, dtype=float
            ).reshape(3)
            W_tan_ee_or_base = tangent_wrench_from_world_force(f_world, R_tcp)

        W_C_base = wrench_ee_to_base(W_C, R_tcp)
        W_tan_base = W_tan_ee_or_base
        W_approach_base = wrench_ee_to_base(W_approach, R_tcp)
        # Air SEEK: spring into the surface along n* (into-surface).
        # Constant Fd alone often loses to static friction under DTC; k·d grows
        # with air gap so the tip closes. Zero residual W_L (above) so τ is not
        # spent on Align.
        if enable_force and in_air and fault_fs < 0.5 and overshoot < 0.5:
            n_star = np.asarray(R_field, dtype=float).reshape(3, 3)[:, 2]
            nn = float(np.linalg.norm(n_star))
            if nn > 1e-12:
                d_gap = float(signed_d)
                f_spring = float(self.approach_stiffness_n_per_m) * d_gap
                f_app = max(
                    float(self.approach_force_n),
                    abs(float(self.desired_force_n)),
                    f_spring,
                )
                f_app = min(f_app, max(float(self.max_wrench_n), abs(float(self.desired_force_n))))
                W_C_base = np.zeros(6, dtype=float)
                W_C_base[:3] = (float(self.force_sign) * f_app * scale) * (n_star / nn)
                W_approach_base = W_C_base.copy()

        tau = assemble_eq56_tau(
            jacobian_ee,
            W_L=W_L,
            W_V=W_V,
            W_C=W_C_base,
            W_tan=W_tan_base,
            gains=self.gains,
        )
        tn = float(np.linalg.norm(tau))
        tau_cap = float(self.max_tau if max_tau_limit is None else max_tau_limit)
        if tn > tau_cap and tn > 1e-12:
            tau = tau * (tau_cap / tn)

        self.last_diag = {
            "W_L_norm": float(np.linalg.norm(W_L)),
            "W_V_norm": float(np.linalg.norm(W_V)),
            "W_C_norm": float(np.linalg.norm(W_C_base)),
            "W_approach_norm": float(np.linalg.norm(W_approach_base)),
            "W_tan_norm": float(np.linalg.norm(W_tan_base)),
            "tau_norm": float(np.linalg.norm(tau)),
            "V_align_norm": float(np.linalg.norm(V)),
            "Fs_ee_z": float(W_m[2]),
            "Fs_comp": float(fs_comp),
            "Fs_raw_N": float(fs_raw),
            "dFs_dt_N_s": float(d_fs_raw),
            "Fd_N": float(self.desired_force_n),
            "force_enable": 1.0 if enable_force else 0.0,
            "force_overshoot": float(overshoot),
            "spike": float(spike),
            "wrong_sign": float(wrong_sign),
            "fault_fs": float(fault_fs),
            "real_contact": 1.0 if real_contact else 0.0,
            "in_air": 1.0 if in_air else 0.0,
            "command_scale": float(scale),
            "max_tau_limit": float(tau_cap),
            "W_L_fx": float(W_L[0]),
            "W_L_fy": float(W_L[1]),
            "W_L_fz": float(W_L[2]),
            "W_L_mx": float(W_L[3]),
            "W_L_my": float(W_L[4]),
            "W_L_mz": float(W_L[5]),
            "W_V_wx": float(W_V[3]),
            "W_V_wy": float(W_V[4]),
            "W_V_wz": float(W_V[5]),
            "f_cmd_base_x": float(W_C_base[0]),
            "f_cmd_base_y": float(W_C_base[1]),
            "f_cmd_base_z": float(W_C_base[2]),
            "m_cmd_base_x": float(W_C_base[3]),
            "m_cmd_base_y": float(W_C_base[4]),
            "m_cmd_base_z": float(W_C_base[5]),
        }
        return tau
