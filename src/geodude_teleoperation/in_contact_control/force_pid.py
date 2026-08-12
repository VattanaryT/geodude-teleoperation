"""
Force loop — PID force tracking along the normal line n*, with optional
real-time auto-stabilize (spike guard + oscillation gain drop).

  n   = unit normal in base (tool +Z at capture / target point)
  Fs  = measured force along n  (from F/T, bias-corrected, optional LPF)
  Fd  = desired force along n   [N]
  Fe  = Fd − Fs                 (drive Fe → 0)
  v∥  = Kp·Fe + Ki·∫Fe + Kd·dFe/dt   (then auto-stabilize shaping)
  v_lin = v∥ · n
  ω = 0

Independent of the align loop (align leaves n* open).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Sequence, Tuple

import numpy as np

from geodude_teleoperation.in_contact_control.types import TwistCommand


@dataclass
class ForcePIDConfig:
    """PID force control along n*: Fe = Fd − Fs → v∥."""

    enable: bool = True
    # Desired force along n* [N]
    Fd_N: float = 3.0
    # PID gains: output is along-line velocity [m/s]
    #   Kp: (m/s) / N
    #   Ki: (m/s) / (N·s)
    #   Kd: (m/s) / (N/s)
    kp: float = 0.001  # 1 mm/s per N (conservative contact default)
    ki: float = 0.0
    kd: float = 0.0
    # Anti-windup: clamp |∫Fe dt| [N·s]
    integral_limit_N_s: float = 20.0
    # Ignore |Fe| below this (N) for P/I/D drive
    force_deadband_N: float = 0.35
    # First-order LPF on Fs (Hz); 0 = off
    force_lpf_hz: float = 5.0
    # Safety
    max_v_along_m_s: float = 0.008  # 8 mm/s (anti-slam / bounce)
    # Sign on measured force: Fs = force_sign * (F−bias)·n
    force_sign: float = 1.0
    # If True, force_base is tool-frame; rotate with R_ee (EE→base) first
    # Default True for cells where Kinova stream behaves as tool-frame.
    wrench_in_tool_frame: bool = True
    # Slice E validation: |Fe| ≤ this → force tracking OK (Fs ≈ Fd)
    contact_band_N: float = 0.5
    # Require |Fs| at least this fraction of |Fd| so free-space isn't "OK"
    contact_min_load_frac: float = 0.25
    contact_min_load_N: float = 0.4

    # --- Real-time auto-stabilize (no manual retune needed) ---
    auto_stabilize: bool = True
    # Soft approach while barely touching (limits free-space slam into surface).
    # Always applied when |Fs| is below contact_threshold — do NOT skip when |Fe|
    # is large (wrong-sign impact makes Fe huge and used to open full speed).
    approach_v_max_m_s: float = 0.005  # 5 mm/s until light contact
    contact_threshold_N: float = 0.6
    recover_fe_N: float = 1.0  # kept for diag/compat; no longer opens full speed
    # Active damping on force rate: v -= k_damp * dFs/dt  (cuts bounce)
    # Only applied when force is *rising* (impact), not when force drops.
    force_rate_damp_m_s_per_Nps: float = 0.00015  # stronger bounce damping
    # Spike: raw Fs *rising* faster than this → retract briefly (impact only).
    # Trips must use fs_raw: 5 Hz LPF hid 400 N/s impacts in contact_motion.csv.
    # Force *drops* must NOT freeze — we need to re-approach quickly.
    spike_dF_dt_N_s: float = 40.0
    spike_hold_s: float = 0.15
    # Overshoot: if Fs_raw > Fd + margin, freeze integral and allow only retract
    overshoot_margin_N: float = 1.0
    # On overshoot/spike, unload at least this fast along −n* (m/s)
    unload_v_m_s: float = 0.004
    # Wrong-sign / impact trip: Fs below −max(2, 0.5|Fd|) → stop approach.
    # Mild negative Fs during bounce must NOT trip (was killing WASD).
    wrong_sign_fs_N: float = -2.0
    wrong_sign_fd_frac: float = 0.5
    # High |Fs| trip: retract-only (never freeze — freeze traps the load).
    fault_abs_fs_N: float = 40.0
    fault_fd_mult: float = 6.0  # also trip at mult*|Fd| if larger
    # Oscillation: many Fe sign flips → scale down Kp/Ki in real time
    osc_window_s: float = 0.45
    osc_sign_flips_trigger: int = 3
    osc_gain_cut: float = 0.4  # multiply scale by this when oscillating
    osc_min_scale: float = 0.15
    osc_recover_per_s: float = 0.25  # scale → 1 when quiet
    osc_hold_v_s: float = 0.15  # zero v_along this long after osc trip


class ForcePIDController:
    """
    Force loop: PID on Fe = Fd − Fs → linear velocity along n only.
    Optional auto-stabilize reduces spikes and oscillations in real time.
    """

    def __init__(self, config: Optional[ForcePIDConfig] = None) -> None:
        self.config = config or ForcePIDConfig()
        self._integral = 0.0  # ∫ Fe dt  [N·s]
        self._fe_prev: Optional[float] = None
        self._fs_prev: Optional[float] = None
        self._fs_raw_prev: Optional[float] = None
        self._f_lpf: Optional[float] = None
        self._f_bias = np.zeros(3, dtype=np.float64)
        self._gain_scale = 1.0
        self._spike_timer = 0.0
        self._osc_hold_timer = 0.0
        self._soft_approach_timer = 0.0
        self._fe_sign_times: Deque[float] = deque()
        self._fe_sign_prev: Optional[float] = None
        self._time_s = 0.0
        self._wrong_sign_prev = False

    def reset(self) -> None:
        self._integral = 0.0
        self._fe_prev = None
        self._fs_prev = None
        self._fs_raw_prev = None
        self._f_lpf = None
        self._gain_scale = 1.0
        self._spike_timer = 0.0
        self._osc_hold_timer = 0.0
        self._soft_approach_timer = 0.0
        self._wrong_sign_prev = False
        self._fe_sign_times.clear()
        self._fe_sign_prev = None
        self._time_s = 0.0

    def set_force_bias(self, fx: float, fy: float, fz: float) -> None:
        self._f_bias = np.array([fx, fy, fz], dtype=np.float64)

    def set_force_bias_from_samples(
        self, samples: Sequence[Sequence[float]]
    ) -> np.ndarray:
        """Each sample is (fx,fy,fz) in base. Returns mean force."""
        if not samples:
            raise ValueError("need wrench samples for bias")
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        mean = arr[:, :3].mean(axis=0)
        self.set_force_bias(float(mean[0]), float(mean[1]), float(mean[2]))
        self._f_lpf = None
        self._fe_prev = None
        self._fs_prev = None
        self._fs_raw_prev = None
        return mean

    def step(
        self,
        *,
        n_base: np.ndarray,
        force_base: Sequence[float],
        dt: float,
        R_ee: Optional[np.ndarray] = None,
    ) -> Tuple[TwistCommand, Dict[str, float]]:
        """
        One PID cycle (+ optional auto-stabilize).

        Parameters
        ----------
        n_base : (3,) unit normal in base (line direction)
        force_base : (3,) raw F/T force (bias subtracted here) → Fs
        dt : s
        R_ee : optional EE→base rotation if wrench is in tool frame
        """
        c = self.config
        n = np.asarray(n_base, dtype=np.float64).reshape(3)
        nn = float(np.linalg.norm(n))
        if nn < 1e-12:
            return TwistCommand(), self._empty_diag()
        n = n / nn

        f_raw = np.asarray(force_base, dtype=np.float64).reshape(3)
        f = f_raw - self._f_bias
        if c.wrench_in_tool_frame and R_ee is not None:
            R = np.asarray(R_ee, dtype=np.float64).reshape(3, 3)
            f = R @ f

        fs_raw = float(np.dot(f, n)) * float(c.force_sign)

        # LPF on Fs (PID only) — safety trips use fs_raw below.
        fs = fs_raw
        hz = float(c.force_lpf_hz)
        dt = float(max(dt, 1e-6))
        self._time_s += dt
        if hz > 0:
            tau_f = 1.0 / (2.0 * np.pi * hz)
            alpha = dt / (tau_f + dt)
            if self._f_lpf is None:
                self._f_lpf = fs_raw
            else:
                self._f_lpf = (1.0 - alpha) * self._f_lpf + alpha * fs_raw
            fs = float(self._f_lpf)

        fd = float(c.Fd_N)
        fe = fd - fs  # Fe = Fd − Fs (filtered)

        # Rates: filtered for damping; raw for spike detection
        if self._fs_prev is None:
            d_fs = 0.0
        else:
            d_fs = (fs - self._fs_prev) / dt
        self._fs_prev = fs
        if self._fs_raw_prev is None:
            d_fs_raw = 0.0
        else:
            d_fs_raw = (fs_raw - self._fs_raw_prev) / dt
        self._fs_raw_prev = fs_raw

        # Deadband on error
        db = float(c.force_deadband_N)
        if abs(fe) < db:
            fe_drive = 0.0
        else:
            fe_drive = fe - np.sign(fe) * db

        if not c.enable:
            self._integral = 0.0
            self._fe_prev = None
            cmd = TwistCommand()
            return cmd, self._diag(
                fs_raw, fs, fd, fe, 0.0, 0.0, 0.0, d_fs_raw, 1.0, cmd
            )

        auto = bool(c.auto_stabilize)
        spike_flag = 0.0
        osc_flag = 0.0

        if auto:
            # --- Oscillation detector: Fe sign flips in a sliding window ---
            sgn = 0.0 if abs(fe) < max(db, 0.15) else float(np.sign(fe))
            if sgn != 0.0 and self._fe_sign_prev is not None and sgn != self._fe_sign_prev:
                self._fe_sign_times.append(self._time_s)
            if sgn != 0.0:
                self._fe_sign_prev = sgn
            win = float(c.osc_window_s)
            t0 = self._time_s - win
            while self._fe_sign_times and self._fe_sign_times[0] < t0:
                self._fe_sign_times.popleft()
            n_flips = len(self._fe_sign_times)
            if n_flips >= int(c.osc_sign_flips_trigger):
                osc_flag = 1.0
                self._gain_scale = max(
                    float(c.osc_min_scale),
                    self._gain_scale * float(c.osc_gain_cut),
                )
                # Clear window so we don't cut every sample
                self._fe_sign_times.clear()
                # Dump integral — main source of post-bounce windup
                self._integral = 0.0
                self._osc_hold_timer = max(
                    self._osc_hold_timer, float(c.osc_hold_v_s)
                )
            else:
                # Slow recovery toward full gain when quiet
                self._gain_scale = min(
                    1.0, self._gain_scale + float(c.osc_recover_per_s) * dt
                )

            # --- Spike guard on RAW rising edge (LPF hid impacts in logs) ---
            if d_fs_raw >= float(c.spike_dF_dt_N_s):
                spike_flag = 1.0
                self._spike_timer = max(self._spike_timer, float(c.spike_hold_s))
                self._integral = 0.0
                self._gain_scale = min(self._gain_scale, 0.5)
            if self._spike_timer > 0.0:
                self._spike_timer = max(0.0, self._spike_timer - dt)
                spike_flag = 1.0

            # Safety trips on RAW Fs (not LPF)
            overshoot = fs_raw > (fd + float(c.overshoot_margin_N))
            if overshoot:
                self._integral = 0.0
            ws_thresh = min(
                float(c.wrong_sign_fs_N),
                -abs(float(c.wrong_sign_fd_frac) * fd),
            )
            wrong_sign = fd >= 0.0 and fs_raw <= ws_thresh
            if wrong_sign:
                self._integral = 0.0
                self._wrong_sign_prev = True
            elif self._wrong_sign_prev:
                # Leaving wrong-sign: keep soft approach so we don't slam back in.
                self._soft_approach_timer = max(self._soft_approach_timer, 0.35)
                self._wrong_sign_prev = False
            if self._soft_approach_timer > 0.0:
                self._soft_approach_timer = max(0.0, self._soft_approach_timer - dt)
            fault_thresh = max(
                float(c.fault_abs_fs_N),
                abs(float(c.fault_fd_mult) * fd),
            )
            fault_fs = abs(fs_raw) >= fault_thresh
            if fault_fs:
                self._integral = 0.0
        else:
            self._gain_scale = 1.0
            self._spike_timer = 0.0
            self._osc_hold_timer = 0.0
            self._soft_approach_timer = 0.0
            overshoot = False
            wrong_sign = False
            fault_fs = False

        scale = float(self._gain_scale) if auto else 1.0
        kp = float(c.kp) * scale
        ki = float(c.ki) * scale
        kd = float(c.kd)

        # Integral with clamp (anti-windup); freeze during spike / fault
        if not (auto and (self._spike_timer > 0.0 or wrong_sign or fault_fs)):
            self._integral += fe_drive * dt
        ilim = float(c.integral_limit_N_s)
        if ilim > 0:
            self._integral = float(np.clip(self._integral, -ilim, ilim))

        if self._fe_prev is None:
            d_fe = 0.0
        else:
            d_fe = (fe - self._fe_prev) / dt
        self._fe_prev = fe

        v_p = kp * fe_drive
        v_i = ki * self._integral
        v_d = kd * d_fe
        v_along = v_p + v_i + v_d

        if auto:
            unload = float(c.unload_v_m_s)
            # Active damping only when force is rising (impact/bounce), not on drop
            k_damp = float(c.force_rate_damp_m_s_per_Nps)
            if d_fs_raw > 0.0:
                v_along -= k_damp * d_fs_raw

            # Soft approach whenever |Fs| is still light, or after wrong-sign exit.
            soft = (
                abs(fs_raw) < float(c.contact_threshold_N)
                or self._soft_approach_timer > 0.0
            )
            if soft:
                v_app = float(c.approach_v_max_m_s)
                if fe > 0 and v_along > v_app:
                    v_along = v_app
                elif fe < 0 and v_along < -v_app:
                    v_along = -v_app

            # Spike: unload if already above Fd; else hold (don't push through impact)
            if self._spike_timer > 0.0:
                if fd >= 0.0 and fs_raw > fd:
                    v_along = -unload
                elif fd < 0.0 and fs_raw < fd:
                    v_along = unload
                else:
                    v_along = 0.0

            # Oscillation hold: freeze normal motion briefly after Fe chatter
            if self._osc_hold_timer > 0.0:
                self._osc_hold_timer = max(0.0, self._osc_hold_timer - dt)
                v_along = 0.0
                osc_flag = 1.0

            # Overshoot: retract (don't just freeze into the load)
            if overshoot and fd >= 0.0:
                v_along = min(v_along, 0.0)
                if v_along > -1e-9:
                    v_along = -unload
            if overshoot and fd < 0.0:
                v_along = max(v_along, 0.0)
                if v_along < 1e-9:
                    v_along = unload

            # Wrong-sign / absolute overload: never push into the load; allow retract
            if wrong_sign and v_along > 0.0 and fd >= 0.0:
                v_along = min(v_along, 0.0)
            if fault_fs:
                # Do not freeze — freezing traps |Fs| high (see FAULT=|Fs| with Fe≪0).
                if fd >= 0.0:
                    v_along = min(v_along, 0.0)
                    if v_along > -1e-9 and fs_raw > fd:
                        v_along = -float(c.approach_v_max_m_s)
                else:
                    v_along = max(v_along, 0.0)
                    if v_along < 1e-9 and fs_raw < fd:
                        v_along = float(c.approach_v_max_m_s)

        # Clip along-line speed
        vmax = float(c.max_v_along_m_s)
        if abs(v_along) > vmax:
            v_along = float(np.sign(v_along) * vmax)

        v_lin = v_along * n
        cmd = TwistCommand(
            linear_x=float(v_lin[0]),
            linear_y=float(v_lin[1]),
            linear_z=float(v_lin[2]),
        )
        return cmd, self._diag(
            fs_raw,
            fs,
            fd,
            fe,
            v_p,
            v_i,
            v_along,
            d_fs_raw,
            scale,
            cmd,
            auto=1.0 if auto else 0.0,
            spike=float(spike_flag),
            osc=float(osc_flag),
            wrong_sign=1.0 if wrong_sign else 0.0,
            fault_fs=1.0 if fault_fs else 0.0,
        )

    def _empty_diag(self) -> Dict[str, float]:
        return self._diag(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, TwistCommand()
        )

    @staticmethod
    def _diag(
        fs_raw: float,
        fs: float,
        fd: float,
        fe: float,
        v_p: float,
        v_i: float,
        v_along: float,
        d_fs: float,
        gain_scale: float,
        cmd: TwistCommand,
        *,
        auto: float = 0.0,
        spike: float = 0.0,
        osc: float = 0.0,
        wrong_sign: float = 0.0,
        fault_fs: float = 0.0,
    ) -> Dict[str, float]:
        return {
            "loop": 1.0,
            "Fs_raw_N": float(fs_raw),
            "Fs_N": float(fs),
            "Fd_N": float(fd),
            "Fe_N": float(fe),
            "F_par_raw_N": float(fs_raw),
            "F_par_N": float(fs),
            "dFs_dt_N_s": float(d_fs),
            "auto_stabilize": float(auto),
            "gain_scale": float(gain_scale),
            "spike": float(spike),
            "osc": float(osc),
            "wrong_sign": float(wrong_sign),
            "fault_fs": float(fault_fs),
            "v_p_mm_s": float(v_p) * 1e3,
            "v_i_mm_s": float(v_i) * 1e3,
            "v_along_m_s": float(v_along),
            "v_along_mm_s": float(v_along) * 1e3,
            "v_des_mm_s": float(v_along) * 1e3,
            "cmd_v_m_s": float(
                np.linalg.norm([cmd.linear_x, cmd.linear_y, cmd.linear_z])
            ),
            "cmd_w_deg_s": 0.0,
        }
