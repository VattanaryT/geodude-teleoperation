"""
Align loop — align TCP tool-Z **line** to a target point's tool-Z **line**.

Geometry (base frame)
---------------------
Target point pose (p*, R*) defines target line::

    n*  = R*[:, 2]          # point's Z-axis in base
    L*  = { p* + s n* }

Measured TCP (p, R) defines EE line::

    n_ee = R[:, 2]
    L_ee = { p + s n_ee }

Errors
------
Lateral (put TCP on L*)::

    e_⊥ = (I − n* n*^T)(p* − p)   # = p_cl − p

Direction (align EE Z to n*)::

    e_R = rotvec that rotates n_ee → n*   (or full Log(R* R^T) if hold_full_R)

Along-line free: linear velocity projected off n* so this loop does not
slide/fight along the normal (force loop owns motion along n*).

No force/torque sensor used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from geodude_teleoperation.in_contact_control.so3 import orientation_error_rotvec_base
from geodude_teleoperation.in_contact_control.spring_damper import (
    SpringDamperGains,
    SpringDamperState,
)
from geodude_teleoperation.in_contact_control.types import PoseSO3, TwistCommand


def _unit(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return a / n


def z_align_rotvec(n_meas: np.ndarray, n_des: np.ndarray) -> np.ndarray:
    """
    Rotation vector (rad, base) that rotates unit n_meas toward n_des.
    Axis = n_meas × n_des, angle = atan2(|cross|, dot).
    """
    a = _unit(n_meas)
    b = _unit(n_des)
    c = np.cross(a, b)
    s = float(np.linalg.norm(c))
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-10:
        if d > 0.0:
            return np.zeros(3, dtype=np.float64)
        # 180°: pick any axis ⟂ a
        tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, tmp)
        axis = _unit(axis)
        return axis * np.pi
    axis = c / s
    angle = float(np.arctan2(s, d))
    return axis * angle


def lateral_error_to_line(
    p: np.ndarray, p_star: np.ndarray, n_star: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """e_perp = p_cl − p, s, p_cl for line p_star + s n_star."""
    n = _unit(n_star)
    p = np.asarray(p, dtype=np.float64).reshape(3)
    p_star = np.asarray(p_star, dtype=np.float64).reshape(3)
    s = float(np.dot(p - p_star, n))
    p_cl = p_star + s * n
    e_perp = p_cl - p
    return e_perp, s, p_cl


@dataclass
class AlignmentAdmittanceConfig:
    enable: bool = True
    # Lateral (put TCP on target Z-line)
    k_perp: float = 100.0
    # Orientation: align EE Z to n* (and optionally full R*)
    k_ori: float = 5.0
    hold_full_orientation: bool = False  # True: Log(R* R^T); False: Z-align only
    free_roll_about_n: bool = True  # when hold_full, zero roll about n*
    # Dynamics
    m_lin: float = 1.0
    m_ang: float = 0.05
    b_lin: float = 45.0
    b_ang: float = 2.5
    # Always free along n*: align never owns the unbound axis (force does).
    free_along_normal: bool = True
    # Clips (loop-local; dual-loop may clip again)
    max_linear_m_s: float = 0.04
    max_angular_deg_s: float = 12.0
    # Deadband
    lat_tol_m: float = 0.0005
    ori_tol_rad: float = 0.01


class AlignmentAdmittance:
    """
    Align loop: spring–damper so EE tool-Z line matches target point's Z-line.
    """

    def __init__(self, config: Optional[AlignmentAdmittanceConfig] = None) -> None:
        self.config = config or AlignmentAdmittanceConfig()
        self._sd = SpringDamperState()
        self._sync_gains()

    def _sync_gains(self) -> None:
        c = self.config
        self._sd.gains = SpringDamperGains(
            M=(c.m_lin, c.m_lin, c.m_lin, c.m_ang, c.m_ang, c.m_ang),
            B=(c.b_lin, c.b_lin, c.b_lin, c.b_ang, c.b_ang, c.b_ang),
            K=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )

    def reset(self) -> None:
        self._sd.reset()

    def geometric_errors(
        self,
        measured: PoseSO3,
        p_star: np.ndarray,
        R_star: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Returns e_perp, e_R, s, n_star, n_ee.
        """
        n_star = _unit(R_star[:, 2])
        n_ee = _unit(measured.R[:, 2])
        e_perp, s, _ = lateral_error_to_line(measured.p, p_star, n_star)

        if self.config.hold_full_orientation:
            e_R = orientation_error_rotvec_base(R_star, measured.R)
            if self.config.free_roll_about_n:
                e_R = e_R - np.dot(e_R, n_star) * n_star
        else:
            e_R = z_align_rotvec(n_ee, n_star)

        return e_perp, e_R, s, n_star, n_ee

    def step(
        self,
        *,
        measured: PoseSO3,
        p_star: np.ndarray,
        R_star: np.ndarray,
        n_base: np.ndarray,
        dt: float,
    ) -> Tuple[TwistCommand, Dict[str, float]]:
        """
        n_base is ignored for geometry (taken from R_star); kept for API compat.
        """
        _ = n_base
        c = self.config
        self._sync_gains()

        if not c.enable:
            self._sd.reset()
            cmd = TwistCommand()
            z = np.zeros(3)
            return cmd, self._diag(z, z, 0.0, z, z, cmd, settled=True)

        e_perp, e_R, s, n_star, n_ee = self.geometric_errors(
            measured, p_star, R_star
        )
        lat = float(np.linalg.norm(e_perp))
        ori = float(np.linalg.norm(e_R))

        if lat < c.lat_tol_m and ori < c.ori_tol_rad:
            self._sd.v *= 0.5
            if float(np.linalg.norm(self._sd.v)) < 1e-4:
                self._sd.reset()
                cmd = TwistCommand()
                return cmd, self._diag(
                    e_perp, e_R, s, n_star, n_ee, cmd, settled=True
                )

        # Lateral + orientation only — never command along n* (force owns that)
        f_lin = c.k_perp * e_perp
        f_ang = c.k_ori * e_R
        f_virt = np.concatenate([f_lin, f_ang])
        v_lin, w_rad = self._sd.step_from_force(f_virt, dt)

        # Always strip n*: keep the normal line open for the force loop
        v_lin = v_lin - np.dot(v_lin, n_star) * n_star
        self._sd.v[:3] = v_lin

        # Clip
        vn = float(np.linalg.norm(v_lin))
        if vn > c.max_linear_m_s and vn > 1e-12:
            v_lin = v_lin * (c.max_linear_m_s / vn)
            self._sd.v[:3] = v_lin
        w_deg = np.degrees(w_rad)
        wn = float(np.linalg.norm(w_deg))
        if wn > c.max_angular_deg_s and wn > 1e-12:
            w_deg = w_deg * (c.max_angular_deg_s / wn)
            self._sd.v[3:] = np.radians(w_deg)

        cmd = TwistCommand.from_vectors(v_lin, w_deg)
        return cmd, self._diag(
            e_perp, e_R, s, n_star, n_ee, cmd, settled=False
        )

    @staticmethod
    def _diag(
        e_perp: np.ndarray,
        e_R: np.ndarray,
        s: float,
        n_star: np.ndarray,
        n_ee: np.ndarray,
        cmd: TwistCommand,
        *,
        settled: bool,
    ) -> Dict[str, float]:
        # Angle between EE Z and target Z (deg)
        d = float(np.clip(np.dot(_unit(n_ee), _unit(n_star)), -1.0, 1.0))
        ang_z_deg = float(np.degrees(np.arccos(d)))
        return {
            "loop": 2.0,
            "enabled": 1.0,
            "stub": 0.0,
            "s_m": float(s),
            "s_mm": float(s) * 1e3,
            "err_lat_m": float(np.linalg.norm(e_perp)),
            "err_lat_mm": float(np.linalg.norm(e_perp) * 1e3),
            "err_ori_rad": float(np.linalg.norm(e_R)),
            "err_ori_deg": float(np.degrees(np.linalg.norm(e_R))),
            "ang_z_deg": ang_z_deg,
            "n_star_x": float(n_star[0]),
            "n_star_y": float(n_star[1]),
            "n_star_z": float(n_star[2]),
            "n_ee_x": float(n_ee[0]),
            "n_ee_y": float(n_ee[1]),
            "n_ee_z": float(n_ee[2]),
            "cmd_v_m_s": float(
                np.linalg.norm([cmd.linear_x, cmd.linear_y, cmd.linear_z])
            ),
            "cmd_w_deg_s": float(
                np.linalg.norm([cmd.angular_x, cmd.angular_y, cmd.angular_z])
            ),
            "settled": 1.0 if settled else 0.0,
        }
