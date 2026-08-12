"""
Dual-loop admittance.

  force  — force along normal n* (F_∥ → v_∥ n*)
  align  — align TCP Z-line to target point Z-line (lateral + orientation)

  v_cmd = v_force + v_align
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from geodude_teleoperation.in_contact_control.alignment_admittance import (
    AlignmentAdmittance,
    AlignmentAdmittanceConfig,
)
from geodude_teleoperation.in_contact_control.force_pid import (
    ForcePIDController,
    ForcePIDConfig,
)
from geodude_teleoperation.in_contact_control.types import PoseSO3, TwistCommand


@dataclass
class DualLoopTarget:
    """
    Target **point** with orientation (x,y,z + theta → R).

    Target line: L* = p* + s * n*,  n* = R*[:,2] (point's Z-axis).
    """

    p_star: np.ndarray
    R_star: np.ndarray

    def __post_init__(self) -> None:
        self.p_star = np.asarray(self.p_star, dtype=np.float64).reshape(3).copy()
        self.R_star = np.asarray(self.R_star, dtype=np.float64).reshape(3, 3).copy()

    @property
    def n(self) -> np.ndarray:
        d = self.R_star[:, 2].copy()
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-12:
            return np.array([0.0, 0.0, 1.0])
        return d / nrm

    @classmethod
    def from_pose(cls, pose: PoseSO3) -> "DualLoopTarget":
        return cls(pose.p.copy(), pose.R.copy())

    def copy(self) -> "DualLoopTarget":
        return DualLoopTarget(self.p_star.copy(), self.R_star.copy())

    def line_description(self) -> str:
        n = self.n
        return (
            f"L*: p*=[{self.p_star[0]:.4f},{self.p_star[1]:.4f},{self.p_star[2]:.4f}] "
            f"n*=[{n[0]:.3f},{n[1]:.3f},{n[2]:.3f}]"
        )


def _default_force() -> ForcePIDConfig:
    cfg = ForcePIDConfig()
    cfg.enable = True  # force on unbound axis (along n*)
    return cfg


def _default_align() -> AlignmentAdmittanceConfig:
    cfg = AlignmentAdmittanceConfig()
    cfg.enable = True
    cfg.hold_full_orientation = False  # Z-line align first
    cfg.free_along_normal = True
    return cfg


@dataclass
class DualLoopConfig:
    """force = PID Fe=Fd−Fs along n*; align = line-to-line geometry."""

    force: ForcePIDConfig = field(default_factory=_default_force)
    align: AlignmentAdmittanceConfig = field(default_factory=_default_align)
    max_linear_m_s: float = 0.04
    max_angular_deg_s: float = 12.0


class DualLoopController:
    """
    1. Capture target point pose → L* through p* along tool +Z.
    2. Each tick: measure TCP → L_ee along tool +Z; optional F/T.
    3. Align: L_ee → L* (lateral + orientation); free along n*.
    4. Force: Fe = Fd − Fs → PID → velocity along n* (unbound axis).
    """

    def __init__(self, config: Optional[DualLoopConfig] = None) -> None:
        self.config = config or DualLoopConfig()
        # Keep config objects shared with loop instances
        self.force = ForcePIDController(self.config.force)
        self.align = AlignmentAdmittance(self.config.align)
        self._target: Optional[DualLoopTarget] = None

    @property
    def target(self) -> Optional[DualLoopTarget]:
        return self._target

    def capture(self, pose: PoseSO3) -> DualLoopTarget:
        """Define target point (x,y,z + orientation) and its Z-axis line."""
        self._target = DualLoopTarget.from_pose(pose)
        self.force.reset()
        self.align.reset()
        return self._target

    def set_target(self, target: DualLoopTarget) -> DualLoopTarget:
        """
        Arm dual-loop from an existing DualLoopTarget (e.g. surface waypoint).

        Same effect as capture(pose): replaces the single active target and
        resets force/align integrators. Control law is unchanged.
        """
        self._target = target.copy()
        self.force.reset()
        self.align.reset()
        return self._target

    def update_tracked_target(self, p_star: np.ndarray, R_star: np.ndarray) -> DualLoopTarget:
        """Update geometric target without resetting Force/Align integrators.

        Used for tip-tracking ``p*`` / live ``n*`` while force contact is held.
        """
        if self._target is None:
            return self.capture(PoseSO3(p=p_star, R=R_star))
        self._target.p_star = np.asarray(p_star, dtype=np.float64).reshape(3).copy()
        self._target.R_star = np.asarray(R_star, dtype=np.float64).reshape(3, 3).copy()
        return self._target

    def clear(self) -> None:
        self._target = None
        self.force.reset()
        self.align.reset()

    def set_force_bias_from_samples(
        self, samples: Sequence[Sequence[float]]
    ) -> np.ndarray:
        return self.force.set_force_bias_from_samples(samples)

    def step(
        self,
        measured: PoseSO3,
        force_base: Optional[Sequence[float]] = None,
        dt: float = 0.02,
    ) -> Tuple[TwistCommand, Dict[str, float]]:
        if self._target is None:
            raise RuntimeError("dual-loop: capture a target point first")

        n = self._target.n
        # Force: PID Fe=Fd−Fs along unbound axis n* (needs wrench)
        if self.config.force.enable and force_base is not None:
            cmd_f, diag_f = self.force.step(
                n_base=n,
                force_base=force_base,
                dt=dt,
                R_ee=measured.R,
            )
        else:
            cmd_f = TwistCommand()
            diag_f = {
                "loop": 1.0,
                "Fs_N": 0.0,
                "Fs_raw_N": 0.0,
                "Fd_N": float(self.config.force.Fd_N),
                "Fe_N": 0.0,
                "F_par_N": 0.0,
                "F_par_raw_N": 0.0,
                "v_along_m_s": 0.0,
                "v_along_mm_s": 0.0,
                "v_des_mm_s": 0.0,
                "cmd_v_m_s": 0.0,
                "disabled": 1.0 if not self.config.force.enable else 2.0,
            }

        # Align always frees n*; only Force may command along the normal line.
        cmd_al, diag_al = self.align.step(
            measured=measured,
            p_star=self._target.p_star,
            R_star=self._target.R_star,
            n_base=n,
            dt=dt,
        )

        v = np.array(
            [
                cmd_f.linear_x + cmd_al.linear_x,
                cmd_f.linear_y + cmd_al.linear_y,
                cmd_f.linear_z + cmd_al.linear_z,
            ],
            dtype=np.float64,
        )
        w = np.array(
            [
                cmd_f.angular_x + cmd_al.angular_x,
                cmd_f.angular_y + cmd_al.angular_y,
                cmd_f.angular_z + cmd_al.angular_z,
            ],
            dtype=np.float64,
        )

        vmax = float(self.config.max_linear_m_s)
        wmax = float(self.config.max_angular_deg_s)
        vn = float(np.linalg.norm(v))
        if vn > vmax and vn > 1e-12:
            v = v * (vmax / vn)
        wn = float(np.linalg.norm(w))
        if wn > wmax and wn > 1e-12:
            w = w * (wmax / wn)

        cmd = TwistCommand(
            linear_x=float(v[0]),
            linear_y=float(v[1]),
            linear_z=float(v[2]),
            angular_x=float(w[0]),
            angular_y=float(w[1]),
            angular_z=float(w[2]),
        )

        s = float(np.dot(measured.p - self._target.p_star, n))
        n_ee = measured.tool_z_base
        p_ee = measured.p

        diag: Dict[str, float] = {
            "s_m": s,
            "s_mm": s * 1e3,
            "n_x": float(n[0]),
            "n_y": float(n[1]),
            "n_z": float(n[2]),
            "n_ee_x": float(n_ee[0]),
            "n_ee_y": float(n_ee[1]),
            "n_ee_z": float(n_ee[2]),
            # Live geometry for line visualization (robot base, m)
            "p_star_x": float(self._target.p_star[0]),
            "p_star_y": float(self._target.p_star[1]),
            "p_star_z": float(self._target.p_star[2]),
            "p_ee_x": float(p_ee[0]),
            "p_ee_y": float(p_ee[1]),
            "p_ee_z": float(p_ee[2]),
            "cmd_v_m_s": float(np.linalg.norm(v)),
            "cmd_w_deg_s": float(np.linalg.norm(w)),
            "force_enable": 1.0 if self.config.force.enable else 0.0,
            "align_enable": 1.0 if self.config.align.enable else 0.0,
        }
        for k, val in diag_f.items():
            diag[f"force_{k}"] = float(val) if not isinstance(val, float) else val
        for k, val in diag_al.items():
            diag[f"align_{k}"] = float(val) if not isinstance(val, float) else val
        # Flatten for UI
        diag["err_lat_mm"] = float(diag_al.get("err_lat_mm", 0.0))
        diag["err_ori_deg"] = float(diag_al.get("err_ori_deg", 0.0))
        diag["ang_z_deg"] = float(diag_al.get("ang_z_deg", 0.0))
        diag["settled"] = float(diag_al.get("settled", 0.0))
        diag["F_par_N"] = float(diag_f.get("F_par_N", diag_f.get("Fs_N", 0.0)))
        diag["F_par_raw_N"] = float(
            diag_f.get("F_par_raw_N", diag_f.get("Fs_raw_N", 0.0))
        )
        diag["Fs_N"] = float(diag_f.get("Fs_N", 0.0))
        diag["Fd_N"] = float(diag_f.get("Fd_N", self.config.force.Fd_N))
        diag["Fe_N"] = float(diag_f.get("Fe_N", 0.0))
        diag["v_along_mm_s"] = float(diag_f.get("v_along_mm_s", 0.0))
        diag["v_des_mm_s"] = float(diag_f.get("v_des_mm_s", 0.0))
        diag["gain_scale"] = float(diag_f.get("gain_scale", 1.0))
        diag["spike"] = float(diag_f.get("spike", 0.0))
        diag["osc"] = float(diag_f.get("osc", 0.0))
        diag["auto_stabilize"] = float(diag_f.get("auto_stabilize", 0.0))
        # Slice E: force tracking OK when |Fe| small and surface is loaded
        if self.config.force.enable and force_base is not None:
            fe = abs(float(diag["Fe_N"]))
            fs = abs(float(diag["Fs_N"]))
            fd = abs(float(diag["Fd_N"]))
            band = float(getattr(self.config.force, "contact_band_N", 0.5))
            min_load = max(
                float(getattr(self.config.force, "contact_min_load_N", 0.4)),
                float(getattr(self.config.force, "contact_min_load_frac", 0.25))
                * max(fd, 1e-6),
            )
            contact_ok = (fe <= band) and (fs >= min_load)
            diag["contact_ok"] = 1.0 if contact_ok else 0.0
            diag["contact_band_N"] = band
        else:
            diag["contact_ok"] = 0.0
            diag["contact_band_N"] = float(
                getattr(self.config.force, "contact_band_N", 0.5)
            )
        return cmd, diag
