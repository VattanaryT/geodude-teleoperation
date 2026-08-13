"""Live Enter → lock TCP orientation to R_field with a startup speed/accel ramp.

Dry-run already snaps ``T_ref`` instantly (no ``speedJ``). Live uses these helpers
so the same lock does not dump saturated ``q̇`` into RTDE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from geodude_teleoperation.se3 import NumpyTwist


@dataclass(frozen=True)
class MotionLimits:
    max_joint_velocity: float
    rtde_accel: float
    max_linear_speed: float
    max_angular_speed: float


def orientation_error_deg(R_a, R_b) -> float:
    """Geodesic angle (deg) between two rotation matrices."""
    Ra = np.asarray(R_a, dtype=float).reshape(3, 3)
    Rb = np.asarray(R_b, dtype=float).reshape(3, 3)
    R_err = Ra @ Rb.T
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(R_err).as_rotvec())))


def blend_startup_motion_limits(
    *,
    t_since_enable: float,
    ramp_s: float,
    startup: MotionLimits,
    general: MotionLimits,
    locking: bool = False,
) -> MotionLimits:
    """Lerp joint/accel/angular from startup → general.

    While ``locking``, ``max_linear_speed`` is 0 so Enter does not chase a
    stale world position (in-place rotation only). ``ramp_s <= 0`` → general
    immediately (still zero linear if locking).
    """
    if float(ramp_s) <= 0.0:
        out = general
    else:
        alpha = min(1.0, max(0.0, float(t_since_enable) / float(ramp_s)))

        def _lerp(a: float, b: float) -> float:
            return float(a) + alpha * (float(b) - float(a))

        out = MotionLimits(
            max_joint_velocity=_lerp(
                startup.max_joint_velocity, general.max_joint_velocity
            ),
            rtde_accel=_lerp(startup.rtde_accel, general.rtde_accel),
            max_linear_speed=float(general.max_linear_speed),
            max_angular_speed=_lerp(
                startup.max_angular_speed, general.max_angular_speed
            ),
        )
    if locking:
        return MotionLimits(
            max_joint_velocity=out.max_joint_velocity,
            rtde_accel=out.rtde_accel,
            max_linear_speed=0.0,
            max_angular_speed=out.max_angular_speed,
        )
    return out


class EnableOrientLock:
    """Latch: after Enter, rotate in place toward a frozen ``R_star``.

    After settle, hold still until a second Enter approves teleop. VF queries
    stay frozen until ``f`` again (pose-hold off unlocks WoS).
    """

    def __init__(self) -> None:
        self.enabled_at: float | None = None
        self.settled: bool = False
        self.R_star: np.ndarray | None = None
        self.approved: bool = False
        self.vf_frozen: bool = True
        self.pose_hold: bool = False

    def on_enable(self, now: float, R_star=None) -> None:
        self.enabled_at = float(now)
        self.settled = False
        self.approved = False
        self.vf_frozen = True
        self.pose_hold = False
        self.R_star = None
        if R_star is not None:
            self.capture_R_star(R_star)

    def approve(self) -> None:
        """Second Enter: allow teleop; keep VF frozen. ``f`` holds pose."""
        self.approved = True
        self.vf_frozen = True
        self.pose_hold = False

    @property
    def awaiting_approve(self) -> bool:
        return (
            self.enabled_at is not None and self.settled and not self.approved
        )

    def motion_hold(self) -> bool:
        """True from first Enter until second-Enter approve (no teleop)."""
        if self.enabled_at is None or self.approved:
            return False
        return True

    def toggle_pose_hold(self) -> bool:
        """``f``: hold pose + freeze VF. ``f`` again: resume teleop + unlock VF."""
        self.pose_hold = not self.pose_hold
        self.vf_frozen = bool(self.pose_hold)
        return self.pose_hold

    def toggle_vf_freeze(self) -> bool:
        self.vf_frozen = not self.vf_frozen
        return self.vf_frozen

    def capture_R_star(self, R) -> np.ndarray | None:
        """Freeze the first valid lock target; later VF updates are ignored."""
        if self.R_star is None and R is not None:
            self.R_star = np.asarray(R, dtype=float).reshape(3, 3).copy()
        return self.R_star

    def t_since_enable(self, now: float) -> float:
        if self.enabled_at is None:
            return 0.0
        return max(0.0, float(now) - self.enabled_at)

    def locking(self, ang_err_deg: float, tol_deg: float) -> bool:
        if self.enabled_at is None or self.settled:
            return False
        if float(ang_err_deg) <= float(tol_deg):
            self.settled = True
            return False
        return True


def apply_orient_lock_reference(kin, tcp_pose, R_star) -> bool:
    """``T_ref`` = current TCP position + frozen ``R_star`` (rotate in place)."""
    from geodude_teleoperation.se3 import tcp_pose_to_matrix

    if R_star is None:
        return False
    T = tcp_pose_to_matrix(tcp_pose)
    T[:3, :3] = np.asarray(R_star, dtype=float).reshape(3, 3)
    kin._T_ref = T
    return True


def recapture_pose_reference(kin, tcp_pose) -> bool:
    """``T_ref`` = live TCP (zero pose error). Approve / ``f`` hold must not chase R*."""
    from geodude_teleoperation.se3 import tcp_pose_to_matrix

    if kin is None:
        return False
    kin._T_ref = tcp_pose_to_matrix(tcp_pose)
    return True


def twist_linear_only(twist) -> NumpyTwist:
    """Drop angular components so Enter-lock is not fought by SpaceMouse/WASD."""
    if isinstance(twist, NumpyTwist):
        return NumpyTwist(np.zeros(3, dtype=float), twist.get_linear())
    if hasattr(twist, "get_linear"):
        return NumpyTwist(
            np.zeros(3, dtype=float),
            np.asarray(twist.get_linear(), dtype=float).reshape(3),
        )
    arr = np.asarray(twist, dtype=float).reshape(6)
    return NumpyTwist(np.zeros(3, dtype=float), arr[3:6])
