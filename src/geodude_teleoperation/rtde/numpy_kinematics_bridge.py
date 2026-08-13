"""RTDE live kinematics backup: vendored UR5e FK/J + numpy SE3 (no gafropy)."""
from __future__ import annotations

import sys
from typing import Sequence

import numpy as np

from geodude_teleoperation.teleop.controller import _adaptive_linear_scale, qdot_from_error
from geodude_teleoperation.teleop.line_tracking import (
    LineTrackingConfig,
    apply_line_tracking_pose_error,
    combine_normal_feedforward,
    field_normal_ff_world,
    qdot_from_task_linear,
    split_field_linear_tangent_normal,
    vf_R_field,
    vf_signed_d,
)
from geodude_teleoperation.se3 import (
    NumpyTwist,
    body_twist_increment,
    map_jacobian_to_point,
    matrix_from_motor,
    motor_from_matrix,
    pose_error_current_minus_ref,
    tcp_pose_to_matrix,
)
from geodude_teleoperation.rtde.ur5e_kinematics import compute_jacobian, forward_kinematics

_TCP_OFFSET_WARN_M = 0.5


class NumpyUrKinematicsBridge:
    """FK/Jacobian backup using pendant TCP for pose and TCP-frame Jacobian."""

    backend_name = "numpy"

    def __init__(self, line_tracking: LineTrackingConfig | None = None):
        self._q = np.zeros(6, dtype=float)
        self._tcp_pose = np.zeros(6, dtype=float)
        self._T_ref: np.ndarray | None = None
        self._tcp_offset_warned = False
        self.line_tracking = line_tracking or LineTrackingConfig()
        self._pending_normal_ff_world = np.zeros(3, dtype=float)
        self._extra_normal_ff_world = np.zeros(3, dtype=float)
        # Diagnostics for motion-trace / wobble isolation.
        self.last_scale_factor: float = 1.0
        self.last_distance_m: float = 0.0
        self.last_twist_linear_field: np.ndarray | None = None
        self.last_cmd_dir_world: np.ndarray | None = None
        self.last_control_diag: dict = {}
        self.last_key: str = ""
        self.last_twist_coeffs: np.ndarray | None = None

    @classmethod
    def try_create(
        cls, line_tracking: LineTrackingConfig | None = None
    ) -> "NumpyUrKinematicsBridge | None":
        try:
            # Smoke FK so import/init failures surface here.
            forward_kinematics(np.zeros(6))
        except Exception:
            return None
        return cls(line_tracking=line_tracking)

    def shutdown(self) -> None:
        pass

    def set_line_tracking(self, cfg: LineTrackingConfig) -> None:
        self.line_tracking = cfg

    def set_extra_normal_ff(self, v_world: Sequence[float] | None) -> None:
        """Phase 3 tactile (or other) normal feedforward, world frame."""
        if v_world is None:
            self._extra_normal_ff_world[:] = 0.0
        else:
            self._extra_normal_ff_world = np.asarray(v_world, dtype=float).reshape(3)

    def sync_joints(self, q_arm: Sequence[float]) -> None:
        self._q = np.asarray(q_arm, dtype=float).reshape(6)

    def set_tcp_pose(self, tcp_pose: Sequence[float]) -> None:
        self._tcp_pose = np.asarray(tcp_pose, dtype=float).reshape(6)

    def ensure_reference_from_tcp(self) -> None:
        if self._T_ref is None:
            self._T_ref = tcp_pose_to_matrix(self._tcp_pose)

    def update_command(
        self,
        twist,
        *,
        vf=None,
        min_scale: float = 1.0,
        max_scale: float = 10.0,
        key: str | None = None,
    ) -> None:
        """Compose keyboard twist onto TCP; with VF, interpret in ``R_field``.

        Matches ``TeleoperationController.update_command`` when a fixture is set:
        input twist is in the fixture / field frame (WASD = field x/y, …), composed
        on the projected EE (field orientation at current TCP position), then
        re-projected so orientation stays ``R_field_base``.

        With Phase 1.5 line tracking: only **tangent** linear components update the
        pose reference; field-z becomes normal feedforward (does not stiffen depth).
        """
        if isinstance(twist, NumpyTwist):
            tw = twist
        elif hasattr(twist, "get_angular") and hasattr(twist, "get_linear"):
            tw = NumpyTwist(twist.get_angular(), twist.get_linear())
        else:
            tw = NumpyTwist.from_array(twist)

        if key:
            self.last_key = str(key)
        self.last_twist_coeffs = tw.to_array()

        T_ee = tcp_pose_to_matrix(self._tcp_pose)
        lin = tw.get_linear()
        self.last_twist_linear_field = lin.copy()
        self._pending_normal_ff_world[:] = 0.0
        cfg = self.line_tracking
        if vf is not None:
            dist = float(vf.get_distance(T_ee[:3, 3]))
            factor = _adaptive_linear_scale(dist, min_scale, max_scale)
            self.last_distance_m = dist
            self.last_scale_factor = float(factor)
            linear_scale = 0.25 * factor
            R_field = np.asarray(
                vf.motion_basis()
                if hasattr(vf, "motion_basis")
                else vf.status["R_field_base"],
                dtype=float,
            ).reshape(3, 3)
            lin_for_pose = lin
            if cfg.enabled:
                lin_t, lin_n = split_field_linear_tangent_normal(lin)
                lin_for_pose = lin_t
                if cfg.keyboard_normal_ff and abs(lin_n) > 1e-12:
                    self._pending_normal_ff_world = field_normal_ff_world(
                        lin_n, R_field, scale=linear_scale
                    )
            # Seed / keep a pose reference. Pure linear WASD must NOT snap
            # orientation to R_field — that caused huge rot error / wild q̇ on
            # a single keypress when the tool was misaligned with the field.
            self.ensure_reference_from_tcp()
            assert self._T_ref is not None
            T_ref0 = np.asarray(self._T_ref, dtype=float).reshape(4, 4).copy()
            dp_world = R_field @ (linear_scale * lin_for_pose)
            T_new = T_ref0.copy()
            T_new[:3, 3] = T_ref0[:3, 3] + dp_world
            ang = tw.get_angular()
            if float(np.linalg.norm(ang)) > 1e-12:
                # Angular keys: interpret ω in field frame, then lock R → R_field.
                T_field = np.eye(4)
                T_field[:3, :3] = R_field
                T_field[:3, 3] = T_new[:3, 3]
                T_new = body_twist_increment(
                    T_field, NumpyTwist(ang, np.zeros(3, dtype=float))
                )
                T_new = matrix_from_motor(vf.project(motor_from_matrix(T_new)))
            # World-frame commanded linear direction (unit), for t̂1/t̂2 alignment.
            v_field = linear_scale * lin_for_pose
            n = float(np.linalg.norm(v_field))
            self.last_cmd_dir_world = (
                (R_field @ v_field) / n if n > 1e-12 else None
            )
        else:
            self.last_distance_m = 0.0
            self.last_scale_factor = 1.0
            tw_scaled = NumpyTwist(tw.get_angular(), lin)
            T_new = body_twist_increment(T_ee, tw_scaled)
            n = float(np.linalg.norm(lin))
            self.last_cmd_dir_world = (
                (T_ee[:3, :3] @ lin) / n if n > 1e-12 else None
            )
        self._T_ref = T_new

    def compute_qdot(
        self,
        *,
        gain: float,
        damping: float,
        max_joint_velocity: float,
        vf=None,
        max_linear_speed: float | None = None,
        max_angular_speed: float | None = None,
    ) -> np.ndarray:
        self.ensure_reference_from_tcp()
        assert self._T_ref is not None

        T_flange, _ = forward_kinematics(self._q)
        T_tcp = tcp_pose_to_matrix(self._tcp_pose)
        t_offset = np.linalg.norm(
            (np.linalg.inv(T_flange) @ T_tcp)[:3, 3]
        )
        if t_offset > _TCP_OFFSET_WARN_M and not self._tcp_offset_warned:
            print(
                f"warning: FK flange→RTDE TCP translation {t_offset:.3f} m "
                f"(> {_TCP_OFFSET_WARN_M}); check pendant TCP / arm model",
                file=sys.stderr,
            )
            self._tcp_offset_warned = True

        J_flange = compute_jacobian(self._q)
        J_tcp = map_jacobian_to_point(J_flange, T_flange[:3, 3], T_tcp[:3, 3])
        err = pose_error_current_minus_ref(T_tcp, self._T_ref)

        cfg = self.line_tracking
        R = vf_R_field(vf) if vf is not None else None
        if cfg.enabled and R is not None:
            err = apply_line_tracking_pose_error(err, R, enabled=True)

        # Cap pose error so gain * err cannot exceed configured task speeds.
        # ``<= 0`` means kill that axis (Enter lock: no translation chase).
        g = max(float(gain), 1e-6)
        if max_linear_speed is not None:
            v_max = float(max_linear_speed)
            if v_max <= 0.0:
                err[:3] = 0.0
            else:
                e_max = v_max / g
                n = float(np.linalg.norm(err[:3]))
                if n > e_max:
                    err[:3] *= e_max / n
        if max_angular_speed is not None:
            w_max = float(max_angular_speed)
            if w_max <= 0.0:
                err[3:6] = 0.0
            else:
                e_max = w_max / g
                n = float(np.linalg.norm(err[3:6]))
                if n > e_max:
                    err[3:6] *= e_max / n

        qdot, diag = qdot_from_error(
            err,
            J_tcp,
            gain=gain,
            damping=damping,
            max_joint_velocity=max_joint_velocity,
            return_diag=True,
        )

        v_ff = np.zeros(3, dtype=float)
        if cfg.enabled and R is not None:
            v_ff = combine_normal_feedforward(
                cfg=cfg,
                signed_d=vf_signed_d(vf),
                R_field=R,
                keyboard_v_world=self._pending_normal_ff_world,
            )
            v_ff = v_ff + self._extra_normal_ff_world
            qdot_ff = qdot_from_task_linear(
                v_ff,
                J_tcp,
                damping=damping,
                max_joint_velocity=max_joint_velocity,
            )
            qdot = qdot + qdot_ff
            speed = float(np.linalg.norm(qdot))
            if speed > float(max_joint_velocity):
                qdot = qdot * (float(max_joint_velocity) / speed)
            diag["line_tracking"] = True
            diag["standoff_vn"] = float(np.linalg.norm(v_ff))
        else:
            diag["line_tracking"] = False

        # One-shot keyboard normal FF; stand-off re-applied every tick via vf.
        self._pending_normal_ff_world[:] = 0.0
        self.last_control_diag = diag
        return qdot
