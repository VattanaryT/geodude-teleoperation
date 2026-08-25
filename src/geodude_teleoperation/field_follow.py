"""Outer-loop field follow: accumulate stick linear, project() at 15–50 Hz.

``compute_control_command`` stays a pose-spring tracker. ``project()`` is not
called from ``update_command`` or the RTDE thread.
"""
from __future__ import annotations

import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from geodude_teleoperation.ee_viz import motor_from_matrix, orientation_error_rad

RTDE_PERIOD_MS = 2.0


class FollowTick:
    __slots__ = (
        "target", "project_ms", "quat_wxyz", "idle", "would_overrun_rtde")

    def __init__(self, *, target=None, project_ms=None, quat_wxyz=None,
                 idle=False, would_overrun_rtde=False):
        self.target = target
        self.project_ms = project_ms
        self.quat_wxyz = quat_wxyz
        self.idle = bool(idle)
        self.would_overrun_rtde = bool(would_overrun_rtde)


def _quat_wxyz_from_motor(motor):
    R = np.asarray(motor.to_transformation_matrix(), dtype=float)[:3, :3]
    xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)


def _motor_at(xyz, R):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return motor_from_matrix(T)


class FieldFollow:
    """Stick linear → xyz; ``vf.project()`` on a capped outer rate; freeze on idle."""

    def __init__(self, vf, *, hz=20.0, zero_smoothing_after_first=True):
        hz = float(hz)
        if hz <= 0.0:
            raise ValueError(f"hz must be positive, got {hz}")
        self._vf = vf
        self._min_dt = 1.0 / hz
        self._zero_smoothing_after_first = bool(zero_smoothing_after_first)
        self._lock = threading.Lock()
        self._pending = np.zeros(3, dtype=float)
        self._idle = True
        self._xyz = None
        self._R = np.eye(3)
        self._last_t = float("-inf")
        self.last_target = None
        self.log = []

    @property
    def idle(self):
        return self._idle

    @property
    def xyz(self):
        if self._xyz is None:
            return None
        return np.array(self._xyz, copy=True)

    def seed(self, xyz, R=None):
        with self._lock:
            self._xyz = np.asarray(xyz, dtype=float).reshape(3).copy()
            self._R = (
                np.eye(3) if R is None
                else np.asarray(R, dtype=float).reshape(3, 3).copy())
            self._pending[:] = 0.0
            self._idle = True
            self._last_t = float("-inf")

    def add_body_linear(self, linear):
        with self._lock:
            self._pending = self._pending + np.asarray(linear, dtype=float).reshape(3)
            self._idle = False

    def set_idle(self):
        with self._lock:
            self._pending[:] = 0.0
            self._idle = True

    def _drain(self):
        with self._lock:
            delta = self._pending.copy()
            self._pending[:] = 0.0
            return delta

    def tick(self, now, proposed_motor):
        now = float(now)
        if self._xyz is None:
            T = np.asarray(proposed_motor.to_transformation_matrix(), dtype=float)
            self._xyz = T[:3, 3].copy()
            self._R = T[:3, :3].copy()
        if self._idle:
            return FollowTick(idle=True)
        if np.isfinite(self._last_t) and (now - self._last_t) < self._min_dt:
            return FollowTick(idle=False)
        delta = self._drain()
        self._last_t = now
        if float(np.linalg.norm(delta)) < 1e-15:
            return FollowTick(idle=False)
        self._xyz = self._xyz + self._R @ delta
        proposed = _motor_at(self._xyz, self._R)
        t0 = time.perf_counter()
        target = self._vf.project(proposed)
        project_ms = (time.perf_counter() - t0) * 1000.0
        if self._zero_smoothing_after_first:
            self._vf._smoothing = 0.0
            self._zero_smoothing_after_first = False
        T = np.asarray(target.to_transformation_matrix(), dtype=float)
        self._R = T[:3, :3].copy()
        self._xyz = T[:3, 3].copy()
        self.last_target = target
        quat = _quat_wxyz_from_motor(target)
        overrun = project_ms > RTDE_PERIOD_MS
        self.log.append({
            "t": now,
            "project_ms": float(project_ms),
            "quat_wxyz": quat,
            "xyz": self._xyz.copy(),
            "would_overrun_rtde": overrun,
        })
        return FollowTick(
            target=target, project_ms=float(project_ms), quat_wxyz=quat,
            idle=False, would_overrun_rtde=overrun)


def drive_field_follow(follow, controller, now, proposed_motor, live_R=None,
                       err_rad=0.05, map_target=None):
    """Apply a follow tick to the pose spring. Idle freezes the last field target."""
    tick = follow.tick(now, proposed_motor)
    target = tick.target
    if target is not None and map_target is not None:
        target = map_target(target)
    if target is not None:
        controller.get_objective().set_target(target)
        controller.set_input_active(True)
        return tick
    if follow.idle:
        if follow.last_target is None:
            controller.set_input_active(False)
            return tick
        if live_R is not None:
            T = np.asarray(
                follow.last_target.to_transformation_matrix(), dtype=float)
            err = orientation_error_rad(live_R, T[:3, :3])
            controller.set_input_active(err > float(err_rad))
        else:
            controller.set_input_active(True)
        return tick
    if follow.last_target is not None:
        controller.set_input_active(True)
    return tick


def field_tcp_to_ee(T_tcp, *, T_flange_hw, T_flange_viz, T_ee_hw):
    """Map a field TCP pose in the object/viz world to the hardware EE matrix.

    Same flange as the orange ghost (``ghost_flange_world``): mesh/flange +Z is
    the field approach. Do not use ``flange_world_from_tcp`` — that composes the
    pendant TCP rotvec and yanks hardware off the sim preview.
    """
    from geodude_teleoperation.ee_viz import ghost_flange_world

    T_tcp = np.asarray(T_tcp, dtype=float).reshape(4, 4)
    T_flange_viz = np.asarray(T_flange_viz, dtype=float).reshape(4, 4)
    T_flange_hw = np.asarray(T_flange_hw, dtype=float).reshape(4, 4)
    offset = T_flange_viz[:3, 3] - T_flange_hw[:3, 3]
    p_tcp_hw = T_tcp[:3, 3] - offset
    T_flange_hw_tgt = ghost_flange_world(
        T_flange_hw, T_tcp[:3, :3], p_tcp_hw, 1.0)
    T_ee_from_flange = np.linalg.inv(T_flange_hw) @ np.asarray(
        T_ee_hw, dtype=float).reshape(4, 4)
    return T_flange_hw_tgt @ T_ee_from_flange
