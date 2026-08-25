"""Outer-loop field follow: accumulate stick linear, project() at 15–50 Hz.

``compute_control_command`` stays a pose-spring tracker. Hardware follow runs
``project()`` in a helper process so the 500 Hz speedJ thread is not stalled
by a ~7 ms query (parent waits on a queue, which releases the GIL).
"""
from __future__ import annotations

import multiprocessing as mp
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from geodude_teleoperation.ee_viz import (
    motor_from_matrix,
    orientation_error_rad,
    slerp_rotation,
)

RTDE_PERIOD_MS = 2.0
_HELPER_READY = "ready"


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


def _limit_rotation(R_prev, R_new, max_rad):
    """Slerp toward ``R_new`` so one project() cannot flip tool Z (TCP-held flange jump)."""
    max_rad = float(max_rad)
    if max_rad <= 0.0:
        return np.asarray(R_new, dtype=float).reshape(3, 3)
    err = orientation_error_rad(R_prev, R_new)
    if err <= max_rad + 1e-12:
        return np.asarray(R_new, dtype=float).reshape(3, 3)
    return slerp_rotation(R_prev, R_new, float(max_rad / err))


def _project_once(vf, xyz, R):
    proposed = _motor_at(xyz, R)
    t0 = time.perf_counter()
    target = vf.project(proposed)
    project_ms = (time.perf_counter() - t0) * 1000.0
    T = np.asarray(target.to_transformation_matrix(), dtype=float)
    return T[:3, 3].copy(), T[:3, :3].copy(), float(project_ms)


def _project_helper_main(kind, payload, in_q, out_q):
    """Child process: owns vf.project(). Parent waits on out_q (GIL released)."""
    if kind == "vf":
        vf = payload
    else:
        import trimesh
        from geodude_teleoperation.virtual_fixture import OrientationField

        vertices, faces, normals, kwargs = payload
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        vf = OrientationField(mesh, **(kwargs or {}))
    out_q.put(_HELPER_READY)
    while True:
        msg = in_q.get()
        if msg is None:
            break
        xyz, R = msg
        out_q.put(_project_once(vf, xyz, R))


class FieldFollow:
    """Stick linear → xyz; ``vf.project()`` on a capped outer rate; freeze on idle."""

    def __init__(self, vf, *, hz=20.0, zero_smoothing_after_first=True,
                 project_in_process=False, mesh=None, max_lead_m=0.02,
                 max_dR_rad=0.12):
        hz = float(hz)
        if hz <= 0.0:
            raise ValueError(f"hz must be positive, got {hz}")
        self._vf = vf
        self._min_dt = 1.0 / hz
        self._max_lead_m = float(max_lead_m)
        self._max_dR_rad = float(max_dR_rad)
        self._zero_smoothing_after_first = bool(zero_smoothing_after_first)
        self._proc = None
        self._in_q = None
        self._out_q = None
        self._lock = threading.Lock()
        if project_in_process:
            self._start_helper(mesh)
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

    def set_idle(self, xyz=None):
        with self._lock:
            self._pending[:] = 0.0
            self._idle = True
            if xyz is not None:
                self._xyz = np.asarray(xyz, dtype=float).reshape(3).copy()
                self.last_target = _motor_at(self._xyz, self._R)

    def _drain(self):
        with self._lock:
            delta = self._pending.copy()
            self._pending[:] = 0.0
            return delta

    def _start_helper(self, mesh):
        ctx = mp.get_context("spawn")
        self._in_q = ctx.Queue(maxsize=1)
        self._out_q = ctx.Queue(maxsize=1)
        if mesh is not None:
            kind = "mesh"
            payload = (
                np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.faces, dtype=np.int64),
                np.asarray(mesh.vertex_normals, dtype=np.float64),
                {},
            )
            print("field follow: starting project() helper process (rebuilds field)...",
                  flush=True)
        else:
            kind = "vf"
            payload = self._vf
        self._proc = ctx.Process(
            target=_project_helper_main,
            args=(kind, payload, self._in_q, self._out_q),
            daemon=True,
        )
        self._proc.start()
        ready = self._out_q.get(timeout=180.0)
        if ready != _HELPER_READY:
            raise RuntimeError(f"project helper failed handshake: {ready!r}")
        print("field follow: project() helper ready (speedJ will not wait on the query)",
              flush=True)

    def close(self):
        if self._proc is None:
            return
        try:
            self._in_q.put(None, timeout=1.0)
        except Exception:
            pass
        self._proc.join(timeout=2.0)
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc = None

    def _project(self, xyz, R):
        if self._proc is not None:
            self._in_q.put((
                np.asarray(xyz, dtype=float).reshape(3).copy(),
                np.asarray(R, dtype=float).reshape(3, 3).copy(),
            ))
            xyz_p, R_p, project_ms = self._out_q.get(timeout=2.0)
            return _motor_at(xyz_p, R_p), float(project_ms)
        t0 = time.perf_counter()
        target = self._vf.project(_motor_at(xyz, R))
        project_ms = (time.perf_counter() - t0) * 1000.0
        if self._zero_smoothing_after_first and hasattr(self._vf, "_smoothing"):
            self._vf._smoothing = 0.0
            self._zero_smoothing_after_first = False
        return target, float(project_ms)

    def _clamp_lead(self, live_xyz):
        if live_xyz is None or self._xyz is None or self._max_lead_m <= 0.0:
            return
        live = np.asarray(live_xyz, dtype=float).reshape(3)
        lead = self._xyz - live
        dist = float(np.linalg.norm(lead))
        if dist > self._max_lead_m:
            self._xyz = live + lead * (self._max_lead_m / dist)

    def tick(self, now, proposed_motor, live_xyz=None):
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
        self._clamp_lead(live_xyz)
        target, project_ms = self._project(self._xyz, self._R)
        T = np.asarray(target.to_transformation_matrix(), dtype=float)
        self._R = _limit_rotation(self._R, T[:3, :3], self._max_dR_rad)
        self._xyz = T[:3, 3].copy()
        target = _motor_at(self._xyz, self._R)
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
                       err_rad=0.05, map_target=None, live_xyz=None,
                       live_ee_xyz=None, max_ee_jump_m=0.08):
    """Apply a follow tick to the pose spring.

    Idle stops the tracker immediately and rebases the carrot to ``live_xyz``
    so a far-ahead field target cannot keep speedJ running after hands-off.
    """
    del live_R, err_rad
    tick = follow.tick(now, proposed_motor, live_xyz=live_xyz)
    target = tick.target
    if target is not None and map_target is not None:
        target = map_target(target)
    if target is not None:
        if live_ee_xyz is not None:
            T = np.asarray(target.to_transformation_matrix(), dtype=float)
            jump = float(np.linalg.norm(T[:3, 3] - np.asarray(live_ee_xyz, dtype=float).reshape(3)))
            if jump > float(max_ee_jump_m):
                controller.set_input_active(False)
                print(
                    f"field follow: skip set_target, EE jump {jump:.3f}m "
                    f"(cap {float(max_ee_jump_m):.3f}m)",
                    flush=True,
                )
                return tick
        controller.get_objective().set_target(target)
        controller.set_input_active(True)
        return tick
    if follow.idle:
        if live_xyz is not None:
            follow.set_idle(live_xyz)
        if follow.last_target is not None:
            frozen = follow.last_target
            if map_target is not None:
                frozen = map_target(frozen)
            controller.get_objective().set_target(frozen)
        controller.set_input_active(False)
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
