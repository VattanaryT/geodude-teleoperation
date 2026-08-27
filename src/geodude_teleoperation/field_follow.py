"""Outer-loop field follow: live TCP + stick step, then project() at 15–50 Hz.

Like free-space teleop, the carrot is a small increment from the live tip so
lag cannot compound. ``compute_control_command`` stays a pose-spring tracker.
Hardware follow runs ``project()`` in a helper process so the 500 Hz speedJ
thread is not stalled by a ~7 ms query (parent waits on a queue, which
releases the GIL).
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


def _is_rotation(R, atol=0.08):
    R = np.asarray(R, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(R)):
        return False
    if abs(float(np.linalg.det(R)) - 1.0) > atol:
        return False
    return float(np.linalg.norm(R @ R.T - np.eye(3))) <= 3.0 * atol


def _is_finite_xyz(xyz):
    p = np.asarray(xyz, dtype=float).reshape(3)
    return bool(np.all(np.isfinite(p))) and float(np.linalg.norm(p)) < 20.0


def _limit_rotation(R_prev, R_new, max_rad):
    """Slerp toward ``R_new`` so one project() cannot flip tool Z (TCP-held flange jump)."""
    max_rad = float(max_rad)
    if max_rad <= 0.0:
        return np.asarray(R_new, dtype=float).reshape(3, 3)
    err = orientation_error_rad(R_prev, R_new)
    if err <= max_rad + 1e-12:
        return np.asarray(R_new, dtype=float).reshape(3, 3)
    return slerp_rotation(R_prev, R_new, float(max_rad / err))


def _snap_normal_to_surface(stick_xyz, xyz_proj, R, max_correction_m=0.02):
    """Take the along-normal component from ``vf.project()``, tangential from the stick.

    ``tick()`` used to keep ``stick_xyz`` wholesale and throw ``xyz_proj`` away,
    so the commanded point rode ``live_xyz + tangential step`` and never
    re-snapped to the mesh -- straight steps across curvature accumulate, and
    the next ``vf.project()`` is then queried off-surface, degrading the
    orientation it returns. Tangential stays with the stick so this cannot fight
    the operator; only the normal offset is corrected, and the force loop owns
    that axis anyway.
    """
    stick_xyz = np.asarray(stick_xyz, dtype=float).reshape(3)
    xyz_proj = np.asarray(xyz_proj, dtype=float).reshape(3)
    n = np.asarray(R, dtype=float).reshape(3, 3)[:, 2]
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        return stick_xyz.copy()
    n = n / n_norm
    # Bounded like _clamp_lead: a bad project at a mesh edge must not teleport
    # the command along the normal, and _clamp_lead has already run by here.
    d = float(np.clip(float((xyz_proj - stick_xyz) @ n),
                      -abs(max_correction_m), abs(max_correction_m)))
    out = stick_xyz + d * n
    return out if _is_finite_xyz(out) else stick_xyz.copy()


def _project_once(vf, xyz, R):
    proposed = _motor_at(xyz, R)
    t0 = time.perf_counter()
    target = vf.project(proposed)
    project_ms = (time.perf_counter() - t0) * 1000.0
    T = np.asarray(target.to_transformation_matrix(), dtype=float)
    return T[:3, 3].copy(), T[:3, :3].copy(), float(project_ms)


def _project_helper_main(kind, payload, in_q, out_q):
    """Child process: owns vf.project(). Parent waits on out_q (GIL released)."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
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
    """Stick linear from live TCP; ``vf.project()`` on a capped outer rate; freeze on idle."""

    def __init__(self, vf, *, hz=20.0, zero_smoothing_after_first=True,
                 project_in_process=False, mesh=None, max_lead_m=0.02,
                 max_dR_rad=0.12, max_step_m=0.0, snap_to_surface=True):
        hz = float(hz)
        if hz <= 0.0:
            raise ValueError(f"hz must be positive, got {hz}")
        self._vf = vf
        self._min_dt = 1.0 / hz
        self._max_lead_m = float(max_lead_m)
        self._max_dR_rad = float(max_dR_rad)
        self._max_step_m = float(max_step_m)
        self._snap_to_surface = bool(snap_to_surface)
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
        self._undo = None
        self.log = []

    @property
    def idle(self):
        return self._idle

    @property
    def xyz(self):
        if self._xyz is None:
            return None
        return np.array(self._xyz, copy=True)

    @property
    def pending_m(self):
        with self._lock:
            return float(np.linalg.norm(self._pending))

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

    def set_idle(self, xyz=None, *, clear_pending=True):
        with self._lock:
            if clear_pending:
                self._pending[:] = 0.0
            self._idle = True
            if xyz is not None:
                self._xyz = np.asarray(xyz, dtype=float).reshape(3).copy()
                self.last_target = _motor_at(self._xyz, self._R)

    def snap_tcp(self, xyz, R=None):
        """Rebase the carrot to ``xyz`` without blocking the next stick tick."""
        with self._lock:
            self._xyz = np.asarray(xyz, dtype=float).reshape(3).copy()
            if R is not None:
                self._R = np.asarray(R, dtype=float).reshape(3, 3).copy()
            self.last_target = _motor_at(self._xyz, self._R)
            self._idle = False

    def _drain(self):
        with self._lock:
            delta = self._pending.copy()
            n = float(np.linalg.norm(delta))
            cap = self._max_step_m
            if cap > 0.0 and n > cap:
                delta = delta * (cap / n)
                self._pending = self._pending - delta
            else:
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

    def revert_tick(self):
        """Undo the last committed follow step (bad project / mapped EE jump)."""
        if self._undo is None:
            return
        xyz, R, last = self._undo
        self._xyz = None if xyz is None else np.array(xyz, copy=True)
        self._R = np.array(R, copy=True)
        self.last_target = last
        self._undo = None
        self._pending[:] = 0.0
        self._idle = True

    def tick(self, now, proposed_motor, live_xyz=None, live_R=None, tangent=False):
        now = float(now)
        if self._xyz is None:
            T = np.asarray(proposed_motor.to_transformation_matrix(), dtype=float)
            self._xyz = T[:3, 3].copy()
            self._R = T[:3, :3].copy()
        if self._idle:
            with self._lock:
                pending_n = float(np.linalg.norm(self._pending))
            if pending_n < 1e-15:
                return FollowTick(idle=True)
            self._idle = False
        if np.isfinite(self._last_t) and (now - self._last_t) < self._min_dt:
            return FollowTick(idle=False)
        delta = self._drain()
        self._last_t = now
        if float(np.linalg.norm(delta)) < 1e-15:
            return FollowTick(idle=False)
        self._undo = (
            None if self._xyz is None else self._xyz.copy(),
            self._R.copy(),
            self.last_target,
        )
        base = self._xyz
        if live_xyz is not None and _is_finite_xyz(live_xyz):
            base = np.asarray(live_xyz, dtype=float).reshape(3)
        # Stick in the live tool frame. Slerp field R from live R so the carrot
        # cannot run ahead and dump leftover wrist error into TCP (chamfer lever).
        R_from = self._undo[1]
        if live_R is not None:
            R_live = np.asarray(live_R, dtype=float).reshape(3, 3)
            if _is_rotation(R_live):
                R_from = R_live
        step = R_from @ delta
        if tangent:
            z_hat = R_from[:, 2]
            step = step - z_hat * float(np.dot(step, z_hat))
        self._xyz = base + step
        self._clamp_lead(live_xyz)
        stick_xyz = self._xyz.copy()
        target, project_ms = self._project(stick_xyz, R_from)
        T = np.asarray(target.to_transformation_matrix(), dtype=float)
        R_raw = T[:3, :3]
        xyz_proj = T[:3, 3].copy()
        live_ok = live_xyz is None or (
            _is_finite_xyz(live_xyz)
            and float(np.linalg.norm(
                stick_xyz - np.asarray(live_xyz, dtype=float).reshape(3))) < 0.5)
        if not (
                _is_finite_xyz(stick_xyz) and _is_finite_xyz(xyz_proj)
                and _is_rotation(R_raw) and live_ok):
            self.revert_tick()
            return FollowTick(idle=True)
        R_new = _limit_rotation(R_from, R_raw, self._max_dR_rad)
        if not _is_rotation(R_new):
            self.revert_tick()
            return FollowTick(idle=True)
        self._R = R_new
        self._xyz = _snap_normal_to_surface(
            stick_xyz, xyz_proj, R_new,
            max_correction_m=self._max_lead_m) if self._snap_to_surface else stick_xyz
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


def format_follow_lag(live_tcp, cmd_tcp, R_live, R_cmd, spring_on,
                      project_ms=None, actuation=None):
    """One-line orange-vs-live lag for the hardware follow print."""
    live = np.asarray(live_tcp, dtype=float).reshape(3)
    cmd = np.asarray(cmd_tcp, dtype=float).reshape(3)
    tcp_mm = float(np.linalg.norm(cmd - live) * 1000.0)
    r_deg = float(orientation_error_rad(R_live, R_cmd) * 180.0 / np.pi)
    spring = "on" if spring_on else "off"
    line = (
        f"field follow: tcp_err={tcp_mm:.1f}mm "
        f"R_err={r_deg:.1f}deg spring={spring}")
    if project_ms is not None:
        line += f" project={float(project_ms):.2f}ms"
    if actuation:
        line += f" {actuation}"
    return line


CATCHUP_TCP_M = 0.003


def _leftover_tcp_m(follow, live_xyz):
    if live_xyz is None:
        return 0.0
    live = np.asarray(live_xyz, dtype=float).reshape(3)
    if follow.last_target is not None:
        cmd = np.asarray(
            follow.last_target.to_transformation_matrix(), dtype=float)[:3, 3]
        return float(np.linalg.norm(cmd - live))
    if follow.xyz is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(follow.xyz, dtype=float).reshape(3) - live))


def _rotation_or_none(R):
    if R is None:
        return None
    Rm = np.asarray(R, dtype=float).reshape(3, 3)
    return Rm if _is_rotation(Rm) else None


def drive_field_follow(follow, controller, now, proposed_motor, live_R=None,
                       err_rad=0.05, map_target=None, live_xyz=None,
                       live_ee_xyz=None, max_ee_jump_m=0.08, stick_quiet=True,
                       tangent=False):
    """Apply a follow tick to the pose spring.

    Stick tick: live TCP + body linear, field R slerped from live R.
    Quiet stick freezes (clears pending) so leftover R cannot keep pushing.
    Stick-live leftover snaps xyz; commanded R is live so the wrist cannot run ahead.
    """
    del err_rad, max_ee_jump_m
    R_live = _rotation_or_none(live_R)
    tick = follow.tick(
        now, proposed_motor, live_xyz=live_xyz, live_R=R_live, tangent=tangent)
    target = tick.target
    if target is not None and map_target is not None:
        target = map_target(target)
    if target is not None:
        if live_ee_xyz is not None:
            T = np.asarray(target.to_transformation_matrix(), dtype=float)
            jump = float(np.linalg.norm(
                T[:3, 3] - np.asarray(live_ee_xyz, dtype=float).reshape(3)))
            # Finite EE lag is the chamfer lever from field R vs live wrist.
            # Skipping it froze speedJ and the stick could not recover.
            if not np.isfinite(jump):
                follow.revert_tick()
                if live_xyz is not None:
                    follow.set_idle(live_xyz)
                controller.set_input_active(False)
                print(
                    "\033[93mfield follow: skip set_target, non-finite EE jump; "
                    "follow state reverted. Lift off, then continue.\033[0m",
                    flush=True,
                )
                return tick
        controller.get_objective().set_target(target)
        controller.set_input_active(True)
        return tick
    leftover = _leftover_tcp_m(follow, live_xyz)
    max_lead = float(getattr(follow, "_max_lead_m", 0.02))
    if stick_quiet:
        if R_live is not None:
            follow._R = R_live
        if live_xyz is not None:
            follow.set_idle(live_xyz, clear_pending=True)
        else:
            follow.set_idle(clear_pending=True)
        if follow.last_target is not None:
            frozen = follow.last_target
            if map_target is not None:
                frozen = map_target(frozen)
            controller.get_objective().set_target(frozen)
        controller.set_input_active(False)
        return tick
    R_hold = R_live if R_live is not None else follow._R
    if leftover > max(max_lead, CATCHUP_TCP_M) + 1e-12:
        if live_xyz is not None:
            follow.snap_tcp(live_xyz, R_hold)
        if follow.last_target is not None:
            frozen = follow.last_target
            if map_target is not None:
                frozen = map_target(frozen)
            controller.get_objective().set_target(frozen)
        controller.set_input_active(True)
        return tick
    if leftover > CATCHUP_TCP_M:
        if follow.xyz is not None:
            follow.last_target = _motor_at(follow.xyz, R_hold)
            follow._R = np.array(R_hold, copy=True)
        if follow.last_target is not None:
            frozen = follow.last_target
            if map_target is not None:
                frozen = map_target(frozen)
            controller.get_objective().set_target(frozen)
        controller.set_input_active(True)
        return tick
    if follow.last_target is not None:
        if follow.xyz is not None:
            follow.last_target = _motor_at(follow.xyz, R_hold)
            follow._R = np.array(R_hold, copy=True)
        frozen = follow.last_target
        if map_target is not None:
            frozen = map_target(frozen)
        controller.get_objective().set_target(frozen)
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
