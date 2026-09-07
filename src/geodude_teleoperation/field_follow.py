"""Outer-loop field follow: live TCP + field-frame stick, full field R carrot.

Like the orange viser ghost (``ghost_flange_world(..., alpha=1)``): stick linear
is expressed in ``approach_basis_at`` / field axes, and the commanded orientation
is the field at the tip — not a 0.02 rad crawl from the live wrist. A live-based
``max_dR`` cap used to pin ``R_err`` at ~1.1° forever. Rate-limiting (when
``max_dR_rad>0``) is field→field only, to blunt a single project() flip.
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


def _ramp_step(cmd, goal, vel, dt):
    """Constant-velocity step of the commanded xyz toward the goal.

    Deliberately *not* an exponential settle like ``_approach_rotation``: a
    first-order lag toward a goal that keeps moving every ``--follow-hz``
    commit never catches up, so it adds a steady-state tracking lag on top of
    the hop it was meant to remove. ``vel`` is set once per commit to
    ``(goal - cmd_at_commit_time) / dt_tick`` -- the constant velocity that
    exactly closes the gap by the time the next commit is expected -- and each
    tick (committed or not) is charged ``vel * dt`` of it. Same total
    displacement as the old snap, spread evenly over the gap instead of
    dumped at its start: zero added lag, no periodic step into the control
    loop. This is also what speedJ has always done for free -- the carrot
    there already *is* a velocity command, integrated by the position servo;
    this reproduces the same shape for the position spring under the wrench
    law. Clamped to stop (not overshoot) the moment the ramp reaches ``goal``,
    since ``dt_tick`` is only an estimate of the next commit's timing.
    """
    cmd = np.asarray(cmd, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    vel = np.asarray(vel, dtype=float).reshape(3)
    if not np.isfinite(dt) or dt <= 0.0 or not np.any(vel):
        return cmd.copy(), vel.copy()
    remaining_before = goal - cmd
    new_cmd = cmd + vel * dt
    remaining_after = goal - new_cmd
    if float(np.dot(remaining_before, remaining_after)) <= 0.0:
        # Reached or passed the goal this step (dt_tick was only an estimate
        # of the real gap) -- stop exactly at the goal, don't overshoot back
        # past it next tick.
        return goal.copy(), np.zeros(3)
    return (new_cmd if _is_finite_xyz(new_cmd) else goal.copy()), vel.copy()


def _ramp_velocity(cmd, goal, dt_tick):
    """Velocity that closes the ``cmd`` -> ``goal`` gap in ``dt_tick``."""
    cmd = np.asarray(cmd, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    if not np.isfinite(dt_tick) or dt_tick <= 1e-9:
        return np.zeros(3)
    return (goal - cmd) / dt_tick


def _approach_rotation(R_cmd, R_goal, dt, tau_s):
    """First-order approach of the commanded R toward the field goal.

    ``tau_s <= 0`` snaps, which is the pre-fix behavior. Otherwise the command
    moves a fraction ``dt / tau_s`` of the remaining angle per tick -- an
    exponential approach, so it converges (no sawtooth on the spring gate, no
    permanent lag) and the one-tick step is bounded by that fraction however far
    off-field the wrist is. Deliberately a rate on the *command's own history*,
    not a cap measured against live R: the live-based cap is what pinned R_err
    at ~1.1 deg forever and was removed in 2eceb0f.
    """
    R_goal = np.asarray(R_goal, dtype=float).reshape(3, 3)
    tau_s = float(tau_s)
    if tau_s <= 0.0 or not np.isfinite(dt) or dt <= 0.0:
        return R_goal.copy()
    frac = float(np.clip(dt / tau_s, 0.0, 1.0))
    if frac >= 1.0:
        return R_goal.copy()
    out = slerp_rotation(R_cmd, R_goal, frac)
    return out if _is_rotation(out) else R_goal.copy()


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
        # Control messages carry a string first: they change the field and
        # answer nothing, so the request/response pairing of a project query is
        # untouched. Anything else is (xyz, R).
        if isinstance(msg[0], str):
            if msg[0] == "roll_flip":
                setter = getattr(vf, "set_roll_flip", None)
                if callable(setter):
                    setter(bool(msg[1]))
            continue
        xyz, R = msg
        out_q.put(_project_once(vf, xyz, R))


class FieldFollow:
    """Stick linear from live TCP; ``vf.project()`` on a capped outer rate; freeze on idle."""

    def __init__(self, vf, *, hz=20.0, zero_smoothing_after_first=True,
                 project_in_process=False, mesh=None, max_lead_m=0.02,
                 max_dR_rad=0.12, max_step_m=0.0, snap_to_surface=True,
                 rebase_on_live=True, rearm_slerp_s=0.0,
                 hold_field_orientation=False, linear_approach=False,
                 max_pending_m=0.0):
        hz = float(hz)
        if hz <= 0.0:
            raise ValueError(f"hz must be positive, got {hz}")
        self._vf = vf
        self._min_dt = 1.0 / hz
        self._max_lead_m = float(max_lead_m)
        self._max_dR_rad = float(max_dR_rad)
        self._max_step_m = float(max_step_m)
        # Bound on the undrained stick backlog. 0 = unbounded (the historical
        # behavior on both backends).
        #
        # ``add_body_linear`` is called once per SpaceMouse poll and ``_drain``
        # removes at most ``max_step_m`` per follow tick, and *the two rates do
        # not match*. Measured on this rig's defaults: the poll thread runs at
        # 50 Hz and the follow gate at ``1/hz`` = 50 ms against a ~50 Hz caller,
        # so the gate actually fires every third call -- 16.7 Hz, not 20 -- and
        # the drain ceiling is 16.7 * 4 mm = **66.7 mm/s**. Full deflection
        # feeds 50 * 1.5 mm = 75 mm/s on one axis, 106 on a two-axis diagonal
        # and 130 on three. So the queue overruns on *every* axis combination
        # at full stick, the surplus is carried in ``_pending`` without bound,
        # and the command falls behind the operator by ``pending/66.7 mm/s`` --
        # 2.4 s after a 5 s two-axis traverse. It is discarded on release
        # (``set_idle(clear_pending=True)``), so it does not show up as the arm
        # over-running a stop; it shows up as the arm getting steadily less
        # responsive the longer you hold the stick, which invites the operator
        # to push harder and is a pilot-induced oscillation generator.
        #
        # This is *not* the delayed-damper pathology: the ``-D*v`` term closes
        # on ``J @ qdot_meas`` at the control rate and never sees this queue,
        # and ``_clamp_lead`` still bounds the commanded lead, so the force is
        # bounded either way. Different failure, different fix -- do not read a
        # backlog number as evidence about zeta.
        self._max_pending_m = float(max_pending_m)
        self._snap_to_surface = bool(snap_to_surface)
        # Where each stick step starts from. True (speedJ) re-bases the carrot
        # on the live TCP every tick, so the command is only ever one drained
        # step ahead: that is a *velocity* command, which is exactly what a
        # position servo wants. Under the direct-torque wrench law the same
        # carrot becomes force = k_lin * (one step) -- force proportional to
        # stick *speed*, capped at k_lin * max_step_m and zero when the stick
        # is held still against a stalled arm. False accumulates instead, so a
        # held stick builds error up to max_lead_m and the spring makes a real,
        # sustained push that decays on its own once the arm catches up.
        self._rebase_on_live = bool(rebase_on_live)
        # Time constant for the commanded orientation's approach to the field
        # goal. 0 snaps, which is what put |W_ang| 0 -> 4.25 N.m and tau on the
        # 50 N.m rail in one tick on dtc_20260829_214019.csv: the target was the
        # live pose while the stick was quiet and the full field pose one tick
        # later. The commanded R is kept separate from the goal (_R_cmd vs _R)
        # rather than rate-limiting against *live*, which is what pinned R_err
        # at ~1.1 deg forever before 2eceb0f -- this one converges.
        self._rearm_slerp_s = max(0.0, float(rearm_slerp_s))
        # Whether the commanded orientation survives a quiet stick. Off (speedJ)
        # collapses it to live R on every idle/catch-up call, which at 50 Hz
        # against a 20 Hz follow tick is most calls -- so the wrist was handed a
        # fresh full-field error on every re-arm and never converged onto the
        # field between them. On, only the *position* re-bases to live.
        self._hold_field_orientation = bool(hold_field_orientation)
        # Whether the *linear* carrot ramps to its goal at a constant
        # velocity spread over the gap to the next --follow-hz commit,
        # advanced on every tick() call (control rate), rather than snapping
        # the full step at the commit. Off (default) is the pre-fix
        # behavior -- the linear carrot never had this, unlike rotation's
        # rearm_slerp_s.
        self._linear_approach = bool(linear_approach)
        self._zero_smoothing_after_first = bool(zero_smoothing_after_first)
        self._proc = None
        self._in_q = None
        self._out_q = None
        self._lock = threading.Lock()
        if project_in_process:
            self._start_helper(mesh)
        self._pending = np.zeros(3, dtype=float)
        # Tool orientation A/B, forwarded to the helper on the NEXT project
        # call rather than written to the queue from whichever thread pressed
        # the key: the queue is a call/response pair owned by the follow tick,
        # and a second producer on it is how a response gets read by the wrong
        # request.
        self._roll_flip_pending = None
        self._idle = True
        self._xyz = None
        self._xyz_cmd = None
        self._xyz_vel = np.zeros(3, dtype=float)
        self._R = np.eye(3)
        self._R_cmd = np.eye(3)
        self._last_t = float("-inf")
        self._last_advance_t = float("-inf")
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
            # Seed is the one place the command is deliberately re-based on the
            # arm (align has just finished, so live R is already on the field),
            # so the re-arm slerp starts from here and not from a stale carrot.
            self._R_cmd = self._R.copy()
            self._xyz_cmd = self._xyz.copy()
            self._xyz_vel[:] = 0.0
            self._pending[:] = 0.0
            self._idle = True
            self._last_t = float("-inf")
            self._last_advance_t = float("-inf")

    def add_body_linear(self, linear):
        with self._lock:
            self._pending = self._pending + np.asarray(linear, dtype=float).reshape(3)
            cap = self._max_pending_m
            if cap > 0.0:
                # Drop the surplus rather than queueing it: a stick is a
                # velocity command, so a *stale* step is worth less than no
                # step. Direction is preserved -- this only shortens the vector.
                n = float(np.linalg.norm(self._pending))
                if n > cap:
                    self._pending = self._pending * (cap / n)
            self._idle = False

    def set_idle(self, xyz=None, *, clear_pending=True):
        with self._lock:
            if clear_pending:
                self._pending[:] = 0.0
            self._idle = True
            if xyz is not None:
                self._xyz = np.asarray(xyz, dtype=float).reshape(3).copy()
                self._xyz_cmd = self._xyz.copy()
                self._xyz_vel[:] = 0.0
                self._last_advance_t = float("-inf")
                self.last_target = _motor_at(self._xyz, self._R_cmd)

    def snap_tcp(self, xyz, R=None):
        """Rebase the carrot to ``xyz`` without blocking the next stick tick."""
        with self._lock:
            self._xyz = np.asarray(xyz, dtype=float).reshape(3).copy()
            self._xyz_cmd = self._xyz.copy()
            self._xyz_vel[:] = 0.0
            self._last_advance_t = float("-inf")
            if R is not None:
                self._R = np.asarray(R, dtype=float).reshape(3, 3).copy()
                self._R_cmd = self._R.copy()
            self.last_target = _motor_at(self._xyz, self._R_cmd)
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

    def set_roll_flip(self, flip) -> bool:
        """Tool orientation A (0 deg) / B (180 deg). Applies to both copies.

        ``vf`` lives here for ``approach_basis_at`` and in the helper process
        for ``project()``; both have to agree or the ghost and the command are
        two different orientations.
        """
        flip = bool(flip)
        setter = getattr(self._vf, "set_roll_flip", None)
        if callable(setter):
            setter(flip)
        if self._proc is not None:
            self._roll_flip_pending = flip
        return flip

    def _project(self, xyz, R):
        if self._proc is not None:
            if self._roll_flip_pending is not None:
                flip, self._roll_flip_pending = self._roll_flip_pending, None
                try:
                    self._in_q.put(("roll_flip", bool(flip)), timeout=1.0)
                except Exception as exc:
                    print(f"field follow: could not forward the tool roll to "
                          f"the project helper ({exc})", flush=True)
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
        xyz, R, last, R_cmd, xyz_cmd, xyz_vel = self._undo
        self._xyz = None if xyz is None else np.array(xyz, copy=True)
        self._R = np.array(R, copy=True)
        self._R_cmd = np.array(R_cmd, copy=True)
        self._xyz_cmd = None if xyz_cmd is None else np.array(xyz_cmd, copy=True)
        self._xyz_vel = np.array(xyz_vel, copy=True)
        self.last_target = last
        self._undo = None
        self._pending[:] = 0.0
        self._idle = True

    def tick(self, now, proposed_motor, live_xyz=None, live_R=None, tangent=False):
        now = float(now)
        if self._xyz is None:
            T = np.asarray(proposed_motor.to_transformation_matrix(), dtype=float)
            self._xyz = T[:3, 3].copy()
            self._xyz_cmd = self._xyz.copy()
            self._xyz_vel[:] = 0.0
            self._R = T[:3, :3].copy()
        if self._idle:
            with self._lock:
                pending_n = float(np.linalg.norm(self._pending))
            if pending_n < 1e-15:
                return FollowTick(idle=True)
            self._idle = False
        if np.isfinite(self._last_t) and (now - self._last_t) < self._min_dt:
            if (not self._linear_approach or self._xyz_cmd is None
                    or self._xyz is None):
                return FollowTick(idle=False)
            # Keep charging the constant-velocity ramp at control-tick rate
            # even though this call did not earn a new project() -- this is
            # the fix for "the carrot is a 20 Hz staircase into a 124 Hz
            # loop" (HANDOFF): the command creeps toward the last committed
            # goal every tick instead of holding then jumping.
            # -inf means no advance has ever run (startup): charge 0 rather
            # than falling back to a full ramp duration, which would snap the
            # whole step on what should be a single control-tick slice.
            adv_dt = (now - self._last_advance_t
                      if np.isfinite(self._last_advance_t) else 0.0)
            adv_dt = float(np.clip(adv_dt, 0.0, 4.0 * self._min_dt))
            self._xyz_cmd, self._xyz_vel = _ramp_step(
                self._xyz_cmd, self._xyz, self._xyz_vel, adv_dt)
            self._last_advance_t = now
            target = _motor_at(self._xyz_cmd, self._R_cmd)
            self.last_target = target
            return FollowTick(
                target=target, quat_wxyz=_quat_wxyz_from_motor(target),
                idle=False)
        delta = self._drain()
        t_prev = self._last_t
        self._last_t = now
        if float(np.linalg.norm(delta)) < 1e-15:
            return FollowTick(idle=False)
        # Bounded to the nominal follow period: the ramp is a per-tick
        # fraction, and an idle gap (t_prev is the last *committed* tick, which
        # can be seconds ago, or -inf on the first) would otherwise clip the
        # fraction to 1 and snap -- exactly the step this exists to remove.
        dt_tick = self._min_dt
        if np.isfinite(t_prev):
            dt_tick = float(
                np.clip(now - t_prev, self._min_dt, 4.0 * self._min_dt))
        self._undo = (
            None if self._xyz is None else self._xyz.copy(),
            self._R.copy(),
            self.last_target,
            self._R_cmd.copy(),
            None if self._xyz_cmd is None else self._xyz_cmd.copy(),
            self._xyz_vel.copy(),
        )
        base = self._xyz
        if (self._rebase_on_live and live_xyz is not None
                and _is_finite_xyz(live_xyz)):
            base = np.asarray(live_xyz, dtype=float).reshape(3)
        # Stick + carrot R match the orange viser ghost: field axes at the tip
        # (approach_basis_at / project), not a slerp from the live wrist.
        # max_dR_rad only rate-limits field→field (mesh flips); 0 = full snap.
        R_prev = self._undo[1]
        R_field = R_prev
        basis_fn = getattr(self._vf, "approach_basis_at", None)
        if callable(basis_fn):
            R_at = np.asarray(basis_fn(base), dtype=float).reshape(3, 3)
            if _is_rotation(R_at):
                R_field = R_at
        step = R_field @ delta
        if tangent:
            z_hat = R_field[:, 2]
            step = step - z_hat * float(np.dot(step, z_hat))
        self._xyz = base + step
        self._clamp_lead(live_xyz)
        stick_xyz = self._xyz.copy()
        target, project_ms = self._project(stick_xyz, R_field)
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
        R_new = _limit_rotation(R_prev, R_raw, self._max_dR_rad)
        if not _is_rotation(R_new):
            self.revert_tick()
            return FollowTick(idle=True)
        self._R = R_new
        # The goal is the field; the *command* approaches it. Everything
        # downstream (last_target, the ghost, the published hold rotation) reads
        # _R_cmd, so there is exactly one commanded orientation.
        self._R_cmd = _approach_rotation(
            self._R_cmd, R_new, dt_tick, self._rearm_slerp_s)
        # Surface snap takes its normal from the *goal*: it is correcting the
        # commanded point onto the mesh, and the mesh normal is the field's, not
        # the ramp's momentary orientation.
        self._xyz = _snap_normal_to_surface(
            stick_xyz, xyz_proj, R_new,
            max_correction_m=self._max_lead_m) if self._snap_to_surface else stick_xyz
        # The goal moves at --follow-hz; the *command* ramps toward it at a
        # constant velocity sized to close the gap by the next expected
        # commit (dt_tick), charged in whatever increment tick() is actually
        # called at (control rate) -- not an exponential settle like
        # _R_cmd/_approach_rotation, which would add steady-state lag chasing
        # a goal that moves again before it arrives. Off (default) snaps,
        # byte-identical to pre-fix behavior.
        if self._linear_approach:
            if self._xyz_cmd is None:
                self._xyz_cmd = self._xyz.copy()
            self._xyz_vel = _ramp_velocity(self._xyz_cmd, self._xyz, dt_tick)
            adv_dt = (now - self._last_advance_t
                      if np.isfinite(self._last_advance_t) else 0.0)
            adv_dt = float(np.clip(adv_dt, 0.0, dt_tick))
            self._xyz_cmd, self._xyz_vel = _ramp_step(
                self._xyz_cmd, self._xyz, self._xyz_vel, adv_dt)
        else:
            self._xyz_cmd = self._xyz.copy()
        self._last_advance_t = now
        target = _motor_at(self._xyz_cmd, self._R_cmd)
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


def _target_changed(follow, motor, eps=1e-9):
    """True when this mapped target differs from the last one sent.

    State lives on ``follow`` so the caller stays stateless. Compared as a 4x4
    rather than by identity: ``map_target`` builds a fresh Motor every call, so
    identity is always False and the value is what matters.
    """
    T = np.asarray(motor.to_transformation_matrix(), dtype=float)
    prev = getattr(follow, "_last_sent_target", None)
    follow._last_sent_target = T
    return prev is None or float(np.abs(T - prev).max()) > eps


def drive_field_follow(follow, controller, now, proposed_motor, live_R=None,
                       err_rad=0.05, map_target=None, live_xyz=None,
                       live_ee_xyz=None, max_ee_jump_m=0.08, stick_quiet=True,
                       tangent=False, rebase_on_idle=True, resend_target=True):
    """Apply a follow tick to the pose spring.

    Stick tick: live TCP + field-frame linear; commanded R is the field at the
    tip (orange-ghost contract), approached at ``rearm_slerp_s``. Quiet stick
    freezes (clears pending) so leftover xyz cannot keep pushing.

    ``resend_target`` (``--no-follow-target-resend`` turns it off) controls
    whether the quiet branch re-writes an *unchanged* target to the objective on
    every call. It writes at the caller's rate -- the main loop's fixed 50 Hz,
    **not** ``--follow-hz``, because ``tick()`` returns idle before its own rate
    gate whenever the stick is untouched -- and it writes from a different
    thread than the 200 Hz control loop that reads it. The value is provably
    constant while quiet (``map_target`` is pose-invariant: at ``alpha=1.0``
    ``ghost_flange_world`` drops the live flange entirely, measured spread
    9e-16), so skipping the redundant write is semantically a no-op and is a
    clean bisection for whether that cross-thread write is behind the 30 Hz
    burst on dtc_20260902_171136.csv. Default True = the behavior every log so
    far was recorded with.

    ``rebase_on_idle`` mirrors ``--dtc-hold-at-tick``: when False (compliance
    hold / push test) a quiet stick clears pending only and keeps the last
    commanded field pose so external pushes create spring error and return on
    release. SpaceMouse motion still drives ``follow.tick()`` via pending
    deltas; only the idle re-base onto ``live_xyz`` is skipped.

    What the quiet and catch-up branches do with *orientation* is
    ``hold_field_orientation``:

    - **off** (speedJ) collapses the commanded R to live R, so a stalled wrist
      is never yanked by field orientation. This path must stay bit-identical --
      it is the speedJ side of the C153 A/B.
    - **on** (dtc wrench law) keeps it. Measured on
      ``dtc_20260829_214019.csv``: this function runs at ~50 Hz against a 20 Hz
      follow tick, so most calls took a catch-up branch and re-based the
      commanded orientation onto the live wrist. Combined with the controller's
      idle hold (target := live pose at 125 Hz) the angular error was erased
      between every stick motion, so the wrists -- which moved 0.06 deg over the
      whole run while j0..j2 moved 8 deg -- were handed the same full-field
      error over and over and never converged onto it. Only the *position*
      re-bases to live here.
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
                if rebase_on_idle and live_xyz is not None:
                    follow.set_idle(live_xyz)
                else:
                    follow.set_idle(clear_pending=True)
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
    hold_R = bool(getattr(follow, "_hold_field_orientation", False))
    if stick_quiet:
        if rebase_on_idle and R_live is not None and not hold_R:
            follow._R = R_live
            follow._R_cmd = np.array(R_live, copy=True)
        if rebase_on_idle:
            if live_xyz is not None:
                follow.set_idle(live_xyz, clear_pending=True)
            else:
                follow.set_idle(clear_pending=True)
        else:
            follow.set_idle(clear_pending=True)
        if follow.last_target is not None:
            frozen = follow.last_target
            if map_target is not None:
                frozen = map_target(frozen)
            if resend_target or _target_changed(follow, frozen):
                controller.get_objective().set_target(frozen)
        controller.set_input_active(False)
        return tick
    if hold_R:
        R_hold = np.array(follow._R_cmd, copy=True)
    else:
        R_hold = R_live if R_live is not None else follow._R
    if leftover > max(max_lead, CATCHUP_TCP_M) + 1e-12:
        if rebase_on_idle and live_xyz is not None:
            # R=None under hold_R: snap_tcp re-bases the command deliberately,
            # and here only the position is being re-based.
            follow.snap_tcp(live_xyz, None if hold_R else R_hold)
        elif follow.xyz is not None:
            follow.last_target = _motor_at(follow.xyz, R_hold)
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
            if not hold_R:
                follow._R = np.array(R_hold, copy=True)
                follow._R_cmd = np.array(R_hold, copy=True)
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
            if not hold_R:
                follow._R = np.array(R_hold, copy=True)
                follow._R_cmd = np.array(R_hold, copy=True)
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
