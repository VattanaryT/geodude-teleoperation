"""Viser picking + execution state for the deterministic path experiment.

The example owns the robot; this owns the *plan*. It answers three questions
per tick and nothing else:

- where should the carrot step next (``tick`` -> world-frame delta), and
- is the tool on the part (``press_engaged``, from ``press_latch``), and
- where should the orange ghost be during a preview (``preview_xyz``).

Every viser callback only *records a request*; the state machine advances on the
caller's own thread inside ``tick``. Viser callbacks run on their own threads and
this state reaches a 500 Hz control loop, so "the button sets a flag" is the
whole concurrency design.
"""

import threading
import time

import numpy as np

from .path_plan import (
    PHASE_DONE,
    PHASE_PLUNGE,
    PathRunner,
    plan_surface_path,
    preview_positions,
    project_to_surface,
)
from .press_latch import (
    AllOf, DistanceLatch, ForceLatch, LatchState, PhaseLatch)

STATE_PICK = "pick"          # waiting for two clicks
STATE_READY = "ready"        # planned, not moving
STATE_PREVIEW = "preview"    # ghost walking the plan, arm still
STATE_RUN = "run"            # arm executing
STATE_DONE = "done"


class PathSession:
    """Pick two points, plan, preview, execute. No hardware, no gafropy."""

    def __init__(self, *, vf, mesh, free_speed_m_s=0.02, contact_speed_m_s=0.01,
                 standoff_m=0.03, ds_m=0.005, engage_gap_m=0.002,
                 release_gap_m=0.005, arrive_m=0.002, depth_m=0.0,
                 touch_n=0.0, max_seek_m=0.01,
                 on_execute_start=None, on_execute_end=None,
                 use_field_normal=True):
        self.vf = vf
        self.mesh = mesh
        self.free_speed = float(free_speed_m_s)
        self.contact_speed = float(contact_speed_m_s)
        self.standoff_m = float(standoff_m)
        self.ds_m = float(ds_m)
        self.arrive_m = float(arrive_m)
        # How far BELOW the nominal surface the plunge and traverse are
        # commanded. The scan is placed by calibration, so the model surface
        # and the real one differ by millimetres; a plan that stops exactly on
        # the model leaves the tool touching lightly or not at all.
        self.depth_m = float(depth_m)
        # Optional force confirmation. 0 disables and the latch is pure
        # geometry. Above 0 the plunge keeps descending until Fs reaches it
        # (bounded by max_seek_m) AND the latch requires it, which is what
        # stops the press handing over while the tool is still in the air --
        # the failure this rig already has a log of: with the normal freed, a
        # pure TCP force rotates the wrist instead of translating the arm.
        self.touch_n = float(max(0.0, touch_n))
        self.max_seek_m = float(max(0.0, max_seek_m))
        self._seeking = False
        self._touched = False
        self.last_fs = float("nan")
        self._on_execute_start = on_execute_start
        self._on_execute_end = on_execute_end
        self._use_field_normal = bool(use_field_normal)

        self._lock = threading.Lock()
        self.state = STATE_PICK
        self.p_start = None
        self.p_end = None
        self.plan = None
        self.runner = None
        latches = [
            PhaseLatch(),
            DistanceLatch(engage_gap_m=engage_gap_m,
                          release_gap_m=release_gap_m),
        ]
        if self.touch_n > 0.0:
            # A threshold, not a regulator: this sensor's spread is well under
            # a newton and the gate sits several newtons up, so it is a
            # trustworthy "is anything there at all" even though it is not
            # trustworthy enough to close a 15 N loop on.
            latches.append(ForceLatch(self.touch_n, engage_frac=1.0,
                                      release_frac=0.3))
        self.latch = AllOf(latches)
        self._distance_latch = self.latch.latches[1]
        self._engaged = False
        self._phase = "-"
        self._request = None          # "execute" | "preview" | "abort" | "reset"
        self._preview_pts = None
        self._preview_t0 = None
        self._live_tcp = None
        self._server = None
        self._gui = {}
        self._status_cb = None

    # ---------------------------------------------------------------- state

    @property
    def press_engaged(self) -> bool:
        return bool(self._engaged)

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def latch_distance_m(self) -> float:
        return float(self._distance_latch.last_distance_m)

    @property
    def running(self) -> bool:
        return self.state == STATE_RUN

    def set_live_tcp(self, xyz):
        if xyz is None:
            return
        self._live_tcp = np.asarray(xyz, dtype=float).reshape(3).copy()

    def status_line(self) -> str:
        d = self.latch_distance_m
        picked = sum(p is not None for p in (self.p_start, self.p_end))
        bias = 0.0 if self.runner is None else self.runner.depth_bias
        extra = ""
        if self.touch_n > 0.0:
            extra = (f" Fs={self.last_fs:+.1f}N "
                     f"{'touched' if self._touched else 'seeking' if self._seeking else 'no-touch'}"
                     f" seek={bias * 1000:.1f}mm")
        return (f"path[{self.state}]: picked={picked}/2 "
                f"phase={self._phase} engaged={int(self._engaged)} "
                f"d_surf={d * 1000.0:+.1f}mm "
                f"wp={self.runner.index if self.runner else 0}"
                f"/{len(self.plan) if self.plan else 0}{extra}")

    # ---------------------------------------------------------------- picking

    def normal_at(self, p):
        """Outward surface normal the plan uses.

        The field's approach Z points *into* the surface, so outward is its
        negative -- taking it from the field rather than the mesh keeps the plan
        and the controller on one definition of "up off the part".
        """
        if self._use_field_normal and self.vf is not None:
            basis = np.asarray(self.vf.approach_basis_at(p), dtype=float)
            n = -basis[:, 2]
            if np.all(np.isfinite(n)) and float(np.linalg.norm(n)) > 1e-9:
                return n / float(np.linalg.norm(n))
        _, n = project_to_surface(self.mesh, p)
        return n

    def pick(self, xyz):
        """Record a picked point. First = start, second = end, third resets."""
        p = np.asarray(xyz, dtype=float).reshape(3)
        with self._lock:
            if self.state in (STATE_RUN, STATE_PREVIEW):
                return self.state
            if self.p_start is None:
                self.p_start = p
                self.p_end = None
                self.plan = None
                self.state = STATE_PICK
            elif self.p_end is None:
                self.p_end = p
            else:
                self.p_start = p
                self.p_end = None
                self.plan = None
                self.state = STATE_PICK
        if self.p_start is not None and self.p_end is not None:
            self.replan()
        self._refresh_scene()
        return self.state

    def replan(self):
        if self.p_start is None or self.p_end is None:
            return None
        plan = plan_surface_path(
            self.mesh, self.p_start, self.p_end,
            standoff_m=self.standoff_m, ds_m=self.ds_m,
            depth_m=self.depth_m, normal_fn=self.normal_at)
        with self._lock:
            self.plan = plan
            self.runner = PathRunner(
                plan, free_speed_m_s=self.free_speed,
                contact_speed_m_s=self.contact_speed, arrive_m=self.arrive_m)
            self.latch.reset()
            self._engaged = False
            self._phase = plan.waypoints[0].phase if len(plan) else PHASE_DONE
            self.state = STATE_READY
            self._preview_pts = None
        print(f"path planned: {len(plan)} waypoints, {plan.length_m() * 100:.1f} cm "
              f"(standoff {self.standoff_m * 100:.1f} cm, ds {self.ds_m * 1000:.0f} mm, "
              f"depth {self.depth_m * 1000:.1f} mm"
              + (f", seek to {self.touch_n:.1f} N within "
                 f"{self.max_seek_m * 1000:.0f} mm" if self.touch_n > 0 else "")
              + ")", flush=True)
        self._refresh_scene()
        return plan

    def reset(self):
        with self._lock:
            self.p_start = None
            self.p_end = None
            self.plan = None
            self.runner = None
            self.latch.reset()
            self._engaged = False
            self._phase = "-"
            self.state = STATE_PICK
            self._preview_pts = None
        self._refresh_scene()

    # ------------------------------------------------------------- requests

    def request(self, what):
        with self._lock:
            self._request = str(what)

    def request_execute(self):
        self.request("execute")

    def request_preview(self):
        self.request("preview")

    def request_abort(self):
        self.request("abort")

    def _take_request(self):
        with self._lock:
            req = self._request
            self._request = None
            return req

    # ------------------------------------------------------------------ tick

    def tick(self, dt, carrot_xyz, tcp_xyz=None, fs=None, allow_execute=True,
             block_reason="", hold=False):
        """Advance the state machine. Returns a ``PathStep`` or ``None``.

        ``carrot_xyz`` is what the caller is driving (field-follow's own
        command), never the arm -- the lead clamp is what bounds the gap between
        them, exactly as it does for a held stick.

        **Call this on every main-loop tick, including the ones that cannot
        move the arm**, with ``allow_execute=False`` on those. Requests are
        recorded by viser's own threads and consumed here, so a tick that is
        skipped is a button that does nothing: previewing, resetting and
        aborting must keep working while startup align is still holding the
        arm, and only EXECUTE has to wait for it.

        ``hold`` freezes the *plan* for a tick without stopping the run: the
        phase and the latch still update, but nothing advances and the touch
        seek does not deepen. The caller sets it when the arm has fallen behind
        the carrot -- a plan that keeps walking into the lead clamp winds the
        command up against a stop, and the clamp pushing back against the plan
        pushing forward is an oscillation nobody asked for.
        """
        self.set_live_tcp(tcp_xyz)
        if fs is not None:
            self.last_fs = float(fs)
        req = self._take_request()
        if req == "execute" and not allow_execute:
            print(f"\033[93mpath: not ready to execute{': ' + block_reason if block_reason else ''}\033[0m",
                  flush=True)
            req = None
        if req == "reset":
            self.reset()
            return None
        if req == "abort":
            self._abort()
            return None
        if req == "preview" and self.plan is not None:
            self._start_preview()
            return None
        if req == "execute" and self.plan is not None:
            self._start_run()

        if self.state != STATE_RUN or self.runner is None:
            return None

        step = self.runner.step(0.0 if hold else dt, carrot_xyz)
        self._phase = step.phase
        if not hold:
            step = self._seek_for_touch(dt, step)
        tcp = self._live_tcp if tcp_xyz is None else np.asarray(
            tcp_xyz, dtype=float).reshape(3)
        self._engaged = bool(self.latch.update(LatchState(
            tcp_xyz=tcp,
            surface_xyz=step.surface_xyz,
            normal=step.normal,
            phase_press=step.press_phase,
            fs=self.last_fs,
        )))
        if step.done:
            self._finish()
        return step

    def _seek_for_touch(self, dt, step):
        """Keep the plunge descending until the force says the tool is there.

        Off unless ``touch_n`` is set. The plan's surface is a laser scan
        placed by calibration; the real part can sit a few millimetres either
        side of it. Descending on geometry alone therefore ends one of two
        ways -- pressed harder than asked, or hovering while the log reports a
        healthy 15 N command that never reaches anything -- and only a force
        reading tells the two apart.

        The seek grows the runner's ``depth_bias``, so the goal keeps sinking
        and the runner never "arrives" at the plunge waypoint. On touch it
        freezes, and every later traverse waypoint inherits the same bias: the
        traverse then follows the surface at the depth contact was actually
        found, not the depth the model predicted.
        """
        if self.touch_n <= 0.0 or self.runner is None or self._touched:
            return step
        if step.phase != PHASE_PLUNGE:
            return step
        if np.isfinite(self.last_fs) and self.last_fs >= self.touch_n:
            self._touched = True
            self._seeking = False
            print(f"path: touch confirmed at Fs={self.last_fs:.1f} N, "
                  f"{self.runner.depth_bias * 1000:.1f} mm past the planned "
                  f"depth", flush=True)
            return step
        if self.runner.depth_bias >= self.max_seek_m:
            print(f"\033[91mpath: no touch within {self.max_seek_m * 1000:.0f} mm "
                  f"of the planned surface (Fs={self.last_fs:.1f} N). The scan "
                  f"and the part disagree by more than that, or the force sign "
                  f"is inverted. Aborting.\033[0m", flush=True)
            self.request_abort()
            return step
        if not self._seeking:
            self._seeking = True
            print(f"path: seeking for contact (Fs={self.last_fs:.1f} N < "
                  f"{self.touch_n:.1f} N), up to "
                  f"{self.max_seek_m * 1000:.0f} mm deeper", flush=True)
        # One tick behind by construction: this tick's delta was computed
        # against the previous goal. At the contact speed that is tens of
        # microns, and being a tick late on a search is the safe direction.
        self.runner.depth_bias = min(
            self.max_seek_m,
            self.runner.depth_bias + self.runner.contact_speed * float(dt))
        return step

    def _start_preview(self):
        with self._lock:
            self._preview_pts = preview_positions(
                self.plan, free_speed_m_s=self.free_speed,
                contact_speed_m_s=self.contact_speed)
            self._preview_t0 = time.monotonic()
            self.state = STATE_PREVIEW
        print(f"path preview: ghost walks {len(self._preview_pts)} samples "
              f"(~{len(self._preview_pts) / 50.0:.1f}s). The arm does not move.",
              flush=True)

    def _start_run(self):
        with self._lock:
            if self.runner is None:
                return
            self.runner.start()
            self.runner.depth_bias = 0.0
            self.latch.reset()
            self._engaged = False
            self._touched = False
            self._seeking = False
            self.state = STATE_RUN
            self._preview_pts = None
        print("path EXECUTE: driving the field-follow carrot along the plan.",
              flush=True)
        if self._on_execute_start is not None:
            try:
                self._on_execute_start()
            except Exception as exc:
                print(f"path on_execute_start failed: {exc}", flush=True)

    def _abort(self):
        was_running = self.state == STATE_RUN
        with self._lock:
            if self.runner is not None:
                self.runner.abort()
            self._engaged = False
            self._phase = PHASE_DONE
            self.state = STATE_READY if self.plan is not None else STATE_PICK
            self._preview_pts = None
        if was_running:
            print("path ABORT: press off, carrot frozen on the live pose.",
                  flush=True)
            self._end_execution()

    def _finish(self):
        with self._lock:
            self._engaged = False
            self._phase = PHASE_DONE
            self.state = STATE_DONE
        print("path DONE: retracted, press off.", flush=True)
        self._end_execution()

    def _end_execution(self):
        if self._on_execute_end is not None:
            try:
                self._on_execute_end()
            except Exception as exc:
                print(f"path on_execute_end failed: {exc}", flush=True)

    # --------------------------------------------------------------- preview

    def preview_xyz(self, now=None):
        """Ghost position during a preview, or ``None`` when not previewing."""
        if self.state != STATE_PREVIEW or self._preview_pts is None:
            return None
        if len(self._preview_pts) == 0:
            return None
        now = time.monotonic() if now is None else float(now)
        i = int((now - self._preview_t0) * 50.0)
        if i >= len(self._preview_pts):
            with self._lock:
                self.state = STATE_READY
                self._preview_pts = None
            return None
        return np.asarray(self._preview_pts[i], dtype=float)

    # ------------------------------------------------------------ viser glue

    def attach(self, server, *, on_status=None):
        """Register the click handler and the GUI panel on a viser server.

        Wrapped in its own try/except per feature: a viser version whose pointer
        events differ must degrade to the numeric GUI inputs, not take the whole
        run down.
        """
        self._server = server
        self._status_cb = on_status
        try:
            self._attach_click(server)
        except Exception as exc:
            print(f"\033[93mpath: scene clicks unavailable ({exc}); use the "
                  f"GUI 'set start/end' fields\033[0m", flush=True)
        try:
            self._attach_gui(server)
        except Exception as exc:
            print(f"\033[93mpath: GUI panel unavailable ({exc})\033[0m",
                  flush=True)
        self._refresh_scene()

    def _attach_click(self, server):
        import trimesh  # ray intersection lives here, not in the mesh module

        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)

        def handle(event):
            origin = np.asarray(event.ray_origin, dtype=float).reshape(1, 3)
            direction = np.asarray(event.ray_direction, dtype=float).reshape(1, 3)
            hits, _, _ = intersector.intersects_location(origin, direction)
            if len(hits) == 0:
                print("path: click missed the object", flush=True)
                return
            d = np.linalg.norm(np.asarray(hits) - origin, axis=1)
            self.pick(np.asarray(hits)[int(np.argmin(d))])
            print(f"path: picked {self.status_line()}", flush=True)

        @server.on_client_connect
        def _(client):
            @client.scene.on_pointer_event(event_type="click")
            def _(event):
                try:
                    handle(event)
                except Exception as exc:
                    print(f"path click failed: {exc}", flush=True)

    def _attach_gui(self, server):
        gui = server.gui
        with gui.add_folder("Deterministic path"):
            status = gui.add_text("status", initial_value="pick 2 points",
                                  disabled=True)
            speed = gui.add_slider("free mm/s", min=2.0, max=60.0, step=1.0,
                                   initial_value=self.free_speed * 1000.0)
            cspeed = gui.add_slider("contact mm/s", min=1.0, max=40.0, step=1.0,
                                    initial_value=self.contact_speed * 1000.0)
            b_start_live = gui.add_button("start = live TCP")
            b_preview = gui.add_button("preview (ghost only)")
            b_execute = gui.add_button("EXECUTE")
            b_abort = gui.add_button("ABORT")
            b_reset = gui.add_button("reset picks")
        self._gui = {"status": status, "speed": speed, "cspeed": cspeed}

        @speed.on_update
        def _(_e):
            self.free_speed = float(speed.value) / 1000.0
            if self.runner is not None:
                self.runner.free_speed = self.free_speed

        @cspeed.on_update
        def _(_e):
            self.contact_speed = float(cspeed.value) / 1000.0
            if self.runner is not None:
                self.runner.contact_speed = self.contact_speed

        @b_start_live.on_click
        def _(_e):
            if self._live_tcp is not None:
                self.pick(self._live_tcp)

        @b_preview.on_click
        def _(_e):
            self.request_preview()

        @b_execute.on_click
        def _(_e):
            self.request_execute()

        @b_abort.on_click
        def _(_e):
            self.request_abort()

        @b_reset.on_click
        def _(_e):
            self.request("reset")

    def refresh_gui(self):
        status = self._gui.get("status")
        if status is not None:
            try:
                status.value = self.status_line()
            except Exception:
                pass
        if self._status_cb is not None:
            try:
                self._status_cb(self.status_line())
            except Exception:
                pass

    def _refresh_scene(self):
        server = self._server
        if server is None:
            return
        try:
            scene = server.scene
            for name, p, color in (("/path_start", self.p_start, (40, 200, 90)),
                                   ("/path_end", self.p_end, (220, 60, 40))):
                if p is None:
                    continue
                scene.add_icosphere(name, radius=0.005, color=color,
                                    position=tuple(float(v) for v in p))
            if self.plan is not None and len(self.plan) > 1:
                pts = self.plan.points
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                scene.add_line_segments("/path_preview", points=segs,
                                        colors=(255, 170, 40),
                                        thickness=0.002)
        except Exception as exc:
            print(f"path: scene draw skipped ({exc})", flush=True)
