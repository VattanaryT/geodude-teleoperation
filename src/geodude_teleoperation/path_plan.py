"""Deterministic surface path: plan it once, then walk the follow carrot.

This is the A-side of the SpaceMouse: same field-follow carrot, same projection,
same DTC wrench law underneath -- the only thing replaced is *where the steps
come from*. A stick emits body-frame increments at whatever rate the HID device
and the follow queue happen to agree on; this emits the same increments from a
plan, at a rate the loop chooses, so any latency left in the arm's behaviour is
the controller's own and not the input path's.

Nothing here talks to hardware, viser or gafropy. A plan is a list of
:class:`Waypoint`, a runner turns it into per-tick world-frame deltas, and the
caller rotates those into the body frame ``FieldFollow.add_body_linear`` wants.

The four phases exist because the press and the pose spring must hand the
contact normal back and forth exactly once each way:

``approach``  free space, above the surface at ``standoff_m``. Press commands 0;
              the spring owns the normal.
``plunge``    straight down the surface normal onto the first contact point.
              Still spring-driven -- a pure TCP force rotates the wrist instead
              of translating the arm (dtc_20260903_151838), so the press must
              never be the thing that crosses the gap.
``traverse``  along the surface between the two picked points. The press owns
              the normal here and field-follow goes tangent-only.
``retract``   back up the normal at the end point. Press off again.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

PHASE_APPROACH = "approach"
PHASE_PLUNGE = "plunge"
PHASE_TRAVERSE = "traverse"
PHASE_RETRACT = "retract"
PHASE_DONE = "done"

#: Phases in which the press is allowed to command force at all. The *latch*
#: (press_latch.DistanceLatch) still has to agree before anything is emitted --
#: this is the plan's half of the AND.
PRESS_PHASES = (PHASE_PLUNGE, PHASE_TRAVERSE)


def _unit(v, fallback=(0.0, 0.0, 1.0)):
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        return np.asarray(fallback, dtype=float).reshape(3)
    return v / n


@dataclass(frozen=True)
class Waypoint:
    """One commanded TCP position, with the surface it belongs to."""

    xyz: np.ndarray            # world TCP position the carrot walks to
    surface_xyz: np.ndarray    # the planned contact point this waypoint serves
    normal: np.ndarray         # outward unit surface normal at surface_xyz
    phase: str

    @property
    def press_phase(self) -> bool:
        return self.phase in PRESS_PHASES


@dataclass
class PathPlan:
    waypoints: List[Waypoint] = field(default_factory=list)
    start_surface: Optional[np.ndarray] = None
    end_surface: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.waypoints)

    @property
    def points(self) -> np.ndarray:
        """(N, 3) commanded positions -- what the viser preview line draws."""
        if not self.waypoints:
            return np.zeros((0, 3))
        return np.asarray([w.xyz for w in self.waypoints], dtype=float)

    def length_m(self) -> float:
        p = self.points
        if len(p) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    def phase_at(self, index: int) -> str:
        if index < 0 or index >= len(self.waypoints):
            return PHASE_DONE
        return self.waypoints[index].phase


def mesh_surface_normal_fn(mesh) -> Callable[[np.ndarray], np.ndarray]:
    """``p -> outward unit normal`` from the nearest triangle of ``mesh``."""
    import trimesh  # local: planning is optional for a headless force run

    def normal_at(p):
        p = np.asarray(p, dtype=float).reshape(1, 3)
        _, _, tri = trimesh.proximity.closest_point(mesh, p)
        return _unit(np.asarray(mesh.face_normals[int(tri[0])], dtype=float))

    return normal_at


def project_to_surface(mesh, p):
    """Closest point on ``mesh`` to ``p`` and the outward normal there."""
    import trimesh

    p = np.asarray(p, dtype=float).reshape(1, 3)
    closest, _, tri = trimesh.proximity.closest_point(mesh, p)
    n = _unit(np.asarray(mesh.face_normals[int(tri[0])], dtype=float))
    return np.asarray(closest[0], dtype=float), n


def plan_surface_path(mesh, p_start, p_end, *, standoff_m=0.03, ds_m=0.005,
                      depth_m=0.0, normal_fn=None, live_tcp=None,
                      axial_approach=True) -> PathPlan:
    """Plan approach -> plunge -> traverse -> retract between two picked points.

    ``p_start`` / ``p_end`` are anywhere near the surface (viser click rays land
    *on* it); both are projected onto the mesh. ``normal_fn`` overrides the mesh
    normal -- pass the *field* normal (``-vf.approach_basis_at(p)[:, 2]``) so the
    plan and the controller agree on which way is up off the part.

    ``depth_m`` sinks the commanded plunge and traverse points that far BELOW
    the surface, along its normal. The mesh is a laser scan placed by
    calibration, so the real part is not exactly where the model says; a
    commanded pose on the nominal surface therefore leaves the tool touching
    lightly, or not at all. ``surface_xyz`` keeps the true projected point, so
    the latch still measures distance to the *surface* and ``d_surf`` still
    reads 0 there and goes negative when the tool is commanded into the part.

    ``live_tcp`` with ``axial_approach`` (the default) splits the travel to the
    start point into the two moves the field frame is made of, instead of one
    diagonal: first **along the normal** from wherever the tool is up to the
    standoff plane, then **purely tangential** across that plane to sit above
    the start point. Only then does the plunge run, and it is pure ``-n``. A
    diagonal lead-in mixes all three axes and can carry the tool toward the
    part while it is still travelling sideways; this cannot. Without
    ``live_tcp`` the plan starts at the standoff point and the runner walks to
    it however it likes.
    """
    p0, n0 = project_to_surface(mesh, p_start)
    p1, n1 = project_to_surface(mesh, p_end)
    if normal_fn is not None:
        n0 = _unit(normal_fn(p0), fallback=n0)
        n1 = _unit(normal_fn(p1), fallback=n1)

    standoff_m = float(max(0.0, standoff_m))
    ds_m = float(max(1e-4, ds_m))
    depth_m = float(depth_m)

    wps: List[Waypoint] = []
    if axial_approach and live_tcp is not None:
        live = np.asarray(live_tcp, dtype=float).reshape(3)
        # 1. Rise (or fall) along the normal to the standoff plane, staying
        #    over wherever the tool already is.
        d_live = float(np.dot(n0, live - p0))
        lift = live + (standoff_m - d_live) * n0
        wps.append(Waypoint(lift, p0, n0, PHASE_APPROACH))
    wps.append(Waypoint(p0 + standoff_m * n0, p0, n0, PHASE_APPROACH))
    wps.append(Waypoint(p0 - depth_m * n0, p0, n0, PHASE_PLUNGE))

    span = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    dist = float(np.linalg.norm(span))
    n_seg = max(1, int(np.ceil(dist / ds_m)))
    for i in range(1, n_seg + 1):
        p_lin = p0 + span * (i / n_seg)
        p_s, n_s = project_to_surface(mesh, p_lin)
        if normal_fn is not None:
            n_s = _unit(normal_fn(p_s), fallback=n_s)
        wps.append(Waypoint(p_s - depth_m * n_s, p_s, n_s, PHASE_TRAVERSE))

    wps.append(Waypoint(p1 + standoff_m * n1, p1, n1, PHASE_RETRACT))
    return PathPlan(waypoints=wps, start_surface=p0, end_surface=p1)


@dataclass
class PathStep:
    """One tick of the runner."""

    delta_world: np.ndarray
    phase: str
    index: int
    target_xyz: np.ndarray
    surface_xyz: np.ndarray
    normal: np.ndarray
    press_phase: bool
    done: bool
    running: bool


class PathRunner:
    """Walks a :class:`PathPlan` at a bounded speed, one carrot step per tick.

    The runner never reads the arm -- it reads the *carrot* it is driving, so a
    stalled arm cannot make it run away (the lead clamp in ``FieldFollow`` is
    what bounds the gap between the two, exactly as it does for a held stick).
    """

    def __init__(self, plan: PathPlan, *, free_speed_m_s=0.02,
                 contact_speed_m_s=0.01, arrive_m=0.002):
        self.plan = plan
        self.free_speed = abs(float(free_speed_m_s))
        self.contact_speed = abs(float(contact_speed_m_s))
        self.arrive_m = abs(float(arrive_m))
        # Extra depth, along each waypoint's own normal, applied to the press
        # phases only. The caller grows it while seeking for a real touch (the
        # scan is not exactly where the part is) and freezes it once found, so
        # the traverse stays seated at the depth where contact actually
        # happened rather than the depth the model predicted.
        self.depth_bias = 0.0
        self._i = 0
        self._running = False
        self._done = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def done(self) -> bool:
        return self._done

    @property
    def index(self) -> int:
        return self._i

    def start(self):
        self._i = 0
        self._running = True
        self._done = False

    def abort(self):
        self._running = False

    def phase(self) -> str:
        if self._done or not self.plan.waypoints:
            return PHASE_DONE
        return self.plan.phase_at(min(self._i, len(self.plan) - 1))

    def speed_for(self, phase: str) -> float:
        return self.contact_speed if phase in PRESS_PHASES else self.free_speed

    def goal_of(self, wp) -> np.ndarray:
        """Commanded point for ``wp``, with the live depth bias applied."""
        if not wp.press_phase or self.depth_bias == 0.0:
            return np.asarray(wp.xyz, dtype=float)
        return (np.asarray(wp.xyz, dtype=float)
                - float(self.depth_bias) * np.asarray(wp.normal, dtype=float))

    def step(self, dt, carrot_xyz) -> PathStep:
        dt = float(max(0.0, dt))
        carrot = np.asarray(carrot_xyz, dtype=float).reshape(3)
        if not self.plan.waypoints or self._done or not self._running:
            wp = (self.plan.waypoints[-1] if self.plan.waypoints
                  else Waypoint(carrot, carrot, np.array([0.0, 0.0, 1.0]),
                                PHASE_DONE))
            return PathStep(np.zeros(3), PHASE_DONE if self._done else wp.phase,
                            self._i, wp.xyz, wp.surface_xyz, wp.normal,
                            False, self._done, self._running)

        # Retire every waypoint already inside the arrival ball, so a coarse
        # ``ds_m`` next to a fast tick cannot leave the runner chasing a point
        # it has passed.
        while self._i < len(self.plan):
            wp = self.plan.waypoints[self._i]
            if float(np.linalg.norm(self.goal_of(wp) - carrot)) > self.arrive_m:
                break
            self._i += 1
        if self._i >= len(self.plan):
            self._done = True
            self._running = False
            last = self.plan.waypoints[-1]
            return PathStep(np.zeros(3), PHASE_DONE, self._i, last.xyz,
                            last.surface_xyz, last.normal, False, True, False)

        wp = self.plan.waypoints[self._i]
        goal = self.goal_of(wp)
        to_go = goal - carrot
        d = float(np.linalg.norm(to_go))
        v = self.speed_for(wp.phase)
        reach = v * dt
        delta = to_go if (d <= reach or d < 1e-12) else to_go * (reach / d)
        return PathStep(delta, wp.phase, self._i, goal, wp.surface_xyz,
                        wp.normal, wp.press_phase, False, True)


def preview_positions(plan: PathPlan, *, free_speed_m_s=0.02,
                      contact_speed_m_s=0.01, hz=50.0) -> np.ndarray:
    """Positions a ghost would visit walking ``plan`` -- pure, for the preview."""
    if not plan.waypoints:
        return np.zeros((0, 3))
    runner = PathRunner(plan, free_speed_m_s=free_speed_m_s,
                        contact_speed_m_s=contact_speed_m_s)
    runner.start()
    dt = 1.0 / max(1.0, float(hz))
    p = np.array(plan.waypoints[0].xyz, dtype=float)
    out = [p.copy()]
    for _ in range(int(hz * 600)):  # 10 minutes of walk is a hard stop
        s = runner.step(dt, p)
        if s.done:
            break
        p = p + s.delta_world
        out.append(p.copy())
    return np.asarray(out, dtype=float)


def phase_normal(plan: PathPlan, index: int) -> Sequence[float]:
    if not plan.waypoints:
        return np.array([0.0, 0.0, 1.0])
    return plan.waypoints[int(np.clip(index, 0, len(plan) - 1))].normal
