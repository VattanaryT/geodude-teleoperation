"""Viser-only EE cosmetics. Does not change gafropy FK or the UR TCP."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

_REPO = Path(__file__).resolve().parent.parent.parent
CHAMFER_MESH = _REPO / "assets" / "robots" / "geodude" / "assets" / "chamfer_ee.stl"
CHAMFER_MESH_SCALE = 0.001  # stl vertices are mm
RIGHT_WRIST3 = "right_ur5e/wrist_3_link"

# Same as geodude.yaml right_ur5e/gripper/_fixed: wrist_3 → flange.
FLANGE_FROM_WRIST3_XYZ = np.array([0.0, 0.1, 0.0], dtype=float)
FLANGE_FROM_WRIST3_QUAT_WXYZ = np.array(
    [-0.7071067811865475, 0.7071067811865475, 0.0, 0.0], dtype=float)
# CAD already has the mount face on z=0 (xy ≈ 75 mm, tool along +z ≈ 120 mm).
# A 90° X or Y rotation puts that face vertical (flat facing sideways).
# Yaw about the flange axis after seating (positive = CCW looking along +z).
CHAMFER_MESH_RZ_DEG = 40
# Pendant chamfer TCP in UR tool0 / flange (geodude-teleoperation calibration/right.yaml).
# Do not stack the 30 mm plates on top of this — they are already in the taught TCP.
CHAMFER_TCP_FROM_FLANGE_XYZ = np.array([-0.0049, -0.00029, 0.14422], dtype=float)
CHAMFER_TCP_FROM_FLANGE_ROTVEC = np.array([1.20695, 1.20049, 1.20171], dtype=float)


def is_gripper_visual_link(name: str) -> bool:
    return "/gripper" in name


def hide_gripper_visuals(robot) -> int:
    """Hide Robotiq meshes. Leaves kinematics (and EE frame) unchanged."""
    n = 0
    for handle, (link_name, _, _) in zip(robot._handles, robot.visuals):
        if is_gripper_visual_link(link_name):
            handle.visible = False
            n += 1
    return n


def _quat_wxyz_matrix(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(wxyz, dtype=float).reshape(4)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def flange_from_wrist3() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _quat_wxyz_matrix(FLANGE_FROM_WRIST3_QUAT_WXYZ)
    T[:3, 3] = FLANGE_FROM_WRIST3_XYZ
    return T


def chamfer_tcp_from_flange() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(CHAMFER_TCP_FROM_FLANGE_ROTVEC).as_matrix()
    T[:3, 3] = CHAMFER_TCP_FROM_FLANGE_XYZ
    return T


def chamfer_tcp_from_wrist3() -> np.ndarray:
    return flange_from_wrist3() @ chamfer_tcp_from_flange()


def chamfer_tcp_world_from_wrist3(T_wrist3: np.ndarray) -> np.ndarray:
    return np.asarray(T_wrist3, dtype=float).reshape(4, 4) @ chamfer_tcp_from_wrist3()


def motor_from_matrix(T: np.ndarray):
    import gafropy as ga

    return ga.Motor.from_matrix(np.asarray(T, dtype=float).reshape(4, 4))


def chamfer_tcp_motor(system, q):
    """Chamfer pendant TCP in the gafropy world (vention-overlaid q)."""
    fk = system.compute_forward_kinematics(np.asarray(q, dtype=float))
    T_wrist = np.asarray(
        fk.get_link_pose(RIGHT_WRIST3).to_transformation_matrix())
    return motor_from_matrix(chamfer_tcp_world_from_wrist3(T_wrist))


def matrix_from_motor(motor) -> np.ndarray:
    return np.asarray(motor.to_transformation_matrix(), dtype=float)


def flange_world_from_tcp(T_tcp: np.ndarray) -> np.ndarray:
    return np.asarray(T_tcp, dtype=float).reshape(4, 4) @ np.linalg.inv(
        chamfer_tcp_from_flange())


def slerp_rotation(R0: np.ndarray, R1: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    key = Rotation.concatenate([
        Rotation.from_matrix(np.asarray(R0, dtype=float).reshape(3, 3)),
        Rotation.from_matrix(np.asarray(R1, dtype=float).reshape(3, 3)),
    ])
    return Slerp([0.0, 1.0], key)(alpha).as_matrix()


def pingpong_alpha(now: float, period: float = 4.0) -> float:
    """0 at live, 1 at field, back to 0. Period is a full round trip."""
    period = float(period)
    x = (float(now) % period) / period
    return 1.0 - abs(2.0 * x - 1.0)


class SimTcpProbe:
    """Viser-only TCP. Stick linear is applied in the current EE frame by the ticker."""

    def __init__(self, xyz):
        self._lock = threading.Lock()
        self._xyz = np.asarray(xyz, dtype=float).reshape(3).copy()
        self._pending = np.zeros(3, dtype=float)

    def add_body_linear(self, linear):
        with self._lock:
            self._pending = self._pending + np.asarray(linear, dtype=float).reshape(3)

    def clear_pending(self):
        with self._lock:
            self._pending[:] = 0.0

    def xyz(self):
        with self._lock:
            return self._xyz.copy()

    def step(self, R_ee):
        with self._lock:
            delta = self._pending
            self._pending = np.zeros(3, dtype=float)
            self._xyz = self._xyz + np.asarray(R_ee, dtype=float).reshape(3, 3) @ delta
            return self._xyz.copy()


def live_tcp_xyz_from_flange(T_flange: np.ndarray) -> np.ndarray:
    """Tip along the chamfer +Z (mesh axis), not the pendant TCP rotvec."""
    T = np.asarray(T_flange, dtype=float).reshape(4, 4)
    return T[:3, 3] + T[:3, :3] @ CHAMFER_TCP_FROM_FLANGE_XYZ


def ghost_flange_world(T_live_flange: np.ndarray, R_field: np.ndarray, p_tcp: np.ndarray,
                       alpha: float) -> np.ndarray:
    """Slerp flange rotation to the field, keeping the live TCP xyz.

    ``R_field`` column 2 is EE +Z (into the surface), opposite the blue
    /field_z arrows (object +normal). Do not compose the pendant TCP rotvec
    — that swings the 3mf through the table.

    Holding TCP parks the flange on the far side of the tip (a retract). Startup
    align uses ``align_flange_world`` instead so the tip moves toward the work.
    """
    T_live = np.asarray(T_live_flange, dtype=float).reshape(4, 4)
    R = slerp_rotation(T_live[:3, :3], R_field, alpha)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p_tcp, dtype=float).reshape(3) - R @ CHAMFER_TCP_FROM_FLANGE_XYZ
    return T


def align_flange_world(T_live_flange: np.ndarray, R_field: np.ndarray,
                       alpha: float) -> np.ndarray:
    """Slerp flange rotation to the field, keeping live flange xyz.

    Tool +Z into the surface then moves the tip toward the work. Same
    ``R_field`` as ``ghost_flange_world``; different translation.
    """
    T_live = np.asarray(T_live_flange, dtype=float).reshape(4, 4)
    T = np.eye(4)
    T[:3, :3] = slerp_rotation(T_live[:3, :3], R_field, alpha)
    T[:3, 3] = T_live[:3, 3]
    return T


def orientation_error_rad(R0: np.ndarray, R1: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices."""
    rel = Rotation.from_matrix(R0).inv() * Rotation.from_matrix(R1)
    return float(rel.magnitude())


class AlignTick:
    __slots__ = ("phase", "T_ee", "input_active", "finished", "alpha")

    def __init__(self, phase, T_ee=None, input_active=False, finished=False, alpha=0.0):
        self.phase = phase
        self.T_ee = T_ee
        self.input_active = input_active
        self.finished = finished
        self.alpha = float(alpha)


class StartupAlign:
    """One-shot flange slerp to the field. Spring tracks a moving target; no project() snap.

    Wait until ``trigger()`` (``delay_s is None``) or ``delay_s`` seconds, then
    ramp ``duration_s``, then hold when the live flange is close to the field
    or ``settle_s`` elapses. Ramp keeps live flange xyz so the tip approaches.
    """

    def __init__(self, *, delay_s=3.0, duration_s=4.0, err_rad=0.05, settle_s=2.0):
        self.delay_s = None if delay_s is None else float(delay_s)
        self.duration_s = max(float(duration_s), 1e-6)
        self.err_rad = float(err_rad)
        self.settle_s = float(settle_s)
        self.phase = "idle"
        self._t0 = None
        self._T_flange0 = None
        self._R_field = None
        self._T_ee_from_flange = None
        self._goal_flange = None
        self._triggered = False

    @property
    def finished(self):
        return self.phase == "done"

    def arm(self, now):
        if self.phase == "idle":
            self.phase = "wait"
            self._t0 = float(now)

    def trigger(self):
        if self.phase in ("idle", "wait"):
            self._triggered = True

    def desired_flange(self):
        """Snap pose the orange ghost should show (alpha=1). None until wait ends."""
        if self._T_flange0 is None or self._R_field is None:
            return None
        return align_flange_world(self._T_flange0, self._R_field, 1.0)

    def step(self, now, T_flange_live, R_field, T_ee_live):
        now = float(now)
        T_flange_live = np.asarray(T_flange_live, dtype=float).reshape(4, 4)
        T_ee_live = np.asarray(T_ee_live, dtype=float).reshape(4, 4)
        R_field = np.asarray(R_field, dtype=float).reshape(3, 3)

        if self.phase == "idle":
            return AlignTick("idle")

        if self.phase == "wait":
            ready = (
                self._triggered if self.delay_s is None
                else (now - self._t0) >= self.delay_s)
            if not ready:
                return AlignTick("wait")
            self._T_flange0 = T_flange_live.copy()
            self._R_field = R_field.copy()
            self._T_ee_from_flange = np.linalg.inv(T_flange_live) @ T_ee_live
            self._t0 = now
            self.phase = "ramp"

        if self.phase == "ramp":
            alpha = float(np.clip((now - self._t0) / self.duration_s, 0.0, 1.0))
            T_flange = align_flange_world(
                self._T_flange0, self._R_field, alpha)
            T_ee = T_flange @ self._T_ee_from_flange
            if alpha >= 1.0:
                self._goal_flange = T_flange
                self.phase = "settle"
                self._t0 = now
            return AlignTick("ramp" if alpha < 1.0 else "settle",
                             T_ee=T_ee, input_active=True, alpha=alpha)

        if self.phase == "settle":
            T_ee = self._goal_flange @ self._T_ee_from_flange
            err = orientation_error_rad(T_flange_live[:3, :3], self._R_field)
            if err <= self.err_rad or (now - self._t0) >= self.settle_s:
                self.phase = "done"
                return AlignTick("done", input_active=False, finished=True, alpha=1.0)
            return AlignTick("settle", T_ee=T_ee, input_active=True, alpha=1.0)

        return AlignTick("done", finished=True, alpha=1.0)


def seat_mesh_on_xy_origin(vertices: np.ndarray) -> np.ndarray:
    """Put the lowest face on z=0 and center xy on the origin (flange axis)."""
    v = np.asarray(vertices, dtype=float)
    lo = v.min(axis=0)
    hi = v.max(axis=0)
    shift = np.array(
        [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2]], dtype=float)
    return v - shift


def prepare_chamfer_vertices_m(vertices_mm: np.ndarray) -> np.ndarray:
    """Scale mm→m, seat the CAD z=0 mount on the flange, then yaw about +z."""
    v = seat_mesh_on_xy_origin(
        np.asarray(vertices_mm, dtype=float) * CHAMFER_MESH_SCALE)
    if CHAMFER_MESH_RZ_DEG:
        v = Rotation.from_euler("z", CHAMFER_MESH_RZ_DEG, degrees=True).apply(v)
    return v


def chamfer_world_from_wrist3(T_wrist3: np.ndarray) -> np.ndarray:
    return np.asarray(T_wrist3, dtype=float).reshape(4, 4) @ flange_from_wrist3()


def _set_mesh_pose(handle, world: np.ndarray) -> None:
    import trimesh

    world = np.asarray(world, dtype=float).reshape(4, 4)
    handle.wxyz = trimesh.transformations.quaternion_from_matrix(world)
    handle.position = world[:3, 3]


def _load_seated_chamfer(color_rgba):
    import trimesh

    raw = trimesh.load(str(CHAMFER_MESH), force="mesh")
    mesh = raw.copy()
    mesh.vertices = prepare_chamfer_vertices_m(np.asarray(raw.vertices))
    mesh.visual.face_colors = np.tile(
        np.asarray(color_rgba, dtype=np.uint8).reshape(1, 4),
        (len(mesh.faces), 1))
    return mesh


def add_right_chamfer_visual(viz, system, *, q):
    """Draw the chamfer on the right flange. Returns an updater(q)."""
    handle = viz.scene.add_mesh_trimesh(
        "/right_chamfer_ee", _load_seated_chamfer((160, 160, 160, 255)))

    def update(q_now):
        fk = system.compute_forward_kinematics(np.asarray(q_now, dtype=float))
        T_wrist = np.asarray(
            fk.get_link_pose(RIGHT_WRIST3).to_transformation_matrix())
        _set_mesh_pose(handle, chamfer_world_from_wrist3(T_wrist))

    update(q)
    return update


def add_ghost_chamfer_visual(viz, *, visible=True):
    """Orange ghost mesh. Pass a flange-frame world pose to the updater."""
    handle = viz.scene.add_mesh_trimesh(
        "/right_chamfer_ee_field_ghost", _load_seated_chamfer((255, 140, 0, 140)))
    handle.visible = bool(visible)

    def update(T_flange_world):
        _set_mesh_pose(handle, T_flange_world)

    return update
