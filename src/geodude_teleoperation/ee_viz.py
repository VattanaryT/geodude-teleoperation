"""Viser-only EE cosmetics. Does not change gafropy FK or the UR TCP."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

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


def add_right_chamfer_visual(viz, system, *, q):
    """Draw the chamfer on the right flange. Returns an updater(q)."""
    import trimesh

    raw = trimesh.load(str(CHAMFER_MESH), force="mesh")
    mesh = raw.copy()
    mesh.vertices = prepare_chamfer_vertices_m(np.asarray(raw.vertices))
    handle = viz.scene.add_mesh_trimesh("/right_chamfer_ee", mesh)

    def update(q_now):
        fk = system.compute_forward_kinematics(np.asarray(q_now, dtype=float))
        T_wrist = np.asarray(
            fk.get_link_pose(RIGHT_WRIST3).to_transformation_matrix())
        world = chamfer_world_from_wrist3(T_wrist)
        handle.wxyz = trimesh.transformations.quaternion_from_matrix(world)
        handle.position = world[:3, 3]

    update(q)
    return update
