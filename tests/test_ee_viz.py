import unittest

import numpy as np

from geodude_teleoperation.ee_viz import (
    FLANGE_FROM_WRIST3_XYZ,
    chamfer_world_from_wrist3,
    flange_from_wrist3,
    is_gripper_visual_link,
    prepare_chamfer_vertices_m,
    seat_mesh_on_xy_origin,
)


class EeVizTest(unittest.TestCase):
    def test_gripper_links_are_hidden_left_and_right(self):
        self.assertTrue(is_gripper_visual_link("left_ur5e/gripper/base"))
        self.assertTrue(is_gripper_visual_link("right_ur5e/gripper/left_driver"))
        self.assertFalse(is_gripper_visual_link("right_ur5e/wrist_3_link"))
        self.assertFalse(is_gripper_visual_link("left_ur5e/endeffector_link"))

    def test_chamfer_uses_yaml_flange_not_rviz_wrist_offset(self):
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        world = chamfer_world_from_wrist3(T)
        np.testing.assert_allclose(world[:3, 3], [1.0, 2.0, 3.0] + FLANGE_FROM_WRIST3_XYZ)

    def test_flange_rotation_is_wrist3_rx(self):
        R = flange_from_wrist3()[:3, :3]
        # yaml quat w=-0.707 x=0.707 → ẑ_wrist → ŷ_flange (MuJoCo UR flange).
        np.testing.assert_allclose(R @ [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], atol=1e-9)

    def test_seated_mesh_sits_on_z0_and_xy_origin(self):
        verts = np.array([
            [0.10, 0.14, 0.00],
            [0.18, 0.22, 0.12],
        ], dtype=float)
        seated = seat_mesh_on_xy_origin(verts)
        np.testing.assert_allclose(seated[:, 2].min(), 0.0, atol=1e-12)
        lo, hi = seated.min(axis=0), seated.max(axis=0)
        np.testing.assert_allclose(0.5 * (lo[:2] + hi[:2]), [0.0, 0.0], atol=1e-12)

    def test_prepare_chamfer_seats_real_stl(self):
        import trimesh
        from geodude_teleoperation.ee_viz import CHAMFER_MESH

        raw = trimesh.load(str(CHAMFER_MESH), force="mesh")
        v = prepare_chamfer_vertices_m(np.asarray(raw.vertices))
        self.assertAlmostEqual(float(v[:, 2].min()), 0.0, places=6)
        lo, hi = v.min(axis=0), v.max(axis=0)
        np.testing.assert_allclose(0.5 * (lo[:2] + hi[:2]), [0.0, 0.0], atol=1e-6)
        # Mount on z=0: tool length is the long axis (~120 mm), not a 75 mm side face.
        z_ext = float(hi[2] - lo[2])
        xy_ext = hi[:2] - lo[:2]
        self.assertGreater(z_ext, 0.11)
        self.assertLess(z_ext, 0.13)
        self.assertTrue(np.all(xy_ext < 0.09))

    def test_chamfer_rz_yaws_about_flange_z(self):
        import geodude_teleoperation.ee_viz as ee_viz

        verts = np.array([[100.0, 140.0, 0.0], [180.0, 220.0, 120.0]], dtype=float)
        old = ee_viz.CHAMFER_MESH_RZ_DEG
        ee_viz.CHAMFER_MESH_RZ_DEG = 90.0
        try:
            v = ee_viz.prepare_chamfer_vertices_m(verts)
        finally:
            ee_viz.CHAMFER_MESH_RZ_DEG = old
        # 90° about z: seated (x, y) → (−y, x); z unchanged.
        np.testing.assert_allclose(v[:, 2], [0.0, 0.12], atol=1e-12)
        np.testing.assert_allclose(v[0, :2], [0.04, -0.04], atol=1e-12)
