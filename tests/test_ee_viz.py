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

    def test_chamfer_tcp_is_pendant_offset_not_robotiq_length(self):
        from geodude_teleoperation.ee_viz import (
            chamfer_tcp_world_from_wrist3,
            flange_from_wrist3,
        )

        T_wrist = np.eye(4)
        flange = flange_from_wrist3()[:3, 3]
        tcp = chamfer_tcp_world_from_wrist3(T_wrist)[:3, 3]
        delta = tcp - flange
        self.assertAlmostEqual(float(np.linalg.norm(delta)), 0.1443, places=3)
        self.assertGreater(float(np.linalg.norm(delta)), 0.12)
        self.assertLess(float(np.linalg.norm(delta)), 0.16)

    def test_motor_from_matrix_keeps_chamfer_tcp_xyz(self):
        from geodude_teleoperation.ee_viz import (
            chamfer_tcp_world_from_wrist3,
            motor_from_matrix,
        )

        T = chamfer_tcp_world_from_wrist3(np.eye(4))
        motor = motor_from_matrix(T)
        tr = motor.get_translator()
        np.testing.assert_allclose(
            [tr.x(), tr.y(), tr.z()], T[:3, 3], atol=1e-9)

    def test_ghost_flange_at_field_keeps_tcp_and_matches_field_z(self):
        from geodude_teleoperation.ee_viz import (
            CHAMFER_TCP_FROM_FLANGE_XYZ,
            ghost_flange_world,
        )

        T_live = np.eye(4)
        T_live[:3, 3] = [0.4, -0.2, 0.9]
        p_tcp = T_live[:3, 3] + T_live[:3, :3] @ CHAMFER_TCP_FROM_FLANGE_XYZ
        R_field = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        T0 = ghost_flange_world(T_live, R_field, p_tcp, 0.0)
        T1 = ghost_flange_world(T_live, R_field, p_tcp, 1.0)
        np.testing.assert_allclose(T0, T_live, atol=1e-9)
        np.testing.assert_allclose(T1[:3, :3], R_field, atol=1e-9)
        np.testing.assert_allclose(
            T1[:3, 3] + T1[:3, :3] @ CHAMFER_TCP_FROM_FLANGE_XYZ, p_tcp, atol=1e-9)
        np.testing.assert_allclose(T1[:3, :3] @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], atol=1e-9)

    def test_ghost_into_surface_puts_flange_above_tcp(self):
        from geodude_teleoperation.ee_viz import (
            CHAMFER_TCP_FROM_FLANGE_XYZ,
            ghost_flange_world,
            live_tcp_xyz_from_flange,
        )

        T_live = np.eye(4)
        T_live[:3, 3] = [0.0, 0.0, 1.0]
        p_tcp = live_tcp_xyz_from_flange(T_live)
        R_into = np.diag([1.0, -1.0, -1.0])
        T1 = ghost_flange_world(T_live, R_into, p_tcp, 1.0)
        np.testing.assert_allclose(
            T1[:3, :3] @ [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], atol=1e-9)
        self.assertGreater(float(T1[2, 3]), float(p_tcp[2]))
        np.testing.assert_allclose(
            T1[:3, 3] + T1[:3, :3] @ CHAMFER_TCP_FROM_FLANGE_XYZ, p_tcp, atol=1e-9)

    def test_pingpong_alpha_hits_live_and_field(self):
        from geodude_teleoperation.ee_viz import pingpong_alpha

        self.assertAlmostEqual(pingpong_alpha(0.0, period=4.0), 0.0)
        self.assertAlmostEqual(pingpong_alpha(2.0, period=4.0), 1.0)
        self.assertAlmostEqual(pingpong_alpha(4.0, period=4.0), 0.0)

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
