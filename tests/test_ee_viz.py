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

    def test_align_flange_keeps_xyz_so_tip_approaches_not_retracts(self):
        from geodude_teleoperation.ee_viz import (
            CHAMFER_TCP_FROM_FLANGE_XYZ,
            align_flange_world,
            ghost_flange_world,
            live_tcp_xyz_from_flange,
        )

        T_live = np.eye(4)
        T_live[:3, 3] = [0.0, 0.0, 1.0]
        p_tcp = live_tcp_xyz_from_flange(T_live)
        R_into = np.diag([1.0, -1.0, -1.0])
        held_tcp = ghost_flange_world(T_live, R_into, p_tcp, 1.0)
        held_flange = align_flange_world(T_live, R_into, 1.0)
        np.testing.assert_allclose(held_flange[:3, 3], T_live[:3, 3], atol=1e-9)
        np.testing.assert_allclose(held_flange[:3, :3], R_into, atol=1e-9)
        tip = held_flange[:3, 3] + held_flange[:3, :3] @ CHAMFER_TCP_FROM_FLANGE_XYZ
        self.assertLess(float(tip[2]), float(p_tcp[2]))
        self.assertGreater(float(held_tcp[2, 3]), float(T_live[2, 3]))

    def test_pingpong_alpha_hits_live_and_field(self):
        from geodude_teleoperation.ee_viz import pingpong_alpha

        self.assertAlmostEqual(pingpong_alpha(0.0, period=4.0), 0.0)
        self.assertAlmostEqual(pingpong_alpha(2.0, period=4.0), 1.0)
        self.assertAlmostEqual(pingpong_alpha(4.0, period=4.0), 0.0)

    def test_sim_tcp_steps_along_ee_z(self):
        from geodude_teleoperation.ee_viz import SimTcpProbe

        probe = SimTcpProbe([0.0, 0.0, 1.0])
        probe.add_body_linear([0.0, 0.0, 0.1])
        xyz = probe.step(np.diag([1.0, -1.0, -1.0]))
        np.testing.assert_allclose(xyz, [0.0, 0.0, 0.9])

    def test_sim_tcp_idle_clears_queued_step(self):
        from geodude_teleoperation.ee_viz import SimTcpProbe

        probe = SimTcpProbe([0.0, 0.0, 1.0])
        probe.add_body_linear([0.0, 0.0, 0.5])
        probe.clear_pending()
        xyz = probe.step(np.eye(3))
        np.testing.assert_allclose(xyz, [0.0, 0.0, 1.0])

    def test_startup_align_waits_then_slerps_without_snap(self):
        from geodude_teleoperation.ee_viz import (
            StartupAlign,
            orientation_error_rad,
        )

        T0 = np.eye(4)
        T0[:3, 3] = [0.0, 0.0, 1.0]
        R_field = np.diag([1.0, -1.0, -1.0])
        T_ee = T0.copy()
        align = StartupAlign(delay_s=1.0, duration_s=2.0, err_rad=0.05, settle_s=0.5)
        align.arm(0.0)
        wait = align.step(0.5, T0, R_field, T_ee)
        self.assertEqual(wait.phase, "wait")
        self.assertIsNone(wait.T_ee)
        self.assertFalse(wait.input_active)

        start = align.step(1.0, T0, R_field, T_ee)
        self.assertTrue(start.input_active)
        np.testing.assert_allclose(start.T_ee[:3, :3], T0[:3, :3], atol=1e-9)

        mid = align.step(2.0, T0, R_field, T_ee)
        err_mid = orientation_error_rad(T0[:3, :3], mid.T_ee[:3, :3])
        err_full = orientation_error_rad(T0[:3, :3], R_field)
        self.assertGreater(err_mid, 0.1)
        self.assertLess(err_mid, err_full - 0.1)

        end = align.step(3.0, T0, R_field, T_ee)
        from geodude_teleoperation.ee_viz import align_flange_world
        T_goal = align_flange_world(T0, R_field, 1.0)
        np.testing.assert_allclose(end.T_ee[:3, :3], T_goal[:3, :3], atol=1e-9)
        np.testing.assert_allclose(
            end.T_ee[:3, 3], T0[:3, 3], atol=1e-9)

        done = align.step(3.6, T_goal, R_field, T_goal)
        self.assertTrue(done.finished)
        self.assertFalse(done.input_active)

    def test_startup_align_waits_for_trigger_not_time(self):
        from geodude_teleoperation.ee_viz import StartupAlign, align_flange_world

        T0 = np.eye(4)
        T0[:3, 3] = [0.0, 0.0, 1.0]
        R_field = np.diag([1.0, -1.0, -1.0])
        align = StartupAlign(delay_s=None, duration_s=1.0, err_rad=0.05, settle_s=0.5)
        align.arm(0.0)
        still = align.step(30.0, T0, R_field, T0)
        self.assertEqual(still.phase, "wait")
        self.assertFalse(still.input_active)
        align.trigger()
        start = align.step(30.1, T0, R_field, T0)
        self.assertTrue(start.input_active)
        np.testing.assert_allclose(start.T_ee[:3, 3], T0[:3, 3], atol=1e-9)
        end = align.step(31.1, T0, R_field, T0)
        np.testing.assert_allclose(
            end.T_ee, align_flange_world(T0, R_field, 1.0), atol=1e-9)
        np.testing.assert_allclose(
            align.desired_flange(), align_flange_world(T0, R_field, 1.0), atol=1e-9)

    def test_trigger_before_arm_still_starts(self):
        from geodude_teleoperation.ee_viz import StartupAlign

        T0 = np.eye(4)
        R_field = np.diag([1.0, -1.0, -1.0])
        align = StartupAlign(delay_s=None, duration_s=1.0)
        align.trigger()
        align.arm(0.0)
        start = align.step(0.0, T0, R_field, T0)
        self.assertTrue(start.input_active)

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
