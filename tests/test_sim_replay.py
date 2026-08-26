import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from geodude_teleoperation.sim_replay import (
    absolute_error_mm,
    apply_mode_edge,
    bag_tcp_flange_velocity,
    body_twist_for_plant,
    format_tracking_table,
    path_follow_contact,
    relative_error_mm,
    replay_mode_label,
    score_legend,
    summarize_tracking,
    ur_reaction_fz,
)


class PlantTwistTest(unittest.TestCase):
    def test_approach_sends_displacement_not_velocity(self):
        v = np.array([0.10, 0.0, 0.0])
        tw = body_twist_for_plant(
            v, dt_s=0.1, contact=False, R_flange=np.eye(3), R_plant=np.eye(3))
        np.testing.assert_allclose(tw[:3], 0.0)
        np.testing.assert_allclose(tw[3:], [0.01, 0.0, 0.0])

    def test_contact_sends_velocity(self):
        v = np.array([0.10, 0.0, 0.0])
        tw = body_twist_for_plant(
            v, dt_s=0.1, contact=True, R_flange=np.eye(3), R_plant=np.eye(3))
        np.testing.assert_allclose(tw[3:], [0.10, 0.0, 0.0])

    def test_converts_flange_body_into_yawed_plant_ee(self):
        R_plant = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        v_flange = np.array([0.10, 0.0, 0.0])
        tw = body_twist_for_plant(
            v_flange, dt_s=0.1, contact=True,
            R_flange=np.eye(3), R_plant=R_plant)
        np.testing.assert_allclose(tw[3:], [0.0, -0.10, 0.0], atol=1e-12)


class BagTcpFlangeVelocityTest(unittest.TestCase):
    def test_identity_flange_is_world_delta_over_dt(self):
        xyz = np.array([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [0.10, 0.20, 0.0]])
        R = np.stack([np.eye(3), np.eye(3), np.eye(3)])
        v = bag_tcp_flange_velocity(xyz, R, dt_s=0.1)
        np.testing.assert_allclose(v, [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    def test_yawed_flange_does_not_use_field_frame(self):
        xyz = np.array([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]])
        R_flange = np.array([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
        v = bag_tcp_flange_velocity(xyz, R_flange, dt_s=0.1)
        np.testing.assert_allclose(v, [[0.0, -1.0, 0.0]])


class VelocityIkTest(unittest.TestCase):
    def test_error_maps_to_negative_task_velocity(self):
        from geodude_teleoperation.sim_replay import task_twist_from_error

        err = np.array([0.0, 0.0, 0.0, 0.02, 0.0, 0.0])
        tw = task_twist_from_error(err, kp=10.0)
        np.testing.assert_allclose(tw[3:], [-0.20, 0.0, 0.0])
        np.testing.assert_allclose(tw[:3], 0.0)

    def test_contact_replaces_body_z_with_dualloop(self):
        from geodude_teleoperation.sim_replay import task_twist_from_error

        err = np.array([0.1, 0.0, 0.0, 0.02, 0.01, 0.05])
        tw = task_twist_from_error(err, kp=10.0, z_velocity=-0.004)
        np.testing.assert_allclose(tw[5], -0.004)
        np.testing.assert_allclose(tw[3], -0.20)
        np.testing.assert_allclose(tw[0], -1.0)

    def test_feedforward_adds_on_xy_dualloop_keeps_z(self):
        from geodude_teleoperation.sim_replay import task_twist_from_error

        err = np.zeros(6)
        ff = np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.02])
        tw = task_twist_from_error(err, kp=10.0, z_velocity=-0.004, feedforward=ff)
        np.testing.assert_allclose(tw[3], 0.05)
        np.testing.assert_allclose(tw[5], -0.004)

    def test_damped_qdot_tracks_six_task_rows(self):
        from geodude_teleoperation.sim_replay import damped_least_squares_qdot

        J = np.eye(6, 7)
        twist = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
        qdot = damped_least_squares_qdot(J, twist)
        self.assertEqual(qdot.shape, (7,))
        self.assertAlmostEqual(float(qdot[3]), 0.1, places=6)
        self.assertAlmostEqual(float(qdot[0]), 0.0, places=6)


class ModeEdgeTest(unittest.TestCase):
    def test_raw_edges_enter_and_exit_once(self):
        enters, exits = [], []
        mode = {"active": "approach"}
        raw = [False, False, True, True, False, False]
        modes = []
        for i, flag in enumerate(raw):
            apply_mode_edge(
                mode, flag,
                on_enter=lambda i=i: enters.append(i),
                on_exit=lambda i=i: exits.append(i),
            )
            modes.append(mode["active"])
        self.assertEqual(enters, [2])
        self.assertEqual(exits, [4])
        self.assertEqual(
            modes,
            ["approach", "approach", "contact", "contact", "approach", "approach"],
        )


class TrackingErrorTest(unittest.TestCase):
    def test_relative_path_ignores_start_offset(self):
        bag0 = np.array([1.0, 2.0, 3.0])
        live0 = np.array([0.0, 0.0, 0.0])
        xy, z, n = relative_error_mm(
            live_xyz=np.array([0.01, 0.0, 0.0]),
            bag_xyz=np.array([1.01, 2.0, 3.0]),
            live0=live0,
            bag0=bag0,
        )
        self.assertAlmostEqual(xy, 0.0)
        self.assertAlmostEqual(z, 0.0)
        self.assertAlmostEqual(n, 0.0)

    def test_absolute_error_is_live_minus_bag(self):
        xy, z, n = absolute_error_mm(
            live_xyz=np.array([1.01, 2.0, 3.05]),
            bag_xyz=np.array([1.0, 2.0, 3.0]),
        )
        self.assertAlmostEqual(xy, 10.0)
        self.assertAlmostEqual(z, 50.0)
        self.assertAlmostEqual(n, float(np.hypot(10.0, 50.0)))

    def test_table_lists_mode_and_errors(self):
        from geodude_teleoperation.sim_replay import TrackingRow

        rows = [
            TrackingRow(0, 0.0, "approach", 1.0, 0.5, 1.1, 0.2, 0.0),
            TrackingRow(1, 0.1, "contact", 2.0, 3.0, 3.6, 40.0, 0.005),
        ]
        text = format_tracking_table(rows)
        self.assertIn("approach", text)
        self.assertIn("contact", text)
        self.assertIn("40.00", text)

    def test_controller_score_is_contact_xy_not_3d(self):
        from geodude_teleoperation.sim_replay import TrackingRow

        rows = [
            TrackingRow(0, 0.0, "approach", 1.0, 0.5, 1.1, 0.2, 0.0),
            TrackingRow(1, 0.1, "contact", 2.0, 40.0, 40.1, 40.0, 0.005),
            TrackingRow(2, 0.2, "contact", 4.0, 40.0, 40.2, 40.0, 0.005),
        ]
        text = summarize_tracking(rows)
        legend = score_legend(follow_path=False)
        self.assertIn("mean_xy", text)
        self.assertIn("XY", legend)
        self.assertIn("DualLoop", legend)
        self.assertNotIn("3D", legend)

    def test_path_follow_keeps_press_and_skips_dualloop(self):
        raw = np.array([False, True, True, False])
        np.testing.assert_array_equal(path_follow_contact(raw, follow_path=True), False)
        np.testing.assert_array_equal(path_follow_contact(raw, follow_path=False), raw)
        self.assertEqual(replay_mode_label(True, follow_path=True), "path")
        self.assertEqual(replay_mode_label(False, follow_path=True), "path")
        self.assertEqual(replay_mode_label(True, follow_path=False), "contact")
        self.assertIn("path", score_legend(follow_path=True))
        self.assertIn("err_mm", score_legend(follow_path=True))

    def test_write_tracking_session_saves_npz(self):
        import tempfile
        from pathlib import Path

        from geodude_teleoperation.sim_replay import (
            TrackingRow,
            write_tracking_session,
        )

        rows = [
            TrackingRow(0, 0.0, "approach", 1.0, 0.5, 1.1, 0.2, 0.0, 0.1),
            TrackingRow(1, 0.1, "contact", 2.0, 3.0, 3.6, 40.0, 0.005, -12.0),
        ]
        tmp = Path(tempfile.mkdtemp())
        png = tmp / "tracking.png"
        out = write_tracking_session(rows, png)
        npz = png.with_suffix(".npz")
        self.assertTrue(npz.is_file())
        z = np.load(npz)
        np.testing.assert_allclose(z["err_xy_mm"], [1.0, 2.0])
        np.testing.assert_allclose(z["meas_fz"], [0.1, -12.0])
        self.assertIsNotNone(out)


class SlowReplayTest(unittest.TestCase):
    def test_start_gap_refuses_when_far(self):
        from geodude_teleoperation.sim_replay import (
            refuse_start_gap,
            start_gap_mm,
        )

        gap = start_gap_mm(np.array([0.0, 0.0, 0.0]), np.array([0.030, 0.0, 0.0]))
        self.assertAlmostEqual(gap, 30.0)
        with self.assertRaises(RuntimeError):
            refuse_start_gap(gap, limit_mm=25.0)
        refuse_start_gap(10.0, limit_mm=25.0)

    def test_wait_until_start_gap_returns_when_close(self):
        from geodude_teleoperation.sim_replay import wait_until_start_gap

        gaps = [80.0, 50.0, 20.0]
        i = {"n": 0}

        def live():
            g = gaps[min(i["n"], len(gaps) - 1)]
            i["n"] += 1
            return g

        ticks: list[float] = []
        g = wait_until_start_gap(
            live, limit_mm=25.0, timeout_s=5.0, dt_s=0.0, on_tick=ticks.append)
        self.assertAlmostEqual(g, 20.0)
        self.assertEqual(ticks, [80.0, 50.0])

    def test_wait_until_start_gap_times_out(self):
        from geodude_teleoperation.sim_replay import wait_until_start_gap

        with self.assertRaises(RuntimeError):
            wait_until_start_gap(
                lambda: 63.9, limit_mm=25.0, timeout_s=0.05, dt_s=0.01)

    def test_ramp_duration_is_distance_over_speed(self):
        from geodude_teleoperation.sim_replay import ramp_duration_s

        self.assertAlmostEqual(ramp_duration_s(0.10, speed_mps=0.02), 5.0)
        self.assertAlmostEqual(ramp_duration_s(0.01, speed_mps=0.02, min_s=2.0), 2.0)

    def test_blend_pose_takes_short_arc_when_quat_signs_differ(self):
        from geodude_teleoperation.sim_replay import blend_pose_matrix
        from scipy.spatial.transform import Rotation

        T0 = np.eye(4)
        T1 = np.eye(4)
        T1[:3, :3] = Rotation.from_euler("z", 10, degrees=True).as_matrix()
        T1_flip = T1.copy()
        q = Rotation.from_matrix(T1[:3, :3]).as_quat()
        T1_flip[:3, :3] = Rotation.from_quat(-q).as_matrix()
        mid = blend_pose_matrix(T0, T1_flip, 0.5)
        ang = float(np.degrees(np.linalg.norm(
            Rotation.from_matrix(mid[:3, :3]).as_rotvec())))
        self.assertLess(ang, 10.0)

    def test_blend_pose_midpoint_is_halfway(self):
        from geodude_teleoperation.sim_replay import blend_pose_matrix

        T0 = np.eye(4)
        T1 = np.eye(4)
        T1[:3, 3] = [0.10, 0.0, 0.0]
        T = blend_pose_matrix(T0, T1, 0.5)
        np.testing.assert_allclose(T[:3, 3], [0.05, 0.0, 0.0])
        np.testing.assert_allclose(blend_pose_matrix(T0, T1, 0.0)[:3, 3], T0[:3, 3])
        np.testing.assert_allclose(blend_pose_matrix(T0, T1, 1.0)[:3, 3], T1[:3, 3])

    def test_blend_translation_keeps_start_rotation(self):
        from geodude_teleoperation.sim_replay import blend_translation_matrix
        from scipy.spatial.transform import Rotation

        T0 = np.eye(4)
        T1 = np.eye(4)
        T1[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        T1[:3, 3] = [0.10, 0.0, 0.0]
        T = blend_translation_matrix(T0, T1, 1.0)
        np.testing.assert_allclose(T[:3, 3], T1[:3, 3])
        np.testing.assert_allclose(T[:3, :3], T0[:3, :3])

    def test_speedj_stalled_when_cmd_without_motion(self):
        from geodude_teleoperation.sim_replay import speedj_stalled

        self.assertTrue(speedj_stalled(0.10, 0.0))
        self.assertFalse(speedj_stalled(0.10, 0.05))
        self.assertFalse(speedj_stalled(0.0, 0.0))

    def test_slow_clock_is_five_times_longer_at_speed_0_2(self):
        from geodude_teleoperation.sim_replay import slow_replay_clock

        t = np.array([0.0, 0.1, 0.2])
        t_wall, t_src = slow_replay_clock(t, speed=0.2, cmd_hz=10.0)
        self.assertAlmostEqual(float(t_wall[-1]), 1.0)
        np.testing.assert_allclose(t_src, t_wall * 0.2)
        self.assertGreaterEqual(len(t_wall), 10)

    def test_interpolated_path_moves_at_scaled_speed(self):
        from geodude_teleoperation.sim_replay import (
            interpolate_tcp_xyzw,
            slow_replay_clock,
        )

        t = np.array([0.0, 0.1, 0.2])
        poses = np.zeros((3, 7))
        poses[:, 6] = 1.0
        poses[:, 0] = [0.0, 0.015, 0.030]
        t_wall, t_src = slow_replay_clock(t, speed=0.2, cmd_hz=10.0)
        out = interpolate_tcp_xyzw(poses, t, t_src)
        self.assertAlmostEqual(float(out[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(out[-1, 0]), 0.030, places=6)
        dt = float(np.diff(t_wall).mean())
        v = np.diff(out[:, 0]) / dt
        self.assertAlmostEqual(float(np.mean(v)), 0.030, places=3)


class BagForceAsUrTest(unittest.TestCase):
    def test_contact_is_negative_reaction(self):
        self.assertAlmostEqual(ur_reaction_fz(40.0, in_contact=True), -40.0)
        self.assertAlmostEqual(ur_reaction_fz(-12.0, in_contact=True), -12.0)
        self.assertAlmostEqual(ur_reaction_fz(40.0, in_contact=False), 0.0)


class SpeedjHandoffTest(unittest.TestCase):
    def test_enter_keeps_pose_plant_and_starts_force_overlay(self):
        from geodude_teleoperation.sim_replay import (
            enter_speedj_contact,
            exit_speedj_contact,
        )

        class Approach:
            def __init__(self):
                self.vz = 0.0
                self.held = False
                self.active = True
                self.z_compliant = False

            def hold_live_pose(self):
                self.held = True

            def set_input_active(self, active):
                self.active = active

            def set_tool_z_velocity(self, vz):
                self.vz = vz

            def set_z_compliant(self, compliant):
                self.z_compliant = compliant

        class Contact:
            def __init__(self):
                self.started = False
                self.on_vz = None

            def set_on_vz(self, cb):
                self.on_vz = cb

            def start(self):
                self.started = True

            def stop(self):
                self.started = False

        approach = Approach()
        contact = Contact()
        mode = {"active": "approach"}
        enter_speedj_contact(
            mode=mode, contact_controller=contact, approach_controller=approach)
        self.assertEqual(mode["active"], "contact")
        self.assertTrue(contact.started)
        self.assertEqual(contact.on_vz, approach.set_tool_z_velocity)
        self.assertTrue(approach.z_compliant)
        exit_speedj_contact(
            mode=mode, contact_controller=contact, approach_controller=approach)
        self.assertEqual(mode["active"], "approach")
        self.assertFalse(contact.started)
        self.assertFalse(approach.z_compliant)

    def test_enter_keep_pose_xy_does_not_freeze_or_slide(self):
        from geodude_teleoperation.sim_replay import enter_speedj_contact

        class Approach:
            def __init__(self):
                self.vz = 0.0
                self.held = False
                self.active = False
                self.z_compliant = False
                self.force_owns_z = False

            def hold_live_pose(self):
                self.held = True

            def set_input_active(self, active):
                self.active = active

            def set_tool_z_velocity(self, vz):
                self.vz = vz

            def set_z_compliant(self, compliant):
                self.z_compliant = compliant

            def set_force_owns_z(self, owns):
                self.force_owns_z = owns

        class Contact:
            def __init__(self):
                self.started = False
                self.on_vz = None

            def set_on_vz(self, cb):
                self.on_vz = cb

            def start(self):
                self.started = True

        approach = Approach()
        contact = Contact()
        mode = {"active": "approach"}
        enter_speedj_contact(
            mode=mode, contact_controller=contact, approach_controller=approach,
            keep_pose_xy=True)
        self.assertEqual(mode["active"], "contact")
        self.assertTrue(contact.started)
        self.assertFalse(approach.held)
        self.assertTrue(approach.active)
        self.assertFalse(approach.z_compliant)
        self.assertFalse(approach.force_owns_z)


class ForceOwnsZTest(unittest.TestCase):
    def test_zeros_kz_without_velocity_slide(self):
        from geodude_teleoperation.controller import AdmittanceTeleoperationController

        c = AdmittanceTeleoperationController(
            stiffness=np.diag([2000.0] * 6))
        self.assertAlmostEqual(float(c._stiffness[5, 5]), 2000.0)
        c.set_force_owns_z(True)
        self.assertAlmostEqual(float(c._stiffness[5, 5]), 0.0)
        self.assertFalse(c._z_compliant)
        c.set_force_owns_z(False)
        self.assertAlmostEqual(float(c._stiffness[5, 5]), 2000.0)


if __name__ == "__main__":
    unittest.main()
