import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from geodude_teleoperation.sim_replay import (
    absolute_error_mm,
    apply_mode_edge,
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


if __name__ == "__main__":
    unittest.main()
