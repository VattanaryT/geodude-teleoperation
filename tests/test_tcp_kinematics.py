import unittest
from pathlib import Path

import numpy as np

from geodude_teleoperation.tcp_kinematics import (
    MODEL,
    apply_right_ur6,
    poses_in_right_base,
    right_base_points_to_world,
    rotation_error_deg,
    ur_arm_xyz_in_right_base,
    wrap_ur6_to_pm_pi,
    write_geodude_model_with_right_ur6,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bimanual_0_tcp_fk_samples.npz"


class TcpKinematicsFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gafropy as ga

        cls.system = ga.SystemSerialization.load(str(MODEL))
        data = np.load(_FIXTURE)
        cls.q_ur6 = np.asarray(data["q_ur6"], dtype=float)
        cls.tcp = np.asarray(data["tcp_pose_xyzw"], dtype=float)

    def test_chamfer_position_matches_bag_tcp_within_3mm(self):
        from scipy.spatial.transform import Rotation

        q0 = np.asarray(self.system.get_default_configuration(), dtype=float)
        for q6, pose in zip(self.q_ur6, self.tcp):
            q = apply_right_ur6(self.system, q0, q6)
            frames = poses_in_right_base(self.system, q)
            err_mm = np.linalg.norm(frames["chamfer"][:3, 3] - pose[:3]) * 1e3
            self.assertLess(err_mm, 3.0, msg=f"chamfer vs bag tcp {err_mm:.2f} mm")
            R_bag = Rotation.from_quat(pose[3:]).as_matrix()
            self.assertLess(rotation_error_deg(frames["flange"][:3, :3], R_bag), 1.0)
            ee_mm = np.linalg.norm(frames["yaml_ee"][:3, 3] - pose[:3]) * 1e3
            self.assertGreater(ee_mm, 50.0)

    def test_wrap_puts_bag_wrists_inside_pm_pi_same_chamfer(self):
        q6 = self.q_ur6[0]
        wrapped = wrap_ur6_to_pm_pi(q6)
        self.assertTrue(np.any(np.abs(q6) > np.pi))
        np.testing.assert_array_less(np.abs(wrapped), np.pi + 1e-12)
        q0 = np.asarray(self.system.get_default_configuration(), dtype=float)
        p_raw = poses_in_right_base(self.system, apply_right_ur6(self.system, q0, q6))
        p_wrap = poses_in_right_base(self.system, apply_right_ur6(self.system, q0, wrapped))
        err_mm = np.linalg.norm(p_raw["chamfer"][:3, 3] - p_wrap["chamfer"][:3, 3]) * 1e3
        self.assertLess(err_mm, 0.1)
        bag_mm = np.linalg.norm(p_wrap["chamfer"][:3, 3] - self.tcp[0, :3]) * 1e3
        self.assertLess(bag_mm, 3.0)

    def test_shortest_q_delta_does_not_take_the_long_way_around_pi(self):
        from geodude_teleoperation.tcp_kinematics import shortest_q_delta

        q_from = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_to = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dq = shortest_q_delta(q_from, q_to)
        self.assertLess(abs(float(dq[0])), np.pi)
        self.assertLess(abs(float(dq[0])), 0.5)

    def test_joint_lerp_to_bag_q_walks_chamfer_off_the_start_line(self):
        from geodude_teleoperation.tcp_kinematics import (
            BAG_JOINT_TO_GAFRO,
            UR6_ORDER,
            limit_safe_q_delta,
        )

        q0 = np.asarray(self.system.get_default_configuration(), dtype=float)
        by_name = {
            j.get_name(): j.get_index()
            for j in self.system.get_joints() if j.is_actuated()
        }
        idx = np.array([by_name[BAG_JOINT_TO_GAFRO[n]] for n in UR6_ORDER])
        q_from = wrap_ur6_to_pm_pi(q0[idx])
        q_to = wrap_ur6_to_pm_pi(self.q_ur6[0])
        dq = limit_safe_q_delta(q_from, q_to)
        p0 = poses_in_right_base(self.system, q0)["chamfer"][:3, 3]
        p1 = self.tcp[0, :3]
        v = p1 - p0
        max_off = 0.0
        for a in np.linspace(0.0, 1.0, 21):
            qk = q0.copy()
            qk[idx] = q_from + a * dq
            p = poses_in_right_base(self.system, qk)["chamfer"][:3, 3]
            t = float(np.clip(np.dot(p - p0, v) / np.dot(v, v), 0.0, 1.0))
            max_off = max(max_off, float(np.linalg.norm(p - (p0 + t * v)) * 1e3))
        self.assertGreater(max_off, 100.0)

    def test_chamfer_pose_blend_stays_on_the_start_line(self):
        from geodude_teleoperation.sim_replay import blend_pose_matrix
        from scipy.spatial.transform import Rotation

        q0 = np.asarray(self.system.get_default_configuration(), dtype=float)
        T0 = poses_in_right_base(self.system, q0)["chamfer"]
        T1 = np.eye(4)
        T1[:3, 3] = self.tcp[0, :3]
        T1[:3, :3] = Rotation.from_quat(self.tcp[0, 3:]).as_matrix()
        p0, p1 = T0[:3, 3], T1[:3, 3]
        v = p1 - p0
        max_off = 0.0
        for a in np.linspace(0.0, 1.0, 21):
            p = blend_pose_matrix(T0, T1, a)[:3, 3]
            t = float(np.clip(np.dot(p - p0, v) / np.dot(v, v), 0.0, 1.0))
            max_off = max(max_off, float(np.linalg.norm(p - (p0 + t * v)) * 1e3))
        self.assertLess(max_off, 0.5)

    def test_limit_safe_q_delta_does_not_cross_pm_pi(self):
        from geodude_teleoperation.tcp_kinematics import limit_safe_q_delta

        q_from = np.array([2.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_to = np.array([-2.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        dq = limit_safe_q_delta(q_from, q_to)
        end = q_from[0] + dq[0]
        self.assertLessEqual(abs(float(end)), np.pi + 1e-9)
        mids = q_from[0] + np.linspace(0.0, 1.0, 9) * dq[0]
        self.assertTrue(np.all(np.abs(mids) <= np.pi + 1e-9))

    def test_right_arm_chain_is_ur5e_fk(self):
        q0 = np.asarray(self.system.get_default_configuration(), dtype=float)
        q = apply_right_ur6(self.system, q0, wrap_ur6_to_pm_pi(self.q_ur6[0]))
        arms = ur_arm_xyz_in_right_base(self.system, q)
        self.assertEqual(set(arms), {"left", "right"})
        self.assertEqual(len(arms["right"]), 7)
        upper_to_forearm = np.linalg.norm(arms["right"][3] - arms["right"][2])
        self.assertAlmostEqual(upper_to_forearm, 0.425, delta=0.02)
        wrist3 = poses_in_right_base(self.system, q)
        tip = arms["right"][-1]
        self.assertAlmostEqual(
            np.linalg.norm(tip - wrist3["flange"][:3, 3]) * 1e3, 100.0, delta=2.0)

    def test_right_base_roundtrip_to_gafro_world(self):
        from geodude_teleoperation.ee_viz import chamfer_tcp_world_from_wrist3

        q0 = np.asarray(self.system.get_default_configuration(), dtype=float)
        q = apply_right_ur6(self.system, q0, wrap_ur6_to_pm_pi(self.q_ur6[0]))
        p_rb = poses_in_right_base(self.system, q)["chamfer"][:3, 3]
        p_w = right_base_points_to_world(self.system, q, p_rb)[0]
        fk = self.system.compute_forward_kinematics(q)
        T_wrist = np.asarray(
            fk.get_link_pose("right_ur5e/wrist_3_link").to_transformation_matrix())
        p_fk = chamfer_tcp_world_from_wrist3(T_wrist)[:3, 3]
        self.assertLess(np.linalg.norm(p_w - p_fk) * 1e3, 0.1)

    def test_patched_ready_keyframe_matches_bag_start_joints(self):
        import tempfile

        import gafropy as ga

        q6 = wrap_ur6_to_pm_pi(self.q_ur6[0])
        dest = Path(tempfile.mkdtemp()) / "geodude.yaml"
        write_geodude_model_with_right_ur6(self.q_ur6[0], dest, src=MODEL)
        patched = ga.SystemSerialization.load(str(dest))
        ready = np.asarray(patched.get_keyframe("ready").qpos, dtype=float)
        home = np.asarray(patched.get_keyframe("home").qpos, dtype=float)
        q0 = np.asarray(patched.get_default_configuration(), dtype=float)
        want = apply_right_ur6(patched, q0, q6)
        by_name = {
            j.get_name(): j.get_index()
            for j in patched.get_joints() if j.is_actuated()
        }
        from geodude_teleoperation.tcp_kinematics import BAG_JOINT_TO_GAFRO, UR6_ORDER

        idx = [by_name[BAG_JOINT_TO_GAFRO[n]] for n in UR6_ORDER]
        np.testing.assert_allclose(ready[idx], q6, atol=1e-9)
        np.testing.assert_allclose(home[idx], q6, atol=1e-9)
        self.assertLessEqual(float(np.max(np.abs(ready[idx]))), np.pi)
        frames = poses_in_right_base(patched, want)
        err_mm = np.linalg.norm(frames["chamfer"][:3, 3] - self.tcp[0, :3]) * 1e3
        self.assertLess(err_mm, 3.0)

    def test_match_t_right_mount_puts_chamfer_on_worktop_stringer(self):
        """Rail stays 0.25 m. Lifting the yaml origin (not the carriage) meets T_right."""
        import tempfile

        import gafropy as ga

        from geodude_teleoperation.ee_viz import chamfer_tcp_world_from_wrist3
        from geodude_teleoperation.laser_frames import right_base_points_to_vention
        from geodude_teleoperation.vention import (
            DEFAULT_VENTION_HEIGHT,
            apply_vention_heights,
        )

        dest = Path(tempfile.mkdtemp()) / "geodude.yaml"
        write_geodude_model_with_right_ur6(
            self.q_ur6[0], dest, src=MODEL, match_t_right_mount=True)
        system = ga.SystemSerialization.load(str(dest))
        q0 = np.asarray(system.get_default_configuration(), dtype=float)
        q = apply_vention_heights(
            system,
            apply_right_ur6(system, q0, wrap_ur6_to_pm_pi(self.q_ur6[0])),
            DEFAULT_VENTION_HEIGHT,
            DEFAULT_VENTION_HEIGHT,
        )
        fk = system.compute_forward_kinematics(q)
        T_wrist = np.asarray(
            fk.get_link_pose("right_ur5e/wrist_3_link").to_transformation_matrix())
        chamfer_w = chamfer_tcp_world_from_wrist3(T_wrist)[:3, 3]
        overlay = right_base_points_to_vention(self.tcp[0, :3])[0]
        err_mm = float(np.linalg.norm(chamfer_w - overlay) * 1e3)
        self.assertLess(err_mm, 25.0, msg=f"chamfer world vs T_right overlay {err_mm:.1f} mm")
        self.assertGreater(float(chamfer_w[2]), 0.735)
        self.assertLess(float(chamfer_w[2]), 0.955)


class YamlEeVsChamferTest(unittest.TestCase):
    def test_yaml_ee_is_not_the_bag_tcp(self):
        import gafropy as ga

        system = ga.SystemSerialization.load(str(MODEL))
        q = np.asarray(system.get_default_configuration(), dtype=float)
        frames = poses_in_right_base(system, q)
        gap_mm = np.linalg.norm(
            frames["yaml_ee"][:3, 3] - frames["chamfer"][:3, 3]) * 1e3
        self.assertGreater(gap_mm, 50.0)
        self.assertLess(gap_mm, 80.0)


class BagTcpToYamlEeTest(unittest.TestCase):
    def test_bag_tcp_rebuilds_yaml_ee_at_recorded_q(self):
        import gafropy as ga

        from geodude_teleoperation.tcp_kinematics import (
            RIGHT_EE,
            apply_right_ur6,
            bag_tcp_to_yaml_ee_world,
            rotation_error_deg,
            wrap_ur6_to_pm_pi,
        )

        system = ga.SystemSerialization.load(str(MODEL))
        data = np.load(_FIXTURE)
        q0 = np.asarray(system.get_default_configuration(), dtype=float)
        q = apply_right_ur6(system, q0, wrap_ur6_to_pm_pi(data["q_ur6"][0]))
        T_des = bag_tcp_to_yaml_ee_world(system, q, data["tcp_pose_xyzw"][0])
        fk = system.compute_forward_kinematics(q)
        T_ee = np.asarray(
            fk.get_link_pose(RIGHT_EE).to_transformation_matrix(), dtype=float)
        err_mm = np.linalg.norm(T_des[:3, 3] - T_ee[:3, 3]) * 1e3
        self.assertLess(err_mm, 3.0)
        self.assertLess(rotation_error_deg(T_des[:3, :3], T_ee[:3, :3]), 1.0)


class RightUr6NameLookupTest(unittest.TestCase):
    def test_ur6_lookup_is_by_name_not_array_order(self):
        from geodude_teleoperation.tcp_kinematics import (
            UR6_ORDER,
            right_ur6_from_names_positions,
        )

        q = np.arange(6, dtype=float)
        np.testing.assert_allclose(right_ur6_from_names_positions(UR6_ORDER, q), q)
        order = [0, 1, 3, 5, 4, 2]
        names = [UR6_ORDER[i] for i in order]
        pos = [q[i] for i in order]
        np.testing.assert_allclose(right_ur6_from_names_positions(names, pos), q)

    def test_missing_joint_name_raises(self):
        from geodude_teleoperation.tcp_kinematics import right_ur6_from_names_positions

        with self.assertRaises(KeyError):
            right_ur6_from_names_positions(["right_shoulder_pan_joint"], [0.0])
