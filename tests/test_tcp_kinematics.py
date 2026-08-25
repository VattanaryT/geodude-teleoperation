import unittest
from pathlib import Path

import numpy as np

from geodude_teleoperation.tcp_kinematics import (
    MODEL,
    apply_right_ur6,
    poses_in_right_base,
    rotation_error_deg,
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

    def test_patched_ready_keyframe_matches_bag_start_joints(self):
        import tempfile

        import gafropy as ga

        q6 = self.q_ur6[0]
        dest = Path(tempfile.mkdtemp()) / "geodude.yaml"
        write_geodude_model_with_right_ur6(q6, dest, src=MODEL)
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
        frames = poses_in_right_base(patched, want)
        err_mm = np.linalg.norm(frames["chamfer"][:3, 3] - self.tcp[0, :3]) * 1e3
        self.assertLess(err_mm, 3.0)


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
