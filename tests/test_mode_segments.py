import unittest
from pathlib import Path

import numpy as np

from geodude_teleoperation.mode_segments import (
    body_linear_from_local,
    format_segment_table,
    sandwich_local_actions,
    segments_from_contact,
    sticky_contact,
)

_DATA = Path("/mnt/data/boeing/local_frames_data/08_16_half_stringer_local_frames")
_ACTUAL = _DATA / "bags_08_16_half_stringer_bimanual_0_cga_actual.npz"
_TRAIN0 = _DATA / "training" / "bags_08_16_half_stringer_bimanual_0.npz"


class StickyContactTest(unittest.TestCase):
    def test_exits_only_after_lost_s(self):
        raw = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        # 10 Hz, lost_s=0.3 → 3 free samples before exit
        out = sticky_contact(raw, dt_s=0.1, lost_s=0.3)
        np.testing.assert_array_equal(
            out.astype(int),
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        )

    def test_enters_on_first_true(self):
        raw = np.array([0, 0, 1, 1], dtype=bool)
        out = sticky_contact(raw, dt_s=0.1, lost_s=1.0)
        np.testing.assert_array_equal(out.astype(int), [0, 0, 1, 1])


class BodyLinearTest(unittest.TestCase):
    def test_field_step_in_aligned_ee_is_body_velocity(self):
        R_field = np.eye(3)
        R_ee = np.eye(3)
        a = np.array([0.01, -0.002, 0.004])
        v = body_linear_from_local(R_field, R_ee, a, dt_s=0.1, contact=False)
        np.testing.assert_allclose(v, [0.10, -0.02, 0.04])

    def test_contact_zeros_body_z_and_field_normal(self):
        R_field = np.eye(3)
        R_ee = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        a = np.array([0.01, 0.0, 0.05])
        v = body_linear_from_local(R_field, R_ee, a, dt_s=0.1, contact=True)
        np.testing.assert_allclose(v, [0.0, -0.10, 0.0])


class SegmentTest(unittest.TestCase):
    def test_half_open_ranges_use_pose_timestamps(self):
        flags = np.array([0, 0, 1, 1, 0], dtype=bool)
        t = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        segs = segments_from_contact(flags, t)
        self.assertEqual([(s.mode, s.i0, s.i1, s.t0, s.t1) for s in segs], [
            ("approach", 0, 2, 0.0, 0.2),
            ("contact", 2, 4, 0.2, 0.4),
            ("approach", 4, 5, 0.4, 0.5),
        ])


class SandwichTest(unittest.TestCase):
    def test_contact_body_z_is_zero_and_twist_is_gafro_order(self):
        n_pose = 5
        t = np.linspace(0.0, 0.4, n_pose)
        R = np.stack([np.eye(3)] * n_pose)
        a = np.array([[0.01, 0.0, 0.02]] * (n_pose - 1))
        raw = np.array([0, 0, 1, 1], dtype=bool)
        f_n = np.array([0.0, 0.0, 8.0, 9.0])
        contact, v_body, twist, segs, stats = sandwich_local_actions(
            R_field=R, R_ee=R, a_local=a, in_contact=raw, timestamps=t,
            f_n=f_n, lost_s=1.0)
        np.testing.assert_array_equal(contact.astype(int), [0, 0, 1, 1])
        np.testing.assert_allclose(v_body[:2], [[0.10, 0.0, 0.20], [0.10, 0.0, 0.20]])
        np.testing.assert_allclose(v_body[2:], [[0.10, 0.0, 0.0], [0.10, 0.0, 0.0]])
        np.testing.assert_allclose(twist[:, :3], 0.0)
        np.testing.assert_allclose(twist[:, 3:], v_body)
        self.assertEqual([s.mode for s in segs], ["approach", "contact"])
        self.assertAlmostEqual(stats[1].mean_abs_fn, 8.5)
        self.assertAlmostEqual(stats[1].mean_abs_v_body[2], 0.0)
        table = format_segment_table(stats)
        self.assertIn("approach", table)
        self.assertIn("contact", table)


@unittest.skipUnless(_ACTUAL.is_file() and _TRAIN0.is_file(), "local-frames data not mounted")
class RealSandwichTest(unittest.TestCase):
    def test_sticky_contact_zeros_body_z(self):
        from scipy.spatial.transform import Rotation

        from geodude_teleoperation.local_frames_npz import load_cga_episode

        ep = load_cga_episode(_ACTUAL, training_path=_TRAIN0)
        n = len(ep.a_local)
        R_ee = Rotation.from_quat(ep.poses[: n + 1, 3:7]).as_matrix()
        contact, v_body, twist, segs, stats = sandwich_local_actions(
            R_field=ep.R_field, R_ee=R_ee, a_local=ep.a_local,
            in_contact=ep.in_contact, timestamps=ep.timestamps,
            f_n=ep.f_n, lost_s=1.0)
        self.assertGreaterEqual(int(contact.sum()), int(ep.in_contact[:n].sum()))
        self.assertTrue(any(s.mode == "approach" for s in segs))
        self.assertTrue(any(s.mode == "contact" for s in segs))
        contact_vz = np.abs(v_body[contact, 2])
        self.assertLess(float(contact_vz.max()), 1e-12)
        self.assertEqual(twist.shape, (n, 6))
        contact_stats = [s for s in stats if s.mode == "contact"]
        self.assertGreater(float(np.mean([s.mean_abs_fn for s in contact_stats])), 1.0)

