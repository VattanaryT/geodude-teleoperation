import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geodude_teleoperation.local_frames_npz import (
    integrate_desired_xyz,
    load_cga_episode,
    local_actions_from_xyz,
    replay_arrays,
    world_deltas_from_local,
    write_cga_actual_tcp,
)

_DATA = Path("/mnt/data/boeing/local_frames_data/08_16_half_stringer_local_frames")
_CGA0 = _DATA / "bags_08_16_half_stringer_bimanual_0_cga.npz"
_TRAIN0 = _DATA / "training" / "bags_08_16_half_stringer_bimanual_0.npz"


def _rotation_about_z(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class WorldDeltaConventionTest(unittest.TestCase):
    def test_local_action_is_field_transpose_times_world_delta(self):
        R = np.stack([_rotation_about_z(0.0), _rotation_about_z(0.4)])
        a = np.array([[0.01, -0.002, 0.003]])
        dp = world_deltas_from_local(R, a)
        np.testing.assert_allclose(dp[0], R[0] @ a[0])
        np.testing.assert_allclose(R[0].T @ dp[0], a[0])

    def test_integrate_recovers_desired_path(self):
        R = np.stack([np.eye(3), _rotation_about_z(0.3), np.eye(3)])
        a = np.array([[0.10, 0.0, 0.0], [0.0, 0.05, 0.0]])
        start = np.array([1.0, 2.0, 3.0])
        xyz = integrate_desired_xyz(start, R, a)
        np.testing.assert_allclose(xyz[0], start)
        np.testing.assert_allclose(xyz[1], start + np.array([0.10, 0.0, 0.0]))
        np.testing.assert_allclose(xyz[2], xyz[1] + R[1] @ a[1])

    def test_local_actions_from_xyz_inverts_world_deltas(self):
        R = np.stack([_rotation_about_z(0.0), _rotation_about_z(0.4), np.eye(3)])
        xyz = np.array([[0.0, 0.0, 0.0], [0.01, -0.002, 0.003], [0.01, 0.01, 0.003]])
        a = local_actions_from_xyz(R, xyz)
        np.testing.assert_allclose(a[0], R[0].T @ (xyz[1] - xyz[0]))
        rec = integrate_desired_xyz(xyz[0], R, a)
        np.testing.assert_allclose(rec, xyz)


class LoadCgaNpzTest(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_cga_episode(Path("/no/such/episode_cga.npz"))

    def test_synthetic_npz_roundtrip(self):
        n = 4
        R = np.stack([_rotation_about_z(0.1 * i) for i in range(n)])
        desired = np.cumsum(np.vstack([np.zeros(3), np.array([0.01, 0.0, 0.002]) * np.ones((n - 1, 3))]), axis=0)
        a = np.einsum("nji,nj->ni", R[:-1], np.diff(desired, axis=0))
        poses = np.zeros((n, 7))
        poses[:, :3] = desired + 0.05
        poses[:, 6] = 1.0
        meta = {
            "contact_threshold": 0.01,
            "pose_frame": "right_base",
            "action_source": "desired",
            "convention": "a_local = R_field^T (x[t+1]-x[t]); columns [t1, t2, n]",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ep_cga.npz"
            np.savez_compressed(
                path,
                a_local_cga=a,
                R_field_cga=R,
                poses=poses,
                poses_desired=desired,
                signed_cga=np.array([0.02, 0.001, -0.004, 0.03]),
                timestamps=np.linspace(0.0, 0.3, n),
                meta=json.dumps(meta),
            )
            ep = load_cga_episode(path)
        self.assertEqual(ep.pose_frame, "right_base")
        np.testing.assert_allclose(ep.a_local, a)
        rec = integrate_desired_xyz(ep.poses_desired_xyz[0], ep.R_field, ep.a_local)
        np.testing.assert_allclose(rec, desired, atol=1e-12)
        np.testing.assert_array_equal(ep.in_contact, [False, True, True, False])
        series = replay_arrays(ep)
        self.assertEqual(len(series["a_t1"]), 3)
        self.assertEqual(len(series["t_s"]), 4)
        np.testing.assert_array_equal(series["in_contact"], [False, True, True])


@unittest.skipUnless(_CGA0.is_file(), "local-frames data not mounted")
class RealCgaEpisodeTest(unittest.TestCase):
    def test_bimanual_0_reconstructs_desired_not_actual(self):
        ep = load_cga_episode(_CGA0, training_path=_TRAIN0)
        rec = integrate_desired_xyz(ep.poses_desired_xyz[0], ep.R_field, ep.a_local)
        np.testing.assert_allclose(rec, ep.poses_desired_xyz, atol=1e-9)
        actual_err = np.linalg.norm(rec - ep.poses[:, :3], axis=1)
        self.assertGreater(float(actual_err.mean()), 0.01)
        self.assertEqual(ep.hz, 10)
        self.assertEqual(ep.a_local.shape[1], 3)
        self.assertEqual(ep.action.shape, (358, 6))
        np.testing.assert_allclose(ep.action[:, :2], ep.a_local[:, :2])
        self.assertEqual(ep.f_n.shape, (358,))
        self.assertGreater(float(ep.in_contact.mean()), 0.5)

    def test_rewrite_actual_tcp_reconstructs_measured_path_and_keeps_source(self):
        src_before = _CGA0.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "bimanual_0_cga_actual.npz"
            write_cga_actual_tcp(_CGA0, dst)
            ep = load_cga_episode(dst)
        self.assertEqual(_CGA0.read_bytes(), src_before)
        self.assertEqual(ep.meta.get("action_source"), "actual")
        rec = integrate_desired_xyz(ep.action_xyz[0], ep.R_field, ep.a_local)
        np.testing.assert_allclose(rec, ep.poses[:, :3], atol=1e-9)
        self.assertGreater(
            float(np.linalg.norm(rec - ep.poses_desired_xyz, axis=1).mean()), 0.01)
