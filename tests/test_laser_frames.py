import unittest
from pathlib import Path

import numpy as np

from geodude_teleoperation.laser_frames import (
    DEFAULT_T_RIGHT,
    compose_T_tag0_rtde,
    is_laser_left_base_object,
    load_SE3,
    load_t_vention,
    place_laser_object,
    place_laser_points,
    right_base_points_to_vention,
    right_base_R_to_vention,
    transform_points,
)

_REPO = Path(__file__).resolve().parents[1]
_MERGED = _REPO / "assets" / "laser_profilometer" / "merged.ply"


class LaserFrameDetectTest(unittest.TestCase):
    def test_merged_ply_is_left_base(self):
        self.assertTrue(is_laser_left_base_object(_MERGED))
        self.assertTrue(is_laser_left_base_object(
            _REPO / "assets" / "laser_profilometer" / "merged_voxel0.0020.ply"))

    def test_hemisphere_is_not_left_base(self):
        self.assertFalse(is_laser_left_base_object(
            _REPO / "assets" / "objects" / "hemisphere.stl"))


class LaserPlacementTest(unittest.TestCase):
    def test_left_base_origin_lands_on_left_mount_plus_laser_nudge(self):
        T_left = load_SE3(_REPO / "assets" / "frames" / "T_left_base_vention.json")
        t_laser = load_t_vention(
            _REPO / "assets" / "laser_profilometer" / "merged.laser_offset.json")
        placed = place_laser_points(
            np.zeros((1, 3)),
            T_left_base_vention=T_left,
            t_laser_vention_m=t_laser,
            t_tag0_vention_m=np.zeros(3),
        )
        expected = T_left[:3, 3] + t_laser
        np.testing.assert_allclose(placed.points_vention[0], expected, atol=1e-9)

    def test_tag0_rtde_is_inv_right_times_tag0_vention(self):
        T_right = load_SE3(_REPO / "assets" / "frames" / "T_right_base_vention.json")
        T_tag0 = load_SE3(_REPO / "assets" / "laser_profilometer" / "crop_meta.json")
        t_tag0 = load_t_vention(
            _REPO / "assets" / "laser_profilometer" / "crop_meta.object_offset.json")
        placed = place_laser_points(
            np.zeros((1, 3)),
            t_tag0_vention_m=t_tag0,
        )
        expected = compose_T_tag0_rtde(placed.T_tag0_vention, T_right)
        np.testing.assert_allclose(placed.T_tag0_rtde, expected)
        np.testing.assert_allclose(
            placed.T_tag0_vention[:3, 3], T_tag0[:3, 3] + t_tag0)

    def test_roundtrip_vention_tag0(self):
        placed = place_laser_points(np.array([[0.1, -0.2, 0.3]]))
        back = transform_points(placed.T_tag0_vention, placed.points_tag0)
        np.testing.assert_allclose(back, placed.points_vention, atol=1e-9)

    def test_merged_ply_stays_out_of_right_base(self):
        placed = place_laser_object(_MERGED)
        T_right = placed.T_right_base_vention
        # Overlay is vention, not right_base: left-base origin is not right mount.
        left_origin_vention = placed.T_left_base_vention[:3, 3]
        right_origin_vention = T_right[:3, 3]
        self.assertGreater(
            np.linalg.norm(left_origin_vention - right_origin_vention), 0.5)
        self.assertFalse(np.allclose(
            placed.points_vention.mean(axis=0), right_origin_vention, atol=0.05))

    def test_skipping_left_to_vention_is_not_this_placement(self):
        """Raw left_base mean is not the vention overlay (would look ~90° off)."""
        from geodude_teleoperation.mesh_fixture import load_points

        raw = load_points(_MERGED)
        placed = place_laser_object(_MERGED)
        self.assertGreater(
            np.linalg.norm(raw.mean(axis=0) - placed.points_vention.mean(axis=0)),
            0.5,
        )


class RightBaseVentionOverlayTest(unittest.TestCase):
    """Hardware draws right_base clouds with T_right, not yaml UR FK.

    Yaml mount at rail=0.25 is 280 mm below T_right, so FK overlay puts the
    stringer under the worktop (z=0.755). Calibrated overlay sits on it.
    """

    @classmethod
    def setUpClass(cls):
        cls.tcp = np.asarray(
            np.load(
                Path(__file__).resolve().parent
                / "fixtures"
                / "bimanual_0_tcp_fk_samples.npz"
            )["tcp_pose_xyzw"][0, :3],
            dtype=float,
        )
        cls.T_right = load_SE3(DEFAULT_T_RIGHT)
        cls.worktop_z = 0.755

    def test_calibrated_overlay_puts_bag_tcp_on_worktop(self):
        p = right_base_points_to_vention(self.tcp)[0]
        self.assertGreater(p[2], self.worktop_z - 0.02)
        self.assertLess(p[2], self.worktop_z + 0.20)
        np.testing.assert_allclose(
            p, transform_points(self.T_right, self.tcp)[0], atol=1e-12)

    def test_yaml_fk_overlay_is_the_under_table_bug(self):
        import gafropy as ga

        from geodude_teleoperation.tcp_kinematics import (
            MODEL,
            apply_right_ur6,
            right_base_points_to_world,
            wrap_ur6_to_pm_pi,
        )
        from geodude_teleoperation.vention import (
            DEFAULT_VENTION_HEIGHT,
            apply_vention_heights,
        )

        data = np.load(
            Path(__file__).resolve().parent
            / "fixtures"
            / "bimanual_0_tcp_fk_samples.npz"
        )
        system = ga.SystemSerialization.load(str(MODEL))
        q0 = np.asarray(system.get_default_configuration(), dtype=float)
        q = apply_vention_heights(
            system,
            apply_right_ur6(system, q0, wrap_ur6_to_pm_pi(data["q_ur6"][0])),
            DEFAULT_VENTION_HEIGHT,
            DEFAULT_VENTION_HEIGHT,
        )
        p_fk = right_base_points_to_world(system, q, self.tcp)[0]
        self.assertLess(p_fk[2], self.worktop_z - 0.10)

    def test_R_maps_with_T_right_rotation(self):
        R = np.eye(3)
        np.testing.assert_allclose(
            right_base_R_to_vention(R), self.T_right[:3, :3], atol=1e-12)
