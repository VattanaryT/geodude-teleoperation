import unittest
from pathlib import Path

import numpy as np

from geodude_teleoperation.laser_frames import (
    compose_T_tag0_rtde,
    is_laser_left_base_object,
    load_SE3,
    load_t_vention,
    place_laser_object,
    place_laser_points,
    transform_points,
)

_REPO = Path(__file__).resolve().parents[1]
# Frames and scan metadata are package data (they ship in the wheel);
# the bulk meshes/clouds under assets/ are not.
_DATA = _REPO / "src" / "geodude_teleoperation" / "data"
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
        T_left = load_SE3(_DATA / "frames" / "T_left_base_vention.json")
        t_laser = load_t_vention(
            _DATA / "laser_profilometer" / "merged.laser_offset.json")
        placed = place_laser_points(
            np.zeros((1, 3)),
            T_left_base_vention=T_left,
            t_laser_vention_m=t_laser,
            t_tag0_vention_m=np.zeros(3),
        )
        expected = T_left[:3, 3] + t_laser
        np.testing.assert_allclose(placed.points_vention[0], expected, atol=1e-9)

    def test_tag0_rtde_is_inv_right_times_tag0_vention(self):
        T_right = load_SE3(_DATA / "frames" / "T_right_base_vention.json")
        T_tag0 = load_SE3(_DATA / "laser_profilometer" / "crop_meta.json")
        t_tag0 = load_t_vention(
            _DATA / "laser_profilometer" / "crop_meta.object_offset.json")
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


class MissingCalibrationIsLoudTest(unittest.TestCase):
    """A silently-zeroed calibration is the failure this repo keeps repeating.

    The two offset sidecars are tracked on some branches and not others, so a
    branch switch deletes them -- and load_t_vention's zero fallback made the
    scan sit 31 mm out with every control number still looking healthy.
    """

    def test_the_calibrated_offsets_are_present_on_this_branch(self):
        from geodude_teleoperation.laser_frames import (
            DEFAULT_CROP_META, DEFAULT_LASER_OFFSET, DEFAULT_T_LEFT,
            DEFAULT_T_RIGHT, DEFAULT_TAG0_OFFSET)
        for p in (DEFAULT_CROP_META, DEFAULT_LASER_OFFSET, DEFAULT_TAG0_OFFSET,
                  DEFAULT_T_LEFT, DEFAULT_T_RIGHT):
            self.assertTrue(Path(p).is_file(), f"missing: {p}")

    def test_the_offsets_are_the_calibrated_values_not_zero(self):
        from geodude_teleoperation.laser_frames import (
            load_t_vention, DEFAULT_LASER_OFFSET, DEFAULT_TAG0_OFFSET)
        np.testing.assert_allclose(load_t_vention(DEFAULT_LASER_OFFSET),
                                   [0.0, 0.0, -0.010], atol=1e-6)
        np.testing.assert_allclose(load_t_vention(DEFAULT_TAG0_OFFSET),
                                   [0.0, 0.031, -0.014], atol=1e-5)

    def test_a_missing_offset_warns_before_returning_zero(self):
        import io
        import contextlib
        from geodude_teleoperation import laser_frames as lf
        lf._MISSING_OFFSETS_WARNED.discard("does_not_exist.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = lf.load_t_vention("does_not_exist.json")
        np.testing.assert_allclose(out, np.zeros(3))
        self.assertIn("CALIBRATION MISSING", buf.getvalue())
