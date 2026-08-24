import importlib.util
import pathlib
import unittest

import numpy as np

from geodude_teleoperation.virtual_fixture import (
    DEFAULT_MAX_BOUNDARY_SAMPLES,
    DEFAULT_T_SPATIAL_EDGE_FACTOR,
    boundary_sample_indices,
)

_EXAMPLE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples"
    / "spacemouse_rotor_field_teleop_example.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "spacemouse_rotor_field_teleop_example", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BoundarySampleIndicesTest(unittest.TestCase):
    def test_small_mesh_keeps_every_vertex(self):
        indices = boundary_sample_indices(100)
        np.testing.assert_array_equal(indices, np.arange(100))

    def test_large_cloud_is_capped_by_default(self):
        indices = boundary_sample_indices(37754)
        self.assertLessEqual(len(indices), DEFAULT_MAX_BOUNDARY_SAMPLES)
        self.assertGreater(len(indices), DEFAULT_MAX_BOUNDARY_SAMPLES // 2)

    def test_explicit_count_is_honored(self):
        indices = boundary_sample_indices(37754, num_boundary_samples=500)
        self.assertLessEqual(len(indices), 500)


class ProbeMotionScaleTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load_example()

    def test_hemisphere_keeps_legacy_linear_scale(self):
        scale = self.mod.resolve_probe_linear_scale(bbox_diag=3.0, frequency=50.0)
        self.assertAlmostEqual(scale, 0.05)

    def test_merged_ply_scale_is_object_sized(self):
        scale = self.mod.resolve_probe_linear_scale(bbox_diag=0.459, frequency=50.0)
        self.assertLess(scale, 0.02)
        self.assertGreater(scale, 1e-4)
        # Full deflection should not exceed ~1 object-length per second.
        self.assertLessEqual(scale * 50.0, 0.459 * 1.01)

    def test_explicit_linear_scale_wins(self):
        scale = self.mod.resolve_probe_linear_scale(
            bbox_diag=0.459, frequency=50.0, explicit=0.05)
        self.assertAlmostEqual(scale, 0.05)

    def test_capped_drain_rate_limits_without_losing_remainder(self):
        acc = self.mod.LinearAccumulator()
        acc.add(np.array([0.25, 0.0, 0.0]))
        first = acc.take_capped(0.01)
        np.testing.assert_allclose(first, [0.01, 0.0, 0.0])
        leftover = acc.drain()
        np.testing.assert_allclose(leftover, [0.24, 0.0, 0.0])

    def test_releasing_stick_drops_queued_remainder(self):
        acc = self.mod.LinearAccumulator()
        acc.add(np.array([0.25, 0.0, 0.0]))
        acc.take_capped(0.01)
        self.mod.enqueue_or_stop(acc, np.zeros(3), stop_eps=1e-6)
        np.testing.assert_allclose(acc.drain(), [0.0, 0.0, 0.0])

    def test_held_stick_keeps_queueing(self):
        acc = self.mod.LinearAccumulator()
        self.mod.enqueue_or_stop(acc, np.array([0.01, 0.0, 0.0]), stop_eps=1e-6)
        np.testing.assert_allclose(acc.drain(), [0.01, 0.0, 0.0])

    def test_probe_arrows_use_object_size_not_mean_edge(self):
        # Laser mean_edge ~2 mm would make 3*edge arrows invisible on a 0.46 m scan.
        scale = self.mod.probe_arrow_scale(mean_edge=0.002, bbox_diag=0.459)
        self.assertGreater(scale, 0.02)
        self.assertLess(scale, 0.1)

    def test_flip_yz_negates_green_and_blue_keeps_right_handed(self):
        R = np.eye(3)
        flipped = self.mod.flip_yz_axes(R)
        np.testing.assert_allclose(flipped[:, 0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(flipped[:, 1], [0.0, -1.0, 0.0])
        np.testing.assert_allclose(flipped[:, 2], [0.0, 0.0, -1.0])
        self.assertAlmostEqual(np.linalg.det(flipped), 1.0)
        mean_edge = 0.002
        wide = (DEFAULT_T_SPATIAL_EDGE_FACTOR * mean_edge) ** 2
        self.assertGreater(wide, mean_edge ** 2)
        self.assertAlmostEqual(wide / (mean_edge ** 2), 25.0)


class LaserCloudPlacementTest(unittest.TestCase):
    def test_merged_scan_rests_flat_on_z(self):
        from geodude_teleoperation.mesh_fixture import (
            compute_resting_orientation, load_points)

        path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "assets" / "laser_profilometer" / "merged.ply")
        points = load_points(path)
        rotated = points @ compute_resting_orientation(points).T
        span = rotated.max(axis=0) - rotated.min(axis=0)
        self.assertEqual(int(np.argmin(span)), 2)
        self.assertLess(span[2], 0.5 * max(span[0], span[1]))
