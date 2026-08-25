import unittest

import numpy as np
import trimesh

from geodude_teleoperation.stringer_field import (
    bases_from_tangent_and_normal,
    end_cap_source_sink,
    length_tangents,
)
from geodude_teleoperation.virtual_fixture import OrientationField


def _long_box():
    mesh = trimesh.creation.box(extents=[0.40, 0.04, 0.02])
    return mesh.subdivide().subdivide()


class EndCapBandTest(unittest.TestCase):
    def test_length_flow_bands_are_on_short_end_caps(self):
        rng = np.random.default_rng(0)
        pts = np.column_stack([
            rng.uniform(-0.2, 0.2, 400),
            rng.uniform(-0.02, 0.02, 400),
            rng.uniform(-0.005, 0.005, 400),
        ])
        sink, source, axis = end_cap_source_sink(pts, fraction=0.06, flow="length")
        self.assertGreater(abs(axis[0]), 0.9)
        self.assertGreater(len(sink), 0)
        self.assertGreater(len(source), 0)
        self.assertGreater(
            abs(float(pts[source, 0].mean()) - float(pts[sink, 0].mean())), 0.15)

    def test_default_bands_run_along_the_long_edges(self):
        rng = np.random.default_rng(1)
        pts = np.column_stack([
            rng.uniform(-0.2, 0.2, 800),
            rng.uniform(-0.05, 0.05, 800),
            rng.uniform(-0.005, 0.005, 800),
        ])
        sink, source, axis = end_cap_source_sink(pts)
        self.assertGreater(abs(axis[1]), 0.9)
        self.assertGreater(float(np.ptp(pts[source, 0])), 0.25)
        self.assertGreater(float(np.ptp(pts[sink, 0])), 0.25)

    def test_swap_end_lines_flips_source_and_sink(self):
        pts = np.column_stack([
            np.linspace(-0.2, 0.2, 50),
            np.zeros(50),
            np.zeros(50),
        ])
        sink, source, _ = end_cap_source_sink(pts, fraction=0.1, flow="length")
        sink_s, source_s, _ = end_cap_source_sink(pts, fraction=0.1, swap=True, flow="length")
        np.testing.assert_array_equal(sink_s, source)
        np.testing.assert_array_equal(source_s, sink)


class LengthTangentTest(unittest.TestCase):
    def test_box_tangents_run_along_length(self):
        mesh = _long_box()
        tangent, sink, source, axis = length_tangents(mesh, flow="length")
        self.assertGreater(abs(axis[0]), 0.9)
        self.assertGreater(len(sink), 0)
        self.assertGreater(len(source), 0)
        keep = np.ones(len(tangent), dtype=bool)
        keep[sink] = False
        keep[source] = False
        along = np.abs(tangent[keep] @ np.array([1.0, 0.0, 0.0]))
        self.assertGreater(float(along.mean()), 0.85)

    def test_bases_are_orthonormal_right_handed(self):
        t = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        B = bases_from_tangent_and_normal(t, n)
        for i in range(2):
            np.testing.assert_allclose(B[i].T @ B[i], np.eye(3), atol=1e-9)
            self.assertGreater(np.linalg.det(B[i]), 0.9)


class OrientationFieldLengthTest(unittest.TestCase):
    def test_near_box_approach_x_follows_length(self):
        mesh = _long_box()
        vf = OrientationField(mesh, key_direction=np.array([0.0, 1.0, 0.0]), flow="length")
        p = np.array([0.0, 0.0, 0.02])
        R = vf.approach_basis_at(p)
        self.assertGreater(abs(float(R[0, 0])), 0.7)
        self.assertLess(float(np.dot(R[:, 2], [0.0, 0.0, 1.0])), 0.0)

    def test_source_and_sink_points_are_exposed(self):
        mesh = _long_box()
        vf = OrientationField(mesh)
        self.assertGreater(len(vf.source_points), 0)
        self.assertGreater(len(vf.sink_points), 0)
        self.assertEqual(vf.source_points.shape[1], 3)

    def test_queries_are_not_walk_on_spheres(self):
        import geodude_teleoperation.virtual_fixture as vf_mod

        self.assertFalse(hasattr(vf_mod, "WalkOnSpheresDiffusion"))
        from pathlib import Path

        src = Path(vf_mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("WalkOnSpheres", src)
        self.assertIn("RotorFieldDiffusion", src)


if __name__ == "__main__":
    unittest.main()
