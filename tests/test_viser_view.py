import unittest
from pathlib import Path

import numpy as np

from geodude_teleoperation.viser_view import framed_camera, sitting_up_axis


class SittingUpAxisTest(unittest.TestCase):
    def test_thin_y_slab_sits_with_minus_y_up(self):
        rng = np.random.default_rng(0)
        pts = np.column_stack([
            rng.uniform(-0.2, 0.2, 400),
            rng.uniform(-0.02, 0.02, 400),
            rng.uniform(-0.1, 0.1, 400),
        ])
        up = sitting_up_axis(pts)
        np.testing.assert_allclose(np.abs(up), [0.0, 1.0, 0.0], atol=0.15)
        self.assertLess(float(up[1]), 0.0)


class FramedCameraTest(unittest.TestCase):
    def test_eye_looks_at_centroid_from_nearby_above(self):
        rng = np.random.default_rng(1)
        pts = np.column_stack([
            rng.uniform(-0.2, 0.2, 200) - 0.1,
            rng.uniform(-0.02, 0.02, 200) + 0.86,
            rng.uniform(-0.1, 0.1, 200) + 0.5,
        ])
        center, up, eye = framed_camera(pts)
        np.testing.assert_allclose(center, pts.mean(axis=0), atol=1e-12)
        self.assertGreater(float((eye - center) @ up), 0.0)
        dist = float(np.linalg.norm(eye - center))
        span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        self.assertLess(dist, 2.5 * span)
        self.assertGreater(dist, 0.15)
        self.assertGreater(float(np.linalg.norm(center)), 0.5)
        # PCA long-axis sign is arbitrary; pin so right_base +X is not yawed 180°.
        self.assertGreater(float((eye - center)[0]), 0.0)


@unittest.skipUnless(
    Path("/mnt/data/boeing/local_frames_data/08_16_half_stringer_local_frames"
         "/merged_rightbase.ply").is_file(),
    "stringer ply not mounted",
)
class RealStringerViewTest(unittest.TestCase):
    def test_ply_sits_on_thin_y_and_camera_is_close(self):
        from geodude_teleoperation.mesh_fixture import load_points

        ply = Path(
            "/mnt/data/boeing/local_frames_data/08_16_half_stringer_local_frames"
            "/merged_rightbase.ply")
        pts = load_points(ply)
        center, up, eye = framed_camera(pts)
        self.assertGreater(float(abs(up[1])), 0.9)
        self.assertLess(float(up[1]), 0.0)
        dist = float(np.linalg.norm(eye - center))
        self.assertLess(dist, 1.0)
        self.assertGreater(dist, 0.25)
        self.assertGreater(float((eye - center)[0]), 0.0)

