import unittest
from pathlib import Path

import numpy as np

from geodude_teleoperation.wrist_cam import (
    annotate_wrist_frame,
    nearest_indices,
    overlay_caption,
    WRIST_CAM_TOPIC,
)

_BAG = Path("/mnt/data/boeing/data/bags_08_16_half_stringer/bimanual_0")
_DATA = Path("/mnt/data/boeing/local_frames_data/08_16_half_stringer_local_frames")
_NPZ = _DATA / "bags_08_16_half_stringer_bimanual_0_cga_actual.npz"
_CAM = _DATA / "bags_08_16_half_stringer_bimanual_0_wrist_cam.npz"


class NearestIndexTest(unittest.TestCase):
    def test_picks_closest_source_stamp(self):
        source = np.array([0.0, 0.10, 0.20, 0.30])
        query = np.array([0.0, 0.14, 0.29])
        np.testing.assert_array_equal(
            nearest_indices(query, source), [0, 1, 3])

    def test_empty_source_raises(self):
        with self.assertRaises(ValueError):
            nearest_indices(np.array([0.0]), np.array([]))


class OverlayTest(unittest.TestCase):
    def test_caption_includes_mode_time_force(self):
        self.assertEqual(
            overlay_caption(mode="contact", t_s=12.3, f_n=34.14),
            "contact  t=12.30s  fn=34.1N",
        )

    def test_caption_handles_missing_force(self):
        self.assertEqual(
            overlay_caption(mode="approach", t_s=0.0, f_n=None),
            "approach  t=0.00s  fn=n/a",
        )

    def test_annotate_paints_caption_on_copy(self):
        rgb = np.zeros((80, 160, 3), dtype=np.uint8)
        rgb[:] = (10, 20, 30)
        out = annotate_wrist_frame(
            rgb, mode="contact", t_s=2.6, f_n=34.1)
        self.assertEqual(out.shape, rgb.shape)
        np.testing.assert_array_equal(rgb[0, 0], [10, 20, 30])
        self.assertGreater(int(out.max()), 30)
        self.assertEqual(overlay_caption("contact", 2.6, 34.1), "contact  t=2.60s  fn=34.1N")

    def test_annotate_bar_scales_with_frame_width(self):
        small = annotate_wrist_frame(
            np.zeros((80, 160, 3), dtype=np.uint8), mode="free", t_s=0.0, f_n=0.0)
        wide = annotate_wrist_frame(
            np.zeros((480, 848, 3), dtype=np.uint8), mode="free", t_s=0.0, f_n=0.0)
        def bar_h(img):
            # first row that is not the green/red bar
            col = img[:, 0]
            painted = np.any(col != 0, axis=1)
            return int(np.argmax(~painted)) if (~painted).any() else img.shape[0]
        self.assertGreater(bar_h(wide), bar_h(small) + 20)


@unittest.skipUnless(_BAG.exists() and _NPZ.exists(), "bimanual_0 bag/npz not mounted")
class BagWristCamTest(unittest.TestCase):
    def test_images_cover_episode_and_align_within_one_cam_period(self):
        from geodude_teleoperation.wrist_cam import load_wrist_cam

        ep_t = np.load(_NPZ)["timestamps"]
        try:
            stamps, payloads = load_wrist_cam(_BAG, WRIST_CAM_TOPIC)
        except ImportError:
            self.skipTest("rclpy/rosbag2 not available")
        self.assertGreater(len(payloads), 100)
        self.assertEqual(len(stamps), len(payloads))
        self.assertLess(float(stamps[0]), float(ep_t[0]) + 0.2)
        self.assertGreater(float(stamps[-1]), float(ep_t[-1]) - 0.2)
        idx = nearest_indices(ep_t, stamps)
        err = np.abs(ep_t - stamps[idx])
        self.assertLess(float(err.max()), 0.08)
        self.assertIn(b"\xff\xd8", bytes(payloads[0][:3]))


class WristCamCacheTest(unittest.TestCase):
    def test_npz_roundtrip_without_ros(self):
        import tempfile

        import cv2

        from geodude_teleoperation.wrist_cam import load_wrist_cam

        rgb = np.zeros((24, 32, 3), dtype=np.uint8)
        rgb[5:10, 5:10] = (200, 10, 10)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.assertTrue(ok)
        payload = bytes(buf)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrist.npz"
            np.savez_compressed(
                path,
                stamps=np.array([1.0, 1.1]),
                jpeg=np.array([payload, payload], dtype=object),
            )
            stamps, payloads = load_wrist_cam(path)
        np.testing.assert_allclose(stamps, [1.0, 1.1])
        self.assertEqual(payloads[0][:2], b"\xff\xd8")


@unittest.skipUnless(_CAM.is_file() and _NPZ.is_file(), "wrist-cam cache not extracted")
class CachedWristCamTest(unittest.TestCase):
    def test_sidecar_aligns_to_episode(self):
        from geodude_teleoperation.wrist_cam import decode_jpeg, load_wrist_cam

        ep_t = np.load(_NPZ)["timestamps"]
        stamps, payloads = load_wrist_cam(_CAM)
        idx = nearest_indices(ep_t, stamps)
        err = np.abs(ep_t - stamps[idx])
        self.assertLess(float(err.max()), 0.08)
        rgb = decode_jpeg(payloads[int(idx[0])])
        self.assertEqual(rgb.ndim, 3)
        self.assertEqual(rgb.shape[2], 3)
