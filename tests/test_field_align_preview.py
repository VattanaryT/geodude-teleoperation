import unittest

import numpy as np

from geodude_teleoperation.virtual_fixture import (
    FieldAlignPreview,
    _orient_approach_into_surface,
    field_z_segments,
)


class _FakeField:
    def __init__(self):
        self.calls = []

    def project(self, live_ee):
        self.calls.append(live_ee)
        return f"projected:{live_ee}"


class FieldAlignPreviewTest(unittest.TestCase):
    def test_first_update_projects_live_ee(self):
        vf = _FakeField()
        preview = FieldAlignPreview(vf, min_dt=1.0 / 15.0)
        motor = preview.update("ee0", now=0.0)
        self.assertEqual(motor, "projected:ee0")
        self.assertEqual(vf.calls, ["ee0"])

    def test_ticker_faster_than_min_dt_does_not_reproject(self):
        vf = _FakeField()
        preview = FieldAlignPreview(vf, min_dt=1.0 / 15.0)
        preview.update("ee0", now=0.0)
        motor = preview.update("ee1", now=0.05)
        self.assertEqual(motor, "projected:ee0")
        self.assertEqual(vf.calls, ["ee0"])

    def test_reprojects_after_min_dt(self):
        vf = _FakeField()
        preview = FieldAlignPreview(vf, min_dt=1.0 / 15.0)
        preview.update("ee0", now=0.0)
        motor = preview.update("ee1", now=0.07)
        self.assertEqual(motor, "projected:ee1")
        self.assertEqual(vf.calls, ["ee0", "ee1"])


class FieldZArrowTest(unittest.TestCase):
    def test_approach_basis_z_is_negated_field_z(self):
        R = np.eye(3)
        B = _orient_approach_into_surface(R)
        np.testing.assert_allclose(B[:, 2], [0.0, 0.0, -1.0])

    def test_field_z_segments_are_start_to_end(self):
        origins = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        z_dirs = np.array([[0.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
        segs = field_z_segments(origins, z_dirs, 0.1)
        self.assertEqual(segs.shape, (2, 2, 3))
        np.testing.assert_allclose(segs[0, 0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(segs[0, 1], [0.0, 0.0, 0.1])
        np.testing.assert_allclose(segs[1, 1], [1.0, 0.1, 0.0])

    def test_display_dirs_outward_negates_into_surface_approach(self):
        from geodude_teleoperation.virtual_fixture import field_z_display_dirs

        into_surface = np.array([[0.0, 0.0, -1.0]])
        np.testing.assert_allclose(
            field_z_display_dirs(into_surface, outward=True), [[0.0, 0.0, 1.0]])
        np.testing.assert_allclose(
            field_z_display_dirs(into_surface, outward=False), into_surface)

