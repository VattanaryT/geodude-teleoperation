import unittest
from unittest.mock import patch

import numpy as np

from geodude_teleoperation.virtual_fixture import (
    FieldAlignPreview,
    _orient_approach_into_surface,
    field_z_display_dirs,
    field_z_segments,
)


class _FakeField:
    def __init__(self):
        self.calls = []
        self.basis_calls = []
        self.approach_calls = []

    def project(self, live_ee):
        self.calls.append(live_ee)
        return f"projected:{live_ee}"

    def display_basis_at(self, xyz):
        self.basis_calls.append(tuple(np.asarray(xyz, dtype=float).reshape(3)))
        return np.eye(3)

    def approach_basis_at(self, xyz):
        self.approach_calls.append(tuple(np.asarray(xyz, dtype=float).reshape(3)))
        return np.diag([1.0, -1.0, -1.0])


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

    def test_update_display_basis_is_throttled(self):
        vf = _FakeField()
        preview = FieldAlignPreview(vf, min_dt=1.0 / 15.0)
        b0 = preview.update_display_basis(np.array([1.0, 2.0, 3.0]), now=0.0)
        b1 = preview.update_display_basis(np.array([9.0, 9.0, 9.0]), now=0.05)
        np.testing.assert_allclose(b0, np.eye(3))
        np.testing.assert_allclose(b1, np.eye(3))
        self.assertEqual(len(vf.basis_calls), 1)

    def test_update_approach_basis_is_throttled_and_flipped(self):
        vf = _FakeField()
        preview = FieldAlignPreview(vf, min_dt=1.0 / 15.0)
        b0 = preview.update_approach_basis(np.array([1.0, 2.0, 3.0]), now=0.0)
        b1 = preview.update_approach_basis(np.array([9.0, 9.0, 9.0]), now=0.05)
        np.testing.assert_allclose(b0, np.diag([1.0, -1.0, -1.0]))
        np.testing.assert_allclose(b1, np.diag([1.0, -1.0, -1.0]))
        self.assertEqual(len(vf.approach_calls), 1)
        self.assertEqual(len(vf.basis_calls), 0)


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

    def test_display_basis_z_matches_outward_arrows(self):
        local = np.eye(3)
        into = _orient_approach_into_surface(local)
        outward_z = field_z_display_dirs(into[:, 2], outward=True)
        np.testing.assert_allclose(outward_z, local[:, 2])

    def test_display_dirs_outward_negates_into_surface_approach(self):
        from geodude_teleoperation.virtual_fixture import field_z_display_dirs

        into_surface = np.array([[0.0, 0.0, -1.0]])
        np.testing.assert_allclose(
            field_z_display_dirs(into_surface, outward=True), [[0.0, 0.0, 1.0]])
        np.testing.assert_allclose(
            field_z_display_dirs(into_surface, outward=False), into_surface)


def _stub_orientation_field():
    from geodude_teleoperation.virtual_fixture import OrientationField

    vf = OrientationField.__new__(OrientationField)
    vf._t_spatial = 1.0
    vf._max_iterations = 50
    vf._has_last = False
    vf._last_basis = np.eye(3)
    vf._table_flat = np.eye(3)
    vf._far_near = 0.04
    vf._far_far = 0.20
    vf.get_planar_signed_distance = lambda position: -1.0
    return vf


class _AlwaysSucceeds:
    def diffuse(self, position, t, max_iterations=50):
        del position, t, max_iterations
        return "rotor"


class _AlwaysUnderflow:
    def diffuse(self, position, t, max_iterations=50):
        del position, t, max_iterations
        raise RuntimeError("underflow")


class FarFieldBlendTest(unittest.TestCase):
    def test_far_field_weight_is_zero_near_and_one_far(self):
        from geodude_teleoperation.virtual_fixture import far_field_weight

        self.assertEqual(far_field_weight(0.0, near=0.04, far=0.20), 0.0)
        self.assertEqual(far_field_weight(0.04, near=0.04, far=0.20), 0.0)
        self.assertEqual(far_field_weight(0.20, near=0.04, far=0.20), 1.0)
        self.assertEqual(far_field_weight(1.0, near=0.04, far=0.20), 1.0)
        mid = far_field_weight(0.12, near=0.04, far=0.20)
        self.assertGreater(mid, 0.0)
        self.assertLess(mid, 1.0)

    def test_ease_out_has_weight_at_the_rim(self):
        from geodude_teleoperation.virtual_fixture import far_field_weight

        # Smoothstep would still be ~0.03 here and keep object tilt through the table.
        w = far_field_weight(0.05, near=0.0, far=0.45)
        self.assertGreater(w, 0.2)
        self.assertLess(w, 0.5)

    def test_overlap_starts_inside_the_hull(self):
        from geodude_teleoperation.virtual_fixture import (
            FAR_FIELD_FAR,
            FAR_FIELD_NEAR,
            far_field_weight,
        )

        self.assertLess(FAR_FIELD_NEAR, 0.0)
        rim = far_field_weight(0.0, near=FAR_FIELD_NEAR, far=FAR_FIELD_FAR)
        self.assertGreater(rim, 0.0)
        self.assertLess(rim, 0.5)
        core = far_field_weight(FAR_FIELD_NEAR, near=FAR_FIELD_NEAR, far=FAR_FIELD_FAR)
        self.assertEqual(core, 0.0)

    def test_planar_hull_signed_distance_inside_and_outside(self):
        from geodude_teleoperation.virtual_fixture import planar_hull_signed_distance

        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        self.assertLess(planar_hull_signed_distance(square, [0.5, 0.5]), 0.0)
        outside = planar_hull_signed_distance(square, [1.5, 0.5])
        self.assertGreater(outside, 0.0)
        self.assertAlmostEqual(outside, 0.5, places=5)

    def test_near_object_keeps_object_approach(self):
        import geodude_teleoperation.virtual_fixture as vf_mod

        local = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        vf = _stub_orientation_field()
        vf._field = _AlwaysSucceeds()
        vf.get_planar_signed_distance = lambda position: -1.0
        with patch.object(vf_mod, "_fiber_rotor_to_rotation_matrix", return_value=local):
            near = vf.approach_basis_at([0.0, 0.0, 0.0])
        np.testing.assert_allclose(near, vf_mod._orient_approach_into_surface(local))

    def test_far_from_object_is_table_flat_approach(self):
        import geodude_teleoperation.virtual_fixture as vf_mod

        local = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        vf = _stub_orientation_field()
        vf._field = _AlwaysSucceeds()
        vf.get_planar_signed_distance = lambda position: 1.0
        with patch.object(vf_mod, "_fiber_rotor_to_rotation_matrix", return_value=local):
            far = vf.approach_basis_at([1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            far, vf_mod._orient_approach_into_surface(np.eye(3)))
        np.testing.assert_allclose(far[:, 2], [0.0, 0.0, -1.0])

    def test_underflow_far_away_is_table_flat_not_last_tilt(self):
        from geodude_teleoperation.virtual_fixture import _orient_approach_into_surface

        vf = _stub_orientation_field()
        vf._field = _AlwaysUnderflow()
        vf.get_planar_signed_distance = lambda position: 1.0
        far = vf.approach_basis_at([1.0, 0.0, 0.0])
        np.testing.assert_allclose(far, _orient_approach_into_surface(np.eye(3)))


class WorkstationFieldArrowTest(unittest.TestCase):
    def test_workstation_arrows_point_up_and_skip_object(self):
        from geodude_teleoperation.virtual_fixture import workstation_field_z_arrows

        pts = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
        segs = workstation_field_z_arrows(
            pts, pad=0.2, spacing=0.1, length=0.05, exclude_radius=0.05)
        self.assertGreater(len(segs), 0)
        np.testing.assert_allclose(segs[:, 0, 2], 0.0)
        dirs = segs[:, 1] - segs[:, 0]
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        np.testing.assert_allclose(dirs, np.tile([0.0, 0.0, 1.0], (len(segs), 1)))
        for origin in segs[:, 0]:
            self.assertGreater(
                float(np.linalg.norm(pts - origin, axis=1).min()), 0.05 - 1e-9)

