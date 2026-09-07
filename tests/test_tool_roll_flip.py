"""Tool orientation A/B: exactly 180 deg about tool Z, and nothing else moves."""

import unittest

import numpy as np
import trimesh

from geodude_teleoperation.virtual_fixture import OrientationField


def _slab():
    box = trimesh.creation.box(extents=(0.30, 0.06, 0.02))
    box.apply_translation([0.0, 0.0, 0.01])
    return box


class ToolRollFlipTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vf = OrientationField(_slab(), num_boundary_samples=64)
        cls.p = np.array([0.0, 0.0, 0.05])

    def test_flip_is_exactly_180_about_tool_z(self):
        self.vf.set_roll_flip(False)
        A = np.array(self.vf.approach_basis_at(self.p), copy=True)
        self.vf.set_roll_flip(True)
        B = np.array(self.vf.approach_basis_at(self.p), copy=True)
        # R_A^T R_B must be Rz(pi) in the tool frame.
        M = A.T @ B
        np.testing.assert_allclose(M, np.diag([-1.0, -1.0, 1.0]), atol=1e-9)

    def test_tool_z_and_the_surface_normal_are_untouched(self):
        self.vf.set_roll_flip(False)
        zA = np.array(self.vf.approach_basis_at(self.p)[:, 2], copy=True)
        self.vf.set_roll_flip(True)
        zB = np.array(self.vf.approach_basis_at(self.p)[:, 2], copy=True)
        # The press latch measures distance along this axis, so a flip must not
        # move it by even a hair.
        np.testing.assert_allclose(zA, zB, atol=1e-12)

    def test_it_round_trips(self):
        self.vf.set_roll_flip(False)
        A = np.array(self.vf.approach_basis_at(self.p), copy=True)
        self.vf.set_roll_flip(True)
        self.vf.set_roll_flip(False)
        np.testing.assert_allclose(self.vf.approach_basis_at(self.p), A, atol=1e-12)

    def test_the_constructor_flag_matches_the_setter(self):
        flipped = OrientationField(_slab(), num_boundary_samples=64, roll_flip=True)
        self.vf.set_roll_flip(True)
        np.testing.assert_allclose(flipped.approach_basis_at(self.p),
                                   self.vf.approach_basis_at(self.p), atol=1e-9)
        self.assertTrue(flipped.roll_flip)


if __name__ == "__main__":
    unittest.main()
