import unittest

import numpy as np

from geodude_teleoperation.vention import (
    LEFT_VENTION_JOINT,
    RIGHT_VENTION_JOINT,
    apply_vention_heights,
    vention_dof_indices,
)


class FakeJoint:
    def __init__(self, name, index, actuated=True):
        self._name = name
        self._index = index
        self._actuated = actuated

    def get_name(self):
        return self._name

    def get_index(self):
        return self._index

    def is_actuated(self):
        return self._actuated


class FakeSystem:
    def __init__(self, joints):
        self._joints = joints

    def get_joints(self):
        return self._joints


class VentionOverlayTest(unittest.TestCase):
    def setUp(self):
        self.system = FakeSystem([
            FakeJoint(LEFT_VENTION_JOINT, 0),
            FakeJoint("left_ur5e/shoulder_pan_joint", 1),
            FakeJoint(RIGHT_VENTION_JOINT, 15),
            FakeJoint("fixed", 99, actuated=False),
        ])

    def test_indices_are_actuated_vention_dofs(self):
        self.assertEqual(
            vention_dof_indices(self.system),
            {LEFT_VENTION_JOINT: 0, RIGHT_VENTION_JOINT: 15},
        )

    def test_apply_writes_both_rails_and_leaves_other_dofs(self):
        q = np.arange(30, dtype=float)
        out = apply_vention_heights(self.system, q, left=0.25, right=0.4)
        self.assertEqual(out[0], 0.25)
        self.assertEqual(out[15], 0.4)
        self.assertEqual(out[1], 1.0)
        np.testing.assert_array_equal(q, np.arange(30, dtype=float))

    def test_missing_joint_raises(self):
        system = FakeSystem([FakeJoint(LEFT_VENTION_JOINT, 0)])
        with self.assertRaises(KeyError):
            apply_vention_heights(system, np.zeros(2), left=0.25, right=0.25)


if __name__ == "__main__":
    unittest.main()
