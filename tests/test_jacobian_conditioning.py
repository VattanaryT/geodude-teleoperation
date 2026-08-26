"""The joint-velocity path near a singularity.

The UR pendant reported C153A1 "position deviates from path" on the shoulder
while the contact loop was healthy (Fs=+9.67 N against Fd=10.0, v_z=+0.0005 m/s,
no guards). That is kinematic, and nothing in this path was ever measured: the
damped-least-squares regularizer was a hardcoded 1e-6 inherited from
ufactory_demos, the velocity clamp was on norm(qdot) only, and sigma_min was
never recorded, so a protective stop left no evidence of the pose that caused it.
"""
import unittest

import numpy as np

from geodude_teleoperation.controller import AdmittanceTeleoperationController


def _jacobian(sigma_min, n_joints=7):
    """6xN jacobian whose smallest singular value is sigma_min."""
    j = np.zeros((6, n_joints))
    j[:6, :6] = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, sigma_min])
    return j


def _qdot(j, v_task, lam, max_v=0.8, per_joint=True):
    jinv = np.linalg.inv(j.T @ j + lam ** 2 * np.eye(j.shape[1]))
    q = jinv @ j.T @ v_task
    speed = np.linalg.norm(q)
    if speed > max_v:
        q = q * (max_v / speed)
    return np.clip(q, -max_v, max_v) if per_joint else q


class SigmaMinTest(unittest.TestCase):
    def test_uses_j_jT_so_joint_redundancy_is_not_read_as_singular(self):
        """With 7 joints and 6 task DOF, J.T@J is rank-deficient by construction.

        Reading sigma_min off it reports every pose as singular -- the extra
        eigenvalue is the redundancy nullspace, not ill-conditioning.
        """
        ctrl = AdmittanceTeleoperationController()
        for sigma in (0.5, 0.2, 0.05, 0.005):
            with self.subTest(sigma=sigma):
                ctrl._note_conditioning(_jacobian(sigma, n_joints=7))
                self.assertAlmostEqual(ctrl._last_sigma_min, sigma, places=6)

    def test_a_square_jacobian_reads_the_same(self):
        ctrl = AdmittanceTeleoperationController()
        ctrl._note_conditioning(_jacobian(0.05, n_joints=6))
        self.assertAlmostEqual(ctrl._last_sigma_min, 0.05, places=6)

    def test_conditioning_is_reported_for_the_contact_log(self):
        ctrl = AdmittanceTeleoperationController()
        ctrl._note_conditioning(_jacobian(0.03))
        c = ctrl.conditioning()
        self.assertAlmostEqual(c["sigma_min"], 0.03, places=6)
        self.assertIn("clamp_scale", c)
        self.assertIn("qdot_max", c)


class DlsDampingTest(unittest.TestCase):
    """lambda=1e-3 peaks the inverse gain at 1/(2*lambda) = 500."""

    V = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0052])   # 5.2 mm/s, the logged case

    def test_well_conditioned_poses_barely_change(self):
        # The reduction to the previously-working case: free-space teleop lives
        # here, and it must not feel different.
        j = _jacobian(0.2)
        old = np.max(np.abs(_qdot(j, self.V, 0.001)))
        new = np.max(np.abs(_qdot(j, self.V, 0.05)))
        self.assertLess(abs(new - old) / old, 0.10)

    def test_an_ordinary_near_singular_pose_no_longer_pins_the_clamp(self):
        j = _jacobian(0.005)
        old = np.max(np.abs(_qdot(j, self.V, 0.001)))
        new = np.max(np.abs(_qdot(j, self.V, 0.05)))
        self.assertAlmostEqual(old, 0.8, places=6)      # saturated
        self.assertLess(new, 0.05)                      # not saturated

    def test_the_default_matches_the_sibling_controller(self):
        # TeleoperationController has always used damping=0.05 for the same
        # math; the admittance controller hardcoded 2500x less.
        self.assertAlmostEqual(
            AdmittanceTeleoperationController()._dls_damping, 0.05, places=9)

    def test_the_old_value_is_still_reachable_for_a_bisect(self):
        ctrl = AdmittanceTeleoperationController(dls_damping=0.001)
        self.assertAlmostEqual(ctrl._dls_damping, 0.001, places=9)


class PerJointClampTest(unittest.TestCase):
    def test_norm_clamp_alone_lets_one_joint_take_the_whole_budget(self):
        j = _jacobian(0.005)
        q = _qdot(j, DlsDampingTest.V, 0.001, per_joint=False)
        self.assertAlmostEqual(np.linalg.norm(q), 0.8, places=6)
        self.assertGreater(np.max(np.abs(q)), 0.79)

    def test_per_joint_clamp_is_on_by_default_and_disableable(self):
        self.assertTrue(AdmittanceTeleoperationController()._per_joint_velocity)
        self.assertFalse(
            AdmittanceTeleoperationController(
                per_joint_velocity=False)._per_joint_velocity)


if __name__ == "__main__":
    unittest.main()
