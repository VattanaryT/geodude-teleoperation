import unittest

import numpy as np

from geodude_teleoperation.controller import slew_qdot


class SlewQdotTest(unittest.TestCase):
    def test_first_command_is_unchanged(self):
        desired = np.array([0.55, 0.0])
        got = slew_qdot(None, desired, dt=0.001, max_qddot=5.0)
        np.testing.assert_allclose(got, desired)

    def test_idle_zero_does_not_step_in_one_tick(self):
        prev = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        got = slew_qdot(prev, np.zeros(6), dt=0.0011, max_qddot=5.0)
        step = float(np.linalg.norm(got - prev))
        self.assertAlmostEqual(step, 5.0 * 0.0011, places=9)
        self.assertGreater(float(np.linalg.norm(got)), 0.5)

    def test_hitch_dt_is_capped_so_stall_cannot_dump_delta(self):
        prev = np.zeros(2)
        desired = np.array([0.8, 0.0])
        got = slew_qdot(prev, desired, dt=0.040, max_qddot=5.0, max_dt=0.004)
        self.assertAlmostEqual(float(np.linalg.norm(got)), 5.0 * 0.004, places=9)

    def test_small_step_passes_through(self):
        prev = np.array([0.01, 0.0])
        desired = np.array([0.012, 0.0])
        got = slew_qdot(prev, desired, dt=0.001, max_qddot=5.0)
        np.testing.assert_allclose(got, desired)


    def test_lowpass_does_not_follow_a_one_tick_flip(self):
        from geodude_teleoperation.controller import lowpass_qdot
        a = np.array([0.4, 0.0])
        b = np.array([-0.4, 0.0])
        filt = lowpass_qdot(None, a, dt=0.001, tau=0.03)
        filt = lowpass_qdot(filt, b, dt=0.001, tau=0.03)
        self.assertGreater(float(filt[0]), 0.3)
