"""The feed-forward press and the external latch that owns it."""

import unittest

import gafropy as ga
import numpy as np

from geodude_teleoperation.controller import ContactTeleoperationController


def _make(**kwargs):
    config = ga.control.VelocityAdmittanceControllerConfig()
    config.dt = 0.005
    config.max_linear_speed = 0.005
    config.max_angular_speed = 0.15
    config.admittance = ga.control.AdmittanceConfig(
        inertia=np.eye(6),
        damping=np.diag([40.0] * 5 + [200.0]),
        stiffness=np.diag([400.0] * 5 + [0.0]),
    )
    params = dict(
        vf=None, config=config, target_force=15.0, log_period_s=0.0,
        press_slew_n_s=0.0,   # no rate limit: assert the law, not the ramp
        press_feedforward=True,
    )
    params.update(kwargs)
    return ContactTeleoperationController(**params)


class FeedforwardPressTest(unittest.TestCase):
    def test_zero_until_the_external_latch_engages(self):
        c = _make()
        c.set_press_external(False)
        self.assertEqual(c._press_force_n(0.0, 0.0, 0.005), 0.0)
        self.assertFalse(c._press_engaged)

    def test_commands_fd_flat_regardless_of_fs(self):
        c = _make()
        c.set_press_external(True)
        # Same command at every plausible reading of a noisy sensor.
        for fs in (-1.0, 0.0, 3.7, 14.9, 15.0, 21.0):
            self.assertAlmostEqual(c._press_force_n(fs, 0.0, 0.005), 15.0)
        self.assertTrue(c._press_engaged)

    def test_wrong_sign_does_not_zero_the_feedforward_press(self):
        # The wrong-sign rule is a statement about a measurement this path does
        # not use. A slightly negative Fs is exactly what an unloaded, noisy,
        # tared sensor reads, and it must not stop the press.
        c = _make()
        c.set_press_external(True)
        self.assertAlmostEqual(c._press_force_n(-2.5, 0.0, 0.005), 15.0)

    def test_overload_latches_off_and_recovers_with_hysteresis(self):
        c = _make(press_overload_n=30.0)
        c.set_press_external(True)
        self.assertAlmostEqual(c._press_force_n(10.0, 0.0, 0.005), 15.0)
        self.assertEqual(c._press_force_n(31.0, 0.0, 0.005), 0.0)
        self.assertFalse(c._press_engaged)
        # Still latched off between half and full overload.
        self.assertEqual(c._press_force_n(20.0, 0.0, 0.005), 0.0)
        self.assertAlmostEqual(c._press_force_n(14.0, 0.0, 0.005), 15.0)

    def test_overload_uses_magnitude_so_an_inverted_sign_still_trips(self):
        c = _make(press_overload_n=30.0)
        c.set_press_external(True)
        self.assertEqual(c._press_force_n(-31.0, 0.0, 0.005), 0.0)

    def test_default_overload_is_twice_fd(self):
        self.assertAlmostEqual(_make()._press_overload_n, 30.0)

    def test_press_max_still_bounds_the_command(self):
        c = _make(target_force=40.0, press_max_n=25.0)
        c.set_press_external(True)
        self.assertAlmostEqual(c._press_force_n(0.0, 0.0, 0.005), 25.0)

    def test_external_latch_without_feedforward_keeps_the_hold_trim(self):
        c = _make(press_feedforward=False, press_kp_hold=0.5)
        c.set_press_external(True)
        # Fd + kp*(Fd - Fs) = 15 + 0.5*(15 - 11) = 17
        self.assertAlmostEqual(c._press_force_n(11.0, 0.0, 0.005), 17.0)

    def test_none_restores_the_internal_fs_latch(self):
        c = _make(press_feedforward=True)
        c.set_press_external(True)
        c._press_force_n(0.0, 0.0, 0.005)
        self.assertTrue(c._press_engaged)
        c.set_press_external(None)
        # Fs = 0 is under engage_frac*Fd, so the Fs latch releases and the
        # spring -- not the press -- owns the approach again.
        self.assertEqual(c._press_force_n(0.0, 0.0, 0.005), 0.0)
        self.assertFalse(c._press_engaged)

    def test_reset_clears_the_overload_latch(self):
        c = _make(press_overload_n=30.0)
        c.set_press_external(True)
        c._press_force_n(31.0, 0.0, 0.005)
        self.assertTrue(c._press_overloaded)
        c._reset_press_loop()
        self.assertFalse(c._press_overloaded)

    def test_slew_still_rate_limits_the_rise(self):
        # A real run always ticks unengaged first (the approach phase), which
        # seeds the slew at 0 -- so the step onto the part is ramped.
        c = _make(press_slew_n_s=200.0)
        c.set_press_external(False)
        self.assertEqual(c._press_force_n(0.0, 0.0, 0.005), 0.0)
        c.set_press_external(True)
        first = c._press_force_n(0.0, 0.0, 0.005)
        self.assertAlmostEqual(first, 1.0)   # 200 N/s * 5 ms
        self.assertAlmostEqual(c._press_force_n(0.0, 0.0, 0.005), 2.0)


class LatchDebugTest(unittest.TestCase):
    def test_latch_debug_is_recorded_for_the_log(self):
        c = _make()
        c.set_press_latch_debug(distance_m=0.0031, phase="traverse")
        self.assertAlmostEqual(c._latch_d_m, 0.0031)
        self.assertEqual(c._latch_phase, "traverse")

    def test_log_fields_carry_the_latch_columns(self):
        self.assertIn("d_surf", ContactTeleoperationController._LOG_FIELDS)
        self.assertIn("phase", ContactTeleoperationController._LOG_FIELDS)


if __name__ == "__main__":
    unittest.main()


class ExternalLatchReachesTheSpringTest(unittest.TestCase):
    """The external latch must drive the SAME bool the Fs latch used to.

    Three things have to agree or contact deadlocks: whether the pose spring
    keeps the contact normal (free_along_normal), whether the press commands
    anything, and whether field-follow strips the normal out of a step. They
    all read one bool, and under --path-mode that bool comes from the plan.
    """

    def test_engagement_is_published_to_the_spring(self):
        seen = []
        c = _make()
        c.set_on_press_engaged(seen.append)
        c.set_on_press_wrench(lambda w: None)
        c.set_press_external(False)
        c._emit_press_wrench(c._press_force_n(0.0, 0.0, 0.005))
        c.set_press_external(True)
        c._emit_press_wrench(c._press_force_n(0.0, 0.0, 0.005))
        self.assertEqual(seen, [False, True])

    def test_an_overload_hands_the_normal_back_to_the_spring(self):
        seen = []
        c = _make(press_overload_n=30.0)
        c.set_on_press_engaged(seen.append)
        c.set_on_press_wrench(lambda w: None)
        c.set_press_external(True)
        c._emit_press_wrench(c._press_force_n(5.0, 0.0, 0.005))
        c._emit_press_wrench(c._press_force_n(40.0, 0.0, 0.005))
        self.assertEqual(seen, [True, False])
