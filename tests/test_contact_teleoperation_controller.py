import inspect
import pathlib
import time
import unittest

import gafropy as ga
import numpy as np

from geodude_teleoperation.controller import ContactTeleoperationController


def _make_controller(**kwargs):
    config = ga.control.VelocityAdmittanceControllerConfig()
    config.dt = 0.01
    config.max_linear_speed = 0.005
    config.max_angular_speed = 0.15
    config.admittance = ga.control.AdmittanceConfig(
        inertia=np.eye(6),
        damping=np.diag([40.0] * 5 + [200.0]),
        stiffness=np.diag([400.0] * 5 + [0.0]),
    )
    params = dict(
        vf=None,
        config=config,
        # These cases were written against the bare PID and assert its exact
        # numbers, so pin them there; the dtc loop has its own cover in
        # test_force_pid.py and in DtcPressLoopTest below.
        press_loop="legacy",
        target_force=15.0,
        force_sign=1.0,
        log_period_s=0.0,
        dead_wrench_abort_s=0.4,
        kp_force=0.001,
        ki_force=0.0,
        v_approach=0.005,
        max_v_hold=0.005,
        z_damping=200.0,
        v_slew_mps2=0.0,
        lost_s=1.0,
        sit_s=0.0,
        contact_release_n=0.4,
        fz_tau_s=0.0,
        force_limit=25.0,
    )
    params.update(kwargs)
    return ContactTeleoperationController(**params)


def _vz(ctrl, wrench):
    return float(np.asarray(wrench)[5]) / ctrl._z_damping


class ContactForceLoopTest(unittest.TestCase):
    def test_aborts_if_force_stays_near_zero(self):
        ctrl = _make_controller(measured_fz_fn=lambda: 0.008)
        ctrl._t_start = time.monotonic() - 1.0
        ctrl._compute_force_command()
        self.assertTrue(ctrl._force_aborted)

    def test_does_not_abort_after_force_was_seen(self):
        samples = iter([-5.16, 0.008])
        ctrl = _make_controller(measured_fz_fn=lambda: next(samples))
        ctrl._t_start = time.monotonic() - 1.0
        ctrl._compute_force_command()
        ctrl._compute_force_command()
        self.assertFalse(ctrl._force_aborted)

    def test_in_air_is_latched_contact_pid_not_approach_feed(self):
        # Button 0 starts DualLoop already in contact. Air must PID at max_v_hold,
        # not the 5 mm/s DualLoop approach feed (that was free-space + C153).
        ctrl = _make_controller(
            measured_fz_fn=lambda: 0.09, force_sign=1.0,
            v_approach=0.005, max_v_hold=0.002)
        ctrl._t_start = time.monotonic()
        desired = ctrl._compute_force_command()
        self.assertEqual(ctrl._z_mode, "contact")
        vz = _vz(ctrl, desired)
        self.assertGreater(vz, 0.0)
        self.assertLessEqual(vz, 0.002 + 1e-9)
        np.testing.assert_allclose(ctrl._external_wrench, np.zeros(6))

    def test_positive_fz_stays_latched_contact(self):
        # UR ActualTCPForce is the reaction: +Z press loads Fz < 0. A +Fz blip
        # must not drop DualLoop out of the button-latched session.
        ctrl = _make_controller(measured_fz_fn=lambda: 10.39, force_sign=1.0)
        ctrl._t_start = time.monotonic()
        desired = ctrl._compute_force_command()
        self.assertEqual(ctrl._z_mode, "contact")
        self.assertLessEqual(abs(_vz(ctrl, desired)), ctrl._max_v_hold + 1e-9)

    def test_negative_fz_engages_plus_z_press(self):
        ctrl = _make_controller(measured_fz_fn=lambda: -10.39, force_sign=1.0)
        ctrl._t_start = time.monotonic()
        ctrl._compute_force_command()
        self.assertEqual(ctrl._z_mode, "contact")
        self.assertEqual(ctrl._press_sign, 1.0)

    def test_raw_spike_engages_without_signed_lpf(self):
        seq = [0.0] * 40 + [-17.08]
        samples = iter(seq)
        ctrl = _make_controller(
            measured_fz_fn=lambda: next(samples), force_sign=1.0, fz_tau_s=0.03)
        ctrl._t_start = time.monotonic()
        for _ in seq:
            ctrl._compute_force_command()
        self.assertEqual(ctrl._z_mode, "contact")

    def test_signed_pid_underforce_presses(self):
        # UR: Fz=-5.16, sign=+1 → Fs=+5.16, Fe=9.84 → clip max_v_hold into +Z.
        ctrl = _make_controller(measured_fz_fn=lambda: -5.16, force_sign=1.0, sit_s=0.0)
        ctrl._t_start = time.monotonic()
        desired = ctrl._compute_force_command()
        np.testing.assert_allclose(_vz(ctrl, desired), 0.005)
        np.testing.assert_allclose(ctrl._external_wrench, np.zeros(6))

    def test_light_contact_does_not_rearm_approach_feed(self):
        # Live: Fs=5.71 N → v_z=+5 mm/s approach while already in contact (C153).
        ctrl = _make_controller(
            measured_fz_fn=lambda: -5.71, force_sign=1.0, sit_s=0.0,
            v_approach=0.005, max_v_hold=0.002)
        ctrl._t_start = time.monotonic()
        desired = ctrl._compute_force_command()
        self.assertEqual(ctrl._z_mode, "contact")
        vz = _vz(ctrl, desired)
        self.assertGreater(vz, 0.0)
        self.assertLessEqual(vz, 0.002 + 1e-9)

    def test_signed_pid_overload_retracts(self):
        ctrl = _make_controller(measured_fz_fn=lambda: -20.0, force_sign=1.0, sit_s=0.0)
        ctrl._t_start = time.monotonic()
        desired = ctrl._compute_force_command()
        self.assertLess(_vz(ctrl, desired), 0.0)

    def test_c153_log_stops_pressing_into_reaction_load(self):
        # Live: Fz=-21…-35 N with v_z still +1 mm/s (C153). Must retract/hold, not press.
        ctrl = _make_controller(
            measured_fz_fn=lambda: -21.59, force_sign=1.0, sit_s=0.0, max_v_hold=0.001)
        ctrl._t_start = time.monotonic()
        desired = ctrl._compute_force_command()
        self.assertEqual(ctrl._z_mode, "contact")
        self.assertLessEqual(_vz(ctrl, desired), 0.0)

    def test_unload_past_lost_s_stays_contact(self):
        # Live: DualLoop dropped to approach after |Fs|<0.4 for 1 s, then drifted
        # off in free space. Button 0 must keep DualLoop contact until stop().
        samples = iter([-10.39, 0.0, 0.0])
        ctrl = _make_controller(
            measured_fz_fn=lambda: next(samples), force_sign=1.0, sit_s=0.0,
            v_approach=0.005, max_v_hold=0.002, lost_s=1.0)
        ctrl._t_start = time.monotonic()
        ctrl._compute_force_command()
        self.assertEqual(ctrl._z_mode, "contact")
        ctrl._unload_since = time.monotonic() - 1.1
        desired = ctrl._compute_force_command()
        self.assertEqual(ctrl._z_mode, "contact")
        vz = _vz(ctrl, desired)
        self.assertGreater(vz, 0.0)
        self.assertLessEqual(vz, 0.002 + 1e-9)

    def test_start_latches_contact(self):
        ctrl = _make_controller(measured_fz_fn=lambda: 0.0, log_period_s=0.0)
        self.assertEqual(ctrl._z_mode, "contact")
        ctrl._z_mode = "approach"
        ctrl.start()
        try:
            self.assertEqual(ctrl._z_mode, "contact")
        finally:
            ctrl.stop()

    def test_fs_sign_flip_stays_contact_and_does_not_press(self):
        # Sliding can flip Fz. Stay in contact and do not re-arm DualLoop
        # approach. The second sample is 17 N against a 15 N target, so the
        # command must not be a press: at or over target the answer is 0 or
        # retract, and the wrong-sign guard answers a bounce the same way.
        # This asserted `vz > 0` when the loop still PID'd through an
        # overshoot; that is the behavior the C153 work removed.
        samples = iter([-15.0, 17.0])
        ctrl = _make_controller(
            measured_fz_fn=lambda: next(samples), force_sign=1.0, sit_s=0.0,
            max_v_hold=0.002, v_approach=0.005)
        ctrl._t_start = time.monotonic()
        ctrl._compute_force_command()
        second = ctrl._compute_force_command()
        self.assertEqual(ctrl._z_mode, "contact")
        vz = _vz(ctrl, second)
        self.assertLessEqual(vz, 0.0)
        # unload_v_m_s is a floor on the escape speed, not a ceiling, so a
        # retract is allowed to exceed max_v_hold -- only a press is not.

    def test_under_target_after_light_contact_still_pids(self):
        # Fs dip / sign noise must not retract off the part (free-space drift).
        samples = iter([-10.39, 1.50])
        ctrl = _make_controller(
            measured_fz_fn=lambda: next(samples), force_sign=1.0, sit_s=0.0,
            max_v_hold=0.002, v_approach=0.005)
        ctrl._t_start = time.monotonic()
        ctrl._compute_force_command()
        second = ctrl._compute_force_command()
        self.assertEqual(ctrl._z_mode, "contact")
        vz = _vz(ctrl, second)
        self.assertGreater(vz, 0.0)
        self.assertLessEqual(vz, 0.002 + 1e-9)

    def test_loaded_fs_dip_pids_without_rearming_approach(self):
        # Sliding unloads Fs. Must keep PID (≤ max_v_hold), not sit at 0 or +5 mm/s.
        samples = iter([-13.0, -1.4])
        ctrl = _make_controller(
            measured_fz_fn=lambda: next(samples), force_sign=1.0, sit_s=0.0,
            max_v_hold=0.002, v_approach=0.005)
        ctrl._t_start = time.monotonic()
        first = ctrl._compute_force_command()
        self.assertEqual(ctrl._z_mode, "contact")
        self.assertGreater(_vz(ctrl, first), 0.0)
        second = ctrl._compute_force_command()
        vz = _vz(ctrl, second)
        self.assertGreater(vz, 0.0)
        self.assertLessEqual(vz, 0.002 + 1e-9)

    def test_sit_only_dwells_when_actually_loaded(self):
        # Live: sit at ~2 N with v_z=0 for 2 s, never reached 15 N.
        ctrl = _make_controller(
            measured_fz_fn=lambda: -2.41, force_sign=1.0, sit_s=2.0)
        ctrl._t_start = time.monotonic()
        # Force a contact state without a 3.75 N spike.
        ctrl._z_mode = "contact"
        ctrl._contact_since = time.monotonic()
        desired = ctrl._compute_force_command()
        self.assertGreater(_vz(ctrl, desired), 0.0)

    def test_contact_gate_sticky_on_bounce(self):
        samples = iter([-10.39, -0.5])
        ctrl = _make_controller(measured_fz_fn=lambda: next(samples), force_sign=1.0)
        ctrl._t_start = time.monotonic()
        ctrl._compute_force_command()
        ctrl._compute_force_command()
        self.assertEqual(ctrl._z_mode, "contact")

    def test_loaded_then_zero_does_not_reenable_approach_speed(self):
        samples = iter([-10.39, 0.0])
        ctrl = _make_controller(
            measured_fz_fn=lambda: next(samples), force_sign=1.0, sit_s=2.0)
        ctrl._t_start = time.monotonic()
        ctrl._compute_force_command()
        ctrl._compute_force_command()
        self.assertEqual(ctrl._z_mode, "contact")

    def test_negative_force_sign_in_air_feeds_minus_z(self):
        ctrl = _make_controller(
            measured_fz_fn=lambda: -0.09, force_sign=-1.0, max_v_hold=0.005)
        ctrl._t_start = time.monotonic()
        desired = ctrl._compute_force_command()
        self.assertEqual(ctrl._z_mode, "contact")
        np.testing.assert_allclose(_vz(ctrl, desired), -0.005)

    def test_sit_does_not_zero_press_under_fd(self):
        # DualLoop sit at v_z=0 never reached Fd; TCP floats off.
        ctrl = _make_controller(
            measured_fz_fn=lambda: -11.04, force_sign=1.0, sit_s=2.0,
            max_v_hold=0.002)
        ctrl._t_start = time.monotonic()
        desired = ctrl._compute_force_command()
        self.assertEqual(ctrl._z_mode, "contact")
        vz = _vz(ctrl, desired)
        self.assertGreater(vz, 0.0)
        self.assertLessEqual(vz, 0.002 + 1e-9)

    def test_on_vz_callback_pids_under_target(self):
        seen = []
        ctrl = _make_controller(measured_fz_fn=lambda: -11.04, force_sign=1.0, sit_s=2.0)
        ctrl.set_on_vz(seen.append)
        ctrl._t_start = time.monotonic()
        ctrl._compute_force_command()
        self.assertEqual(ctrl._z_mode, "contact")
        self.assertGreater(seen[-1], 0.0)


class DtcPressLoopTest(unittest.TestCase):
    """The dtc delegation, mostly about signs. The loop itself is covered in
    test_force_pid.py; what matters here is that the base-frame press axis and
    the tool-Z velocity handed to _on_vz still mean what legacy meant."""

    def _dtc(self, **kwargs):
        params = dict(
            press_loop="dtc", max_v_hold=0.008, v_approach=0.005,
            log_period_s=0.0, dead_wrench_abort_s=0.0)
        params.update(kwargs)
        return _make_controller(**params)

    def test_reduces_to_legacy_axis_without_a_normal(self):
        # No set_contact_normal yet: press must still be press_sign * tool Z.
        for sign in (1.0, -1.0):
            with self.subTest(sign=sign):
                ctrl = self._dtc(
                    force_sign=sign,
                    measured_force_fn=lambda: (0.0, 0.0, 0.0))
                ctrl.set_contact_normal(None)
                ctrl._R_tool_to_base = np.eye(3)
                ctrl._t_start = time.monotonic()
                seen = []
                ctrl.set_on_vz(seen.append)
                ctrl._compute_force_command()
                # Free air seeks the surface along the sign the CLI selected.
                self.assertGreater(seen[-1] * sign, 0.0)

    def test_compression_reads_positive_through_the_reaction_convention(self):
        # UR reports the reaction, so a +tool-Z press reads fz < 0 and must land
        # on Fs > 0 -- same convention as _fs() in the legacy path.
        ctrl = self._dtc(
            force_sign=1.0, measured_force_fn=lambda: (0.0, 0.0, -15.0))
        ctrl._R_tool_to_base = np.eye(3)
        ctrl._t_start = time.monotonic()
        for _ in range(200):
            ctrl._compute_force_command()
        self.assertGreater(ctrl._last_diag["Fs_N"], 0.0)

    def test_over_target_retracts(self):
        ctrl = self._dtc(
            force_sign=1.0, measured_force_fn=lambda: (0.0, 0.0, -30.0))
        ctrl._R_tool_to_base = np.eye(3)
        ctrl._t_start = time.monotonic()
        seen = []
        ctrl.set_on_vz(seen.append)
        for _ in range(200):
            ctrl._compute_force_command()
        self.assertLess(seen[-1], 0.0)

    def test_field_normal_cannot_flip_the_press_hemisphere(self):
        # A field sign convention must never invert the press: the normal
        # refines direction, --force-sign still owns which way is "into".
        ctrl = self._dtc(
            force_sign=-1.0, measured_force_fn=lambda: (0.0, 0.0, 0.0))
        ctrl._R_tool_to_base = np.eye(3)
        ctrl._t_start = time.monotonic()
        seen = []
        ctrl.set_on_vz(seen.append)
        ctrl.set_contact_normal([0.0, 0.0, 1.0], np.eye(3))
        ctrl._compute_force_command()
        self.assertLess(seen[-1], 0.0)
        ctrl.set_contact_normal([0.0, 0.0, -1.0], np.eye(3))
        ctrl._compute_force_command()
        self.assertLess(seen[-1], 0.0)

    def test_tilted_normal_scales_the_tool_z_component(self):
        # v_z is the tool-Z component of motion along the base-frame press axis.
        ctrl = self._dtc(
            force_sign=1.0, measured_force_fn=lambda: (0.0, 0.0, 0.0))
        ctrl._R_tool_to_base = np.eye(3)
        ctrl._t_start = time.monotonic()
        seen = []
        ctrl.set_on_vz(seen.append)
        ctrl.set_contact_normal([0.0, 0.0, 1.0], np.eye(3))
        ctrl._compute_force_command()
        straight = seen[-1]
        ctrl2 = self._dtc(
            force_sign=1.0, measured_force_fn=lambda: (0.0, 0.0, 0.0))
        ctrl2._R_tool_to_base = np.eye(3)
        ctrl2._t_start = time.monotonic()
        seen2 = []
        ctrl2.set_on_vz(seen2.append)
        a = np.deg2rad(30.0)
        ctrl2.set_contact_normal([np.sin(a), 0.0, np.cos(a)], np.eye(3))
        ctrl2._compute_force_command()
        self.assertAlmostEqual(seen2[-1], straight * np.cos(a), places=6)

    def test_reset_press_loop_clears_the_ported_loop(self):
        ctrl = self._dtc(measured_force_fn=lambda: (0.0, 0.0, 0.0))
        ctrl._pid._integral = 5.0
        ctrl._pid._gain_scale = 0.2
        ctrl._reset_press_loop()
        self.assertEqual(ctrl._pid._integral, 0.0)
        self.assertEqual(ctrl._pid._gain_scale, 1.0)

    def test_start_resets_the_ported_loop(self):
        """start() must reset before the seek thread gets going.

        _integral cannot be the witness here: start() spawns the seek thread,
        which begins integrating immediately, so it is nonzero again by the
        time stop() returns -- that raced, it never caught a bug.
        _gain_scale is honest, because recovery is osc_recover_per_s (0.25/s)
        and a few ms of thread cannot lift a stale 0.2 anywhere near 1.0.
        See test_reset_press_loop_clears_the_ported_loop for the integral.
        """
        ctrl = self._dtc(
            measured_force_fn=lambda: (0.0, 0.0, 0.0), force_tare=False)
        ctrl._pid._gain_scale = 0.2
        ctrl.start()
        ctrl.stop()
        self.assertEqual(ctrl._pid._gain_scale, 1.0)

    def test_rejects_an_unknown_press_loop(self):
        with self.assertRaises(ValueError):
            _make_controller(press_loop="nope")



class ForceTareTest(unittest.TestCase):
    """The free-space force offset has to be subtracted before either seek law.

    UR's ActualTCPForce is not gravity compensated unless the payload is
    configured, and the speedj backend never issues zero_ftsensor() -- only the
    force_mode URScript does. So a perfectly live sensor reads the tool's own
    weight with nothing touching, and the dtc loop's safety trips fire on it the
    instant contact latches.
    """

    OFFSET = (0.0, 0.0, 4.0)  # tool-frame payload weight seen in free space
    FD = 5.0

    def _ctrl(self, **kwargs):
        params = dict(
            press_loop="dtc", target_force=self.FD, force_sign=1.0,
            max_v_hold=0.008, v_approach=0.005, log_period_s=0.0,
            dead_wrench_abort_s=0.0,
            measured_force_fn=lambda: self.OFFSET,
            force_tare_samples=3, force_tare_period=0.0)
        params.update(kwargs)
        ctrl = _make_controller(**params)
        ctrl._R_tool_to_base = np.eye(3)
        ctrl._t_start = time.monotonic()
        return ctrl

    def test_untared_offset_retracts_instead_of_approaching(self):
        # +4 N of tool weight reads Fs = -4 N, past wrong_sign
        # (min(-2, -0.5*Fd) = -2.5 at Fd=5), and the loop then commands
        # -unload_v_m_s forever: button 0 walked the arm away from the part.
        ctrl = self._ctrl(force_tare=False)
        seen = []
        ctrl.set_on_vz(seen.append)
        ctrl._compute_force_command()
        self.assertLess(seen[-1], 0.0)
        self.assertTrue(ctrl._last_diag["wrong_sign"])

    def test_tare_turns_the_same_offset_into_an_approach(self):
        ctrl = self._ctrl()
        ctrl.tare_force()
        seen = []
        ctrl.set_on_vz(seen.append)
        ctrl._compute_force_command()
        self.assertGreater(seen[-1], 0.0)
        self.assertAlmostEqual(ctrl._last_diag["Fs_raw_N"], 0.0, places=9)

    def test_tare_also_corrects_the_legacy_reading(self):
        # Legacy never tripped on the offset, but it regulated the offset
        # instead of the contact force -- same bug, quieter symptom.
        ctrl = self._ctrl(press_loop="legacy", measured_fz_fn=lambda: 4.0)
        ctrl.tare_force()
        seen = []
        ctrl.set_on_vz(seen.append)
        ctrl._compute_force_command()
        self.assertGreater(seen[-1], 0.0)

    def test_start_tares_once_per_session(self):
        ctrl = self._ctrl()
        ctrl.start()
        ctrl.stop()
        self.assertTrue(ctrl._force_bias_valid)
        np.testing.assert_allclose(ctrl._force_bias, self.OFFSET)
        first = ctrl._force_bias.copy()
        # Re-entering contact must not re-tare: by then the tool can already be
        # loaded, and taring there would zero real contact force and press Fd on
        # top of it.
        ctrl._measured_force_fn = lambda: (0.0, 0.0, 99.0)
        ctrl.start()
        ctrl.stop()
        np.testing.assert_allclose(ctrl._force_bias, first)
        ctrl.clear_force_tare()
        self.assertFalse(ctrl._force_bias_valid)
        np.testing.assert_allclose(ctrl._force_bias, np.zeros(3))

    def test_tare_does_not_hide_a_live_sensor_from_the_dead_feedback_guard(self):
        # The abort reads fz_raw on purpose: a live sensor sitting on a constant
        # offset must not look dead once the offset is tared out (C153A1).
        ctrl = self._ctrl(dead_wrench_abort_s=0.05)
        ctrl.tare_force()
        ctrl._t_start = time.monotonic() - 1.0
        ctrl._compute_force_command()
        self.assertTrue(ctrl._force_feedback_seen)
        self.assertFalse(ctrl._force_aborted)

    def test_no_force_reader_means_no_tare(self):
        ctrl = self._ctrl(measured_force_fn=None, measured_fz_fn=None)
        self.assertIsNone(ctrl.tare_force())
        self.assertFalse(ctrl._force_bias_valid)



class LegacyWrongSignTest(unittest.TestCase):
    """The bare PID must not answer a flipped reading with full approach.

    fe = Fd - Fs grows without bound as Fs goes negative, so on hardware a
    -11.58 N read produced kp*21.58 = +21.6 mm/s, clipped to the +8 mm/s
    ceiling and driven into a rigid part; the C153 protective stop followed on
    the next tick. The over-target guard never saw it -- Fs was nowhere near Fd.
    """

    FD = 10.0

    def _ctrl(self, fz, **kwargs):
        params = dict(
            press_loop="legacy", target_force=self.FD, force_sign=1.0,
            max_v_hold=0.008, log_period_s=0.0, dead_wrench_abort_s=0.0,
            v_slew_mps2=0.0, force_tare=False,
            measured_fz_fn=lambda: fz)
        params.update(kwargs)
        ctrl = _make_controller(**params)
        ctrl._R_tool_to_base = np.eye(3)
        ctrl._t_start = time.monotonic()
        return ctrl

    def _v_z(self, ctrl):
        seen = []
        ctrl.set_on_vz(seen.append)
        ctrl._compute_force_command()
        return seen[-1]

    def test_wrong_sign_reading_unloads(self):
        # Fs = -press_sign * fz, so fz=+11.58 is the Fs=-11.58 from the log.
        ctrl = self._ctrl(11.58)
        self.assertLess(self._v_z(ctrl), 0.0)

    def test_threshold_matches_the_dtc_loop(self):
        ctrl = self._ctrl(0.0)
        self.assertEqual(ctrl._wrong_sign_threshold(), min(-2.0, -0.5 * self.FD))

    def test_a_mild_negative_dip_still_approaches(self):
        # Bounce chatter must not trip it -- retract-on-any-Fs<0 is what walked
        # the TCP off into free space before.
        ctrl = self._ctrl(1.0)
        self.assertGreater(self._v_z(ctrl), 0.0)

    def test_the_integral_is_dumped_while_wrong_sign(self):
        ctrl = self._ctrl(11.58)
        ctrl._integral = 5.0
        ctrl._compute_force_command()
        self.assertEqual(ctrl._integral, 0.0)

    def test_press_slew_default_allows_a_prompt_reversal(self):
        """0.05 m/s^2 needed 0.32 s to swing +8 -> -8 mm/s, so the loop kept
        pressing ~1.3 mm after it had decided to back off. That is why v_z
        printed +0.0023 m/s at Fs=21.7 N with Fd=10 N."""
        default = inspect.signature(
            ContactTeleoperationController.__init__
        ).parameters["v_slew_mps2"].default
        self.assertLess(0.016 / default, 0.06)


class BaseFrameTareTest(unittest.TestCase):
    """The tare tracks the wrist, because the offset it removes is gravity.

    Two runs minutes apart on the same hardware tared at fz=+3.74 N and
    fz=+10.42 N. The tool's weight did not change; the wrist orientation did,
    and a tool-frame constant cannot represent a base-frame vector.
    """

    WEIGHT_BASE = np.array([0.0, 0.0, 4.0])

    def _ctrl(self, state):
        ctrl = _make_controller(
            press_loop="dtc", target_force=5.0, force_sign=1.0,
            max_v_hold=0.008, v_approach=0.005, log_period_s=0.0,
            dead_wrench_abort_s=0.0, force_tare_samples=3, force_tare_period=0.0,
            measured_force_fn=lambda: state["R"].T @ self.WEIGHT_BASE)
        ctrl._R_tool_to_base = state["R"]
        ctrl._t_start = time.monotonic()
        return ctrl

    def test_rotating_the_wrist_keeps_the_tared_reading_at_zero(self):
        state = {"R": np.eye(3)}
        ctrl = self._ctrl(state)
        ctrl.tare_force()
        np.testing.assert_allclose(ctrl._read_force_tool(), np.zeros(3), atol=1e-9)

        # Flip the tool 180 deg about X: the same base-frame weight now reads
        # fz = -4 N in the tool frame.
        state["R"] = np.diag([1.0, -1.0, -1.0])
        ctrl._R_tool_to_base = state["R"]
        np.testing.assert_allclose(
            ctrl._read_force_tool(), np.zeros(3), atol=1e-9)
        # The frozen tool-frame snapshot would have been off by 8 N here.
        self.assertAlmostEqual(float(ctrl._force_bias[2]), 4.0, places=9)
        self.assertAlmostEqual(float(ctrl._bias_tool_now()[2]), -4.0, places=9)

    def test_falls_back_to_the_tool_frame_when_no_pose_is_known(self):
        state = {"R": np.eye(3)}
        ctrl = self._ctrl(state)
        ctrl._R_tool_to_base = None
        ctrl.tare_force()
        self.assertIsNone(ctrl._force_bias_base)
        np.testing.assert_allclose(ctrl._bias_tool_now(), ctrl._force_bias)

    def test_clear_drops_the_base_frame_copy_too(self):
        state = {"R": np.eye(3)}
        ctrl = self._ctrl(state)
        ctrl.tare_force()
        self.assertIsNotNone(ctrl._force_bias_base)
        ctrl.clear_force_tare()
        self.assertIsNone(ctrl._force_bias_base)


class ContactLogTest(unittest.TestCase):
    def test_start_writes_a_streaming_csv_that_survives_an_abort(self):
        import csv as _csv
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "nested" / "contact.csv"
            ctrl = _make_controller(
                press_loop="dtc", target_force=5.0, log_period_s=0.0,
                dead_wrench_abort_s=0.0, force_tare=False,
                measured_force_fn=lambda: (0.0, 0.0, 0.0))
            ctrl._R_tool_to_base = np.eye(3)
            # Contact mode always has an on_vz overlay; without one _run takes
            # the set_desired_wrench branch, which is not what production does.
            ctrl.set_on_vz(lambda vz: None)
            ctrl.set_contact_log(path)
            ctrl.start()
            time.sleep(0.1)
            ctrl.stop()
            rows = list(_csv.DictReader(path.open()))
        self.assertGreater(len(rows), 0)
        self.assertIn("Fs", rows[0])
        self.assertIn("spike", rows[0])
        self.assertEqual(rows[0]["press_loop"], "dtc")

    def test_none_disables_it(self):
        ctrl = _make_controller(force_tare=False, log_period_s=0.0)
        ctrl.set_contact_log(None)
        ctrl._open_contact_log()
        self.assertIsNone(ctrl._log_writer)



class TickPeriodTest(unittest.TestCase):
    """The seek laws must integrate on real time, not on update_period.

    ``_run`` sleeps update_period *after* the work, and the work includes a
    blocking RTDE read, so the true period is always longer than nominal.
    Nominal dt rescales the integral and the LPF pole, and inflates the spike
    guard's rate by the same ratio -- which is what pinned SPIKE on almost
    every tick in contact and left v_along at 0 short of Fd.
    """

    def _ctrl(self, **kw):
        params = dict(
            press_loop="dtc", target_force=10.0, update_period=0.005,
            log_period_s=0.0, dead_wrench_abort_s=0.0, force_tare=False,
            measured_force_fn=lambda: (0.0, 0.0, 0.0))
        params.update(kw)
        ctrl = _make_controller(**params)
        ctrl._R_tool_to_base = np.eye(3)
        return ctrl

    def test_first_tick_falls_back_to_nominal(self):
        ctrl = self._ctrl()
        self.assertEqual(ctrl._tick_dt(100.0), 0.005)

    def test_later_ticks_use_the_measured_gap(self):
        ctrl = self._ctrl()
        ctrl._tick_dt(100.0)
        self.assertAlmostEqual(ctrl._tick_dt(100.008), 0.008, places=9)

    def test_a_scheduling_hiccup_is_clamped(self):
        ctrl = self._ctrl()
        ctrl._tick_dt(100.0)
        self.assertEqual(ctrl._tick_dt(101.0), 0.05)      # 10x nominal
        ctrl._tick_dt(200.0)
        self.assertEqual(ctrl._tick_dt(200.0), 0.001)     # 0.2x nominal

    def test_start_clears_the_tick_clock(self):
        ctrl = self._ctrl()
        ctrl._tick_dt(100.0)
        ctrl._reset_press_loop()
        self.assertIsNone(ctrl._t_tick)

    def test_the_slew_uses_the_measured_period(self):
        # 0.4 m/s^2 over a real 10 ms tick is 4 mm/s of allowed change, twice
        # what the nominal 5 ms would have permitted.
        ctrl = self._ctrl(v_slew_mps2=0.4)
        ctrl._prev_v_z = 0.0
        self.assertAlmostEqual(ctrl._slew_vz(1.0, 0.010), 0.004, places=9)
        ctrl._prev_v_z = 0.0
        self.assertAlmostEqual(ctrl._slew_vz(1.0, 0.005), 0.002, places=9)



class SafetyModeHaltTest(unittest.TestCase):
    """A protective stop must stop the press, not queue one up.

    The UR discards our speedJ stream while stopped, but this loop kept
    computing press velocity against a robot that was not moving. The force
    error therefore stayed large, and whatever it was holding would have been
    applied the instant someone cleared the stop on the pendant.
    """

    def _ctrl(self, state):
        ctrl = _make_controller(
            press_loop="dtc", target_force=10.0, force_sign=1.0,
            max_v_hold=0.008, v_approach=0.005, log_period_s=0.0,
            dead_wrench_abort_s=0.0, force_tare=False, v_slew_mps2=0.0,
            measured_force_fn=lambda: (0.0, 0.0, 0.0),
            safety_mode_fn=lambda: state["mode"])
        ctrl._R_tool_to_base = np.eye(3)
        ctrl._t_start = time.monotonic()
        return ctrl

    def _v_z(self, ctrl, n=1):
        seen = []
        ctrl.set_on_vz(seen.append)
        for _ in range(n):
            ctrl._compute_force_command()
        return seen[-1]

    def test_a_protective_stop_zeroes_the_overlay(self):
        state = {"mode": 1}
        ctrl = self._ctrl(state)
        self.assertGreater(self._v_z(ctrl, n=20), 0.0)
        state["mode"] = 3                       # PROTECTIVE_STOP
        self.assertEqual(self._v_z(ctrl), 0.0)

    def test_clearing_the_stop_resumes_from_zero_not_mid_ramp(self):
        state = {"mode": 1}
        ctrl = self._ctrl(state)
        self._v_z(ctrl, n=20)
        state["mode"] = 3
        self._v_z(ctrl)
        self.assertEqual(ctrl._prev_v_z, 0.0)
        state["mode"] = 1
        self.assertGreater(self._v_z(ctrl), 0.0)

    def test_reduced_mode_still_allows_the_press(self):
        state = {"mode": 2}                     # REDUCED is a live robot
        self.assertGreater(self._v_z(self._ctrl(state), n=20), 0.0)

    def test_a_missing_or_failing_reader_is_not_treated_as_a_stop(self):
        # Refusing to press because a status read failed would be a worse
        # failure than the one this guards against.
        ctrl = self._ctrl({"mode": 1})
        ctrl.set_safety_mode_fn(None)
        self.assertGreater(self._v_z(ctrl, n=20), 0.0)
        def boom():
            raise RuntimeError("rtde hiccup")
        ctrl.set_safety_mode_fn(boom)
        self.assertGreater(self._v_z(ctrl), 0.0)


if __name__ == "__main__":
    unittest.main()


class WrenchPressAdapterTest(unittest.TestCase):
    """set_on_press_wrench: the press as a force, for --dtc-law wrench.

    No gafropy Wrench type is involved on this path -- _run already skips
    set_desired_wrench whenever an overlay callback is set, so the plain numpy
    3-vector goes straight to compute_control_wrench.
    """

    def _ctrl(self, **kw):
        kw.setdefault("target_force", 10.0)
        kw.setdefault("press_slew_n_s", 0.0)  # off, so the law is visible
        return _make_controller(**kw)

    def test_feedforward_alone_commands_fd(self):
        c = self._ctrl(press_kp_n_per_n=0.0)
        self.assertAlmostEqual(c._press_force_n(fs=0.0, now=0.0, dt=0.005), 10.0)
        self.assertAlmostEqual(c._press_force_n(fs=10.0, now=0.0, dt=0.005), 10.0)

    def test_the_trim_pushes_harder_when_under_target(self):
        c = self._ctrl(press_kp_n_per_n=0.5)
        # Fd + 0.5*(Fd - Fs)
        self.assertAlmostEqual(c._press_force_n(fs=4.0, now=0.0, dt=0.005), 13.0)
        self.assertAlmostEqual(c._press_force_n(fs=16.0, now=0.0, dt=0.005), 7.0)

    def test_the_command_never_goes_negative(self):
        """A wrench-law press pulls the tool off the part if it goes negative;
        lift-off is the pose spring's job, not the force loop's."""
        c = self._ctrl(press_kp_n_per_n=2.0)
        self.assertAlmostEqual(c._press_force_n(fs=40.0, now=0.0, dt=0.005), 0.0)

    def test_the_ceiling_defaults_to_three_times_fd(self):
        c = self._ctrl(press_kp_n_per_n=10.0)
        self.assertAlmostEqual(c._press_max_n, 30.0)
        self.assertAlmostEqual(c._press_force_n(fs=0.0, now=0.0, dt=0.005), 30.0)
        c = self._ctrl(press_kp_n_per_n=10.0, press_max_n=12.0)
        self.assertAlmostEqual(c._press_force_n(fs=0.0, now=0.0, dt=0.005), 12.0)

    def test_wrong_sign_commands_no_press(self):
        """Fs at or below min(-2, -0.5*Fd) is a bounce or a flipped reading,
        not more force owed. Both seek laws answer it the same way."""
        c = self._ctrl(press_kp_n_per_n=0.5)
        self.assertAlmostEqual(c._press_force_n(fs=-6.0, now=0.0, dt=0.005), 0.0)

    def test_the_slew_bounds_the_step_in_newtons(self):
        c = self._ctrl(press_kp_n_per_n=0.0, press_slew_n_s=200.0)
        self.assertAlmostEqual(c._press_force_n(fs=0.0, now=0.0, dt=0.005), 10.0)
        c._prev_press_n = 0.0
        self.assertAlmostEqual(
            c._press_force_n(fs=0.0, now=0.0, dt=0.005), 1.0)  # 200 N/s * 5 ms

    def test_the_body_vector_is_the_normal_rotated_by_the_live_tool(self):
        c = self._ctrl()
        R = np.array([[0.0, -1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]])          # 90 deg about base Z
        n_base = np.array([0.0, 0.0, 1.0])       # along tool Z, as in operation
        c.set_contact_normal(n_base, R)
        w = c._press_wrench_body(12.0)
        np.testing.assert_allclose(w, R.T @ (12.0 * n_base), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(w)), 12.0)

    def test_the_force_sign_hemisphere_still_wins(self):
        """--force-sign is the convention the operator validated; a field normal
        may only pick the *direction*, never flip the hemisphere."""
        c = self._ctrl(force_sign=-1.0)
        R = np.eye(3)
        c.set_contact_normal(np.array([0.0, 0.0, 1.0]), R)
        np.testing.assert_allclose(
            c._press_wrench_body(10.0), [0.0, 0.0, -10.0], atol=1e-12)

    def test_a_near_perpendicular_normal_refuses_to_press(self):
        """At 90 deg the hemisphere is decided by floating-point noise and the
        axis flips tick to tick. The velocity path falls back to tool Z at
        mm/s; a force cannot afford that."""
        c = self._ctrl()
        R = np.eye(3)
        c.set_contact_normal(np.array([1.0, 0.0, 0.0]), R)   # perpendicular
        self.assertIsNone(c._press_wrench_body(12.0))
        sent = []
        c.set_on_press_wrench(sent.append)
        c._emit_press_wrench(12.0)
        np.testing.assert_allclose(sent[-1], np.zeros(3))

    def test_no_pose_yet_means_no_press_rather_than_a_guessed_axis(self):
        """The velocity overlay fell back to tool Z at millimetres per second.
        A mis-aimed *force* is delivered at full magnitude immediately."""
        c = self._ctrl()
        self.assertIsNone(c._press_wrench_body(12.0))
        sent = []
        c.set_on_press_wrench(sent.append)
        c._emit_press_wrench(12.0)
        np.testing.assert_allclose(sent[-1], np.zeros(3))

    def test_emit_never_raises_into_the_force_loop(self):
        c = self._ctrl()
        c.set_on_press_wrench(lambda w: (_ for _ in ()).throw(RuntimeError("x")))
        c._emit_press_wrench(0.0)  # must not propagate

    def test_the_gafropy_plant_is_not_stepped_on_this_path(self):
        """_run must skip set_desired_wrench for the press-wrench callback the
        same way it does for set_on_vz -- two loops authoritative on one axis
        is the fight that produces drift."""
        src = inspect.getsource(ContactTeleoperationController._run)
        self.assertIn(
            "self._on_vz is None and self._on_press_wrench is None", src)


class PressTwoRegimeTest(unittest.TestCase):
    """Approach and hold need opposite gains, latched on Fs with hysteresis.

    contact_20260903_140311.csv ran one gain for both: the 20.6 N median command
    broke the tool free and then drove Fs to a p95 of 19.6 and a max of 33.2
    against Fd = 5.
    """

    def _ctrl(self, **kw):
        kw.setdefault("target_force", 10.0)
        kw.setdefault("press_slew_n_s", 0.0)
        kw.setdefault("press_kp_n_per_n", 1.8)   # breakaway-sized
        kw.setdefault("press_kp_hold", 0.5)
        return _make_controller(**kw)

    def test_approach_uses_the_breakaway_gain(self):
        c = self._ctrl()
        # Fs ~ 0: 10 + 1.8*(10-0) = 28
        self.assertAlmostEqual(c._press_force_n(fs=0.0, now=0.0, dt=0.005), 28.0)

    def test_touching_switches_to_the_hold_gain(self):
        c = self._ctrl()
        c._press_force_n(fs=0.0, now=0.0, dt=0.005)      # approach
        # Fs past 0.5*Fd latches hold: 10 + 0.5*(10-6) = 12
        self.assertAlmostEqual(c._press_force_n(fs=6.0, now=0.0, dt=0.005), 12.0)

    def test_the_latch_has_hysteresis(self):
        c = self._ctrl()
        c._press_force_n(fs=6.0, now=0.0, dt=0.005)      # engage
        self.assertTrue(c._press_engaged)
        c._press_force_n(fs=2.0, now=0.0, dt=0.005)      # above release, stays
        self.assertTrue(c._press_engaged)
        c._press_force_n(fs=1.0, now=0.0, dt=0.005)      # below 0.15*Fd
        self.assertFalse(c._press_engaged)

    def test_backing_off_is_instant_but_rising_is_limited(self):
        c = self._ctrl(press_slew_n_s=200.0)
        c._prev_press_n = 25.0
        # A large drop must land in one tick -- 90 ms of a stale breakaway
        # command is what drove Fs to 33 N on hardware.
        self.assertAlmostEqual(
            c._press_force_n(fs=30.0, now=0.0, dt=0.005), 0.0)
        c._prev_press_n = 0.0
        self.assertAlmostEqual(
            c._press_force_n(fs=0.0, now=0.0, dt=0.005), 1.0)   # 200 N/s * 5 ms

    def test_wrong_sign_still_drops_the_latch(self):
        c = self._ctrl()
        c._press_force_n(fs=6.0, now=0.0, dt=0.005)
        self.assertTrue(c._press_engaged)
        self.assertAlmostEqual(c._press_force_n(fs=-9.0, now=0.0, dt=0.005), 0.0)
        self.assertFalse(c._press_engaged)
