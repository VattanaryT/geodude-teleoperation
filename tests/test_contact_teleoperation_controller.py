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

    def test_fs_sign_flip_stays_contact_and_pids(self):
        # Sliding can flip Fz. Stay in contact and keep PID (≤ max_v_hold), do not
        # retract into free space or re-arm DualLoop approach.
        samples = iter([-15.0, 17.0])
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

    def test_start_resets_the_ported_loop(self):
        ctrl = self._dtc(measured_force_fn=lambda: (0.0, 0.0, 0.0))
        ctrl._pid._integral = 5.0
        ctrl._pid._gain_scale = 0.2
        ctrl.start()
        ctrl.stop()
        self.assertEqual(ctrl._pid._integral, 0.0)
        self.assertEqual(ctrl._pid._gain_scale, 1.0)

    def test_rejects_an_unknown_press_loop(self):
        with self.assertRaises(ValueError):
            _make_controller(press_loop="nope")


if __name__ == "__main__":
    unittest.main()
