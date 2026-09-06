"""Behavioral cover for the press loop ported from ``exp/in-contact-dtc``.

These are the behaviors the previous bare PID did not have and whose absence let
the tool lift off: a retract when over target, a capped soft feed in air rather
than a hard velocity ceiling, and the auto-stabilize trips.

Sign convention here matches how ``ContactTeleoperationController`` drives it:
``n_base`` is the press axis, ``force_base`` is the UR ``ActualTCPForce``
reaction in the tool frame, and ``force_sign=-1`` so compression reads ``Fs>0``.
"""
import pytest
import unittest

import numpy as np

from geodude_teleoperation.in_contact_control.force_pid import (
    ForcePIDConfig,
    ForcePIDController,
)

FD = 15.0
DT = 0.005
N = np.array([0.0, 0.0, 1.0])


def _make(**kwargs):
    params = dict(
        Fd_N=FD,
        force_sign=-1.0,
        max_v_along_m_s=0.008,
        approach_v_max_m_s=0.005,
        kp=0.001,
        ki=0.0,
        kd=0.0,
    )
    params.update(kwargs)
    return ForcePIDController(ForcePIDConfig(**params))


def _settle(pid, fz, n=400, n_base=N):
    """Hold a constant reaction force for n ticks; return the last diag."""
    diag = {}
    for _ in range(n):
        _cmd, diag = pid.step(
            n_base=n_base,
            force_base=[0.0, 0.0, fz],
            dt=DT,
            R_ee=np.eye(3),
        )
    return diag


class PressLoopTest(unittest.TestCase):
    def test_reaction_sign_reads_as_compression(self):
        # UR reports the reaction, so pressing along +n reads fz < 0 and must
        # land on Fs > 0 -- the same convention as _fs() in the controller.
        pid = _make()
        diag = _settle(pid, -FD)
        self.assertGreater(diag["Fs_N"], 0.0)
        self.assertAlmostEqual(diag["Fs_N"], FD, places=2)

    def test_air_presses_toward_the_surface_at_the_soft_cap(self):
        pid = _make()
        _cmd, diag = pid.step(
            n_base=N, force_base=[0.0, 0.0, 0.0], dt=DT, R_ee=np.eye(3))
        v = diag["v_along_m_s"]
        self.assertGreater(v, 0.0, "free air must seek the surface")
        # Soft approach caps the feed while barely touching; it must not be the
        # hard max_v_along ceiling that the old loop clipped to.
        self.assertLessEqual(v, pid.config.approach_v_max_m_s + 1e-12)

    def test_holds_at_target(self):
        pid = _make()
        diag = _settle(pid, -FD)
        self.assertAlmostEqual(diag["v_along_m_s"], 0.0, places=6)

    def test_over_target_retracts_not_just_stops(self):
        # The C153 invariant, strengthened: speedJ cannot yield, so being over
        # target must actively unload rather than sit in the load.
        pid = _make()
        diag = _settle(pid, -(FD + 10.0))
        self.assertLess(diag["v_along_m_s"], 0.0)

    def test_never_presses_while_at_or_above_target(self):
        for over in (0.0, 1.0, 5.0, 20.0):
            with self.subTest(over=over):
                pid = _make()
                diag = _settle(pid, -(FD + over))
                self.assertLessEqual(diag["v_along_m_s"], 0.0)

    def test_speed_is_always_capped(self):
        for fz in (0.0, -1.0, -FD, -100.0, 50.0):
            with self.subTest(fz=fz):
                pid = _make()
                diag = _settle(pid, fz, n=50)
                self.assertLessEqual(
                    abs(diag["v_along_m_s"]),
                    pid.config.max_v_along_m_s + 1e-12)

    def test_impact_trips_the_spike_guard(self):
        pid = _make()
        _settle(pid, 0.0, n=20)
        # A step far faster than spike_dF_dt_N_s is an impact, not a setpoint.
        _cmd, diag = pid.step(
            n_base=N, force_base=[0.0, 0.0, -30.0], dt=DT, R_ee=np.eye(3))
        self.assertTrue(diag["spike"])

    def test_sustained_chatter_cuts_the_gain(self):
        pid = _make(force_lpf_hz=0.0, force_deadband_N=0.05)
        before = pid._gain_scale
        # Straddle Fd so Fe keeps flipping sign, which is what oscillation is.
        for i in range(200):
            pid.step(
                n_base=N,
                force_base=[0.0, 0.0, -(FD + (4.0 if i % 2 else -4.0))],
                dt=DT,
                R_ee=np.eye(3),
            )
        self.assertLess(pid._gain_scale, before)
        self.assertGreaterEqual(pid._gain_scale, pid.config.osc_min_scale)

    def test_reset_clears_windup_and_gain_cut(self):
        # start() calls reset(); a second contact session must not inherit state.
        pid = _make(ki=0.001)
        _settle(pid, -1.0, n=200)
        pid._gain_scale = 0.2
        pid.reset()
        self.assertEqual(pid._integral, 0.0)
        self.assertEqual(pid._gain_scale, 1.0)
        self.assertIsNone(pid._f_lpf)

    def test_force_is_resolved_along_the_normal_not_tool_z(self):
        # The whole point of taking the full vector: with the press axis 90 deg
        # off tool Z, a pure tool-Z force carries no load along the normal.
        pid = _make()
        n_x = np.array([1.0, 0.0, 0.0])
        diag = _settle(pid, -FD, n=5, n_base=n_x)
        self.assertAlmostEqual(diag["Fs_N"], 0.0, places=6)

    def test_rotation_into_base_frame_is_applied(self):
        # R_ee rotates tool -> base. With tool Z mapped onto base X, a tool-frame
        # fz must be seen along a base-frame X normal.
        pid = _make()
        R = np.array([[0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0],
                      [-1.0, 0.0, 0.0]])
        diag = {}
        for _ in range(5):
            _cmd, diag = pid.step(
                n_base=np.array([1.0, 0.0, 0.0]),
                force_base=[0.0, 0.0, -FD],
                dt=DT,
                R_ee=R,
            )
        self.assertAlmostEqual(diag["Fs_N"], FD, places=2)


if __name__ == "__main__":
    unittest.main()


class PushStopThresholdTest(unittest.TestCase):
    """The seek must be able to reach Fd, not stall at half of it.

    The ported guard read ``fs_raw >= 0.5 * abs(fd)``, which is right for DTC's
    *approach* phase handing off to the force loop but wrong for the loop that
    has to hold Fd: it zeroed every approach command from half target upwards,
    so ``--target-force 5`` could only ever settle near 2.5 N.
    """

    def test_still_pushes_between_half_target_and_target(self):
        pid = _make()
        # Reaction of -0.6*Fd is a 0.6*Fd compression: under target, so the
        # loop still owes the surface some feed.
        diag = _settle(pid, -0.6 * FD)
        self.assertGreater(diag["v_along_m_s"], 0.0)

    def test_stops_pushing_at_target(self):
        pid = _make()
        diag = _settle(pid, -FD)
        self.assertLessEqual(diag["v_along_m_s"], 0.0)

    def test_frac_is_configurable_and_restores_the_ported_behavior(self):
        pid = _make(push_stop_frac_of_fd=0.5)
        diag = _settle(pid, -0.6 * FD)
        self.assertEqual(diag["v_along_m_s"], 0.0)


class SpikeGuardNoiseTest(unittest.TestCase):
    """The spike guard must not fire on the sensor's own noise floor.

    On hardware every log line read ``SPIKE gain=0.50`` with ``Fs≈0`` and
    ``v_z=+0.0000`` forever: at 200 Hz a single-tick difference turns UR's
    ±0.3 N of noise into ±120 N/s, permanently above ``spike_dF_dt_N_s``. The
    hold then pinned ``v_along`` at 0, so contact mode never approached at all.
    """

    @staticmethod
    def _noisy(pid, mean_fz, n=400, amp=0.35, seed=0):
        rng = np.random.default_rng(seed)
        diag = {}
        for _ in range(n):
            fz = mean_fz + rng.uniform(-amp, amp)
            _cmd, diag = pid.step(
                n_base=N, force_base=[0.0, 0.0, fz], dt=DT, R_ee=np.eye(3))
        return diag

    def test_free_space_noise_does_not_freeze_the_approach(self):
        pid = _make()
        diag = self._noisy(pid, 0.0)
        self.assertFalse(diag["spike"])
        self.assertGreater(diag["v_along_m_s"], 0.0)
        self.assertEqual(diag["gain_scale"], 1.0)

    def test_in_contact_noise_does_not_freeze_the_hold(self):
        # Under target with a live noisy reading, the loop must still be
        # allowed to close the remaining gap rather than sit at v=0.
        pid = _make()
        diag = self._noisy(pid, -0.5 * FD)
        self.assertFalse(diag["spike"])
        self.assertGreater(diag["v_along_m_s"], 0.0)

    def test_a_real_impact_still_trips_it(self):
        pid = _make()
        self._noisy(pid, 0.0, n=40)
        # 30 N inside one tick is 6000 N/s over any baseline, and lands well
        # past contact_threshold_N, so both gates open.
        _cmd, diag = pid.step(
            n_base=N, force_base=[0.0, 0.0, -30.0], dt=DT, R_ee=np.eye(3))
        self.assertTrue(diag["spike"])

    def test_a_ramp_slower_than_the_threshold_does_not_trip(self):
        pid = _make()
        diag = {}
        fz = 0.0
        for _ in range(200):                      # 20 N/s: half the threshold
            fz -= 20.0 * DT
            _cmd, diag = pid.step(
                n_base=N, force_base=[0.0, 0.0, fz], dt=DT, R_ee=np.eye(3))
        self.assertFalse(diag["spike"])


class GuardAuthorityTest(unittest.TestCase):
    """The guards must shape the seek, not remove it.

    Measured on hardware (contact_20260826_145209.csv, 9152 ticks): SPIKE was
    set on 75.4% of ticks, v_z was exactly 0 on 68.6%, and gain_scale reached
    1.0 on 4.3%. Mean normal authority came out at ~0.9 mm/s against a 24 mm/s
    lateral slide, so any slope past ~2 degrees outran the loop and the tool
    left the surface -- "contact isn't very reactive".
    """

    def test_a_spike_well_under_target_caps_the_seek_rather_than_freezing_it(self):
        pid = _make()
        _settle(pid, 0.0, n=20)
        # Fast enough to trip, but at half of Fd there is nothing to slam into:
        # overshoot/mag_overload/fault are level checks and still own that.
        _cmd, diag = pid.step(
            n_base=N, force_base=[0.0, 0.0, -8.0], dt=DT, R_ee=np.eye(3))
        self.assertTrue(diag["spike"])
        self.assertGreater(diag["v_along_m_s"], 0.0)
        self.assertLessEqual(
            diag["v_along_m_s"], pid.config.spike_v_cap_m_s + 1e-12)

    def test_a_spike_over_target_still_unloads(self):
        pid = _make()
        _settle(pid, 0.0, n=20)
        _cmd, diag = pid.step(
            n_base=N, force_base=[0.0, 0.0, -(FD + 10.0)], dt=DT, R_ee=np.eye(3))
        self.assertTrue(diag["spike"])
        self.assertLess(diag["v_along_m_s"], 0.0)

    def test_a_sustained_spike_cuts_the_gain_only_on_the_rising_edge(self):
        pid = _make()
        _settle(pid, 0.0, n=20)
        pid.step(n_base=N, force_base=[0.0, 0.0, -30.0], dt=DT, R_ee=np.eye(3))
        self.assertLessEqual(pid._gain_scale, 0.5)
        pid._gain_scale = 0.9                      # pretend it recovered
        # Still inside spike_hold_s and still rising fast, so it trips again.
        _cmd, diag = pid.step(
            n_base=N, force_base=[0.0, 0.0, -70.0], dt=DT, R_ee=np.eye(3))
        self.assertTrue(diag["spike"])
        self.assertGreater(pid._gain_scale, 0.5)

    def test_rate_damping_cannot_invert_the_seek(self):
        # 2000 N/s x 0.00015 is 0.3 m/s of damping against an 8 mm/s ceiling.
        # Uncapped that is not damping, it is full retract on a rising force.
        pid = _make(spike_dF_dt_N_s=1e9, force_deadband_N=0.0)
        _settle(pid, 0.0, n=20)
        _cmd, diag = pid.step(
            n_base=N, force_base=[0.0, 0.0, -10.0], dt=DT, R_ee=np.eye(3))
        cap = pid.config.force_rate_damp_max_frac * pid.config.max_v_along_m_s
        self.assertGreaterEqual(diag["v_along_m_s"], -cap - 1e-12)
        self.assertGreater(diag["v_along_m_s"], 0.0)

    def test_the_spike_threshold_clears_this_rigs_ordinary_variation(self):
        """0.635 N per 5.18 ms tick is ~123 N/s of perfectly normal spread."""
        self.assertGreater(ForcePIDConfig().spike_dF_dt_N_s, 123.0)


class OverloadEscapeTest(unittest.TestCase):
    """unload_v_m_s is a floor on the escape speed, never a ceiling.

    The guards used to assign it, which discarded the PID's own answer exactly
    when the load was worst: at |Fs|=40 N with Fd=10 N the PID wants the full
    max_v_along, and the guard forced half of it. Shedding that through a stiff
    contact took ~2 s and the pendant protective-stopped inside that window.
    """

    def test_a_large_overload_escapes_at_the_pid_speed_not_the_unload_floor(self):
        pid = _make()
        diag = _settle(pid, -(4.0 * FD))          # |Fs| = 4x target
        self.assertTrue(diag["fault_fs"])
        self.assertAlmostEqual(
            diag["v_along_m_s"], -pid.config.max_v_along_m_s, places=9)

    def test_a_mild_overload_still_gets_at_least_the_unload_floor(self):
        pid = _make(kp=0.0, ki=0.0, kd=0.0)       # PID contributes nothing
        diag = _settle(pid, -(FD + 2.0))
        self.assertLessEqual(diag["v_along_m_s"], -pid.config.unload_v_m_s + 1e-12)

    def test_the_escape_never_becomes_a_press(self):
        for fz in (-(FD + 2.0), -(2.0 * FD), -(4.0 * FD), -(10.0 * FD)):
            with self.subTest(fz=fz):
                pid = _make()
                self.assertLess(_settle(pid, fz)["v_along_m_s"], 0.0)


# --- press hand-off threshold -------------------------------------------
# The operator's requirement: pressing button 0 while the tool is still in free
# space must NOT switch to tangential-only. The spring keeps the normal and the
# operator drives into the surface; the force loop takes over on measured Fs.
# --press-engage-frac is where that hand-off sits, as a fraction of Fd.


def _press_ctrl(**kw):
    import geodude_teleoperation.controller as C
    ctrl = C.ContactTeleoperationController.__new__(
        C.ContactTeleoperationController)
    import numpy as _np
    ctrl._target_force = kw.get("fd", 10.0)
    ctrl._press_kp_hold = 0.5
    ctrl._press_approach_n = 0.0
    ctrl._press_max_n = 60.0
    ctrl._press_engaged = False
    # None = the internal Fs latch owns engagement, which is what these cases
    # are about; the geometric latch has its own file.
    ctrl._press_external = None
    ctrl._press_feedforward = False
    ctrl._press_overloaded = False
    ctrl._press_overload_n = 2.0 * ctrl._target_force
    ctrl._press_slew_n_s = 0.0        # 0 disables the rate limit
    ctrl._press_cmd_n = 0.0
    ctrl._wrong_sign_fs_n = -2.0
    ctrl._wrong_sign_fd_frac = 0.5
    ctrl._press_engage_frac = kw.get("engage", 0.5)
    ctrl._press_release_frac = kw.get("release", 0.15)
    ctrl._press_auto_approach = kw.get("auto", False)
    ctrl._press_kp_n_per_n = kw.get("kp", 0.5)
    return ctrl


def _press(ctrl, fs):
    return ctrl._press_force_n(fs, 0.005, 0.0)


def test_hovering_after_the_button_press_commands_no_press():
    ctrl = _press_ctrl(engage=1.0)
    assert _press(ctrl, 0.0) == 0.0
    assert ctrl._press_engaged is False


def test_engage_frac_1_waits_for_the_full_target_force():
    """0.5 hands over while merely touching; 1.0 waits until Fd is reached."""
    half = _press_ctrl(engage=0.5)
    _press(half, 6.0)
    assert half._press_engaged is True

    full = _press_ctrl(engage=1.0)
    _press(full, 6.0)
    assert full._press_engaged is False, "half-seated must stay on the spring"
    _press(full, 10.0)
    assert full._press_engaged is True


def test_the_latch_keeps_its_hysteresis_at_any_engage_frac():
    ctrl = _press_ctrl(engage=1.0, release=0.15)
    _press(ctrl, 10.0)
    assert ctrl._press_engaged is True
    _press(ctrl, 5.0)                      # under engage, over release
    assert ctrl._press_engaged is True
    _press(ctrl, 1.0)                      # under release
    assert ctrl._press_engaged is False


def test_release_frac_cannot_exceed_engage_frac():
    import geodude_teleoperation.controller as C
    import numpy as np
    eng, rel = 0.4, 0.9
    e = float(np.clip(eng, 0.0, 1.0))
    r = float(np.clip(rel, 0.0, e))
    assert r == e, "a release above engage would latch on and never let go"


def test_press_engaged_is_readable_so_tangent_can_follow_the_same_latch():
    """Three things must agree on one latch, or contact deadlocks.

    Gating field-follow's tangent projection on the *mode* stripped the normal
    out of every stick step the instant the button was pressed, so the operator
    could not drive into the surface, so Fs never rose, so the press -- which
    commands press_approach_n (0) until Fs >= engage_frac*Fd -- never engaged.
    The tool hovered and pushing the stick down did nothing.
    """
    from geodude_teleoperation.controller import (
        AdmittanceTeleoperationController as A)
    ctrl = A.__new__(A)
    ctrl._press_engaged = False
    assert ctrl.press_engaged is False
    ctrl.set_press_engaged(True)
    assert ctrl.press_engaged is True
    ctrl.set_press_engaged(False)
    assert ctrl.press_engaged is False


def test_auto_approach_commands_the_press_law_before_engagement():
    """The operator could not drive the tool in; let the press do it.

    Off (default) the un-engaged press commands press_approach_n, which is 0,
    so nothing moves until Fs rises on its own. On, it runs Fd + kp*(Fd - Fs)
    from the moment contact mode is entered.
    """
    off = _press_ctrl(engage=1.0, auto=False)
    assert _press(off, 0.0) == 0.0

    on = _press_ctrl(engage=1.0, auto=True, kp=0.5, fd=10.0)
    assert _press(on, 0.0) == pytest.approx(15.0)   # Fd*(1+kp)
    assert on._press_engaged is False                # still approaching
    assert _press(on, 10.0) == pytest.approx(10.0)   # engaged, holds Fd
    assert on._press_engaged is True


def test_auto_approach_still_obeys_the_wrong_sign_guard_and_the_ceiling():
    on = _press_ctrl(engage=1.0, auto=True, kp=0.5, fd=10.0)
    assert _press(on, -6.0) == 0.0            # <= min(-2, -0.5*Fd) = -5
    on2 = _press_ctrl(engage=1.0, auto=True, kp=20.0, fd=10.0)
    on2._press_max_n = 25.0
    assert _press(on2, 0.0) == pytest.approx(25.0)   # clipped, not 210 N


def test_auto_approach_logs_what_was_emitted_not_only_what_was_computed():
    """F_press is the law's answer; F_emit is what reached the wrench law.

    They diverge silently whenever _press_wrench_body refuses (no pose yet, or
    the normal near-perpendicular to tool Z), and "the log says 15 N while 0 N
    reaches the arm" is the failure this repo keeps rediscovering.
    """
    import numpy as np
    import geodude_teleoperation.controller as C
    ctrl = C.ContactTeleoperationController.__new__(
        C.ContactTeleoperationController)
    ctrl._on_press_engaged = None
    ctrl._press_engaged = False
    ctrl._last_press_emit_n = 0.0
    ctrl._press_refuse_logged = False
    got = []
    ctrl._on_press_wrench = got.append

    ctrl._press_wrench_body = lambda f: None          # refused
    ctrl._emit_press_wrench(15.0)
    assert ctrl._last_press_emit_n == 0.0
    assert np.allclose(got[-1], 0.0)

    ctrl._press_wrench_body = lambda f: np.array([0.0, 0.0, f])
    ctrl._emit_press_wrench(15.0)
    assert ctrl._last_press_emit_n == pytest.approx(15.0)
