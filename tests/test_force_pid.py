"""Behavioral cover for the press loop ported from ``exp/in-contact-dtc``.

These are the behaviors the previous bare PID did not have and whose absence let
the tool lift off: a retract when over target, a capped soft feed in air rather
than a hard velocity ceiling, and the auto-stabilize trips.

Sign convention here matches how ``ContactTeleoperationController`` drives it:
``n_base`` is the press axis, ``force_base`` is the UR ``ActualTCPForce``
reaction in the tool frame, and ``force_sign=-1`` so compression reads ``Fs>0``.
"""
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
