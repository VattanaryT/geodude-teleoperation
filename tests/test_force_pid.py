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
