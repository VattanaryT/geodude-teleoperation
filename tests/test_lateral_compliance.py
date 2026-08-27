"""The lateral axes, which the force loop cannot see.

The contact loop regulates the surface normal and is healthy: `Fs` median
7.94 N against `Fd` 10, guards nearly silent. The UR still protective-stops
with C153A1 "robot could not follow path", naming the shoulder, and one lateral
direction trips it far more than the other. At the instant of one stop the force
loop read `Fs=+9.67 N`, `v_z=+0.0005`, no guards -- so the fault is in the
plane, not along the normal.

The mechanism: the lateral axes are pure position control. The pose spring
commands `dt*K/M*e` whether or not the tool is free to move, `e` pins at
field-follow's 5 mm lead clamp when the tool grabs, and the arm is asked for
~8 mm/s it cannot deliver until the UR's own joint deviation trips. `fx`/`fy`
came off the wrist F/T sensor every tick and were thrown away.

These cover the three limiters that now bound the tangential command, and the
two ways a limiter like this goes wrong: latching on forever, and braking a
direction that was never stuck.
"""
import time
import unittest

import numpy as np

from geodude_teleoperation.controller import AdmittanceTeleoperationController


DT = 0.005
# Identity task jacobian on 6 joints: qdot is then the body twist itself, so a
# test can state the measured TCP velocity directly.
J = np.eye(6)


def _qdot_for(v_lin):
    q = np.zeros(6)
    q[3:6] = np.asarray(v_lin, dtype=float)
    return q


def _ctrl(**kw):
    ctrl = AdmittanceTeleoperationController(**kw)
    ctrl.set_z_compliant(True)
    # Surface normal along body Z: the tangent plane is then body X/Y, which is
    # where the stick writes in contact.
    ctrl._n_body = np.array([0.0, 0.0, 1.0])
    return ctrl


def _push(ctrl, v_cmd, v_meas, f_tool=None, fs=None, fd=None, ticks=1):
    """Run the limiter `ticks` times and return the last applied lateral speed."""
    out = None
    for _ in range(ticks):
        if f_tool is not None or fs is not None:
            ctrl.set_contact_force(
                np.zeros(3) if f_tool is None else f_tool, fs=fs, fd=fd)
        out = ctrl._limit_lateral(
            np.asarray(v_cmd, dtype=float), J, _qdot_for(v_meas), dt=DT)
    return out


class FreeSlideTest(unittest.TestCase):
    def test_a_slide_the_arm_is_following_is_not_touched(self):
        """Nothing here may cost speed when the tool is moving as commanded."""
        ctrl = _ctrl()
        v = np.array([0.008, 0.0, 0.0])
        out = _push(ctrl, v, v, f_tool=np.zeros(3), fs=10.0, fd=10.0, ticks=200)
        np.testing.assert_allclose(out, v, atol=1e-9)

    def test_the_normal_component_is_removed_not_scaled(self):
        """Force owns the normal. The spring must not command along it at all."""
        ctrl = _ctrl()
        out = _push(ctrl, [0.008, 0.0, 0.02], [0.008, 0.0, 0.0],
                    f_tool=np.zeros(3), fs=10.0, fd=10.0)
        self.assertAlmostEqual(float(out[2]), 0.0, places=12)


class ForceYieldTest(unittest.TestCase):
    def test_a_grab_opposing_the_push_cancels_it(self):
        ctrl = _ctrl(lateral_yield_n=20.0, lateral_yield_deadband_n=6.0,
                     lateral_stall_yield=False)
        # UR ActualTCPForce is the reaction, so resisting a +X push reads -X.
        out = _push(ctrl, [0.008, 0.0, 0.0], [0.008, 0.0, 0.0],
                    f_tool=[-20.0, 0.0, 0.0], fs=10.0, fd=10.0)
        self.assertAlmostEqual(float(np.linalg.norm(out)), 0.0, places=9)

    def test_the_same_force_does_not_brake_the_opposite_direction(self):
        """One lateral direction trips the UR far more than the other.

        A grab in +X is no reason to refuse a slide in -X -- backing out is how
        the tool gets free.
        """
        ctrl = _ctrl(lateral_stall_yield=False)
        out = _push(ctrl, [-0.008, 0.0, 0.0], [-0.008, 0.0, 0.0],
                    f_tool=[-20.0, 0.0, 0.0], fs=10.0, fd=10.0)
        np.testing.assert_allclose(out, [-0.008, 0.0, 0.0], atol=1e-9)

    def test_steady_sliding_friction_costs_nothing(self):
        """mu*Fd is present on every real slide and is not a grab.

        The absolute deadband covers it at first, and the ambient baseline takes
        over as it settles -- which is also what makes a few N of tare error
        survivable, since neither is knowable in advance.
        """
        ctrl = _ctrl(lateral_stall_yield=False)
        v = np.array([0.008, 0.0, 0.0])
        out = _push(ctrl, v, v, f_tool=[-5.5, 0.0, 0.0], fs=10.0, fd=10.0,
                    ticks=400)
        np.testing.assert_allclose(out, v, atol=1e-9)

    def test_a_rise_above_ambient_friction_still_yields(self):
        """A tool sliding at 12 N of drag has no absolute headroom left.

        Without the ambient baseline the absolute deadband alone would already
        have cancelled the slide, so the operator could not move at all on a
        high-friction surface; with it, 12 N is ordinary and 12 -> 24 N is a
        grab.
        """
        ctrl = _ctrl(lateral_stall_yield=False)
        v = np.array([0.008, 0.0, 0.0])
        _push(ctrl, v, v, f_tool=[-12.0, 0.0, 0.0], fs=10.0, fd=10.0, ticks=2000)
        cruise = _push(ctrl, v, v, f_tool=[-12.0, 0.0, 0.0], fs=10.0, fd=10.0)
        np.testing.assert_allclose(cruise, v, atol=1e-9)
        grabbed = _push(ctrl, v, v, f_tool=[-26.0, 0.0, 0.0], fs=10.0, fd=10.0)
        self.assertAlmostEqual(float(np.linalg.norm(grabbed)), 0.0, places=9)

    def test_stale_force_reverts_to_no_yield(self):
        """A dead reader must not read as a large reaction and brake the slide."""
        ctrl = _ctrl(lateral_stall_yield=False, force_stale_s=0.01)
        ctrl.set_contact_force([-40.0, 0.0, 0.0], fs=10.0, fd=10.0)
        ctrl._f_tool_t = time.perf_counter() - 1.0
        out = ctrl._limit_lateral(
            np.array([0.008, 0.0, 0.0]), J, _qdot_for([0.008, 0.0, 0.0]), dt=DT)
        np.testing.assert_allclose(out, [0.008, 0.0, 0.0], atol=1e-9)


class StallYieldTest(unittest.TestCase):
    def test_a_command_the_arm_is_not_following_decays(self):
        """This is the quantity C153A1 fires on, read before the UR's threshold."""
        ctrl = _ctrl(lateral_force_yield=False)
        v = np.array([0.008, 0.0, 0.0])
        out = _push(ctrl, v, [0.0, 0.0, 0.0], fs=10.0, fd=10.0,
                    ticks=int(0.5 / DT))
        self.assertLess(float(np.linalg.norm(out)), 0.1 * float(np.linalg.norm(v)))

    def test_it_does_not_latch_once_the_command_is_gone(self):
        """Testing the *raw* command would latch forever.

        Once the scale is near zero the arm is correctly not moving, so a raw
        test reads that as a stall and never lets go. Testing the applied
        command instead leaves a small creep that breaks stiction.
        """
        ctrl = _ctrl(lateral_force_yield=False)
        v = np.array([0.008, 0.0, 0.0])
        _push(ctrl, v, [0.0, 0.0, 0.0], fs=10.0, fd=10.0, ticks=int(2.0 / DT))
        floor = float(np.linalg.norm(
            _push(ctrl, v, [0.0, 0.0, 0.0], fs=10.0, fd=10.0)))
        self.assertGreater(floor, 0.0)
        # ...and it comes all the way back once the tool is free again.
        out = _push(ctrl, v, v, fs=10.0, fd=10.0, ticks=int(5.0 / DT))
        np.testing.assert_allclose(out, v, atol=1e-6)

    def test_a_slow_but_real_slide_is_not_called_a_stall(self):
        ctrl = _ctrl(lateral_force_yield=False, lateral_stall_ratio=0.35)
        v = np.array([0.008, 0.0, 0.0])
        out = _push(ctrl, v, [0.006, 0.0, 0.0], fs=10.0, fd=10.0,
                    ticks=int(1.0 / DT))
        np.testing.assert_allclose(out, v, atol=1e-9)


class ContactGateTest(unittest.TestCase):
    def test_slide_is_slowed_while_the_press_is_out_of_band(self):
        """~2.8 mm/s of normal authority cannot chase an 8 mm/s slide.

        Backing the slide off exactly when the force error says the press is
        losing is what keeps the tool on the surface over curvature.
        """
        ctrl = _ctrl(lateral_force_yield=False, lateral_stall_yield=False,
                     contact_gate_floor=0.25)
        v = np.array([0.008, 0.0, 0.0])
        out = _push(ctrl, v, v, fs=0.0, fd=10.0)
        self.assertAlmostEqual(
            float(np.linalg.norm(out)), 0.25 * float(np.linalg.norm(v)), places=9)

    def test_tracking_inside_the_band_costs_nothing(self):
        ctrl = _ctrl(lateral_force_yield=False, lateral_stall_yield=False)
        v = np.array([0.008, 0.0, 0.0])
        out = _push(ctrl, v, v, fs=9.0, fd=10.0)
        np.testing.assert_allclose(out, v, atol=1e-9)

    def test_it_is_off_when_the_press_loop_never_reported(self):
        ctrl = _ctrl(lateral_force_yield=False, lateral_stall_yield=False)
        out = ctrl._limit_lateral(
            np.array([0.008, 0.0, 0.0]), J, _qdot_for([0.008, 0.0, 0.0]), dt=DT)
        np.testing.assert_allclose(out, [0.008, 0.0, 0.0], atol=1e-9)


class NeverAddsMotionTest(unittest.TestCase):
    def test_a_steady_lateral_bias_cannot_drive_the_arm(self):
        """True vector admittance would let a tare error walk the tool sideways.

        These limiters only ever scale the commanded direction down, so with no
        command there is no motion no matter what the sensor reads.
        """
        ctrl = _ctrl()
        out = _push(ctrl, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                    f_tool=[30.0, -30.0, 0.0], fs=10.0, fd=10.0, ticks=100)
        self.assertAlmostEqual(float(np.linalg.norm(out)), 0.0, places=12)

    def test_output_never_exceeds_the_commanded_speed(self):
        ctrl = _ctrl()
        rng = np.random.default_rng(0)
        v = np.array([0.008, 0.0, 0.0])
        for _ in range(500):
            f = rng.normal(0.0, 15.0, 3)
            out = _push(ctrl, v, rng.normal(0.0, 0.01, 3), f_tool=f,
                        fs=float(rng.normal(10.0, 5.0)), fd=10.0)
            self.assertLessEqual(
                float(np.linalg.norm(out)), float(np.linalg.norm(v)) + 1e-12)


class SessionScopeTest(unittest.TestCase):
    def test_leaving_contact_forgets_the_friction_baseline(self):
        """Carrying it across an exit would start the next press already braking."""
        ctrl = _ctrl(lateral_stall_yield=False)
        v = np.array([0.008, 0.0, 0.0])
        _push(ctrl, v, v, f_tool=[-14.0, 0.0, 0.0], fs=10.0, fd=10.0, ticks=2000)
        self.assertGreater(ctrl._lat_opp_base, 5.0)
        ctrl.set_z_compliant(False)
        self.assertEqual(ctrl._lat_opp_base, 0.0)
        self.assertEqual(ctrl._lat_scale, 1.0)


if __name__ == "__main__":
    unittest.main()
