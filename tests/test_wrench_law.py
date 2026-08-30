"""Approach A: the pose spring reaches the arm as a wrench, not a velocity.

The first DTC port kept the whole speedJ velocity pipeline in front of the
torque law -- `wrench_net` -> /inertia -> *dt -> DLS inverse -> joint clamps ->
`tau = kv*(qdot_cmd - qdot_meas)`. Those caps exist to stop a *position servo*
over-shoving, and under direct torque they are what starves the arm: a 4 mm
error becomes ~8 mm/s becomes ~0.02 rad/s becomes ~1 N.m of `kv*qdot_err`,
against a UR5e breakaway of several N.m. So the arm floats, tracks, and will
not slide.

`compute_control_wrench` stops at the wrench the velocity law already computes,
and `DirectTorqueBackend` sends `tau = clip(J.T @ W, +/-tau_max)`. Lateral
force is then `K * e` in newtons, set on purpose.

The reduction that licenses this: both laws read the same
`objective_state.get_error()` and the same `task_space_state.get_jacobian()`,
and the velocity law's own output is `(J.T J + lam^2 I)^-1 J.T * (dt/M) *
wrench_net`. Frames and signs are shared by construction. These tests hold that
line -- that the speedJ path is untouched, that the wrench carries the same
compliance invariants, and that the shove is bounded before it becomes force.
"""
import time
import unittest

import numpy as np

from geodude_teleoperation.controller import AdmittanceTeleoperationController


# Identity task jacobian on 6 joints: qdot is then the body twist itself, and
# `J.T @ W` is `W`, so a test can state forces directly. Blade order is
# [angular, linear], so 0:3 is angular and 3:6 is linear.
J = np.eye(6)


class FakeRobotState:
    def __init__(self, qdot):
        self._qdot = np.asarray(qdot, dtype=float)

    def get_velocity(self):
        return self._qdot


class FakeObjectiveState:
    def __init__(self, error, error_derivative=None):
        self._e = np.asarray(error, dtype=float)
        self._ed = (np.zeros(6) if error_derivative is None
                    else np.asarray(error_derivative, dtype=float))

    def get_error(self):
        return self._e

    def get_error_derivative(self):
        return self._ed


class FakeTaskSpaceState:
    def __init__(self, jacobian=None, jacobian_derivative=None):
        self._j = J if jacobian is None else np.asarray(jacobian, dtype=float)
        self._jd = (np.zeros((6, 6)) if jacobian_derivative is None
                    else np.asarray(jacobian_derivative, dtype=float))

    def get_jacobian(self):
        return self._j

    def get_jacobian_derivative(self):
        return self._jd


def _ctrl(k_lin=3000.0, k_ang=50.0, damping=0.0, **kw):
    """A controller with the wrench-law gain split and (by default) no damping,
    so a test can read `K*e` straight out."""
    k = [k_ang] * 3 + [k_lin] * 3
    d = damping if np.ndim(damping) else [float(damping)] * 6
    return AdmittanceTeleoperationController(
        stiffness=np.diag(k), damping=np.diag(d), **kw)


def _wrench(ctrl, error, qdot=None, error_derivative=None, jacobian=None):
    return np.asarray(ctrl.compute_control_wrench(
        FakeRobotState(np.zeros(6) if qdot is None else qdot),
        FakeObjectiveState(error, error_derivative),
        FakeTaskSpaceState(jacobian),
    ), dtype=float)


def _lin_error(v):
    e = np.zeros(6)
    e[3:6] = v
    return e


class WrenchIsForceNotVelocityTest(unittest.TestCase):
    """The whole point of A: `K * e` lands in newtons, undivided."""

    def test_lateral_force_is_k_times_error(self):
        # 5 mm of field-follow lead at K_lin=3000 N/m is the ~15 N a slide
        # needs. Under the velocity law the same error produced ~8 mm/s and,
        # through kv, ~1 N.m -- which is why the arm would not break away.
        w = _wrench(_ctrl(k_lin=3000.0), _lin_error([0.005, 0.0, 0.0]))
        self.assertAlmostEqual(w[3], -15.0, places=9)
        np.testing.assert_allclose(w[[0, 1, 2, 4, 5]], 0.0, atol=1e-12)

    def test_the_spring_opposes_the_error(self):
        """`-K e` drives toward the target, the same sign the velocity law uses.

        This is the convention check: gafro's error is oriented so that the
        existing `wrench_net = -K e - D e_dot` moves the arm to the objective.
        Flipping to OSC's `+Kp e` would drive it away.
        """
        for axis in range(3):
            w = _wrench(_ctrl(), _lin_error(np.eye(3)[axis] * 0.004))
            self.assertLess(w[3 + axis], 0.0)

    def test_angular_and_linear_gains_are_different_units(self):
        """N.m/rad on blades 0..2, N/m on 3..5 -- a flat number is not shared."""
        # Both inside their clips, so this reads the gains and not the caps.
        e = np.zeros(6)
        e[0] = 0.1       # rad
        e[3] = 0.005     # m
        w = _wrench(_ctrl(k_lin=3000.0, k_ang=50.0), e)
        self.assertAlmostEqual(w[0], -5.0, places=9)     # 50 * 0.1 N.m
        self.assertAlmostEqual(w[3], -15.0, places=9)    # 3000 * 0.005 N
        # A flat gain would have made these equal for equal numbers; the ratio
        # is what the 20:1 split buys.
        self.assertAlmostEqual(w[0] / 0.1, -50.0, places=9)
        self.assertAlmostEqual(w[3] / 0.005, -3000.0, places=9)


class ErrorClipTest(unittest.TestCase):
    """The clip is the shove cap, replacing max_step_m. Sized against K."""

    def test_the_clip_bounds_the_force(self):
        ctrl = _ctrl(k_lin=3000.0, err_clip_lin=0.01)
        w = _wrench(ctrl, _lin_error([1.0, 0.0, 0.0]))   # a metre of error
        self.assertAlmostEqual(w[3], -30.0, places=9)    # 3000 * 0.01, not 3000

    def test_ordinary_sliding_stays_proportional(self):
        """At 5 mm of lead the clip must not bind, or force goes bang-bang."""
        ctrl = _ctrl(k_lin=3000.0, err_clip_lin=0.01)
        w = _wrench(ctrl, _lin_error([0.005, 0.0, 0.0]))
        self.assertAlmostEqual(w[3], -15.0, places=9)

    def test_the_angular_clip_is_separate(self):
        ctrl = _ctrl(k_ang=50.0, err_clip_rot=0.2)
        e = np.zeros(6)
        e[0] = 3.0
        self.assertAlmostEqual(_wrench(ctrl, e)[0], -10.0, places=9)

    def test_clipping_does_not_mutate_the_objective_error(self):
        ctrl = _ctrl(err_clip_lin=0.01)
        state = FakeObjectiveState(_lin_error([1.0, 0.0, 0.0]))
        ctrl.compute_control_wrench(
            FakeRobotState(np.zeros(6)), state, FakeTaskSpaceState())
        self.assertAlmostEqual(float(state.get_error()[3]), 1.0, places=12)

    def test_the_clip_defaults_are_sized_against_the_default_stiffness(self):
        """A clip copied from a rig with a different K is a hardware surprise."""
        ctrl = AdmittanceTeleoperationController()
        self.assertAlmostEqual(ctrl._err_clip_lin, 0.01, places=9)
        self.assertAlmostEqual(ctrl._err_clip_rot, 0.2, places=9)


class DampingTest(unittest.TestCase):
    """A pure damper on `J @ qdot`, not on `get_error_derivative()`."""

    def test_damping_opposes_measured_motion(self):
        qdot = np.zeros(6)
        qdot[3] = 0.01                       # 10 mm/s along body X
        w = _wrench(_ctrl(damping=100.0), np.zeros(6), qdot=qdot)
        self.assertAlmostEqual(w[3], -1.0, places=9)     # -100 * 0.01

    def test_target_motion_does_not_kick_the_wrench(self):
        """`error_derivative` carries target motion and would fight the stick.

        Same error, same measured velocity, wildly different error rate: the
        wrench must not move. The velocity law deliberately does use
        `error_derivative`; this is the one place the two laws differ in what
        they damp, and it must not leak back.
        """
        ctrl = _ctrl(damping=100.0)
        still = _wrench(ctrl, _lin_error([0.004, 0, 0]), error_derivative=np.zeros(6))
        moving = _wrench(ctrl, _lin_error([0.004, 0, 0]),
                         error_derivative=_lin_error([5.0, 0, 0]))
        np.testing.assert_allclose(still, moving, atol=1e-12)

    def test_damping_reads_the_control_group_slice(self):
        """A 7-long velocity (arm + vention rail) must not shift the blades.

        get_joint_indices() is 7 long and system-global; the jacobian is 6x6.
        Indexing one with the other is the "index 15 on size 7" crash.
        """
        qdot7 = np.zeros(7)
        qdot7[0] = 99.0        # the rail slot, not in the jacobian
        qdot7[4] = 0.01        # body X of the 6 driven joints
        w = _wrench(_ctrl(damping=100.0), np.zeros(6), qdot=qdot7)
        self.assertAlmostEqual(w[3], -1.0, places=9)


class ContactInvariantsTest(unittest.TestCase):
    """The compliance invariants carry over one conversion earlier."""

    def test_z_compliant_zeroes_the_pose_spring_on_blade_five(self):
        ctrl = _ctrl()
        ctrl.set_z_compliant(True)
        w = _wrench(ctrl, _lin_error([0.0, 0.0, 0.004]))
        self.assertAlmostEqual(w[5], 0.0, places=12)

    def test_free_along_normal_removes_the_normal_component(self):
        """Align never owns the unbound axis -- force does.

        stiffness[5,5]=0 frees body Z of the *pose error*, which is not the
        surface normal: field-follow slerps tool Z behind the field normal on
        purpose, so mid-slide the spring on the remaining axes has a component
        along the true normal.
        """
        ctrl = _ctrl()
        ctrl.set_z_compliant(True)
        n = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
        ctrl._n_body = n
        w = _wrench(ctrl, _lin_error([0.004, 0.004, 0.0]))
        self.assertAlmostEqual(float(w[3:6] @ n), 0.0, places=12)
        self.assertGreater(np.linalg.norm(w[3:6]), 0.0)

    def test_free_along_normal_is_off_in_free_space(self):
        ctrl = _ctrl()
        ctrl._n_body = np.array([0.0, 0.0, 1.0])
        w = _wrench(ctrl, _lin_error([0.0, 0.0, 0.004]))
        self.assertLess(w[5], 0.0)

    def test_it_can_be_disabled(self):
        ctrl = _ctrl(free_along_normal=False)
        ctrl.set_z_compliant(True)
        n = np.array([0.0, 1.0, 0.0])
        ctrl._n_body = n
        w = _wrench(ctrl, _lin_error([0.0, 0.004, 0.0]))
        self.assertNotAlmostEqual(float(w[3:6] @ n), 0.0, places=6)


class PressWrenchTest(unittest.TestCase):
    """The press is along the contact normal, not blade 5."""

    def test_press_adds_along_the_given_body_axis(self):
        ctrl = _ctrl()
        ctrl.set_press_wrench([0.0, 0.0, 10.0])
        w = _wrench(ctrl, np.zeros(6))
        np.testing.assert_allclose(w[3:6], [0.0, 0.0, 10.0], atol=1e-12)

    def test_press_survives_the_free_along_normal_projection(self):
        """Order matters: project the spring, *then* add the press.

        Projecting after would delete the press, which is the only thing on
        that axis. Writing it on blade 5 instead of the normal would mix axes,
        because tool Z and the field normal differ by the slerp lead mid-slide.
        """
        ctrl = _ctrl()
        ctrl.set_z_compliant(True)
        n = np.array([0.0, 0.6, 0.8])
        ctrl._n_body = n
        ctrl.set_press_wrench(10.0 * n)
        w = _wrench(ctrl, _lin_error([0.004, 0.004, 0.004]))
        self.assertAlmostEqual(float(w[3:6] @ n), 10.0, places=9)

    def test_no_press_by_default_and_clearing_works(self):
        ctrl = _ctrl()
        np.testing.assert_allclose(_wrench(ctrl, np.zeros(6)), 0.0, atol=1e-12)
        ctrl.set_press_wrench([1.0, 2.0, 3.0])
        ctrl.set_press_wrench(None)
        np.testing.assert_allclose(_wrench(ctrl, np.zeros(6)), 0.0, atol=1e-12)


class SpeedjIsUntouchedTest(unittest.TestCase):
    """`compute_control_command` must stay bit-identical: speedj still flies."""

    def test_the_velocity_law_still_uses_error_derivative(self):
        """The damping swap is A-only. If it leaked, speedj's feel changes."""
        ctrl = _ctrl(damping=100.0)
        args = (FakeRobotState(np.zeros(6)), None, FakeTaskSpaceState())
        ctrl.set_input_active(True)
        still = np.asarray(ctrl.compute_control_command(
            args[0], FakeObjectiveState(_lin_error([0.004, 0, 0])), args[2]))
        ctrl2 = _ctrl(damping=100.0)
        ctrl2.set_input_active(True)
        moving = np.asarray(ctrl2.compute_control_command(
            FakeRobotState(np.zeros(6)),
            FakeObjectiveState(_lin_error([0.004, 0, 0]), _lin_error([5.0, 0, 0])),
            FakeTaskSpaceState()))
        self.assertFalse(np.allclose(still, moving))

    def test_calling_the_wrench_law_does_not_disturb_the_velocity_law(self):
        """Both write the same diagnostic fields; neither may change the other's
        command. This is the only shared state the split introduced."""
        e = _lin_error([0.004, 0.0, 0.0])
        clean = _ctrl()
        clean.set_input_active(True)
        expected = np.asarray(clean.compute_control_command(
            FakeRobotState(np.zeros(6)), FakeObjectiveState(e),
            FakeTaskSpaceState())).copy()

        contaminated = _ctrl()
        contaminated.set_input_active(True)
        contaminated.compute_control_wrench(
            FakeRobotState(np.zeros(6)),
            FakeObjectiveState(_lin_error([9.0, 9.0, 9.0])), FakeTaskSpaceState())
        got = np.asarray(contaminated.compute_control_command(
            FakeRobotState(np.zeros(6)), FakeObjectiveState(e),
            FakeTaskSpaceState()))
        np.testing.assert_allclose(got, expected, atol=0.0, rtol=0.0)

    def test_the_velocity_law_keeps_its_defaults(self):
        ctrl = AdmittanceTeleoperationController()
        np.testing.assert_allclose(np.diag(ctrl._stiffness), 3000.0)
        np.testing.assert_allclose(np.diag(ctrl._damping), 20.0)
        self.assertAlmostEqual(ctrl._dt, 0.001, places=12)
        self.assertAlmostEqual(ctrl._max_joint_velocity, 1.0, places=12)


class DiagnosticsTest(unittest.TestCase):
    def test_the_commanded_force_is_logged(self):
        """|W_lin| against tau_peak is how a run says whether the spring or the
        torque clamp decided the motion."""
        ctrl = _ctrl(k_lin=3000.0)
        _wrench(ctrl, _lin_error([0.004, 0.0, 0.0]))
        cond = ctrl.conditioning()
        self.assertAlmostEqual(cond["w_lin"], 12.0, places=9)
        self.assertAlmostEqual(cond["w_ang"], 0.0, places=12)

    def test_the_unclipped_error_is_what_gets_logged(self):
        """A log that records the clipped error hides the excursion that
        caused the clip."""
        ctrl = _ctrl(err_clip_lin=0.01)
        _wrench(ctrl, _lin_error([0.5, 0.0, 0.0]))
        self.assertAlmostEqual(ctrl.conditioning()["err_x"], 0.5, places=9)


class RearmRampTest(unittest.TestCase):
    """The `input_active` 0->1 edge must not be a one-tick step in the wrench.

    `--rearm-slerp-s` ramps the commanded *rotation* only. Nothing ramped the
    wrench, and while the stick is quiet the idle hold pins the target on the
    live pose (err_lin ~ 0) -- so the first active tick handed back the full
    field-follow carrot at once. Measured on dtc_20260830_115039.csv, 9 edges:
    |W_lin| 9.25 -> 38.30 N and |W_ang| 1.09 -> 10.35 N.m inside one 8 ms tick.

    Deliberately an edge blend and not a slew-rate limit on W: on that same log
    ordinary sliding already runs |dW_lin|/dt of 1005 N/s at p95 and 3268 N/s
    at p99.9, against 613..3641 N/s at the edges, so no rate separates them.
    """

    def _idle_then_active(self, ctrl, error, hold_error=None):
        """One quiet tick (which sets the held wrench), then the first active
        tick, which is the edge under test."""
        ctrl.set_input_active(False)
        w_idle = _wrench(ctrl, _lin_error([0.0, 0.0, 0.0])
                         if hold_error is None else hold_error)
        ctrl.set_input_active(True)
        return w_idle, _wrench(ctrl, error)

    # 5 mm is one field-follow lead, inside the 10 mm err_clip_lin default:
    # these read the gain, not the cap. 1 mm is the residue the idle hold
    # leaves behind, which is what the blend has to open from.
    SLIDE = 0.005    # -> -15 N at k_lin=3000
    HOLD = 0.001     # ->  -3 N

    def test_the_first_active_tick_opens_from_the_held_wrench(self):
        ctrl = _ctrl(k_lin=3000.0, rearm_wrench_s=1e4)  # alpha ~ 0 on tick one
        w_idle, w_edge = self._idle_then_active(
            ctrl, _lin_error([self.SLIDE, 0.0, 0.0]),
            hold_error=_lin_error([self.HOLD, 0.0, 0.0]))
        # Un-ramped this edge is -3 N -> -15 N in one tick. The blend starts
        # at the hold instead.
        self.assertAlmostEqual(w_idle[3], -3.0, places=9)
        np.testing.assert_allclose(w_edge, w_idle, atol=1e-3)

    def test_the_ramp_converges_on_the_full_spring(self):
        ctrl = _ctrl(k_lin=3000.0, rearm_wrench_s=0.15)
        self._idle_then_active(ctrl, _lin_error([self.SLIDE, 0.0, 0.0]),
                               hold_error=_lin_error([self.HOLD, 0.0, 0.0]))
        # Halfway: -3 N blending toward -15 N.
        ctrl._rearm_t0 = time.perf_counter() - 0.075
        w_mid = _wrench(ctrl, _lin_error([self.SLIDE, 0.0, 0.0]))
        self.assertLess(w_mid[3], -3.0)
        self.assertGreater(w_mid[3], -15.0)
        # Past the ramp it is the plain spring again, and the state is dropped.
        ctrl._rearm_t0 = time.perf_counter() - 1.0
        w_end = _wrench(ctrl, _lin_error([self.SLIDE, 0.0, 0.0]))
        self.assertAlmostEqual(w_end[3], -15.0, places=9)
        self.assertIsNone(ctrl._rearm_t0)

    def test_zero_restores_the_one_tick_step(self):
        """Reduction to the previously-working case, and it is exact."""
        ctrl = _ctrl(k_lin=3000.0, rearm_wrench_s=0.0)
        _, w_edge = self._idle_then_active(
            ctrl, _lin_error([self.SLIDE, 0.0, 0.0]))
        self.assertAlmostEqual(w_edge[3], -15.0, places=9)

    def test_the_ramp_is_inactive_during_ordinary_sliding(self):
        """No edge, no blend -- so it adds no lag to the loop that is already
        spending its stability budget on the delayed damper.

        Note the *first* active tick of a fresh controller is itself an edge,
        which is correct: nothing was holding the arm, so the blend opens from
        zero. This drains that one and then checks the steady state."""
        ctrl = _ctrl(k_lin=3000.0, rearm_wrench_s=0.15)
        ctrl.set_input_active(True)
        _wrench(ctrl, _lin_error([0.004, 0.0, 0.0]))
        ctrl._rearm_t0 = time.perf_counter() - 1.0
        _wrench(ctrl, _lin_error([0.004, 0.0, 0.0]))
        self.assertIsNone(ctrl._rearm_t0)
        # A large change *without* an edge now passes straight through.
        w = _wrench(ctrl, _lin_error([self.SLIDE, 0.0, 0.0]))
        self.assertAlmostEqual(w[3], -15.0, places=9)
        w = _wrench(ctrl, _lin_error([0.0, 0.0, 0.0]))
        self.assertAlmostEqual(w[3], 0.0, places=12)

    def test_the_press_passes_through_the_blend_untouched(self):
        """The press is added before the blend and is common to both ends, so
        `W0 + a*(W - W0)` leaves it alone. Ramping it would unload the tool at
        exactly the moment the operator asked the arm to move."""
        ctrl = _ctrl(k_lin=3000.0, rearm_wrench_s=1e4)
        ctrl.set_press_wrench([0.0, 0.0, -10.0])
        _, w_edge = self._idle_then_active(
            ctrl, _lin_error([self.SLIDE, 0.0, 0.0]))
        self.assertAlmostEqual(w_edge[5], -10.0, places=6)

    def test_reset_forgets_the_blend(self):
        """A pendant stop must not queue up a ramp that opens from a pre-stop
        wrench. Same rule as the idle latch and the joint hold."""
        ctrl = _ctrl(k_lin=3000.0, rearm_wrench_s=1e4)
        self._idle_then_active(ctrl, _lin_error([self.SLIDE, 0.0, 0.0]),
                               hold_error=_lin_error([self.HOLD, 0.0, 0.0]))
        ctrl.reset_wrench_rearm()
        self.assertIsNone(ctrl._rearm_t0)
        np.testing.assert_allclose(ctrl._last_wrench, 0.0, atol=1e-12)
        # Still active, but the edge has been forgotten, so the next tick
        # re-arms from zero rather than from the stale wrench.
        w = _wrench(ctrl, _lin_error([self.SLIDE, 0.0, 0.0]))
        np.testing.assert_allclose(w, 0.0, atol=1e-3)

    def test_the_velocity_law_never_sees_the_ramp(self):
        """A-only. speedJ has to stay bit-identical or the C153 A/B is void."""
        ctrl = _ctrl(k_lin=3000.0, rearm_wrench_s=1e4)
        ctrl.set_input_active(True)
        qdot = np.asarray(ctrl.compute_control_command(
            FakeRobotState(np.zeros(6)),
            FakeObjectiveState(_lin_error([0.004, 0.0, 0.0])),
            FakeTaskSpaceState(),
        ), dtype=float)
        self.assertIsNone(ctrl._rearm_t0)
        ref = _ctrl(k_lin=3000.0, rearm_wrench_s=0.0)
        ref.set_input_active(True)
        qdot_ref = np.asarray(ref.compute_control_command(
            FakeRobotState(np.zeros(6)),
            FakeObjectiveState(_lin_error([0.004, 0.0, 0.0])),
            FakeTaskSpaceState(),
        ), dtype=float)
        np.testing.assert_allclose(qdot, qdot_ref, atol=0.0)


if __name__ == "__main__":
    unittest.main()
