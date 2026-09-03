import unittest
import threading
import time

import gafropy as ga
import numpy as np

from geodude_teleoperation.ee_viz import (
    ghost_flange_world,
    live_tcp_xyz_from_flange,
    motor_from_matrix,
    orientation_error_rad,
)
from geodude_teleoperation.field_follow import (
    FieldFollow,
    drive_field_follow,
    field_tcp_to_ee,
    format_follow_lag,
)


def _motor_at(xyz, R=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return motor_from_matrix(T)


class FakeVf:
    def __init__(self):
        self.projects = []
        self._smoothing = 0.75

    def project(self, motor):
        self.projects.append(motor)
        return motor


class SlowBusyVf:
    """CPU-bound project() so a same-process call holds the GIL."""

    def __init__(self):
        self._smoothing = 0.0

    def project(self, motor):
        t_end = time.perf_counter() + 0.04
        x = 0.0
        while time.perf_counter() < t_end:
            x += 1.0
        return motor


class FakeObjective:
    def __init__(self):
        self.target = None

    def set_target(self, motor):
        self.target = motor


class FakeCtrl:
    def __init__(self):
        self.active = False
        self._obj = FakeObjective()

    def get_objective(self):
        return self._obj

    def set_input_active(self, active):
        self.active = bool(active)


class ReArmStepTest(unittest.TestCase):
    """The 0 -> 77 N / 0 -> 4.25 N.m one-tick step on dtc_20260829_214019.csv.

    Cause: the target was the live pose while the stick was quiet (both idle
    holds snap target := live) and the full field pose one tick later, so the
    whole accumulated orientation deviation arrived as a step. Two independent
    fixes, both switchable: keep the commanded orientation through quiet
    (hold_field_orientation), and ramp the command onto the field goal
    (rearm_slerp_s).
    """

    @staticmethod
    def _tilt_vf(R_field):
        class TiltVf:
            def approach_basis_at(self, position):
                return R_field

            def project(self, motor):
                T = np.asarray(motor.to_transformation_matrix(), dtype=float)
                T[:3, :3] = R_field
                return motor_from_matrix(T)

        return TiltVf()

    def test_quiet_no_longer_erases_the_commanded_orientation(self):
        from scipy.spatial.transform import Rotation

        R_field = Rotation.from_rotvec([0.0, 0.4, 0.0]).as_matrix()
        follow = FieldFollow(
            self._tilt_vf(R_field), hz=50.0, max_dR_rad=0.0,
            snap_to_surface=False, hold_field_orientation=True)
        live = np.array([0.0, 0.0, 1.0])
        follow.seed(live, R_field)
        ctrl = FakeCtrl()
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        drive_field_follow(follow, ctrl, 0.0, _motor_at(live), live_xyz=live,
                           live_R=np.eye(3), stick_quiet=False)
        commanded = np.array(follow._R_cmd, copy=True)
        # Stick goes quiet with the wrist still 0.4 rad off the field.
        drive_field_follow(follow, ctrl, 0.1, _motor_at(live), live_xyz=live,
                           live_R=np.eye(3), stick_quiet=True)
        np.testing.assert_allclose(follow._R_cmd, commanded, atol=1e-12)
        T = np.asarray(
            ctrl.get_objective().target.to_transformation_matrix(), dtype=float)
        # Position re-based onto live, orientation kept.
        np.testing.assert_allclose(T[:3, 3], live, atol=1e-9)
        np.testing.assert_allclose(T[:3, :3], commanded, atol=1e-9)

    def test_speedj_still_collapses_the_command_onto_live(self):
        """The speedJ path is the C153 A/B baseline and must not move."""
        from scipy.spatial.transform import Rotation

        R_field = Rotation.from_rotvec([0.0, 0.4, 0.0]).as_matrix()
        follow = FieldFollow(
            self._tilt_vf(R_field), hz=50.0, max_dR_rad=0.0,
            snap_to_surface=False, hold_field_orientation=False)
        live = np.array([0.0, 0.0, 1.0])
        follow.seed(live, R_field)
        ctrl = FakeCtrl()
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        drive_field_follow(follow, ctrl, 0.0, _motor_at(live), live_xyz=live,
                           live_R=np.eye(3), stick_quiet=False)
        drive_field_follow(follow, ctrl, 0.1, _motor_at(live), live_xyz=live,
                           live_R=np.eye(3), stick_quiet=True)
        np.testing.assert_allclose(follow._R_cmd, np.eye(3), atol=1e-9)

    def test_the_command_ramps_onto_the_field_instead_of_snapping(self):
        from scipy.spatial.transform import Rotation

        R_field = Rotation.from_rotvec([0.0, 0.4, 0.0]).as_matrix()
        follow = FieldFollow(
            self._tilt_vf(R_field), hz=50.0, max_dR_rad=0.0,
            snap_to_surface=False, rearm_slerp_s=0.5,
            hold_field_orientation=True)
        live = np.array([0.0, 0.0, 1.0])
        follow.seed(live, np.eye(3))          # wrist starts off the field
        errs = []
        for i in range(1, 151):
            follow.add_body_linear(np.array([0.0001, 0.0, 0.0]))
            follow.tick(0.02 * i, _motor_at(live), live_xyz=live,
                        live_R=np.eye(3))
            errs.append(orientation_error_rad(follow._R_cmd, R_field))
        # First step is a fraction of the 0.4 rad gap (dt/tau = 0.04), not all
        # of it -- that whole gap arriving in one tick is the 4.25 N.m step.
        self.assertLess(0.4 - errs[0], 0.4 * 0.1)
        # Monotone approach that actually converges -- no permanent lag.
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(errs, errs[1:])))
        self.assertLess(errs[-1], 0.01)
        # The goal itself is the field the whole time.
        np.testing.assert_allclose(follow._R, R_field, atol=1e-9)

    def test_zero_slerp_snaps_like_before(self):
        from scipy.spatial.transform import Rotation

        R_field = Rotation.from_rotvec([0.0, 0.4, 0.0]).as_matrix()
        follow = FieldFollow(
            self._tilt_vf(R_field), hz=50.0, max_dR_rad=0.0,
            snap_to_surface=False, rearm_slerp_s=0.0,
            hold_field_orientation=True)
        live = np.array([0.0, 0.0, 1.0])
        follow.seed(live, np.eye(3))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        follow.tick(0.0, _motor_at(live), live_xyz=live, live_R=np.eye(3))
        np.testing.assert_allclose(follow._R_cmd, R_field, atol=1e-9)


class FieldFollowTest(unittest.TestCase):
    def test_idle_does_not_project(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.set_idle()
        tick = follow.tick(0.1, _motor_at([0.0, 0.0, 1.0]))
        self.assertTrue(tick.idle)
        self.assertIsNone(tick.target)
        self.assertEqual(vf.projects, [])

    def test_angular_is_ignored_linear_accumulates(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        tick = follow.tick(0.1, _motor_at([0.0, 0.0, 1.0]))
        self.assertFalse(tick.idle)
        self.assertIsNotNone(tick.target)
        self.assertEqual(len(vf.projects), 1)
        T = np.asarray(tick.target.to_transformation_matrix())
        np.testing.assert_allclose(T[:3, 3], [0.02, 0.0, 1.0], atol=1e-9)

    def test_rate_limit_skips_second_project(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=20.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        follow.tick(0.0, _motor_at([0.0, 0.0, 1.0]))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        tick = follow.tick(0.01, _motor_at([0.0, 0.0, 1.0]))
        self.assertIsNone(tick.target)
        self.assertEqual(len(vf.projects), 1)
        tick = follow.tick(0.06, _motor_at([0.0, 0.0, 1.0]))
        self.assertIsNotNone(tick.target)
        self.assertEqual(len(vf.projects), 2)

    def test_idle_clears_pending_so_rest_does_not_nibble(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.05, 0.0, 0.0]))
        follow.set_idle()
        follow.tick(0.1, _motor_at([0.0, 0.0, 1.0]))
        self.assertEqual(vf.projects, [])
        np.testing.assert_allclose(follow.xyz, [0.0, 0.0, 1.0])

    def test_logs_project_wall_time_and_quaternion(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        tick = follow.tick(0.1, _motor_at([0.0, 0.0, 1.0]))
        self.assertIsNotNone(tick.project_ms)
        self.assertGreaterEqual(tick.project_ms, 0.0)
        self.assertEqual(len(tick.quat_wxyz), 4)
        self.assertEqual(len(follow.log), 1)
        self.assertIn("project_ms", follow.log[0])

    def test_follow_lag_line_reports_tcp_and_rotation(self):
        live = np.array([0.0, 0.0, 1.0])
        cmd = np.array([0.010, 0.0, 1.0])
        R_live = np.eye(3)
        R_cmd = np.diag([1.0, -1.0, -1.0])
        line = format_follow_lag(live, cmd, R_live, R_cmd, True, project_ms=7.1)
        self.assertIn("tcp_err=10.0mm", line)
        self.assertIn("R_err=180.0deg", line)
        self.assertIn("spring=on", line)
        self.assertIn("project=7.10ms", line)

    def test_zeros_smoothing_after_first_project(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0, zero_smoothing_after_first=True)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        follow.tick(0.0, _motor_at([0.0, 0.0, 1.0]))
        self.assertEqual(vf._smoothing, 0.0)

    def test_project_over_2ms_flags_rtde_overrun(self):
        vf = FakeVf()

        def slow_project(motor):
            import time
            time.sleep(0.003)
            return motor

        vf.project = slow_project
        follow = FieldFollow(vf, hz=50.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        tick = follow.tick(0.0, _motor_at([0.0, 0.0, 1.0]))
        self.assertTrue(tick.would_overrun_rtde)
        self.assertGreater(tick.project_ms, 2.0)

    def test_pending_applies_even_if_idle_flag_set(self):
        """Rest must not skip a queued stick step (16 mm freeze + dead stick)."""
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0, max_lead_m=0.02, max_step_m=0.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.004, 0.0, 0.0]))
        follow.set_idle(clear_pending=False)
        tick = follow.tick(
            0.0, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]))
        self.assertTrue(tick.target is not None)
        np.testing.assert_allclose(follow.xyz, [0.004, 0.0, 1.0], atol=1e-9)

    def test_rate_limit_with_pending_does_not_freeze(self):
        """Viser ticks faster than --follow-hz; leftover 0 must not kill the stick."""
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.004, 0.0, 0.0]))
        ctrl = FakeCtrl()
        drive_field_follow(
            follow, ctrl, 0.0, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]))
        self.assertTrue(ctrl.active)
        live = np.array(follow.xyz, copy=True)
        follow.add_body_linear(np.array([0.004, 0.0, 0.0]))
        drive_field_follow(
            follow, ctrl, 0.001, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=live, stick_quiet=False)
        self.assertGreater(follow.pending_m, 0.0)
        self.assertTrue(ctrl.active)

    def test_quiet_clears_pending_and_freezes(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.004, 0.0, 0.0]))
        ctrl = FakeCtrl()
        drive_field_follow(
            follow, ctrl, 0.0, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]))
        follow.add_body_linear(np.array([0.004, 0.0, 0.0]))
        drive_field_follow(
            follow, ctrl, 0.001, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]), stick_quiet=True)
        self.assertEqual(follow.pending_m, 0.0)
        self.assertFalse(ctrl.active)

    def test_live_stick_enables_spring_without_last_target(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        ctrl = FakeCtrl()
        drive_field_follow(
            follow, ctrl, 0.0, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]), stick_quiet=False)
        self.assertTrue(ctrl.active)

    def test_quiet_freezes_even_with_tcp_leftover(self):
        """Hands-off must not keep speedJ on for a few mm leftover (retract)."""
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0, max_lead_m=0.02)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.004, 0.0, 0.0]))
        ctrl = FakeCtrl()
        drive_field_follow(
            follow, ctrl, 0.0, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]))
        follow.set_idle()
        drive_field_follow(
            follow, ctrl, 0.1, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]))
        self.assertFalse(ctrl.active)
        np.testing.assert_allclose(follow.xyz, [0.0, 0.0, 1.0], atol=1e-9)

    def test_live_leftover_hold_uses_live_R_not_field_R(self):
        vf = FakeVf()
        R_field = np.diag([1.0, -1.0, -1.0])
        follow = FieldFollow(vf, hz=50.0, max_lead_m=0.02)
        follow.seed(np.array([0.004, 0.0, 1.0]), R_field)
        follow.last_target = _motor_at([0.004, 0.0, 1.0], R_field)
        ctrl = FakeCtrl()
        drive_field_follow(
            follow, ctrl, 0.0, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]), live_R=np.eye(3),
            stick_quiet=False)
        self.assertTrue(ctrl.active)
        np.testing.assert_allclose(follow._R, np.eye(3), atol=1e-9)

    def test_live_stick_keeps_spring_on_small_leftover(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0, max_lead_m=0.02)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.004, 0.0, 0.0]))
        ctrl = FakeCtrl()
        drive_field_follow(
            follow, ctrl, 0.0, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]))
        drive_field_follow(
            follow, ctrl, 0.1, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]), stick_quiet=False)
        self.assertTrue(ctrl.active)
        self.assertAlmostEqual(float(follow.xyz[0]), 0.004, places=9)

    def test_idle_leftover_beyond_lead_does_not_yank(self):
        """A 3 cm leftover must not stay as a spring target (UR C153)."""
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0, max_lead_m=0.005)
        follow.seed(np.array([0.03, 0.0, 1.0]), np.eye(3))
        follow.last_target = _motor_at([0.03, 0.0, 1.0])
        ctrl = FakeCtrl()
        drive_field_follow(
            follow, ctrl, 0.0, _motor_at([0.03, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]))
        self.assertFalse(ctrl.active)
        np.testing.assert_allclose(follow.xyz, [0.0, 0.0, 1.0], atol=1e-9)

    def test_leftover_beyond_lead_keeps_spring_if_stick_live(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0, max_lead_m=0.005)
        follow.seed(np.array([0.03, 0.0, 1.0]), np.eye(3))
        follow.last_target = _motor_at([0.03, 0.0, 1.0])
        ctrl = FakeCtrl()
        drive_field_follow(
            follow, ctrl, 0.0, _motor_at([0.03, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]), stick_quiet=False)
        self.assertTrue(ctrl.active)
        np.testing.assert_allclose(follow.xyz, [0.0, 0.0, 1.0], atol=1e-9)

    def test_drain_cap_carries_remainder(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0, max_lead_m=0.05, max_step_m=0.003)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.010, 0.0, 0.0]))
        follow.tick(
            0.0, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(follow.xyz, [0.003, 0.0, 1.0], atol=1e-9)
        follow.tick(
            0.1, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(follow.xyz, [0.003, 0.0, 1.0], atol=1e-9)

    def test_drive_idle_stops_even_if_orientation_lags(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        ctrl = FakeCtrl()
        drive_field_follow(
            follow, ctrl, 0.0, _motor_at([0.0, 0.0, 1.0]), live_R=np.eye(3))
        self.assertTrue(ctrl.active)
        follow.set_idle()
        carrot = np.array(follow.xyz, copy=True)
        R_into = np.diag([1.0, -1.0, -1.0])
        drive_field_follow(
            follow, ctrl, 0.1, _motor_at([0.0, 0.0, 1.0], R_into),
            live_R=R_into, live_xyz=carrot)
        self.assertFalse(ctrl.active)

    def test_idle_rebases_carrot_to_live_without_project(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.05, 0.0, 0.0]))
        ctrl = FakeCtrl()
        drive_field_follow(follow, ctrl, 0.0, _motor_at([0.0, 0.0, 1.0]))
        self.assertAlmostEqual(float(follow.xyz[0]), 0.05, places=9)
        n = len(vf.projects)
        follow.set_idle()
        carrot = np.array(follow.xyz, copy=True)
        drive_field_follow(
            follow, ctrl, 0.1, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=carrot)
        self.assertEqual(len(vf.projects), n)
        np.testing.assert_allclose(follow.xyz, carrot, atol=1e-9)
        self.assertFalse(ctrl.active)

    def test_drive_does_not_project_on_idle_settle(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        ctrl = FakeCtrl()
        drive_field_follow(follow, ctrl, 0.0, _motor_at([0.0, 0.0, 1.0]))
        n = len(vf.projects)
        follow.set_idle()
        drive_field_follow(
            follow, ctrl, 0.1, _motor_at([0.0, 0.0, 1.0]), live_R=np.eye(3))
        self.assertEqual(len(vf.projects), n)

    def test_stick_step_is_from_live_tcp_not_accumulated_carrot(self):
        """Like free teleop: target = live + R @ delta, so lag cannot compound."""
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        follow.tick(
            0.0, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(follow.xyz, [0.01, 0.0, 1.0], atol=1e-9)
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        follow.tick(
            0.1, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(follow.xyz, [0.01, 0.0, 1.0], atol=1e-9)
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        follow.tick(
            0.2, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.005, 0.0, 1.0]))
        np.testing.assert_allclose(follow.xyz, [0.015, 0.0, 1.0], atol=1e-9)

    def test_tangent_stick_has_no_tool_z_step(self):
        """Contact slide must stay in the field XY plane so DualLoop owns depth."""
        from scipy.spatial.transform import Rotation

        R = Rotation.from_rotvec([0.0, 0.4, 0.0]).as_matrix()
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0)
        live = np.array([0.1, 0.2, 0.3])
        follow.seed(live, R)
        follow.add_body_linear(np.array([0.01, 0.005, 0.0]))
        follow.tick(
            0.0, _motor_at(live), live_xyz=live, live_R=R, tangent=True)
        step = follow.xyz - live
        z_hat = R[:, 2]
        self.assertAlmostEqual(float(np.dot(step, z_hat)), 0.0, places=9)
        self.assertGreater(float(np.linalg.norm(step)), 1e-6)

    def test_stick_step_uses_approach_basis_at_not_lagged_carrot(self):
        """Retract/slide axes come from the object field at the tip (viser contract)."""
        from scipy.spatial.transform import Rotation

        R_at = Rotation.from_rotvec([0.0, 0.5, 0.0]).as_matrix()
        R_stale = np.eye(3)

        class BasisVf(FakeVf):
            def approach_basis_at(self, position):
                return R_at

            def project(self, motor):
                T = np.asarray(motor.to_transformation_matrix(), dtype=float)
                T[:3, :3] = R_at
                return motor_from_matrix(T)

        follow = FieldFollow(
            BasisVf(), hz=50.0, max_dR_rad=0.0, snap_to_surface=False)
        live = np.array([0.0, 0.0, 1.0])
        follow.seed(live, R_stale)
        follow.add_body_linear(np.array([0.0, 0.0, 0.01]))
        follow.tick(0.0, _motor_at(live), live_xyz=live, live_R=np.eye(3))
        # Stick Z along field approach (= R_at[:,2]), not stale identity Z.
        np.testing.assert_allclose(
            follow.xyz, live + R_at @ np.array([0.0, 0.0, 0.01]), atol=1e-9)
        np.testing.assert_allclose(follow._R, R_at, atol=1e-9)

    def test_lead_is_capped_to_live_tcp(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0, max_lead_m=0.02)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.05, 0.0, 0.0]))
        tick = follow.tick(
            0.0, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]))
        self.assertIsNotNone(tick.target)
        np.testing.assert_allclose(follow.xyz, [0.02, 0.0, 1.0], atol=1e-9)

    def test_project_cannot_flip_tool_z_in_one_tick(self):
        class FlipVf:
            def __init__(self):
                self._smoothing = 0.0

            def project(self, motor):
                T = np.asarray(motor.to_transformation_matrix(), dtype=float)
                T[:3, :3] = np.diag([1.0, -1.0, -1.0])
                return motor_from_matrix(T)

        follow = FieldFollow(FlipVf(), hz=50.0, max_dR_rad=0.12)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        follow.tick(0.0, _motor_at([0.0, 0.0, 1.0]))
        z = follow.last_target.to_transformation_matrix()[:3, 2]
        self.assertGreater(float(z[2]), 0.9)
        err = orientation_error_rad(np.eye(3), follow._R)
        self.assertGreater(err, 0.05)
        self.assertLessEqual(err, 0.12 + 1e-9)
        np.testing.assert_allclose(follow.xyz, [0.01, 0.0, 1.0], atol=1e-9)

    def test_field_R_is_rate_limited_field_to_field(self):
        from scipy.spatial.transform import Rotation

        R_field = Rotation.from_rotvec([0.2, 0.0, 0.0]).as_matrix()

        class TiltVf:
            def project(self, motor):
                T = np.asarray(motor.to_transformation_matrix(), dtype=float)
                T[:3, :3] = R_field
                T[:3, 3] = [9.0, 9.0, 9.0]
                return motor_from_matrix(T)

        # snap_to_surface off: this case is about the R rate limit, and its vf
        # parks the projected position at [9,9,9] purely to prove position did
        # not come from project(). SurfaceSnapTest below covers the snap itself.
        follow = FieldFollow(
            TiltVf(), hz=50.0, max_dR_rad=0.02, snap_to_surface=False)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        follow.tick(0.0, _motor_at([0.0, 0.0, 1.0]))
        self.assertAlmostEqual(
            orientation_error_rad(np.eye(3), follow._R), 0.02, places=6)
        np.testing.assert_allclose(follow.xyz, [0.01, 0.0, 1.0], atol=1e-9)

    def test_field_R_snaps_full_like_orange_ghost(self):
        """Commanded R is the field at the tip, not a crawl from live wrist.

        Old live-based max_dR=0.02 pinned R_err at ~1.1° forever while the
        orange ghost already showed full field orientation.
        """
        from scipy.spatial.transform import Rotation

        R_field = Rotation.from_rotvec([0.2, 0.0, 0.0]).as_matrix()

        class TiltVf:
            def project(self, motor):
                T = np.asarray(motor.to_transformation_matrix(), dtype=float)
                T[:3, :3] = R_field
                return motor_from_matrix(T)

        follow = FieldFollow(TiltVf(), hz=50.0, max_dR_rad=0.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), R_field)
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        follow.tick(
            0.0, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]), live_R=np.eye(3))
        self.assertAlmostEqual(
            orientation_error_rad(R_field, follow._R), 0.0, places=6)
        self.assertGreater(
            orientation_error_rad(np.eye(3), follow._R), 0.15)

    def test_mapped_ee_lever_does_not_kill_tracker(self):
        """Chamfer flange lever from field R vs live R can exceed 8 cm EE.

        That skip used to freeze speedJ, so the stick could not recover.
        """
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        ctrl = FakeCtrl()

        def map_target(motor):
            T = np.asarray(motor.to_transformation_matrix(), dtype=float)
            T = T.copy()
            T[:3, 3] = T[:3, 3] + np.array([0.20, 0.0, 0.0])
            return motor_from_matrix(T)

        drive_field_follow(
            follow, ctrl, 0.0, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]),
            live_ee_xyz=np.array([0.0, 0.0, 1.0]),
            map_target=map_target, max_ee_jump_m=0.08)
        self.assertTrue(ctrl.active)
        self.assertIsNotNone(ctrl._obj.target)

    def test_invalid_project_rotation_is_reverted(self):
        class BoomVf:
            def project(self, motor):
                T = np.asarray(motor.to_transformation_matrix(), dtype=float)
                T[:3, :3] = np.eye(3) * 1.0e6
                return motor_from_matrix(T)

        follow = FieldFollow(BoomVf(), hz=50.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        tick = follow.tick(0.0, _motor_at([0.0, 0.0, 1.0]))
        self.assertTrue(tick.idle)
        np.testing.assert_allclose(follow.xyz, [0.0, 0.0, 1.0], atol=1e-9)


class SurfaceSnapTest(unittest.TestCase):
    """tick() used to keep stick_xyz wholesale and throw vf.project()'s position
    away, so the command rode live + tangential step and never re-snapped to the
    mesh -- the next project() was then queried off-surface."""

    @staticmethod
    def _offset_vf(offset):
        class OffsetVf:
            def project(self, motor):
                T = np.asarray(motor.to_transformation_matrix(), dtype=float)
                T = T.copy()
                T[:3, 3] = T[:3, 3] + np.asarray(offset, dtype=float)
                return motor_from_matrix(T)

        return OffsetVf()

    def _run(self, offset, **kwargs):
        params = dict(hz=50.0, max_lead_m=0.05)
        params.update(kwargs)
        follow = FieldFollow(self._offset_vf(offset), **params)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        follow.tick(0.0, _motor_at([0.0, 0.0, 1.0]))
        return follow

    def test_normal_offset_is_taken_from_project(self):
        # Tool Z is +Z here, so a project() 3 mm along +Z is a normal offset.
        follow = self._run([0.0, 0.0, 0.003])
        np.testing.assert_allclose(follow.xyz, [0.01, 0.0, 1.003], atol=1e-9)

    def test_tangential_offset_is_ignored(self):
        # Only the normal is corrected; tangential stays with the stick so the
        # snap can never fight the operator.
        follow = self._run([0.02, 0.0, 0.0])
        np.testing.assert_allclose(follow.xyz, [0.01, 0.0, 1.0], atol=1e-9)

    def test_correction_is_bounded_by_max_lead(self):
        # A bad project at a mesh edge must not teleport the command.
        follow = self._run([0.0, 0.0, 0.5], max_lead_m=0.02)
        np.testing.assert_allclose(follow.xyz, [0.01, 0.0, 1.02], atol=1e-9)

    def test_snap_can_be_disabled(self):
        follow = self._run([0.0, 0.0, 0.003], snap_to_surface=False)
        np.testing.assert_allclose(follow.xyz, [0.01, 0.0, 1.0], atol=1e-9)

    def test_snap_follows_the_commanded_normal_not_world_z(self):
        from scipy.spatial.transform import Rotation

        # Tool Z rotated 90 deg about Y -> +X. A project() offset along +X is
        # then the normal component, and one along +Z is tangential.
        R = Rotation.from_rotvec([0.0, np.pi / 2.0, 0.0]).as_matrix()

        class TiltOffsetVf:
            def project(self, motor):
                T = np.asarray(motor.to_transformation_matrix(), dtype=float)
                T = T.copy()
                T[:3, :3] = R
                T[:3, 3] = T[:3, 3] + np.array([0.003, 0.0, 0.004])
                return motor_from_matrix(T)

        follow = FieldFollow(
            TiltOffsetVf(), hz=50.0, max_lead_m=0.05, max_dR_rad=10.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), R)
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        follow.tick(0.0, _motor_at([0.0, 0.0, 1.0], R))
        # Body step goes through R, so the stick lands at [0, 0, 0.99]. The
        # commanded normal is now +X, so the 3 mm along X is the correction and
        # the 4 mm along Z (tangential here) is dropped -- if the snap used
        # world Z instead, Z would have picked up the 4 mm.
        np.testing.assert_allclose(follow.xyz, [0.003, 0.0, 0.99], atol=1e-9)

    def test_non_finite_project_position_still_reverts(self):
        class NanVf:
            def project(self, motor):
                T = np.asarray(motor.to_transformation_matrix(), dtype=float)
                T = T.copy()
                T[:3, 3] = np.array([np.nan, 0.0, 0.0])
                return motor_from_matrix(T)

        follow = FieldFollow(NanVf(), hz=50.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        tick = follow.tick(0.0, _motor_at([0.0, 0.0, 1.0]))
        self.assertTrue(tick.idle)
        np.testing.assert_allclose(follow.xyz, [0.0, 0.0, 1.0], atol=1e-9)


class ProjectHelperProcessTest(unittest.TestCase):
    def test_helper_returns_projected_pose(self):
        follow = FieldFollow(FakeVf(), hz=50.0, project_in_process=True)
        try:
            follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
            follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
            tick = follow.tick(0.0, _motor_at([0.0, 0.0, 1.0]))
            self.assertIsNotNone(tick.target)
            T = np.asarray(tick.target.to_transformation_matrix())
            np.testing.assert_allclose(T[:3, 3], [0.01, 0.0, 1.0], atol=1e-9)
        finally:
            follow.close()

    def test_helper_wait_lets_other_threads_run(self):
        follow = FieldFollow(SlowBusyVf(), hz=50.0, project_in_process=True)
        try:
            follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
            follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
            n = {"c": 0}
            stop = threading.Event()

            def spin():
                while not stop.is_set():
                    n["c"] += 1

            th = threading.Thread(target=spin)
            th.start()
            t0 = time.perf_counter()
            follow.tick(0.0, _motor_at([0.0, 0.0, 1.0]))
            elapsed = time.perf_counter() - t0
            stop.set()
            th.join()
            self.assertGreater(elapsed, 0.03)
            self.assertGreater(n["c"], 1000)
        finally:
            follow.close()


class FieldTcpToEeTest(unittest.TestCase):
    def test_hardware_ee_matches_ghost_flange_not_pendant_rotvec(self):
        """Sim ``t`` uses ghost_flange_world (mesh +Z = field). Follow must too.

        flange_world_from_tcp composes the pendant TCP rotvec (~120 deg) and
        was yanking hardware off the orange ghost after Enter.
        """
        T_hw = np.eye(4)
        T_hw[:3, 3] = [0.4, -0.2, 0.4]
        T_viz = np.eye(4)
        T_viz[:3, 3] = [0.4, -0.2, 0.91]
        T_off = np.eye(4)
        T_off[:3, 3] = [0.0, 0.0, 0.2]
        T_ee = T_hw @ T_off
        R_field = np.diag([1.0, -1.0, -1.0])
        p_tcp_viz = live_tcp_xyz_from_flange(T_viz)
        T_tcp = np.eye(4)
        T_tcp[:3, :3] = R_field
        T_tcp[:3, 3] = p_tcp_viz

        got = field_tcp_to_ee(
            T_tcp, T_flange_hw=T_hw, T_flange_viz=T_viz, T_ee_hw=T_ee)
        T_flange_hw_goal = ghost_flange_world(
            T_hw, R_field, live_tcp_xyz_from_flange(T_hw), 1.0)
        np.testing.assert_allclose(got, T_flange_hw_goal @ T_off, atol=1e-9)
        np.testing.assert_allclose(got[:3, :3], R_field, atol=1e-9)


class ComplianceHoldFollowTest(unittest.TestCase):
    """Push test: quiet stick must not drag the spring target onto live TCP."""

    def test_quiet_without_rebase_keeps_command_when_live_moves(self):
        vf = FakeVf()
        follow = FieldFollow(
            vf, hz=50.0, rebase_on_live=False, hold_field_orientation=True)
        live0 = np.array([0.0, 0.0, 1.0])
        follow.seed(live0, np.eye(3))
        ctrl = FakeCtrl()
        follow.add_body_linear(np.array([0.010, 0.0, 0.0]))
        drive_field_follow(
            follow, ctrl, 0.0, _motor_at(live0), live_xyz=live0,
            live_R=np.eye(3), stick_quiet=False, rebase_on_idle=True)
        commanded_x = float(follow.xyz[0])
        self.assertGreater(commanded_x, 0.005)
        pushed = np.array([0.020, 0.0, 1.0])
        drive_field_follow(
            follow, ctrl, 0.1, _motor_at(follow.xyz), live_xyz=pushed,
            live_R=np.eye(3), stick_quiet=True, rebase_on_idle=False)
        np.testing.assert_allclose(follow.xyz[0], commanded_x, atol=1e-9)
        T = np.asarray(
            ctrl.get_objective().target.to_transformation_matrix(), dtype=float)
        self.assertAlmostEqual(float(T[0, 3]), commanded_x, places=9)
        self.assertFalse(ctrl.active)

    def test_quiet_with_rebase_still_snaps_to_live(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0, hold_field_orientation=True)
        live0 = np.array([0.0, 0.0, 1.0])
        follow.seed(live0, np.eye(3))
        ctrl = FakeCtrl()
        follow.add_body_linear(np.array([0.010, 0.0, 0.0]))
        drive_field_follow(
            follow, ctrl, 0.0, _motor_at(live0), live_xyz=live0,
            live_R=np.eye(3), stick_quiet=False)
        pushed = np.array([0.020, 0.0, 1.0])
        drive_field_follow(
            follow, ctrl, 0.1, _motor_at(follow.xyz), live_xyz=pushed,
            live_R=np.eye(3), stick_quiet=True, rebase_on_idle=True)
        np.testing.assert_allclose(follow.xyz, pushed, atol=1e-9)

    def test_stick_still_drives_after_external_deflection(self):
        vf = FakeVf()
        follow = FieldFollow(
            vf, hz=50.0, rebase_on_live=False, hold_field_orientation=True)
        live0 = np.array([0.0, 0.0, 1.0])
        follow.seed(live0, np.eye(3))
        ctrl = FakeCtrl()
        follow.add_body_linear(np.array([0.010, 0.0, 0.0]))
        drive_field_follow(
            follow, ctrl, 0.0, _motor_at(live0), live_xyz=live0,
            live_R=np.eye(3), stick_quiet=False, rebase_on_idle=True)
        commanded_x = float(follow.xyz[0])
        pushed = np.array([0.020, 0.0, 1.0])
        drive_field_follow(
            follow, ctrl, 0.1, _motor_at(follow.xyz), live_xyz=pushed,
            live_R=np.eye(3), stick_quiet=True, rebase_on_idle=False)
        np.testing.assert_allclose(follow.xyz[0], commanded_x, atol=1e-9)
        follow.add_body_linear(np.array([0.005, 0.0, 0.0]))
        drive_field_follow(
            follow, ctrl, 0.2, _motor_at(follow.xyz), live_xyz=pushed,
            live_R=np.eye(3), stick_quiet=False, rebase_on_idle=False)
        self.assertTrue(ctrl.active)
        self.assertGreater(float(follow.xyz[0]), commanded_x + 0.001)

    def test_catch_up_does_not_snap_live_when_rebase_off(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0, max_lead_m=0.005, rebase_on_live=False)
        follow.seed(np.array([0.030, 0.0, 1.0]), np.eye(3))
        follow.last_target = _motor_at([0.030, 0.0, 1.0])
        ctrl = FakeCtrl()
        drive_field_follow(
            follow, ctrl, 0.0, _motor_at([0.030, 0.0, 1.0]),
            live_xyz=np.array([0.0, 0.0, 1.0]), stick_quiet=False,
            rebase_on_idle=False)
        self.assertTrue(ctrl.active)
        np.testing.assert_allclose(follow.xyz, [0.030, 0.0, 1.0], atol=1e-9)


class HardwareFollowRoutingTest(unittest.TestCase):
    def test_example_does_not_attach_vf_to_admittance(self):
        import ast
        import pathlib

        src = (
            pathlib.Path(__file__).resolve().parents[1]
            / "examples" / "hardware_object_contact_teleop.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("set_virtual_fixture", src)
        self.assertIn("FieldFollow", src)
        self.assertIn("drive_field_follow", src)
        self.assertIn("project_in_process=True", src)
        # The field->field rotation cap is a flag again, not a hardcoded 0.0:
        # 0 lets one project() flip at a mesh edge bank a large angular error
        # for the next re-entry. The cap it replaced was measured against
        # *live* R, which is a different (and worse) limit -- that one pinned
        # R_err at ~1.1 deg forever and is not coming back.
        self.assertIn("max_dR_rad=args.follow_max_dr_rad", src)
        self.assertIn("--follow-max-dr-rad", src)
        self.assertIn("hold_field_orientation=", src)
        self.assertIn("rebase_on_idle=follow_rebase_on_idle(", src)
        self.assertIn("rearm_slerp_s=", src)
        self.assertIn("max_qddot=5.0", src)
        self.assertNotIn("set_orientation_compliant", src)
        self.assertIn("live_R=", src)
        self.assertIn("pending_contact", src)
        self.assertIn("spacemouse: button", src)
        self.assertIn("field_follow_in_mode", src)
        self.assertIn("follow_linear", src)
        tree = ast.parse(src)
        self.assertTrue(any(
            isinstance(n, ast.FunctionDef) and n.name == "_make_approach_controller"
            for n in tree.body))
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertIn("field_follow_in_mode", names)
        self.assertIn("follow_linear", names)


if __name__ == "__main__":
    unittest.main()


class RebaseOnLiveTest(unittest.TestCase):
    """Where a stick step starts from decides whether force can build at all.

    speedJ re-bases the carrot on the live TCP every tick, so the command is
    only ever one drained step ahead. That is a velocity command, which is what
    the position servo underneath speedJ wants. Under the direct-torque wrench
    law the same carrot becomes ``force = k_lin * (one step)`` -- force
    proportional to stick *speed*, hard-capped at ``k_lin * max_step_m`` (12 N
    at the shipped 3000 N/m and 4 mm) and *zero* while the stick is held still
    against a stalled arm. Measured on dtc_20260828_182701.csv: |W_lin| max
    14.4 N with --max-lead-m 0.02, which therefore never bound.
    """

    def _held_stick(self, follow, n_ticks, live=(0.0, 0.0, 1.0)):
        """A stalled arm and a stick held down: live never moves."""
        live = np.asarray(live, dtype=float)
        for i in range(n_ticks):
            follow.add_body_linear(np.array([0.004, 0.0, 0.0]))
            follow.tick(0.1 * (i + 1), _motor_at(live), live_xyz=live,
                        live_R=np.eye(3))
        T = np.asarray(follow.last_target.to_transformation_matrix())
        return float(np.linalg.norm(T[:3, 3] - live))

    def test_rebasing_pins_the_error_at_one_step_however_long_you_push(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=1000.0, max_step_m=0.004, max_lead_m=0.02,
                             snap_to_surface=False)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        self.assertAlmostEqual(self._held_stick(follow, 8), 0.004, places=9)

    def test_accumulating_builds_error_up_to_the_lead(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=1000.0, max_step_m=0.004, max_lead_m=0.02,
                             snap_to_surface=False, rebase_on_live=False)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        # Eight 4 mm steps want 32 mm; _clamp_lead trims to max_lead_m, which
        # is what makes --max-lead-m the force ceiling it was documented to be.
        self.assertAlmostEqual(self._held_stick(follow, 8), 0.02, places=9)

    def test_accumulating_still_decays_once_the_arm_catches_up(self):
        """A spring, not a latch: the push falls off on its own when live moves."""
        vf = FakeVf()
        follow = FieldFollow(vf, hz=1000.0, max_step_m=0.004, max_lead_m=0.02,
                             snap_to_surface=False, rebase_on_live=False)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        self._held_stick(follow, 8)
        caught_up = np.array([0.02, 0.0, 1.0])
        follow.tick(2.0, _motor_at(caught_up), live_xyz=caught_up,
                    live_R=np.eye(3))
        T = np.asarray(follow.last_target.to_transformation_matrix())
        self.assertLess(float(np.linalg.norm(T[:3, 3] - caught_up)), 0.02)

    def test_rebasing_is_the_default_so_speedj_is_untouched(self):
        follow = FieldFollow(FakeVf(), hz=1000.0)
        self.assertTrue(follow._rebase_on_live)

    def test_the_lead_clamp_still_bounds_the_accumulated_carrot(self):
        """Without _clamp_lead, accumulating would run away with the stick."""
        vf = FakeVf()
        follow = FieldFollow(vf, hz=1000.0, max_step_m=0.004, max_lead_m=0.006,
                             snap_to_surface=False, rebase_on_live=False)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        self.assertAlmostEqual(self._held_stick(follow, 20), 0.006, places=9)


class PendingBacklogTest(unittest.TestCase):
    """The SpaceMouse -> FieldFollow queue and its (opt-in) bound.

    add_body_linear is called once per poll and _drain removes at most
    max_step_m per follow tick, and the two rates do not match: 50 Hz * 1.5 mm
    feeds up to 130 mm/s on a three-axis diagonal against a 66.7 mm/s drain.
    The surplus is carried in _pending, so the command falls behind the
    operator for as long as the stick is held.
    """

    @staticmethod
    def _follow(**kw):
        class _VF:
            def project(self, motor):
                return motor

            def approach_basis_at(self, p):
                return np.eye(3)

        return FieldFollow(_VF(), hz=20.0, max_step_m=0.004, max_lead_m=0.05,
                           rebase_on_live=False, snap_to_surface=False, **kw)

    @staticmethod
    def _motor_at(xyz):
        T = np.eye(4)
        T[:3, 3] = xyz
        return ga.Motor.from_matrix(T)

    def _hold_stick(self, follow, axes, seconds=5.0, poll_hz=50.0):
        lin = np.asarray(axes, dtype=float) * 0.0015  # STICK_LINEAR_SCALE
        live = np.zeros(3)
        t, dt, last = 0.0, 1.0 / poll_hz, -1e9
        for _ in range(int(seconds * poll_hz)):
            follow.add_body_linear(lin)
            if t - last >= 1.0 / 20.0 - 1e-12:
                follow.tick(t, self._motor_at(live), live_xyz=live)
                last = t
            t += dt
        return follow.pending_m

    def test_unbounded_queue_backlogs_on_a_held_diagonal(self):
        """Documents the default. 0 = the historical unbounded queue."""
        backlog = self._hold_stick(self._follow(), [1.0, 1.0, 0.0])
        self.assertGreater(backlog, 0.05,
                           "a 5 s two-axis hold should queue >50 mm at the "
                           "defaults; if this drops, the feed/drain rates "
                           "changed and the comment in __init__ is stale")

    def test_max_pending_m_bounds_the_backlog(self):
        backlog = self._hold_stick(
            self._follow(max_pending_m=0.02), [1.0, 1.0, 1.0])
        self.assertLessEqual(backlog, 0.02 + 1e-9)

    def test_the_bound_preserves_direction(self):
        follow = self._follow(max_pending_m=0.002)
        for _ in range(50):
            follow.add_body_linear(np.array([0.003, 0.004, 0.0]))
        pending = follow._pending
        self.assertAlmostEqual(float(np.linalg.norm(pending)), 0.002)
        # Same unit vector as the input, only shorter.
        np.testing.assert_allclose(
            pending / np.linalg.norm(pending), [0.6, 0.8, 0.0], atol=1e-9)

    def test_zero_cap_is_the_historical_path(self):
        a = self._hold_stick(self._follow(), [1.0, 0.0, 0.0])
        b = self._hold_stick(self._follow(max_pending_m=0.0), [1.0, 0.0, 0.0])
        self.assertAlmostEqual(a, b)


class QuietTargetResendTest(unittest.TestCase):
    """--no-follow-target-resend: skip the redundant cross-thread write.

    While the stick is quiet the mapped target is constant (map_target is
    pose-invariant), so re-writing it 50x a second to the objective the 200 Hz
    control loop is reading is pure redundancy. This is the bisection for the
    30 Hz burst; default True preserves every recorded log's behavior.
    """

    class _Obj:
        def __init__(self):
            self.n = 0

        def set_target(self, motor):
            self.n += 1

    class _Ctl:
        def __init__(self, obj):
            self._obj = obj
            self.active = None

        def get_objective(self):
            return self._obj

        def set_input_active(self, a):
            self.active = a

    @staticmethod
    def _motor(x=0.0):
        T = np.eye(4)
        T[:3, 3] = [x, 0.0, 0.0]
        return ga.Motor.from_matrix(T)

    def _drive(self, resend, n=10, moving=False):
        class _VF:
            def project(self, m):
                return m

            def approach_basis_at(self, p):
                return np.eye(3)

        follow = FieldFollow(_VF(), hz=20.0, max_step_m=0.004,
                             rebase_on_live=False, snap_to_surface=False)
        follow.last_target = self._motor()
        obj = self._Obj()
        ctl = self._Ctl(obj)
        for i in range(n):
            tgt = self._motor(0.001 * i if moving else 0.0)
            drive_field_follow(
                follow, ctl, 100.0 + i, self._motor(), stick_quiet=True,
                map_target=lambda _m, t=tgt: t, resend_target=resend)
        return obj.n

    def test_default_resends_every_call(self):
        self.assertEqual(self._drive(resend=True, n=10), 10)

    def test_opting_out_writes_an_unchanged_target_once(self):
        self.assertEqual(self._drive(resend=False, n=10), 1)

    def test_opting_out_still_writes_when_the_target_moves(self):
        """It must skip only redundancy, never an actual command."""
        self.assertEqual(self._drive(resend=False, n=10, moving=True), 10)
