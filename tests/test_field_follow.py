import unittest
import threading
import time

import gafropy as ga
import numpy as np

from geodude_teleoperation.ee_viz import (
    ghost_flange_world,
    live_tcp_xyz_from_flange,
    motor_from_matrix,
)
from geodude_teleoperation.field_follow import (
    FieldFollow,
    drive_field_follow,
    field_tcp_to_ee,
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
        R_into = np.diag([1.0, -1.0, -1.0])
        drive_field_follow(
            follow, ctrl, 0.1, _motor_at([0.0, 0.0, 1.0], R_into),
            live_R=R_into, live_xyz=np.array([0.0, 0.0, 1.0]))
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
        drive_field_follow(
            follow, ctrl, 0.1, _motor_at([0.0, 0.0, 1.0]),
            live_xyz=np.array([0.01, 0.0, 1.0]))
        self.assertEqual(len(vf.projects), n)
        np.testing.assert_allclose(follow.xyz, [0.01, 0.0, 1.0], atol=1e-9)
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

    def test_drive_skips_huge_ee_jump(self):
        vf = FakeVf()
        follow = FieldFollow(vf, hz=50.0)
        follow.seed(np.array([0.0, 0.0, 1.0]), np.eye(3))
        follow.add_body_linear(np.array([0.01, 0.0, 0.0]))
        ctrl = FakeCtrl()
        drive_field_follow(
            follow, ctrl, 0.0, _motor_at([0.0, 0.0, 1.0]),
            live_ee_xyz=np.array([0.0, 0.0, 1.0]), max_ee_jump_m=0.005)
        self.assertFalse(ctrl.active)


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
        tree = ast.parse(src)
        self.assertTrue(any(
            isinstance(n, ast.FunctionDef) and n.name == "_make_approach_controller"
            for n in tree.body))


if __name__ == "__main__":
    unittest.main()
