import unittest

import gafropy as ga
import numpy as np

from geodude_teleoperation.force_mode_contact import (
    ForceModeContactSession,
    _force_along_task_z,
    twist_to_speedl_base,
    _pose_rotation,
)
from geodude_teleoperation.spacemouse import is_idle_twist


class ForceModeContactHelpersTest(unittest.TestCase):
    def test_identity_pose_maps_tool_xy(self):
        pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v = twist_to_speedl_base(
            [0.01, 0.0, 0.05], [0.0, 0.0, 0.0], pose, max_linear=1.0)
        np.testing.assert_allclose(v[:3], [0.01, 0.0, 0.05], atol=1e-9)

    def test_tool_z_zeroed_by_session_convention(self):
        pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
        lin = np.array([0.02, -0.01, 0.08])
        lin[2] = 0.0
        v = twist_to_speedl_base(lin, [0.0, 0.0, 0.0], pose, max_linear=1.0)
        np.testing.assert_allclose(v[2], 0.0, atol=1e-9)
        np.testing.assert_allclose(v[:2], [0.02, -0.01], atol=1e-9)

    def test_clamp_linear(self):
        pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v = twist_to_speedl_base([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], pose, max_linear=0.02)
        self.assertAlmostEqual(float(np.linalg.norm(v[:3])), 0.02, places=6)

    def test_pose_rotation_normalized(self):
        R = _pose_rotation([0, 0, 0, 0, 0, np.pi / 2])
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=6)

    def test_force_along_task_z_identity(self):
        task = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        tcp_force = [0.0, 0.0, -5.0, 0.0, 0.0, 0.0]
        fz = _force_along_task_z(tcp_force, task, task)
        self.assertAlmostEqual(fz, -5.0, places=6)

    def test_task_z_stripped_from_speedl(self):
        # Tool +X maps partly onto base Z when tool is tilted 90° about Y.
        pose = [0.0, 0.0, 0.0, 0.0, np.pi / 2, 0.0]
        task = pose
        v = twist_to_speedl_base(
            [0.01, 0.0, 0.0], [0.0, 0.0, 0.0], pose,
            max_linear=1.0, task_frame=task)
        z_task = _pose_rotation(task)[:, 2]
        self.assertAlmostEqual(float(np.dot(v[:3], z_task)), 0.0, places=6)

    def test_seek_defaults_match_force_mode_controller_example(self):
        from geodude_teleoperation.force_mode_contact import (
            DEFAULT_DAMPING, DEFAULT_GAIN_SCALING)

        # ur_robot_driver examples/force_mode.py: damping 0.1, gain 0.8.
        self.assertAlmostEqual(DEFAULT_DAMPING, 0.1)
        self.assertGreaterEqual(DEFAULT_GAIN_SCALING, 0.5)
        self.assertLessEqual(DEFAULT_GAIN_SCALING, 1.0)

    def test_limits_pack_like_force_mode_controller(self):
        from geodude_teleoperation.force_mode_contact import (
            DEFAULT_LIMITS, pack_force_mode_limits)

        packed = pack_force_mode_limits(
            [0, 0, 1, 0, 0, 0],
            [0.0, 0.0, 0.005, 0.0, 0.0, 0.0],
            [0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
        )
        self.assertEqual(packed, [0.005, 0.005, 0.005, 0.005, 0.005, 0.005])
        self.assertEqual(DEFAULT_LIMITS[2], 0.005)

    def test_hold_script_recalls_force_mode_every_cycle(self):
        from geodude_teleoperation.force_mode_contact import build_force_hold_script

        script = build_force_hold_script(
            wrench=[0, 0, 15, 0, 0, 0],
            limits=[0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
            damping=0.1,
            gain_scaling=0.8,
            zero_ft_sensor=True,
        )
        self.assertIn("force_mode(p0", script)
        self.assertIn("while True:", script)
        self.assertIn("sync()", script)
        self.assertNotIn("speedl", script.lower())
        self.assertNotIn("movel", script.lower())
        self.assertGreater(script.index("force_mode(p0"), script.index("while True:"))

    def test_clear_twist_zeros_sticky_command(self):
        session = ForceModeContactSession("127.0.0.1")
        session.set_twist([0.02, 0.01, 0.5], [0.1, 0.0, 0.0])
        session.clear_twist()
        with session._lock:
            np.testing.assert_allclose(session._cmd_linear, 0.0)
            np.testing.assert_allclose(session._cmd_angular, 0.0)

    def test_seek_positive_error_moves_task_plus_z(self):
        from geodude_teleoperation.force_mode_contact import seek_speedl_task_z

        pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v = seek_speedl_task_z(15.0, 1.0, pose, kp=0.0006, vmax=0.008, deadband_n=1.0)
        self.assertGreater(float(v[2]), 0.0)
        self.assertAlmostEqual(float(v[2]), 0.008, places=6)

    def test_seek_deadband_zeros_when_close(self):
        from geodude_teleoperation.force_mode_contact import seek_speedl_task_z

        pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v = seek_speedl_task_z(15.0, 14.5, pose, deadband_n=1.0)
        np.testing.assert_allclose(v, 0.0)

    def test_press_pose_moves_along_plus_z(self):
        from geodude_teleoperation.force_mode_contact import pose_offset_along_task_z

        pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
        out = pose_offset_along_task_z(pose, 0.012)
        np.testing.assert_allclose(out[:3], [0.1, 0.2, 0.312], atol=1e-9)
        np.testing.assert_allclose(out[3:], pose[3:])

    def test_z_limit_is_slow_seek_not_slam(self):
        from geodude_teleoperation.force_mode_contact import DEFAULT_LIMITS

        # Compliant-axis LIMITS[2] is max TCP speed, not Fd. 0.05 was 50 mm/s.
        self.assertGreaterEqual(DEFAULT_LIMITS[2], 0.003)
        self.assertLessEqual(DEFAULT_LIMITS[2], 0.005)

    def test_dashboard_stop_sends_stop(self):
        from unittest.mock import MagicMock, patch
        from geodude_teleoperation.force_mode_contact import dashboard_stop_program

        sock = MagicMock()
        sock.recv.side_effect = [b"Connected\n", b"Stopped\n"]
        with patch("socket.create_connection", return_value=sock):
            dashboard_stop_program("127.0.0.1")
        sent = b"".join(call.args[0] for call in sock.sendall.call_args_list)
        self.assertIn(b"stop", sent)

    def test_wait_runtime_state_matches(self):
        from geodude_teleoperation.force_mode_contact import (
            RUNTIME_PLAYING, wait_runtime_state)

        class _Fake:
            def getRuntimeState(self):
                return RUNTIME_PLAYING

        self.assertEqual(wait_runtime_state(_Fake(), RUNTIME_PLAYING, 0.2), RUNTIME_PLAYING)

    def test_idle_rest_bias_does_not_count_as_motion(self):
        rest_yaw = ga.Twist.from_array(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertTrue(is_idle_twist(rest_yaw))


class ContactInputRoutingTest(unittest.TestCase):
    """Mirrors example on_twist / on_button contracts without hardware."""

    def test_idle_clears_force_session_not_sticky(self):
        session = ForceModeContactSession("127.0.0.1")
        session._active = True
        session.set_twist([0.02, 0.0, 0.0], [0.0, 0.0, 0.0])

        def on_twist(twist):
            if is_idle_twist(twist):
                if session.active:
                    session.clear_twist()
                return
            session.set_twist(twist.get_linear(), twist.get_angular())

        on_twist(ga.Twist.from_array(np.zeros(6)))
        with session._lock:
            np.testing.assert_allclose(session._cmd_linear, 0.0)

    def test_twist_never_triggers_handoff(self):
        handoffs = []

        def on_twist(twist):
            if is_idle_twist(twist):
                return

        def on_button(index):
            if index == 0:
                handoffs.append("toggle")

        on_twist(ga.Twist.from_array(np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0])))
        self.assertEqual(handoffs, [])
        on_button(0)
        self.assertEqual(handoffs, ["toggle"])
        on_button(1)
        self.assertEqual(handoffs, ["toggle"])


if __name__ == "__main__":
    unittest.main()
