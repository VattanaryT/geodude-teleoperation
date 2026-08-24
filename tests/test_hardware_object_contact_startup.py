import importlib.util
import pathlib
import unittest

import gafropy as ga
import numpy as np

_EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "hardware_object_contact_teleop.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("hardware_object_contact_teleop", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HardwareObjectContactStartupTest(unittest.TestCase):
    def test_idle_twist_is_idle(self):
        mod = _load_example()
        self.assertTrue(mod.is_idle_twist(ga.Twist.from_array(np.zeros(6))))

    def test_spacemouse_rest_bias_is_idle(self):
        """Untouched SpaceMouse rest often sits just above the 0.05 raw deadzone.

        Default angular_scale is 0.1, so 0.1 raw yaw becomes a 0.01 angular twist.
        That must still count as hands-off: approach mode's virtual fixture replaces
        orientation on every update_command, so forwarding rest bias servos the wrist.
        """
        mod = _load_example()
        rest_yaw = ga.Twist.from_array(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
        rest_linear = ga.Twist.from_array(np.array([0.0, 0.0, 0.0, 0.001, 0.0, 0.0]))
        self.assertTrue(mod.is_idle_twist(rest_yaw))
        self.assertTrue(mod.is_idle_twist(rest_linear))

    def test_nonzero_twist_is_not_idle(self):
        mod = _load_example()
        twist = ga.Twist.from_array(np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0]))
        self.assertFalse(mod.is_idle_twist(twist))

    def test_deliberate_angular_command_is_not_idle(self):
        mod = _load_example()
        twist = ga.Twist.from_array(np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertFalse(mod.is_idle_twist(twist))


    def test_default_contact_backend_is_speedj(self):
        mod = _load_example()
        args = mod.build_arg_parser().parse_args([])
        self.assertEqual(args.contact_backend, "speedj")

    def test_contact_pid_force_limit_covers_target(self):
        from geodude_teleoperation.controller import contact_pid_force_limit

        self.assertGreaterEqual(contact_pid_force_limit(15.0), 15.0)
        self.assertGreater(contact_pid_force_limit(15.0), 10.0)


    def test_pose_spring_z_compliant_zeros_kz(self):
        from geodude_teleoperation.controller import AdmittanceTeleoperationController

        ctrl = AdmittanceTeleoperationController(stiffness=np.diag([2000.0] * 6))
        ctrl.set_z_compliant(True)
        self.assertEqual(ctrl._stiffness[5, 5], 0.0)
        ctrl.set_z_compliant(False)
        self.assertEqual(ctrl._stiffness[5, 5], 2000.0)

    def test_contact_slide_is_capped_xy_velocity_not_pose_spring(self):
        from geodude_teleoperation.controller import AdmittanceTeleoperationController

        ctrl = AdmittanceTeleoperationController(
            stiffness=np.diag([2000.0] * 6), twist_linear_scale=1.0)
        ctrl.set_z_compliant(True)
        # Blade order [tx,ty,tz,fx,fy,fz]; huge X stick must not become a pose increment.
        twist = ga.Twist.from_array(np.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.5]))
        ctrl.update_command(twist)
        slide = ctrl.contact_slide_twist()
        self.assertEqual(slide[5], 0.0)
        self.assertAlmostEqual(np.linalg.norm(slide[3:5]), 0.015, places=6)
        self.assertGreater(abs(slide[3]), 0.0)

    def test_slide_suppresses_press_keeps_retract(self):
        # Live: Fs 8→22 N with v_z=+2 mm/s while sliding, then C153.
        from geodude_teleoperation.controller import AdmittanceTeleoperationController

        ctrl = AdmittanceTeleoperationController(stiffness=np.diag([2000.0] * 6))
        ctrl.set_z_compliant(True)
        twist = ga.Twist.from_array(np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0]))
        ctrl.update_command(twist)
        merged = ctrl.merged_contact_twist(0.002)
        self.assertEqual(merged[5], 0.0)
        self.assertGreater(abs(merged[3]), 0.0)
        retract = ctrl.merged_contact_twist(-0.001)
        self.assertLess(retract[5], 0.0)
        ctrl.set_input_active(False)
        idle = ctrl.merged_contact_twist(0.002)
        np.testing.assert_allclose(idle[5], 0.002)

    def test_speedj_contact_keeps_pose_plant(self):
        """Button 0 must not switch to C++ line admittance (that shake in air)."""
        mod = _load_example()

        class Iface:
            def __init__(self):
                self.ctrl = "approach"

            def control(self, controller, blocking=False):
                self.ctrl = controller

        class Approach:
            def __init__(self):
                self.vz = 0.0
                self.held = False
                self.active = True
                self.z_compliant = False

            def hold_live_pose(self):
                self.held = True

            def set_input_active(self, active):
                self.active = active

            def set_tool_z_velocity(self, vz):
                self.vz = vz

            def set_z_compliant(self, compliant):
                self.z_compliant = compliant

        class Contact:
            def __init__(self):
                self.started = False
                self.on_vz = None

            def set_on_vz(self, cb):
                self.on_vz = cb

            def start(self):
                self.started = True

            def stop(self):
                self.started = False

        iface = Iface()
        approach = Approach()
        contact = Contact()
        mode = {"active": "approach"}
        mod.enter_speedj_contact(
            mode=mode, contact_controller=contact,
            approach_controller=approach, interface=iface)
        self.assertIs(iface.ctrl, "approach")
        self.assertEqual(mode["active"], "contact")
        self.assertTrue(contact.started)
        self.assertEqual(contact.on_vz, approach.set_tool_z_velocity)
        self.assertTrue(approach.z_compliant)
        self.assertFalse(approach.active)
        self.assertTrue(approach.held)

        mod.exit_speedj_contact(
            mode=mode, contact_controller=contact,
            approach_controller=approach, interface=iface)
        self.assertIs(iface.ctrl, "approach")
        self.assertEqual(mode["active"], "approach")
        self.assertFalse(contact.started)
        self.assertEqual(approach.vz, 0.0)
        self.assertFalse(approach.z_compliant)


if __name__ == "__main__":
    unittest.main()
