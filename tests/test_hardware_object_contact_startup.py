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


    def test_default_startup_align_is_on(self):
        mod = _load_example()
        args = mod.build_arg_parser().parse_args([])
        self.assertTrue(args.startup_align)
        self.assertIsNone(args.align_delay)
        self.assertGreaterEqual(args.align_duration, 2.0)

    def test_snap_ghost_is_tcp_held_field_pose(self):
        from geodude_teleoperation.ee_viz import (
            CHAMFER_TCP_FROM_FLANGE_XYZ,
            ghost_flange_world,
            live_tcp_xyz_from_flange,
        )

        T0 = np.eye(4)
        T0[:3, 3] = [0.4, -0.2, 0.9]
        R_field = np.diag([1.0, -1.0, -1.0])
        p_tcp = live_tcp_xyz_from_flange(T0)
        ghost = ghost_flange_world(T0, R_field, p_tcp, 1.0)
        np.testing.assert_allclose(ghost[:3, :3], R_field, atol=1e-9)
        np.testing.assert_allclose(
            ghost[:3, 3] + ghost[:3, :3] @ CHAMFER_TCP_FROM_FLANGE_XYZ,
            p_tcp, atol=1e-9)
        self.assertFalse(np.allclose(ghost[:3, 3], T0[:3, 3]))

    def test_drive_startup_align_enables_spring_then_holds(self):
        from geodude_teleoperation.ee_viz import StartupAlign

        mod = _load_example()

        class Obj:
            def __init__(self):
                self.target = None

            def set_target(self, motor):
                self.target = motor

        class Ctrl:
            def __init__(self):
                self.active = False
                self.held = False
                self._obj = Obj()

            def get_objective(self):
                return self._obj

            def set_input_active(self, active):
                self.active = active

            def hold_live_pose(self):
                self.held = True

        T0 = np.eye(4)
        T0[:3, 3] = [0.0, 0.0, 1.0]
        R_field = np.diag([1.0, -1.0, -1.0])
        align = StartupAlign(delay_s=0.5, duration_s=0.5, err_rad=0.05, settle_s=0.1)
        align.arm(0.0)
        ctrl = Ctrl()
        wait = mod.drive_startup_align(align, ctrl, 0.1, T0, R_field, T0)
        self.assertEqual(wait.phase, "wait")
        self.assertFalse(ctrl.active)
        ramp = mod.drive_startup_align(align, ctrl, 0.5, T0, R_field, T0)
        self.assertTrue(ctrl.active)
        self.assertIsNotNone(ctrl._obj.target)
        mod.drive_startup_align(align, ctrl, 1.0, T0, R_field, T0)
        T_goal = np.eye(4)
        T_goal[:3, :3] = R_field
        T_goal[:3, 3] = T0[:3, 3]
        done = mod.drive_startup_align(align, ctrl, 1.2, T_goal, R_field, T_goal)
        self.assertTrue(done.finished)
        self.assertFalse(ctrl.active)
        self.assertTrue(ctrl.held)

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

    def test_viser_preview_twist_does_not_call_hardware(self):
        from geodude_teleoperation.ee_viz import SimTcpProbe

        mod = _load_example()
        probe = SimTcpProbe([0.0, 0.0, 1.0])
        hardware = []
        twist = ga.Twist.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.05]))
        consumed = mod.route_spacemouse_twist(
            twist, sim_probe=probe, on_hardware=hardware.append)
        self.assertTrue(consumed)
        self.assertEqual(hardware, [])
        np.testing.assert_allclose(probe.step(np.eye(3)), [0.0, 0.0, 1.05])

    def test_headless_twist_goes_to_hardware(self):
        mod = _load_example()
        hardware = []
        twist = ga.Twist.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.05]))
        consumed = mod.route_spacemouse_twist(
            twist, sim_probe=None, on_hardware=hardware.append)
        self.assertFalse(consumed)
        self.assertEqual(hardware, [twist])

    def test_t_toggles_sim_teleop_until_enter_aligns(self):
        from geodude_teleoperation.ee_viz import StartupAlign

        mod = _load_example()
        align = StartupAlign(delay_s=None, duration_s=1.0)
        sim = {"on": False}
        self.assertEqual(
            mod.apply_terminal_key("t", align=align, sim_teleop=sim), "sim_on")
        self.assertTrue(sim["on"])
        self.assertEqual(
            mod.apply_terminal_key("t", align=align, sim_teleop=sim), "sim_off")
        self.assertFalse(sim["on"])
        sim["on"] = True
        self.assertEqual(
            mod.apply_terminal_key("\n", align=align, sim_teleop=sim), "align")
        self.assertFalse(sim["on"])
        self.assertTrue(align._triggered)
        self.assertIsNone(
            mod.apply_terminal_key("t", align=align, sim_teleop=sim))

    def test_sim_teleop_twist_reaches_probe_before_align(self):
        from geodude_teleoperation.ee_viz import SimTcpProbe, StartupAlign

        probe = SimTcpProbe([0.0, 0.0, 1.0])
        hardware = []
        align = StartupAlign(delay_s=None, duration_s=1.0)
        sim = {"on": True}
        twist = ga.Twist.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.05]))
        # Same gate as on_twist while waiting: stick only if sim teleop is on.
        if align is not None and not align.finished and sim["on"]:
            mod = _load_example()
            mod.route_spacemouse_twist(
                twist, sim_probe=probe, on_hardware=hardware.append)
        self.assertEqual(hardware, [])
        np.testing.assert_allclose(probe.step(np.eye(3)), [0.0, 0.0, 1.05])


if __name__ == "__main__":
    unittest.main()
