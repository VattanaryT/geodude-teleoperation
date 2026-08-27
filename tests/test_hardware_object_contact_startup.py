import importlib.util
import pathlib
import unittest

import gafropy as ga
import numpy as np

_EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "hardware_object_contact_teleop.py"


class _VelState:
    def __init__(self, n=6):
        self._v = np.zeros(n)

    def get_velocity(self):
        return self._v


class _ObjState:
    def __init__(self, error):
        self._error = np.asarray(error, dtype=float)

    def get_error(self):
        return self._error

    def get_error_derivative(self):
        return np.zeros_like(self._error)


class _TaskState:
    def get_jacobian(self):
        return np.eye(6)

    def get_jacobian_derivative(self):
        return np.zeros((6, 6))


class _Iface:
    def get_state(self):
        return self

    def get_position(self):
        return np.zeros(6)


class _TaskSpace:
    def get_joint_indices(self):
        return [0, 1, 2, 3, 4, 5]

    def compute_ee_motor(self, position):
        del position
        return ga.Motor()


class _Objective:
    def __init__(self):
        self.target = None

    def set_target(self, motor):
        self.target = motor


def _pose_ctrl(**kwargs):
    from geodude_teleoperation.controller import AdmittanceTeleoperationController

    params = dict(
        stiffness=np.diag([2000.0] * 6),
        damping=np.diag([20.0] * 6),
        inertia=np.diag([1.0] * 6),
        dt=0.001,
        max_qddot=1e6,
        twist_linear_scale=1.0,
    )
    params.update(kwargs)
    ctrl = AdmittanceTeleoperationController(**params)
    obj = _Objective()
    ctrl.set_interface(_Iface())
    ctrl.get_task_space = lambda: _TaskSpace()
    ctrl.get_objective = lambda: obj
    return ctrl


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

    def test_gentle_stick_z_is_not_idle_at_device_scale(self):
        """Lift is often ~30% deflection; default idle cap ate it as rest."""
        mod = _load_example()
        z = 0.3 * mod.STICK_LINEAR_SCALE
        twist = ga.Twist.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, z]))
        self.assertTrue(mod.is_idle_twist(twist))
        self.assertFalse(mod.stick_is_idle(twist))

    def test_slow_creep_still_has_linear_for_follow(self):
        """10% deflection is idle under the rest cap, but must still field-follow."""
        mod = _load_example()
        z = 0.1 * mod.STICK_LINEAR_SCALE
        twist = ga.Twist.from_array(np.array([0.0, 0.0, 0.0, 0.0, 0.0, z]))
        self.assertTrue(mod.stick_is_idle(twist))
        self.assertTrue(mod.stick_has_linear(twist))

    def test_contact_follow_linear_drops_stick_z(self):
        # Live: contact pose-stepped without project(); TCP walked off the chamfer.
        mod = _load_example()
        lin = np.array([0.001, -0.002, 0.004])
        got = mod.follow_linear(lin, ignore_z=True)
        np.testing.assert_allclose(got, [0.001, -0.002, 0.0])
        np.testing.assert_allclose(mod.follow_linear(lin, ignore_z=False), lin)

    def test_speedj_contact_still_field_follows(self):
        mod = _load_example()
        self.assertTrue(mod.field_follow_in_mode("approach", "speedj"))
        self.assertTrue(mod.field_follow_in_mode("contact", "speedj"))
        self.assertTrue(mod.field_follow_in_mode("approach", "force_mode"))
        self.assertFalse(mod.field_follow_in_mode("contact", "force_mode"))

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

    def test_default_follow_hz_is_between_15_and_50(self):
        mod = _load_example()
        args = mod.build_arg_parser().parse_args([])
        self.assertGreaterEqual(args.follow_hz, 15.0)
        self.assertLessEqual(args.follow_hz, 50.0)

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

    def test_default_speedj_plot_is_on(self):
        mod = _load_example()
        args = mod.build_arg_parser().parse_args([])
        self.assertTrue(args.speedj_plot)
        args = mod.build_arg_parser().parse_args(["--no-speedj-plot"])
        self.assertFalse(args.speedj_plot)

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

    def test_pose_spring_ori_compliant_zeros_angular_k(self):
        from geodude_teleoperation.controller import AdmittanceTeleoperationController

        ctrl = AdmittanceTeleoperationController(stiffness=np.diag([2000.0] * 6))
        ctrl.set_orientation_compliant(True)
        self.assertEqual(ctrl._stiffness[0, 0], 0.0)
        self.assertEqual(ctrl._stiffness[1, 1], 0.0)
        self.assertEqual(ctrl._stiffness[2, 2], 0.0)
        self.assertEqual(ctrl._stiffness[3, 3], 2000.0)
        ctrl.set_orientation_compliant(False)
        self.assertEqual(ctrl._stiffness[0, 0], 2000.0)

    def test_z_compliant_keeps_pose_spring_xy(self):
        # hardware_free_teleop plant: stick XY is a pose error, not a Cartesian twist.
        # Live C153: z_compliant replaced the spring with J# * vz (5 mm/s ram).
        ctrl = _pose_ctrl()
        ctrl.set_z_compliant(True)
        ctrl.set_input_active(True)
        error = np.zeros(6)
        error[3] = 0.01
        qdot = ctrl.compute_control_command(
            _VelState(), _ObjState(error), _TaskState())
        self.assertGreater(abs(float(qdot[3])), 1e-4)
        self.assertAlmostEqual(float(qdot[5]), 0.0, places=6)

    def test_lift_off_fixes_default_on_and_are_all_switchable(self):
        # Each fix must be independently defeatable so a hardware session can
        # bisect which one actually stops the lift-off.
        mod = _load_example()
        args = mod.build_arg_parser().parse_args([])
        self.assertEqual(args.press_loop, "dtc")
        self.assertTrue(args.free_along_normal)
        self.assertTrue(args.snap_follow_to_surface)

        off = mod.build_arg_parser().parse_args([
            "--press-loop", "legacy",
            "--no-free-along-normal",
            "--no-snap-follow-to-surface",
        ])
        self.assertEqual(off.press_loop, "legacy")
        self.assertFalse(off.free_along_normal)
        self.assertFalse(off.snap_follow_to_surface)

    def test_free_along_normal_projects_the_spring_off_the_normal(self):
        # The lift-off fix. stiffness[5,5]=0 only frees the body Z of the pose
        # error; field-follow slerps tool Z behind the field normal on purpose,
        # so mid-slide the spring pushes along the true normal and outruns the
        # press. Jacobian is identity here, so qdot[3:6] is the task linear.
        ctrl = _pose_ctrl()
        ctrl.set_z_compliant(True)
        ctrl.set_input_active(True)
        error = np.zeros(6)
        error[3] = 0.01  # linear X
        before = ctrl.compute_control_command(
            _VelState(), _ObjState(error), _TaskState())
        self.assertGreater(abs(float(before[3])), 1e-4)

        ctrl = _pose_ctrl()
        ctrl.set_z_compliant(True)
        ctrl.set_input_active(True)
        ctrl.set_contact_normal([1.0, 0.0, 0.0], np.eye(3))
        after = ctrl.compute_control_command(
            _VelState(), _ObjState(error), _TaskState())
        self.assertAlmostEqual(float(after[3]), 0.0, places=9)

    def test_free_along_normal_keeps_the_tangential_spring(self):
        # Only the normal component goes; sliding must still be position-held.
        ctrl = _pose_ctrl()
        ctrl.set_z_compliant(True)
        ctrl.set_input_active(True)
        ctrl.set_contact_normal([1.0, 0.0, 0.0], np.eye(3))
        error = np.zeros(6)
        error[3] = 0.01  # along the normal -> removed
        error[4] = 0.01  # tangential -> kept
        qdot = ctrl.compute_control_command(
            _VelState(), _ObjState(error), _TaskState())
        self.assertAlmostEqual(float(qdot[3]), 0.0, places=9)
        self.assertGreater(abs(float(qdot[4])), 1e-4)

    def test_free_along_normal_handles_a_tilted_normal(self):
        ctrl = _pose_ctrl()
        ctrl.set_z_compliant(True)
        ctrl.set_input_active(True)
        n = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        ctrl.set_contact_normal(n, np.eye(3))
        error = np.zeros(6)
        error[3] = 0.01
        error[4] = 0.01
        qdot = ctrl.compute_control_command(
            _VelState(), _ObjState(error), _TaskState())
        self.assertAlmostEqual(float(np.asarray(qdot)[3:6] @ n), 0.0, places=9)

    def test_free_space_is_untouched_by_the_normal(self):
        # Not z_compliant: the projection must not run at all.
        ctrl = _pose_ctrl()
        ctrl.set_input_active(True)
        ctrl.set_contact_normal([1.0, 0.0, 0.0], np.eye(3))
        error = np.zeros(6)
        error[3] = 0.01
        qdot = ctrl.compute_control_command(
            _VelState(), _ObjState(error), _TaskState())
        self.assertGreater(abs(float(qdot[3])), 1e-4)

    def test_leaving_contact_clears_the_normal(self):
        ctrl = _pose_ctrl()
        ctrl.set_z_compliant(True)
        ctrl.set_contact_normal([1.0, 0.0, 0.0], np.eye(3))
        self.assertIsNotNone(ctrl._n_body)
        ctrl.set_z_compliant(False)
        self.assertIsNone(ctrl._n_body)

    def test_free_along_normal_can_be_disabled(self):
        ctrl = _pose_ctrl(free_along_normal=False)
        ctrl.set_z_compliant(True)
        ctrl.set_input_active(True)
        ctrl.set_contact_normal([1.0, 0.0, 0.0], np.eye(3))
        error = np.zeros(6)
        error[3] = 0.01
        qdot = ctrl.compute_control_command(
            _VelState(), _ObjState(error), _TaskState())
        self.assertGreater(abs(float(qdot[3])), 1e-4)

    def test_z_compliant_idle_overlays_tool_z(self):
        ctrl = _pose_ctrl()
        ctrl.set_z_compliant(True)
        ctrl.set_input_active(False)
        ctrl.set_tool_z_velocity(0.005)
        qdot = ctrl.compute_control_command(
            _VelState(), _ObjState(np.zeros(6)), _TaskState())
        # The overlay goes through the DLS inverse, so dls_damping attenuates
        # it by 1/(1+lambda^2): 0.25% at the 0.05 this runs on hardware, vs the
        # exact pass-through the old 1e-3 gave. Assert the overlay arrives
        # essentially intact rather than bit-exact, or this test just pins the
        # regularizer the C153 work deliberately changed.
        self.assertAlmostEqual(float(qdot[5]), 0.005, delta=0.005 * 0.01)
        np.testing.assert_allclose(qdot[:5], 0.0, atol=1e-9)

    def test_z_compliant_update_is_pose_step_like_free_teleop(self):
        ctrl = _pose_ctrl()
        ctrl.set_z_compliant(True)
        twist = ga.Twist.from_array(np.array([0.1, 0.0, 0.0, 0.0015, 0.0, 0.5]))
        ctrl.update_command(twist)
        self.assertTrue(ctrl._input_active)
        self.assertIsNotNone(ctrl.get_objective().target)
        np.testing.assert_allclose(ctrl.contact_slide_twist(), 0.0)

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

    def test_speedj_contact_exits_only_from_toggle(self):
        src = _EXAMPLE.read_text()
        self.assertEqual(src.count("exit_speedj_contact("), 2)
        self.assertIn("def toggle_contact_mode():", src)
        self.assertIn("latched until button", src)

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
