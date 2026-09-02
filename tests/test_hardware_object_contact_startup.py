import math
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

    def test_default_config_is_the_measured_best_dtc_wrench_run(self):
        """The bare command must equal the config that was tuned on hardware.

        Pinned as one test rather than nine, because these numbers only mean
        anything together: --dtc-fn-* are 1/s^2 gains that are wrong without
        --dtc-lambda, and --max-lead-m 0.05 is a force ceiling that is only
        sane under --dtc-law wrench.
        """
        mod = _load_example()
        args = mod.build_arg_parser().parse_args([])
        self.assertEqual(args.contact_backend, "dtc")
        self.assertEqual(args.dtc_law, "wrench")
        self.assertTrue(args.dtc_lambda)
        self.assertTrue(args.dtc_friction_comp)
        self.assertAlmostEqual(args.dtc_fn_lin, 3.0)
        # 0.3 put K_ang 100x under K_lin, so the angular command sat pinned at
        # err_clip_rot on every tick and tool Z never reached the field normal
        # (11.7 deg median, 38 at p95 on dtc_20260829_233332.csv). The re-arm
        # step it was detuned for has its own fix in --rearm-slerp-s. 1.5 was
        # still not enough (4.17 deg median / 22.75 p95 on
        # dtc_20260830_002254.csv); at fn_lin it is 1.33 / 5.28, so under
        # Lambda -- which carries the apparent-M difference the split was
        # papering over -- the two frequencies are tied.
        self.assertIsNone(args.dtc_fn_ang)
        self.assertAlmostEqual(
            mod.resolve_wrench_fn_ang(True, args.dtc_fn_lin, args.dtc_fn_ang),
            3.0)
        self.assertAlmostEqual(args.dtc_err_clip_lin, 0.05)
        self.assertAlmostEqual(args.max_lead_m, 0.05)
        # j3/j4 at 1.0 was below UR5e wrist breakaway: they clipped on 32.9%
        # and 45.3% of ticks, so the two joints that carry a width traverse's
        # rotation could not turn at all. 6.0 cleared it (1.0% / 0.1%). It was
        # briefly 10.0 because 6.0 binds again once fn_ang is untied, and went
        # straight back: during the 24.5 Hz limit cycle on
        # dtc_20260830_005006.csv, tau_max was the only thing bounding how hard
        # the wrist shook, and j4 sat on the 10 N.m rail for 18.9% of ticks.
        self.assertEqual(
            list(args.dtc_tau_max), [50.0, 50.0, 50.0, 6.0, 6.0, 30.0])
        self.assertAlmostEqual(args.align_duration, 4.0)

    def test_speedj_backend_is_still_reachable(self):
        args = _load_example().build_arg_parser().parse_args(
            ["--contact-backend", "speedj"])
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

    def test_contact_normal_body_is_the_conformity_error(self):
        """The read side is stored in the *live* body frame, so its angle to +Z
        is exactly the orientation the arm has failed to track.

        Free space must publish it too -- the projection only runs while
        z_compliant, but the number is the field-tracking diagnostic and the
        dtc log wants it on every tick.
        """
        ctrl = _pose_ctrl()
        self.assertIsNone(ctrl.contact_normal_body)
        # Normal 30 deg off base Z, live tool aligned with base: 30 deg of error.
        a = np.radians(30.0)
        ctrl.set_contact_normal([np.sin(a), 0.0, np.cos(a)], np.eye(3))
        n_body = ctrl.contact_normal_body
        self.assertAlmostEqual(
            float(np.degrees(np.arccos(float(n_body[2])))), 30.0, places=6)
        # Rotating the tool onto the normal drives it to zero.
        R = np.array([[np.cos(a), 0.0, np.sin(a)],
                      [0.0, 1.0, 0.0],
                      [-np.sin(a), 0.0, np.cos(a)]])
        ctrl.set_contact_normal([np.sin(a), 0.0, np.cos(a)], R)
        self.assertAlmostEqual(
            float(ctrl.contact_normal_body[2]), 1.0, places=9)
        ctrl.set_contact_normal(None, None)
        self.assertIsNone(ctrl.contact_normal_body)

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


class WrenchLawWiringTest(unittest.TestCase):
    """--dtc-law wrench gains apply at DTC handoff, not during speedJ align."""

    def _args(self, argv):
        return _load_example().build_arg_parser().parse_args(argv)

    def test_speedj_gets_no_wrench_kwargs_at_all(self):
        mod = _load_example()
        args = self._args(["--contact-backend", "speedj"])
        self.assertEqual(mod._wrench_law_kwargs(args), {})

    def test_velocity_law_gets_no_wrench_kwargs_either(self):
        mod = _load_example()
        args = self._args(["--contact-backend", "dtc", "--dtc-law", "velocity"])
        self.assertEqual(args.dtc_law, "velocity")
        self.assertEqual(mod._wrench_law_kwargs(args), {})

    def test_wrench_law_carries_the_split_gains(self):
        mod = _load_example()
        kw = mod._wrench_law_kwargs(
            self._args(["--contact-backend", "dtc", "--dtc-law", "wrench"]))
        self.assertTrue(kw["wrench_law"])
        # k_lin/k_ang are the un-shaped N/m and N.m/rad gains; --dtc-lambda is
        # on by default, so fn_lin/fn_ang are what actually set the spring.
        self.assertAlmostEqual(kw["k_lin"], 3000.0)
        self.assertAlmostEqual(kw["k_ang"], 50.0)
        self.assertTrue(kw["use_lambda"])
        self.assertAlmostEqual(kw["fn_lin"], 3.0)
        # Under Lambda the two blades share a frequency and a damping ratio:
        # Lambda carries the apparent-M difference the 20:1 split and the 0.3/0.1
        # zeta split were both papering over, so neither correction is still
        # doing work. See resolve_wrench_fn_ang / resolve_wrench_zeta.
        self.assertAlmostEqual(kw["fn_ang"], 3.0)
        self.assertAlmostEqual(kw["err_clip_lin"], 0.05)
        self.assertAlmostEqual(kw["damping_ratio"], mod.WRENCH_ZETA_LIN)
        self.assertAlmostEqual(kw["damping_ratio_ang"], mod.WRENCH_ZETA_ANG)
        # Without Lambda both splits come back and mean what they used to.
        no_lambda = mod._wrench_law_kwargs(self._args(
            ["--contact-backend", "dtc", "--dtc-law", "wrench",
             "--no-dtc-lambda"]))
        self.assertFalse(no_lambda["use_lambda"])
        self.assertAlmostEqual(no_lambda["fn_ang"], mod.WRENCH_FN_ANG_NO_LAMBDA)
        self.assertAlmostEqual(no_lambda["damping_ratio"], mod.WRENCH_ZETA_LIN)
        self.assertAlmostEqual(
            no_lambda["damping_ratio_ang"], mod.WRENCH_ZETA_ANG)
        # An explicit flag still wins over either default.
        explicit = mod._wrench_law_kwargs(self._args(
            ["--contact-backend", "dtc", "--dtc-law", "wrench",
             "--dtc-fn-ang", "0.3", "--dtc-damping-ratio-ang", "0.1"]))
        self.assertAlmostEqual(explicit["fn_ang"], 0.3)
        self.assertAlmostEqual(explicit["damping_ratio_ang"], 0.1)

    def test_handoff_and_reopen_pass_wrench_kwargs(self):
        """Align uses speedJ defaults; wrench kwargs only at DTC handoff + reopen."""
        source = _EXAMPLE.read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("**_wrench_law_kwargs(args)"), 2)
        self.assertIn("defer_dtc_until_align_done", source)
        self.assertIn(
            '**(_wrench_law_kwargs(args) if args.contact_backend != "dtc" else {})',
            source)

    def test_defer_dtc_until_align_done(self):
        mod = _load_example()

        class _Align:
            def __init__(self, done):
                self._done = done

            @property
            def finished(self):
                return self._done

        self.assertTrue(mod.defer_dtc_until_align_done(
            dtc_backend=object(), align=_Align(False)))
        self.assertFalse(mod.defer_dtc_until_align_done(
            dtc_backend=object(), align=_Align(True)))
        self.assertFalse(mod.defer_dtc_until_align_done(
            dtc_backend=object(), align=None))
        self.assertFalse(mod.defer_dtc_until_align_done(
            dtc_backend=None, align=_Align(False)))

    def test_dtc_compliance_hold_defers_dtc_until_align_done(self):
        mod = _load_example()

        class _Align:
            finished = False

        class _Args:
            dtc_compliance_hold = True

        self.assertTrue(mod.defer_dtc_until_align_done(
            dtc_backend=object(), align=_Align(), args=_Args()))

    def test_dtc_compliance_hold_keeps_startup_align(self):
        mod = _load_example()
        args = mod.build_arg_parser().parse_args(["--dtc-compliance-hold"])
        mod.configure_dtc_compliance_hold(args)
        self.assertTrue(args.dtc_compliance_hold)
        self.assertTrue(args.startup_align)
        self.assertFalse(args.dtc_hold_at_tick)
        self.assertTrue(args.dtc_lambda)
        self.assertIsNone(args.dtc_fn_ang)
        self.assertEqual(args.dtc_joint_hold_kp, 0.0)
        self.assertEqual(args.dtc_joint_hold_kd, 0.0)
        self.assertAlmostEqual(args.dtc_fn_lin, 3.0)
        self.assertAlmostEqual(
            mod.resolve_wrench_fn_ang(True, args.dtc_fn_lin, args.dtc_fn_ang),
            3.0)
        self.assertAlmostEqual(args.dtc_damping_ratio,
                               mod.COMPLIANCE_HOLD_ZETA_LIN)
        self.assertAlmostEqual(args.dtc_damping_ratio_ang,
                               mod.COMPLIANCE_HOLD_ZETA_ANG)
        # The angular blade goes back to the teleop default: the push test that
        # produced these gains never loaded it (no field-follow => no commanded
        # rotation => zero angular error), so 0.94 was never validated and
        # ran as D_ang = 12.0 against the 5.65 measured stable.
        self.assertAlmostEqual(args.dtc_damping_ratio_ang, mod.WRENCH_ZETA_ANG)
        # ~90% of the UR5e's rated 150/150/150/28/28/28. Read as an amplitude
        # setting: tau_max is the only bound on the unclipped damping half of
        # the wrench, so a run that rings on j3/j4 brings this down first.
        self.assertEqual(
            list(args.dtc_tau_max),
            [140.0, 140.0, 140.0, 25.0, 25.0, 25.0])
        self.assertTrue(all(t <= 150.0 for t in args.dtc_tau_max[:3]))
        self.assertTrue(all(t <= 28.0 for t in args.dtc_tau_max[3:]),
                        "wrist caps must stay inside the UR5e 28 N.m rating")

    def test_dtc_compliance_hold_gains_are_not_silently_capped(self):
        """What --dtc-compliance-hold configures must be what the loop runs.

        0.94/0.94 was not: _cap_zeta clamped both blades to D = 12.0 1/s
        (zeta 0.318), so the "K/D = 10" the constants claimed never ran and the
        only trace was one line of yellow at startup.
        """
        mod = _load_example()
        args = mod.build_arg_parser().parse_args(["--dtc-compliance-hold"])
        mod.configure_dtc_compliance_hold(args)
        fn_ang = mod.resolve_wrench_fn_ang(
            args.dtc_lambda, args.dtc_fn_lin, args.dtc_fn_ang)
        dmax = mod.resolve_wrench_damping_max(args.dtc_hz, args.dtc_damping_max)
        lin, ang = mod.resolve_wrench_zeta(
            args.dtc_lambda, args.dtc_damping_ratio, args.dtc_damping_ratio_ang,
            fn_lin=args.dtc_fn_lin, fn_ang=fn_ang, damping_max=dmax)
        self.assertAlmostEqual(lin, args.dtc_damping_ratio, places=3)
        self.assertAlmostEqual(ang, args.dtc_damping_ratio_ang, places=3)
        d_lin = 2.0 * lin * 2.0 * math.pi * args.dtc_fn_lin
        self.assertLessEqual(d_lin, dmax + 1e-9)

    def test_dtc_push_test_implies_compliance_hold_and_skips_align(self):
        mod = _load_example()
        args = mod.build_arg_parser().parse_args(["--dtc-push-test"])
        mod.configure_dtc_compliance_hold(args)
        self.assertTrue(mod.dtc_push_test_mode(args))
        self.assertTrue(mod.dtc_compliance_hold_mode(args))
        self.assertFalse(args.startup_align)
        self.assertFalse(args.dtc_hold_at_tick)
        self.assertTrue(args.dtc_lambda)
        self.assertAlmostEqual(args.dtc_damping_ratio,
                               mod.COMPLIANCE_HOLD_ZETA_LIN)
        self.assertAlmostEqual(args.dtc_damping_ratio_ang,
                               mod.COMPLIANCE_HOLD_ZETA_ANG)

    def test_compliance_hold_alone_keeps_startup_align(self):
        """Only the bench push test skips align; teleop compliance hold needs it."""
        mod = _load_example()
        args = mod.build_arg_parser().parse_args(["--dtc-compliance-hold"])
        mod.configure_dtc_compliance_hold(args)
        self.assertFalse(mod.dtc_push_test_mode(args))
        self.assertTrue(args.startup_align)

    def test_push_test_skips_the_align_handoff_defer(self):
        mod = _load_example()

        class _Align:
            finished = False

        class _Args:
            dtc_compliance_hold = True
            dtc_push_test = True

        self.assertFalse(mod.defer_dtc_until_align_done(
            dtc_backend=object(), align=_Align(), args=_Args()))
        _Args.dtc_push_test = False
        self.assertTrue(mod.defer_dtc_until_align_done(
            dtc_backend=object(), align=_Align(), args=_Args()))

    def test_push_test_line_reports_w_ang(self):
        """|W_ang| is the reading that says the angular blade was untested."""
        mod = _load_example()

        class _Ctl:
            _last_err_lin = np.array([0.003, 0.0, 0.004])

        line = mod.push_test_line(
            _Ctl(), {"w_lin": 12.5, "w_ang": 0.0, "tau_peak": 3.25,
                     "hz_meas": 198.0, "n_rail_osc": 0})
        self.assertIn("|e_lin|=5.0mm", line)
        self.assertIn("|W_ang|=0.00Nm", line)
        self.assertIn("railosc=0", line)

    def test_follow_rebase_on_idle_is_dtc_only(self):
        """--dtc-hold-at-tick is a DTC gain; it must not touch the speedJ leg.

        The speedJ pose spring has no _input_active gate, so hold_live_pose on a
        quiet stick is the only thing keeping a hands-off arm off a stale
        target -- and speedJ has to stay bit-identical for the C153 A/B.
        """
        mod = _load_example()
        args = mod.build_arg_parser().parse_args(["--dtc-compliance-hold"])
        mod.configure_dtc_compliance_hold(args)
        self.assertFalse(args.dtc_hold_at_tick)
        self.assertFalse(mod.follow_rebase_on_idle(args, on_dtc=True))
        self.assertTrue(mod.follow_rebase_on_idle(args, on_dtc=False))
        plain = mod.build_arg_parser().parse_args([])
        self.assertTrue(mod.follow_rebase_on_idle(plain, on_dtc=True))
        self.assertTrue(mod.follow_rebase_on_idle(plain, on_dtc=False))

    def test_dtc_compliance_hold_keeps_explicit_fn_ang(self):
        mod = _load_example()
        args = mod.build_arg_parser().parse_args([
            "--dtc-compliance-hold", "--dtc-fn-ang", "2.0"])
        mod.configure_dtc_compliance_hold(args)
        self.assertAlmostEqual(args.dtc_fn_ang, 2.0)
        self.assertAlmostEqual(
            mod.resolve_wrench_fn_ang(True, args.dtc_fn_lin, args.dtc_fn_ang),
            2.0)

    def test_dtc_compliance_hold_keeps_explicit_tau_max(self):
        mod = _load_example()
        args = mod.build_arg_parser().parse_args([
            "--dtc-compliance-hold", "--dtc-tau-max", "25", "25", "25",
            "5", "5", "15"])
        mod.configure_dtc_compliance_hold(args)
        self.assertEqual(
            list(args.dtc_tau_max), [25.0, 25.0, 25.0, 5.0, 5.0, 15.0])

    def test_compliance_hold_wires_field_follow_rebase_on_idle(self):
        src = _EXAMPLE.read_text(encoding="utf-8")
        self.assertIn("rebase_on_idle=follow_rebase_on_idle(", src)

    def test_dtc_compliance_hold_keeps_explicit_damping_ratio(self):
        mod = _load_example()
        args = mod.build_arg_parser().parse_args([
            "--dtc-compliance-hold", "--dtc-damping-ratio", "0.25"])
        mod.configure_dtc_compliance_hold(args)
        self.assertAlmostEqual(args.dtc_damping_ratio, 0.25)

    def test_dtc_compliance_hold_rejects_no_lambda(self):
        mod = _load_example()
        args = mod.build_arg_parser().parse_args([
            "--dtc-compliance-hold", "--no-dtc-lambda"])
        with self.assertRaises(SystemExit):
            mod.configure_dtc_compliance_hold(args)

    def test_dtc_compliance_hold_rejects_non_dtc_backend(self):
        mod = _load_example()
        args = mod.build_arg_parser().parse_args([
            "--dtc-compliance-hold", "--contact-backend", "speedj"])
        with self.assertRaises(SystemExit):
            mod.configure_dtc_compliance_hold(args)

    def test_speedj_align_binds_task_space_to_gafropy_interface(self):
        """Regression: dtc task_space + gafropy control → arm does not move."""
        source = _EXAMPLE.read_text(encoding="utf-8")
        self.assertIn(
            "task_space = interface.get_task_space(args.task_space)",
            source)
        self.assertIn(
            "task_space = dtc_backend.get_task_space(args.task_space)",
            source)
        # Must not prefer the driverless DTC System while speedJ still owns RTDE.
        self.assertNotIn(
            "else dtc_backend.get_task_space(args.task_space))",
            source)

    def test_the_factory_splits_the_units_and_the_damping_ratio(self):
        """K_ang on blades 0..2 (N.m/rad), K_lin on 3..5 (N/m), each with its
        own zeta.

        A flat 2000 across both -- what speedj runs -- is a velocity gain, not
        a stiffness, and is violent on the angular half under tau = J.T @ W.
        A flat *zeta* is the same mistake one level down: D = 2*sqrt(K)*zeta is
        critical damping for M = 1, and M is ~3 kg linear against ~0.05 kg.m^2
        at the wrist.
        """
        import math

        mod = _load_example()
        got = {}

        class _Spy:
            def __init__(self, **kw):
                got.update(kw)

            def __getattr__(self, name):
                return lambda *a, **k: None

        real = mod.AdmittanceTeleoperationController
        real_obj = mod.ga.control.SingleArmMotorControlObjective
        mod.AdmittanceTeleoperationController = _Spy
        # Factory finishes with SingleArmMotorControlObjective(home_ee); the
        # spy already ate the controller, but that ctor still runs and rejects
        # None. Stub it so this test only asserts K/D wiring.
        mod.ga.control.SingleArmMotorControlObjective = lambda *a, **k: object()
        try:
            mod._make_approach_controller(
                None, None, None, wrench_law=True)
        finally:
            mod.AdmittanceTeleoperationController = real
            mod.ga.control.SingleArmMotorControlObjective = real_obj

        k = np.diag(got["stiffness"])
        d = np.diag(got["damping"])
        self.assertAlmostEqual(k[0], 50.0)
        self.assertAlmostEqual(k[3], 3000.0)
        self.assertAlmostEqual(d[0], 2.0 * math.sqrt(50.0) * 0.1)
        self.assertAlmostEqual(d[3], 2.0 * math.sqrt(3000.0) * 0.3)
        # 5 mm of field-follow lead is the sliding force this buys.
        self.assertAlmostEqual(k[3] * 0.005, 15.0)

    def test_the_damping_survives_a_delayed_feedback_at_the_worst_seen_period(self):
        """(D/M)*T is what decides whether a delayed damper shakes the arm.

        The run that ended in ROBOT_EMERGENCY_STOP was measured at a 23 ms tick
        against an 8 ms nominal, and zeta=1 put the angular product at 6.5 --
        a delayed damper diverges past ~1. Both blades must stay under 1 there.

        Under --dtc-lambda M is 1 by construction on both blades, so the same
        product is just D*T = 2*zeta*omega_n*T and the two zetas no longer have
        to differ -- but the bound still has to hold at the worst measured tick.
        """
        import math

        mod = _load_example()
        t_worst = 0.023
        for zeta, k, m, blade in (
                (mod.WRENCH_ZETA_LIN, 3000.0, 3.0, "linear"),
                (mod.WRENCH_ZETA_ANG, 50.0, 0.05, "angular")):
            d = 2.0 * math.sqrt(k) * zeta
            self.assertLess(d / m * t_worst, 1.0, blade)
        args = mod.build_arg_parser().parse_args([])
        self.assertIsNone(args.dtc_damping_ratio)
        self.assertIsNone(args.dtc_damping_ratio_ang)
        # Same two ratios with and without Lambda. They were briefly 0.7/0.7
        # under Lambda on the argument that Lambda makes M = 1 so zeta reads as
        # the true damping ratio -- which is correct and beside the point, see
        # the limit-cycle test below.
        for use_lambda in (False, True):
            self.assertEqual(
                mod.resolve_wrench_zeta(use_lambda, None, None, fn_lin=3.0,
                                        fn_ang=3.0),
                (mod.WRENCH_ZETA_LIN, mod.WRENCH_ZETA_ANG), use_lambda)

    def test_the_damping_cap_refuses_what_limit_cycled_the_arm(self):
        """D, not zeta, is the bounded quantity -- and fn raises D too.

        dtc_20260830_005006.csv ran a 24.5 Hz limit cycle with the stick at
        rest at zeta 0.7/0.7. A damper is -D*v applied one loop delay late, so
        past 90 deg of phase it is *negative* damping; at ~12 ms of delay that
        is everything above ~20 Hz, and this rig has a structural mode right
        there. The rigid-body bound (D*T ~ 1) said 0.21 and was useless.
        """
        import math

        mod = _load_example()
        fn = 3.0
        d = lambda z: 2.0 * z * 2.0 * math.pi * fn
        # The two measured-stable ratios fit under the cap; 0.7 does not.
        self.assertLessEqual(d(mod.WRENCH_ZETA_LIN), mod.WRENCH_DAMPING_MAX_1_S)
        self.assertLessEqual(d(mod.WRENCH_ZETA_ANG), mod.WRENCH_DAMPING_MAX_1_S)
        self.assertGreater(d(0.7), mod.WRENCH_DAMPING_MAX_1_S)
        lin, ang = mod.resolve_wrench_zeta(True, 0.7, 0.7, fn_lin=fn, fn_ang=fn)
        self.assertAlmostEqual(d(lin), mod.WRENCH_DAMPING_MAX_1_S)
        self.assertAlmostEqual(d(ang), mod.WRENCH_DAMPING_MAX_1_S)
        # Raising fn at a fixed zeta walks back into the cycle, so the cap has
        # to take zeta *down* as fn goes up. This is what makes the planned
        # --dtc-fn-lin sweep safe.
        hi, _ = mod.resolve_wrench_zeta(True, None, None, fn_lin=6.0, fn_ang=6.0)
        self.assertLess(hi, mod.WRENCH_ZETA_LIN)
        self.assertAlmostEqual(
            2.0 * hi * 2.0 * math.pi * 6.0, mod.WRENCH_DAMPING_MAX_1_S)
        # Without Lambda D is in N.s/m and the cap must not touch it.
        self.assertEqual(
            mod.resolve_wrench_zeta(False, 0.7, 0.7, fn_lin=fn, fn_ang=fn),
            (0.7, 0.7))
        # 0 disables it, for a deliberate experiment.
        self.assertEqual(
            mod.resolve_wrench_zeta(True, 0.7, 0.7, fn_lin=fn, fn_ang=fn,
                                    damping_max=0.0),
            (0.7, 0.7))
        # The flag defaults to None so resolve_wrench_damping_max owns the
        # number; parse_args must not bake one in behind it.
        self.assertIsNone(mod.build_arg_parser().parse_args([]).dtc_damping_max)

    def test_the_d_ceiling_does_not_move_with_the_control_rate(self):
        """It scaled as 12.0 * hz/125 for one commit, and the arm paid for it.

        The reasoning was that the cap bounds a phase lag, so it should follow
        the delay -- true as far as it goes. What it missed is that
        `f_90 = 1/(4T)` is a sweep line, not a target the modes sit below:
        doubling the rate sweeps it up into structure that was out of reach.
        dtc_20260830_141034.csv ran the 19.2 the scaling allowed (D_lin 19.1)
        and limit-cycled at **32.1 Hz** -- f_90 for this loop to within a bin --
        with tau alternating +50/-50 N.m on the shoulder and elbow, the stick
        quiet and the pose error at 0.0 mm. The arm swung into the workstation.
        """
        mod = _load_example()
        for hz in (125.0, 200.0, 500.0):
            self.assertAlmostEqual(mod.resolve_wrench_damping_max(hz),
                                   mod.WRENCH_DAMPING_MAX_1_S, msg=f"hz={hz}")
        # An explicit ceiling still passes through, at any rate.
        self.assertAlmostEqual(mod.resolve_wrench_damping_max(200.0, 15.0), 15.0)
        self.assertAlmostEqual(mod.resolve_wrench_damping_max(125.0, 0.0), 0.0)

    def test_contact_is_refused_under_the_wrench_law(self):
        """The press reaches the arm through set_tool_z_velocity, which the
        wrench law does not read. Entering contact would hover and log a
        healthy zero force -- refuse until set_press_wrench is wired (A2)."""
        source = _EXAMPLE.read_text(encoding="utf-8")
        self.assertIn('refusing contact under --dtc-law wrench', source)


class DtcDiagnosticsWiringTest(unittest.TestCase):
    """The console line and the CLI knobs the C283A2 run needed and did not have."""

    def test_the_stale_bound_is_tunable_and_defers_to_the_backend(self):
        mod = _load_example()
        args = mod.build_arg_parser().parse_args([])
        self.assertIsNone(args.dtc_stale_tau_ms)
        args = mod.build_arg_parser().parse_args(["--dtc-stale-tau-ms", "50"])
        self.assertEqual(args.dtc_stale_tau_ms, 50.0)

    def test_the_per_tick_dtc_log_is_on_by_default(self):
        mod = _load_example()
        args = mod.build_arg_parser().parse_args([])
        self.assertEqual(args.dtc_log, "auto")
        path = mod._contact_log_path("auto", prefix="dtc")
        self.assertIn("/dtc_", path)
        self.assertTrue(path.endswith(".csv"))
        self.assertIsNone(mod._contact_log_path("none", prefix="dtc"))

    def test_the_contact_log_name_did_not_move(self):
        mod = _load_example()
        self.assertIn("/contact_", mod._contact_log_path("auto"))

    def test_the_console_line_reports_the_publish_side_stale_count(self):
        """It printed the backend's read-stale as 0 while the publisher chopped tau."""
        source = _EXAMPLE.read_text(encoding="utf-8")
        self.assertIn("pub_stale=", source)
        self.assertIn('st.get("streamer")', source)
        self.assertIn("read_stale=", source)

    def test_the_console_line_names_the_railed_joint(self):
        source = _EXAMPLE.read_text(encoding="utf-8")
        self.assertIn("clip_joint_worst", source)


if __name__ == "__main__":
    unittest.main()
