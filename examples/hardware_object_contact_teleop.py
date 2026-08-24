"""Hardware object-contact teleop for geodude.

Free space: gafropy speedJ + SpaceMouse (approach mode).

Contact (default ``speedj``): same pose-spring speedJ plant as free space.
A host DualLoop force loop overlays tool-Z velocity (APPROACH 5 mm/s, sit 2 s,
then ``v = clip(kp*(Fd − Fs))``). Button 0 must not switch to line admittance.

``--contact-backend force_mode`` is UR hybrid force/position (not admittance).
"""
from __future__ import annotations

import argparse
import pathlib
import socket
import time

import gafropy as ga
import numpy as np
import yaml
from gafropy import robot_interface as ri
from gafropy import viz as gafropy_viz

from geodude_teleoperation import (
    AdmittanceTeleoperationController,
    ContactTeleoperationController,
    KeyboardTeleop,
    OrientationField,
    SpaceMouseTeleop,
    apply_vention_heights,
    compute_resting_orientation,
    is_idle_twist,
    load_object_mesh,
    load_points,
    table_placement_translation,
)
from geodude_teleoperation.ee_viz import add_right_chamfer_visual, hide_gripper_visuals
from geodude_teleoperation.force_mode_contact import (
    ForceModeContactSession,
    reopen_gafropy,
    shutdown_gafropy_for_handoff,
)
from geodude_teleoperation.laser_frames import (
    is_laser_left_base_object,
    place_laser_object,
)
from geodude_teleoperation.mesh_fixture import reconstruct_mesh_from_points
from geodude_teleoperation.tcp_force import connect_tcp_force_reader, fz_from_tcp_force

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OBJECT = str(ROOT / "assets" / "laser_profilometer" / "merged.ply")
MODEL = str(ROOT / "assets" / "robots" / "geodude" / "geodude.yaml")
DEFAULT_HARDWARE_CONFIG = str(ROOT / "assets" / "robots" / "geodude" / "geodude_hardware.yaml")

DEFAULT_VENTION_HEIGHT = 0.51


def _dashboard_bring_up(ip: str) -> None:
    sock = socket.create_connection((ip, 29999), timeout=5)
    sock.recv(4096)

    def command(text: str) -> str:
        sock.sendall((text + "\n").encode())
        time.sleep(0.3)
        return sock.recv(4096).decode(errors="replace")

    command("power on")
    time.sleep(2)
    command("brake release")
    for _ in range(30):
        mode = command("robotmode")
        if "RUNNING" in mode:
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"{ip}: robot never reached RUNNING mode")
    sock.close()
    print(f"{ip}: RUNNING")


def _make_approach_controller(interface, task_space, home_ee_motor):
    # Match hardware_free_teleop defaults for smooth free-space tracking.
    # Do NOT attach OrientationField here: vf.project() every stick sample snaps
    # orientation and makes speedJ feel discontinuous vs the SpaceMouse.
    approach = AdmittanceTeleoperationController(
        inertia=np.diag([1.0] * 6),
        damping=np.diag([20.0] * 6),
        stiffness=np.diag([2000.0] * 6),
        dt=0.001,
        max_joint_velocity=0.8,
        twist_linear_scale=1.0,
    )
    approach.set_task_space(task_space)
    approach.set_interface(interface)
    approach.set_objective(ga.control.SingleArmMotorControlObjective(home_ee_motor))
    return approach


def enter_speedj_contact(*, mode, contact_controller, approach_controller, interface=None):
    """Keep the pose-spring speedJ plant. Overlay DualLoop Z. Do not swap to line admittance."""
    del interface  # caller may pass it; we must not interface.control(contact)
    mode["active"] = "contact"
    approach_controller.hold_live_pose()
    approach_controller.set_input_active(False)
    approach_controller.set_tool_z_velocity(0.0)
    approach_controller.set_z_compliant(True)
    contact_controller.set_on_vz(approach_controller.set_tool_z_velocity)
    contact_controller.start()


def exit_speedj_contact(*, mode, contact_controller, approach_controller, interface=None):
    """Drop Z overlay and freeze on the same pose-spring plant."""
    del interface
    contact_controller.stop()
    contact_controller.set_on_vz(None)
    approach_controller.set_tool_z_velocity(0.0)
    approach_controller.set_z_compliant(False)
    approach_controller.hold_live_pose()
    approach_controller.set_input_active(False)
    mode["active"] = "approach"


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--task-space", choices=["left_ee", "right_ee"], default="left_ee")
    parser.add_argument(
        "--device", choices=["keyboard", "spacemouse"], default="spacemouse")
    parser.add_argument(
        "--hardware-config", default=DEFAULT_HARDWARE_CONFIG,
        help="YAML file mapping control groups to driver configs (Configuration.hardware_config)")
    parser.add_argument(
        "--no-bring-up", dest="bring_up", action="store_false",
        help="skip the dashboard power-on/brake-release sequence (already-running real hardware)")
    parser.add_argument(
        "--object", default=DEFAULT_OBJECT,
        help="mesh or point cloud file to place on the table (see teleoperation.py's module docstring)")
    parser.add_argument(
        "--normal-orientation", type=int, choices=[1, -1], default=1,
        help="sign correction for point-cloud objects' PCA-estimated normals (see teleoperation.py)")
    parser.add_argument(
        "--target-force", type=float, default=15.0,
        help="desired contact force (N) along tool Z in contact mode")
    parser.add_argument(
        "--force-sign", type=float, default=-1.0,
        help="approach press sign along tool Z (default -1). Fixed knob; not latched from Fz.",
        )
    parser.add_argument(
        "--no-zero-ft", dest="zero_ft_sensor", action="store_false",
        help="do not call zero_ftsensor() on each force_mode contact entry "
             "(ignored for speedj). Use when already touching so preload is not zeroed.",
        )
    parser.add_argument(
        "--mode-button", type=int, default=0,
        help="spacemouse button index that toggles between approach and contact mode")
    parser.add_argument(
        "--contact-backend", choices=["speedj", "force_mode"], default="speedj",
        help="contact implementation (default: speedj = pose spring + Z force overlay). "
             "'force_mode' is UR hybrid force/position, not admittance.")
    parser.add_argument("--no-viz", action="store_true",
                        help="disable the viser visualization")
    parser.add_argument(
        "--vention-height", type=float, default=DEFAULT_VENTION_HEIGHT,
        help="physical vention rail height in meters for both arms (hardware does "
             "not sense these joints; used for visualization FK).")
    parser.add_argument(
        "--left-vention-height", type=float, default=None,
        help="override --vention-height for the left rail only")
    parser.add_argument(
        "--right-vention-height", type=float, default=None,
        help="override --vention-height for the right rail only")
    return parser


def main():
    args = build_arg_parser().parse_args()
    print(
        f"contact: backend={args.contact_backend}  "
        f"force_sign={args.force_sign:g}  "
        f"Fd={args.force_sign * args.target_force:+.1f}N",
        flush=True,
    )

    with open(args.hardware_config) as f:
        hardware_config = yaml.safe_load(f)

    if args.bring_up:
        for control_group, entry in hardware_config.items():
            _dashboard_bring_up(entry["robot_ip"])

    task_group = "left_ur5e" if args.task_space == "left_ee" else "right_ur5e"
    robot_ip = hardware_config[task_group]["robot_ip"]

    interface = ri.RobotInterface(ri.Configuration(MODEL, "hardware", args.hardware_config))
    interface.initialize()

    left_vention = args.left_vention_height if args.left_vention_height is not None else args.vention_height
    right_vention = args.right_vention_height if args.right_vention_height is not None else args.vention_height
    print(f"hardware does not report vention rails; drawing them at "
          f"left={left_vention:.3f}m right={right_vention:.3f}m")

    def q_with_vention():
        return apply_vention_heights(
            interface.get_system(), interface.get_state().get_position(),
            left_vention, right_vention)

    task_space = interface.get_task_space(args.task_space)
    if task_space is None:
        raise RuntimeError(f"task space '{args.task_space}' not found in {MODEL}")

    joint_idx = np.array(task_space.get_joint_indices())
    live_position = interface.get_state().get_position()[joint_idx]
    home_ee_motor = task_space.compute_ee_motor(live_position)

    viz_position = q_with_vention()[joint_idx]
    viz_ee_motor = task_space.compute_ee_motor(viz_position)
    viz_translator = viz_ee_motor.get_translator()
    home_xyz = np.array(
        [viz_translator.x(), viz_translator.y(), viz_translator.z()])
    rail_height = left_vention if args.task_space == "left_ee" else right_vention
    vention_offset = float(viz_position[0]) - rail_height
    table_reference_xyz = home_xyz - np.array([0.0, 0.0, vention_offset])

    object_points_raw = load_points(args.object)
    if is_laser_left_base_object(args.object):
        # Calibrated Keyence merge: left_base → vention → tag_0. Do not PCA-rest
        # or rewrite into right_base (that rotates the layup ~90°).
        placement = place_laser_object(args.object)
        object_surface = reconstruct_mesh_from_points(placement.points_vention)
        if args.normal_orientation < 0:
            object_surface.invert()
        print(
            "laser overlay: merged.ply left_base → vention "
            f"(+{placement.T_left_base_vention[:3, 3].round(3).tolist()} mount) "
            f"→ tag_0; T_tag0_rtde.t={placement.T_tag0_rtde[:3, 3].round(3).tolist()} "
            "(right_base). Cloud is not copied into the right arm.",
            flush=True,
        )
    else:
        rotation = compute_resting_orientation(object_points_raw)
        translation = table_placement_translation(
            object_points_raw, rotation, table_reference_xyz)
        object_surface = load_object_mesh(
            args.object, rotation=rotation, translation=translation,
            normal_orientation=args.normal_orientation)

    key_direction = np.asarray(home_ee_motor.get_rotor().to_rotation_matrix())[:, 0]
    vf = OrientationField(object_surface, key_direction=key_direction)

    approach_controller = _make_approach_controller(
        interface, task_space, home_ee_motor)

    # --- speedj contact: stay on gafropy (default). force_mode is opt-in. ---
    contact_controller = None
    measured_fz_fn = None
    close_tcp_force = None
    if args.contact_backend == "speedj":
        contact_stiffness = np.diag([400.0] * 5 + [0.0])
        z_damping = 200.0
        contact_damping = np.diag([40.0] * 5 + [z_damping])
        contact_config = ga.control.VelocityAdmittanceControllerConfig()
        contact_config.dt = 0.01
        contact_config.max_linear_speed = 0.005
        contact_config.max_angular_speed = 0.15
        contact_config.admittance = ga.control.AdmittanceConfig(
            inertia=1.0 * np.eye(6), damping=contact_damping, stiffness=contact_stiffness)
        try:
            read_tcp_force, close_tcp_force = connect_tcp_force_reader(robot_ip)
            measured_fz_fn = lambda: fz_from_tcp_force(read_tcp_force())
            print(f"contact FT: RTDE ActualTCPForce from {robot_ip}", flush=True)
        except Exception as exc:
            print(
                f"\033[93mWARNING: cannot open RTDE ActualTCPForce on {robot_ip} "
                f"({exc}). speedj contact will refuse without Fz.\033[0m",
                flush=True,
            )
        contact_controller = ContactTeleoperationController(
            vf, contact_config, target_force=args.target_force,
            kp_force=0.001, ki_force=0.0003,
            v_approach=0.005, max_v_hold=0.002, z_damping=z_damping,
            v_slew_mps2=0.05, lost_s=1.0, sit_s=2.0,
            force_sign=args.force_sign,
            measured_fz_fn=measured_fz_fn)
        contact_controller.set_task_space(task_space)
        contact_controller.set_interface(interface)
        contact_controller.set_objective(
            ga.control.SingleArmLineControlObjective(
                home_ee_motor.transform_line(ga.Line.Z())))

    force_session = None
    if args.contact_backend == "force_mode":
        force_session = ForceModeContactSession(
            robot_ip,
            target_force=args.target_force,
            force_sign=args.force_sign,
            force_ramp_s=2.0,
            zero_ft_sensor=args.zero_ft_sensor,
        )

    mode = {"active": "approach"}
    handoff_busy = {"busy": False}
    device = None
    viz = None
    robot_viz = None
    eeviz = None

    def live_ee_motor():
        position = interface.get_state().get_position()[joint_idx]
        return task_space.compute_ee_motor(position)

    def rebind_after_gafropy_reopen(new_interface):
        nonlocal interface, task_space, joint_idx, approach_controller
        interface = new_interface
        task_space = interface.get_task_space(args.task_space)
        if task_space is None:
            raise RuntimeError(f"task space '{args.task_space}' missing after reopen")
        joint_idx = np.array(task_space.get_joint_indices())
        ee = live_ee_motor()
        approach_controller = _make_approach_controller(interface, task_space, ee)
        if robot_viz is not None:
            robot_viz.update(q_with_vention())
        if eeviz is not None:
            eeviz.update(ee)

    def enter_force_mode_contact():
        nonlocal interface
        if handoff_busy["busy"]:
            return
        handoff_busy["busy"] = True
        try:
            print("-> contact handoff: settling speedJ, then force_mode...", flush=True)
            # Hold current pose (no vf.project snap) while still under gafropy.
            approach_controller.get_objective().set_target(live_ee_motor())
            time.sleep(0.35)
            shutdown_gafropy_for_handoff(interface, settle_s=0.35)
            force_session.enter(settle_s=0.2)
            mode["active"] = "contact"
            print(
                f"-> contact mode (force_mode): Fd={args.force_sign * args.target_force:+.1f}N "
                "tool Z; SpaceMouse XY slide only",
                flush=True,
            )
        except Exception as exc:
            print(f"\033[91mforce_mode handoff failed: {exc}\033[0m", flush=True)
            force_session.exit(settle_s=0.2)
            interface = reopen_gafropy(MODEL, args.hardware_config, robot_ip=robot_ip)
            rebind_after_gafropy_reopen(interface)
            interface.control(approach_controller, blocking=False)
            mode["active"] = "approach"
            print("-> approach mode restored after failed handoff", flush=True)
        finally:
            handoff_busy["busy"] = False

    def enter_approach_from_force_mode():
        nonlocal interface
        if handoff_busy["busy"]:
            return
        handoff_busy["busy"] = True
        try:
            print("-> approach handoff: ending force_mode, restoring speedJ...", flush=True)
            force_session.exit(settle_s=0.35)
            interface = reopen_gafropy(MODEL, args.hardware_config, robot_ip=robot_ip)
            rebind_after_gafropy_reopen(interface)
            # Live pose target — no snap to a stale pre-contact objective.
            approach_controller.set_input_active(False)
            approach_controller.hold_live_pose()
            interface.control(approach_controller, blocking=False)
            mode["active"] = "approach"
            print("-> approach mode: holding live pose", flush=True)
        except Exception as exc:
            print(f"\033[91mapproach handoff failed: {exc}\033[0m", flush=True)
        finally:
            handoff_busy["busy"] = False

    def on_twist(twist):
        if handoff_busy["busy"]:
            return
        # Hands-off / rest bias: freeze — zero speedJ (approach) and clear speedL (contact).
        if is_idle_twist(twist):
            if mode["active"] == "approach":
                approach_controller.set_input_active(False)
                approach_controller.hold_live_pose()
            elif mode["active"] == "contact":
                approach_controller.set_input_active(False)
                approach_controller.hold_live_pose()
            if args.contact_backend == "force_mode" and force_session is not None and force_session.active:
                force_session.clear_twist()
            return
        if mode["active"] == "approach":
            approach_controller.update_command(twist)
            return
        if args.contact_backend == "force_mode":
            if force_session is not None and force_session.active:
                # Spacemouse twist is body/tool frame; force session zeros tool Z.
                force_session.set_twist(twist.get_linear(), twist.get_angular())
            return
        # speedj contact: same pose-spring plant; stick Z ignored inside set_z_compliant.
        approach_controller.update_command(twist)

    def on_button(index):
        # Handoff is rising-edge button only (SpaceMouseTeleop); twists never toggle mode.
        if index != args.mode_button or handoff_busy["busy"]:
            return
        if args.contact_backend == "force_mode":
            if mode["active"] == "approach":
                enter_force_mode_contact()
            else:
                enter_approach_from_force_mode()
            return

        # --- speedj backend (legacy / hidden) ---
        assert contact_controller is not None
        if mode["active"] == "approach":
            if measured_fz_fn is None:
                print(
                    "\033[91mrefusing speedj contact: no ActualTCPForce reader\033[0m",
                    flush=True,
                )
                return
            mode["active"] = "contact"
            enter_speedj_contact(
                mode=mode,
                contact_controller=contact_controller,
                approach_controller=approach_controller,
                interface=interface,
            )
            print(
                f"-> contact mode (speedj): XY slide ≤15mm/s, no press while sliding; "
                f"APPROACH 5mm/s, sit 2s, PID to {args.target_force:.1f}N",
                flush=True,
            )
        else:
            exit_speedj_contact(
                mode=mode,
                contact_controller=contact_controller,
                approach_controller=approach_controller,
                interface=interface,
            )
            print("-> approach mode: holding live pose", flush=True)

    def start_control_and_device():
        nonlocal device
        # Start frozen: no speedJ until a non-idle SpaceMouse reading arrives.
        approach_controller.set_input_active(False)
        approach_controller.hold_live_pose()
        interface.control(approach_controller, blocking=False)
        device = KeyboardTeleop(
            step=0.0025) if args.device == "keyboard" else SpaceMouseTeleop(
                # Slightly finer than default 0.01: with twist_linear_scale=1, full
                # deflection ~= 0.5 m/s command rate (was ~1 m/s with 2x*0.01@50Hz).
                linear_scale=0.005, angular_scale=0.05)
        device.start(on_twist, on_button)
        print(
            "free-space scale: SpaceMouse lin=0.005 ang=0.05, "
            "admittance K=2000 D=20 dt=0.001 max|qdot|=0.8, twist_linear_scale=1 "
            "(no VF project in approach)",
            flush=True,
        )

    try:
        backend_note = (
            "speedj contact (pose spring + Z overlay)"
            if args.contact_backend == "speedj"
            else "force_mode contact (not admittance)"
        )
        if args.no_viz:
            print(f"Teleoperating '{args.task_space}' on real geodude (headless) "
                  f"-- {backend_note}; button {args.mode_button} toggles. Ctrl+C to quit.")
            start_control_and_device()
            while True:
                time.sleep(0.1)
        else:
            viz = gafropy_viz.Visualizer()
            print(f"viser running at http://localhost:{viz.port} -- "
                  "holding pose; open the page before moving the spacemouse.")

            robot_viz = viz.add_robot(interface.get_system(),
                                      name="geodude", joint_sliders=False)
            robot_viz.update(q_with_vention())
            n_hidden = hide_gripper_visuals(robot_viz)
            update_chamfer = add_right_chamfer_visual(
                viz, interface.get_system(), q=q_with_vention())
            print(
                f"ee viz: hid {n_hidden} gripper meshes; "
                "left flange blank, right chamfer on wrist_3 (visual only)",
                flush=True,
            )

            viz.scene.add_point_cloud(
                "/object", points=object_surface.vertices, colors=(200, 200, 200), point_size=0.002
            )

            eeviz = viz.add_motor(viz_ee_motor)

            def update():
                if handoff_busy["busy"] or (
                    mode["active"] == "contact"
                    and args.contact_backend == "force_mode"
                ):
                    # force_mode shuts down gafropy; speedj contact still has live state.
                    return
                q = q_with_vention()
                robot_viz.update(q)
                update_chamfer(q)
                eeviz.update(task_space.compute_ee_motor(q[joint_idx]))

            viz.add_ticker(lambda: update())

            start_control_and_device()

            viz.show(message=f"Teleoperating '{args.task_space}' on real geodude "
                     f"-- {backend_note}; button {args.mode_button} toggles. "
                     f"viser at http://localhost:{viz.port} -- ctrl-c to quit.")
    except KeyboardInterrupt:
        pass
    finally:
        if device is not None:
            device.stop()
        if force_session is not None and force_session.active:
            force_session.exit(settle_s=0.2)
        if contact_controller is not None:
            contact_controller.stop()
        if close_tcp_force is not None:
            close_tcp_force()
        try:
            interface.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
