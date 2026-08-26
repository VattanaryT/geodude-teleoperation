"""Hardware object-contact teleop for geodude.

Free space: gafropy speedJ + SpaceMouse (approach mode).

Contact (default ``speedj``): same pose-spring speedJ plant as free space.
Button 0 latches force-hold until button 0 again — no auto-drop to free-space
approach. The press PID overlays tool-Z; it owns Fd and nothing else does.
Stick XY still field-follows (``project()`` rewrites R; stick Z ignored).
Button 0 must not switch to line admittance.

Three switches exist for the contact lift-off fixes, so a hardware session can
bisect which one matters. ``--press-loop legacy --no-free-along-normal
--no-snap-follow-to-surface`` restores the previous behavior exactly:

* ``--press-loop dtc`` uses the ``in_contact_control`` ForcePIDController
  (LPF + deadband on the measurement, oscillation/spike/overshoot handling)
  instead of the bare ``clip(kp*(Fd-Fs))`` on fz alone.
* free-along-normal projects the pose spring's velocity off the surface normal
  while in contact. ``stiffness[5,5]=0`` alone only frees the body Z of the
  pose error, and field-follow slerps tool Z behind the normal on purpose, so
  the spring otherwise keeps a component along the normal and lifts the tool.
* snap-follow-to-surface takes the along-normal component of the commanded
  point from ``vf.project()`` so the next query stays on the mesh.

``--contact-backend force_mode`` is UR hybrid force/position (not admittance).
"""
from __future__ import annotations

import argparse
import pathlib
import select
import socket
import sys
import termios
import threading
import time
import tty

import gafropy as ga
import numpy as np
import yaml
from gafropy import robot_interface as ri
from gafropy import viz as gafropy_viz

ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC = ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

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
from geodude_teleoperation.ee_viz import (
    add_ghost_chamfer_visual,
    add_right_chamfer_visual,
    chamfer_world_from_wrist3,
    ghost_flange_world,
    hide_gripper_visuals,
    live_tcp_xyz_from_flange,
    matrix_from_motor,
    motor_from_matrix,
    StartupAlign,
    SimTcpProbe,
)
from geodude_teleoperation.field_follow import (
    FieldFollow,
    drive_field_follow,
    field_tcp_to_ee,
    format_follow_lag,
)
from geodude_teleoperation.force_mode_contact import (
    ForceModeContactSession,
    reopen_gafropy,
    shutdown_gafropy_for_handoff,
)
from geodude_teleoperation.laser_frames import (
    is_laser_left_base_object,
    place_laser_object,
)
from geodude_teleoperation.mesh_fixture import (
    clip_mesh_to_scan_points,
    orient_surface_toward,
    reconstruct_mesh_from_points,
)
from geodude_teleoperation.spacemouse import (
    format_spacemouse_paths,
    list_spacemouse_paths,
    resolve_spacemouse_path,
    verify_spacemouse_connection,
)
from geodude_teleoperation.tcp_force import (
    connect_tcp_force_reader,
    force_from_tcp_force,
    fz_from_tcp_force,
)
from geodude_teleoperation.virtual_fixture import FieldAlignPreview, workstation_field_z_arrows

DEFAULT_OBJECT = str(ROOT / "assets" / "laser_profilometer" / "merged.ply")
MODEL = str(ROOT / "assets" / "robots" / "geodude" / "geodude.yaml")
DEFAULT_HARDWARE_CONFIG = str(ROOT / "assets" / "robots" / "geodude" / "geodude_hardware.yaml")

DEFAULT_VENTION_HEIGHT = 0.51
STICK_LINEAR_SCALE = 0.0015
STICK_ANGULAR_SCALE = 0.05


def stick_is_idle(twist):
    """Hands-off check using the same scales as SpaceMouseTeleop in this example.

    ``is_idle_twist`` defaults are linear_scale=0.01. Idle cap uses this
    example's stick scale so a 30% Z lift is not treated as rest.
    """
    return is_idle_twist(
        twist,
        linear_scale=STICK_LINEAR_SCALE,
        angular_scale=STICK_ANGULAR_SCALE,
    )


def stick_has_linear(twist):
    """True if the device already passed its deadzone and sent a linear step.

    Field-follow idle uses a 20% cap at this stick scale, which ate slow creep.
    """
    linear = np.asarray(twist.get_linear(), dtype=float).reshape(3)
    return float(np.linalg.norm(linear)) > 1e-12


def follow_linear(lin, *, ignore_z):
    """Stick linear for FieldFollow. Contact drops Z; DualLoop owns press."""
    out = np.asarray(lin, dtype=float).reshape(3).copy()
    if ignore_z:
        out[2] = 0.0
    return out


def field_follow_in_mode(mode, backend):
    """Approach always follows. speedj contact does too; force_mode does not."""
    if mode == "approach":
        return True
    return mode == "contact" and backend == "speedj"


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


def _make_approach_controller(interface, task_space, home_ee_motor,
                              free_along_normal=True):
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
        max_qddot=5.0,
        free_along_normal=free_along_normal,
    )
    approach.set_task_space(task_space)
    approach.set_interface(interface)
    approach.set_objective(ga.control.SingleArmMotorControlObjective(home_ee_motor))
    return approach


def route_spacemouse_twist(twist, *, sim_probe, on_hardware):
    """Viser probe consumes the stick; hardware only sees it when there is no probe."""
    if sim_probe is not None:
        if stick_is_idle(twist):
            sim_probe.clear_pending()
        else:
            sim_probe.add_body_linear(
                np.asarray(twist.get_linear(), dtype=float).reshape(3))
        return True
    on_hardware(twist)
    return False


def apply_terminal_key(key, *, align, sim_teleop):
    """Enter starts hardware align. ``t`` toggles viser-only stick while waiting."""
    if align is None or align.finished:
        return None
    waiting = align.phase in ("idle", "wait") and not align._triggered
    if key in ("t", "T"):
        if not waiting:
            return None
        sim_teleop["on"] = not sim_teleop["on"]
        return "sim_on" if sim_teleop["on"] else "sim_off"
    if key in ("\n", "\r"):
        if align.phase not in ("idle", "wait"):
            return None
        sim_teleop["on"] = False
        align.trigger()
        return "align"
    return None


def drive_startup_align(align, controller, now, T_flange, R_field, T_ee,
                        T_flange_display=None):
    """Track a slerped flange target. Hold and freeze when the one-shot finishes.

    ``T_flange`` / ``T_ee`` must be in the controller world (hardware q, rail at
    0). ``T_flange_display`` is the vention-overlaid flange the orange ghost uses.
    """
    was_done = align.finished
    cmd = align.step(
        now, T_flange, R_field, T_ee, T_flange_display=T_flange_display)
    if cmd.T_ee is not None:
        controller.get_objective().set_target(motor_from_matrix(cmd.T_ee))
        controller.set_input_active(True)
    elif cmd.finished and not was_done:
        controller.set_input_active(False)
        controller.hold_live_pose()
    return cmd


def _contact_log_path(spec, save_dir="speedj_logs"):
    """Resolve --contact-log. 'none' disables; 'auto' names one by timestamp."""
    if spec in (None, "", "none", "off"):
        return None
    if spec != "auto":
        return spec
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return str(pathlib.Path(save_dir).resolve() / f"contact_{stamp}.csv")


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
        "--task-space", choices=["left_ee", "right_ee"], default="right_ee")
    parser.add_argument(
        "--device", choices=["keyboard", "spacemouse"], default="spacemouse")
    parser.add_argument(
        "--spacemouse-path",
        help="hidraw path for the SpaceMouse (e.g. /dev/hidraw5). "
        "Default: by-id symlink, else newest hidraw when several are connected.")
    parser.add_argument(
        "--spacemouse-index", type=int,
        help="index into sorted connected hidraw paths (0=first). "
        "Ignored when --spacemouse-path is set.")
    parser.add_argument(
        "--check-spacemouse", action="store_true",
        help="list connected SpaceMice, open the selected one, poll 3s, and exit")
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
        "--force-sign", type=float, default=1.0,
        help="press sign along tool Z (default +1, verified on hardware: press "
             "is along +tool Z). Fixed knob; never latched from Fz polarity, and "
             "a field normal refines the press axis without flipping this sign.",
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
    # --- contact lift-off fixes, independently switchable so a hardware run can
    # bisect which one actually matters. All three off reproduces the previous
    # behavior exactly: --press-loop legacy --no-free-along-normal
    # --no-snap-follow-to-surface
    parser.add_argument(
        "--press-loop", choices=["dtc", "legacy"], default="dtc",
        help="speedj press seek law (default: dtc = the in-contact-dtc "
             "ForcePIDController, with LPF/deadband on the measurement plus "
             "oscillation, spike and overshoot handling). 'legacy' is the bare "
             "clip(kp*(Fd-Fs)) PID on fz alone.")
    parser.add_argument(
        "--no-free-along-normal", dest="free_along_normal",
        action="store_false", default=True,
        help="do not project the pose spring's velocity off the surface normal "
             "in contact (stiffness[5,5]=0 alone, the previous behavior)")
    parser.add_argument(
        "--no-snap-follow-to-surface", dest="snap_follow_to_surface",
        action="store_false", default=True,
        help="let field-follow keep discarding vf.project()'s position, so the "
             "commanded point never re-snaps to the mesh (previous behavior)")
    parser.add_argument(
        "--press-v-max", type=float, default=0.008, metavar="M_S",
        help="ceiling on the along-normal seek speed. Raise this only after "
             "the guards are letting the loop run -- check `spike %%` and "
             "`v_z == 0` in analyze_contact_log.py first, or you are just "
             "raising the speed of an impact.")
    parser.add_argument(
        "--approach-v", type=float, default=0.005, metavar="M_S",
        help="soft feed speed while |Fs| is below contact_threshold_N")
    parser.add_argument(
        "--spike-rate", type=float, default=None, metavar="N_S",
        help="override the dtc spike guard's rate threshold (default 250 N/s, "
             "set from this rig's measured tick-to-tick force spread)")
    parser.add_argument(
        "--v-slew", type=float, default=0.4, metavar="M_S2",
        help="press-velocity slew limit (m/s^2). At the old 0.05 a reversal "
             "took 0.32s, so the loop kept pressing ~1.3mm after it had "
             "decided to back off. Joint-space ramp is separately bounded by "
             "max_qddot.")
    parser.add_argument(
        "--contact-log", default="auto", metavar="PATH",
        help="per-tick CSV of the force loop (Fs, Fd, v_z, tare, dtc flags). "
             "'auto' writes next to the speedj logs, 'none' disables. Streamed "
             "to disk, so an aborted or protective-stopped run keeps its data.")
    parser.add_argument(
        "--no-force-tare", dest="force_tare", action="store_false", default=True,
        help="do not sample the free-space force offset when contact first "
             "latches. UR ActualTCPForce is not gravity compensated unless the "
             "payload is set, and the speedj backend never issues "
             "zero_ftsensor(), so without the tare the dtc loop trips "
             "wrong_sign/mag_overload on the tool weight and retracts instead "
             "of approaching")
    parser.add_argument("--no-viz", action="store_true",
                        help="disable the viser visualization")
    parser.add_argument(
        "--startup-align", dest="startup_align", action="store_true", default=True,
        help="after viser is up, slerp the live flange onto the field (one-shot)")
    parser.add_argument(
        "--no-startup-align", dest="startup_align", action="store_false",
        help="skip the one-shot field orientation align")
    parser.add_argument(
        "--follow-hz", type=float, default=20.0,
        help="after align, project() rate for stick field-follow (15–50 Hz). "
             "Not the RTDE/speedJ rate.")
    parser.add_argument(
        "--align-delay", type=float, default=None,
        help="seconds to wait after viser starts before the align slerp "
             "(default: wait for Enter; orange ghost shows the snap first)")
    parser.add_argument(
        "--align-duration", type=float, default=4.0,
        help="seconds to slerp live orientation onto the field (no snap)")
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
    parser.add_argument(
        "--no-speedj-plot", dest="speedj_plot", action="store_false",
        help="do not open the live speedJ qdot / qddot / Δt plot")
    parser.set_defaults(speedj_plot=True)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.check_spacemouse:
        paths = list_spacemouse_paths()
        if not paths:
            print("spacemouse: none connected", flush=True)
            raise SystemExit(1)
        print(f"spacemouse: connected: {format_spacemouse_paths()}", flush=True)
        path = resolve_spacemouse_path(
            args.spacemouse_path, device_index=args.spacemouse_index)
        print(f"spacemouse: opening {path} — move the cap for 3s…", flush=True)
        saw_motion, peak = verify_spacemouse_connection(path, wait_s=3.0)
        if saw_motion:
            print(f"spacemouse: OK (peak axis={peak:.3f})", flush=True)
            raise SystemExit(0)
        print(
            f"spacemouse: no motion detected (peak={peak:.3f}). "
            "Wrong device? Try --spacemouse-path or --spacemouse-index.",
            flush=True,
        )
        raise SystemExit(1)

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
        n_poisson = len(object_surface.vertices)
        object_surface = clip_mesh_to_scan_points(
            object_surface, placement.points_vention)
        orient_surface_toward(object_surface, np.array([0.0, 0.0, 1.0]))
        if args.normal_orientation < 0:
            object_surface.invert()
        print(
            "laser overlay: merged.ply left_base → vention "
            f"(+{placement.T_left_base_vention[:3, 3].round(3).tolist()} mount) "
            f"→ tag_0; T_tag0_rtde.t={placement.T_tag0_rtde[:3, 3].round(3).tolist()} "
            "(right_base). Cloud is not copied into the right arm. "
            f"scan surface {len(object_surface.vertices)}/{n_poisson} verts "
            "(Poisson underside dropped, normals toward +Z).",
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
    field_preview = FieldAlignPreview(vf)
    follow = FieldFollow(
        vf, hz=args.follow_hz, project_in_process=True, mesh=object_surface,
        max_lead_m=0.005, max_step_m=0.004, max_dR_rad=0.02,
        snap_to_surface=args.snap_follow_to_surface)
    follow_log_t = {"t": float("-inf")}
    stick = {
        "t": float("-inf"), "t_nz": float("-inf"),
        "lin": np.zeros(3), "n": 0, "nz": 0,
    }

    approach_controller = _make_approach_controller(
        interface, task_space, home_ee_motor,
        free_along_normal=args.free_along_normal)

    # --- speedj contact: stay on gafropy (default). force_mode is opt-in. ---
    contact_controller = None
    measured_fz_fn = None
    measured_force_fn = None
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
            # The dtc loop resolves force along the field normal, not tool Z, so
            # it needs all three linear components.
            measured_force_fn = lambda: force_from_tcp_force(read_tcp_force())
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
            v_approach=args.approach_v, max_v_hold=args.press_v_max,
            z_damping=z_damping,
            v_slew_mps2=args.v_slew, lost_s=1.0, sit_s=2.0,
            spike_dF_dt_N_s=args.spike_rate,
            force_sign=args.force_sign,
            press_loop=args.press_loop,
            force_tare=args.force_tare,
            measured_fz_fn=measured_fz_fn,
            measured_force_fn=measured_force_fn)
        contact_controller.set_contact_log(_contact_log_path(args.contact_log))
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
    pending_contact = {"on": False}
    device = None
    speedj_trace = None
    viz = None
    robot_viz = None
    eeviz = None
    sim_probe = {"p": None}
    sim_teleop = {"on": False}
    align = (
        StartupAlign(delay_s=args.align_delay, duration_s=args.align_duration)
        if args.startup_align else None)
    align_phase = {"name": None}

    def live_ee_motor():
        position = interface.get_state().get_position()[joint_idx]
        return task_space.compute_ee_motor(position)

    def chamfer_flange_from_q(q):
        fk = interface.get_system().compute_forward_kinematics(np.asarray(q, dtype=float))
        T_wrist = np.asarray(
            fk.get_link_pose("right_ur5e/wrist_3_link").to_transformation_matrix())
        return chamfer_world_from_wrist3(T_wrist)

    def snap_target_flange(T_flange_viz):
        """Field orientation at the live TCP, with the tip held (same as sim ``t``)."""
        p_tcp = live_tcp_xyz_from_flange(T_flange_viz)
        R_field = vf.approach_basis_at(p_tcp)
        return R_field, ghost_flange_world(T_flange_viz, R_field, p_tcp, 1.0)

    def run_field_follow(T_flange_viz, T_ee):
        if follow.xyz is None:
            p_tcp = live_tcp_xyz_from_flange(T_flange_viz)
            follow.seed(p_tcp, vf.approach_basis_at(p_tcp))
        T_flange_hw = chamfer_flange_from_q(interface.get_state().get_position())
        T_ee_m = np.asarray(T_ee, dtype=float).reshape(4, 4)

        def map_target(motor):
            return motor_from_matrix(field_tcp_to_ee(
                matrix_from_motor(motor),
                T_flange_hw=T_flange_hw,
                T_flange_viz=T_flange_viz,
                T_ee_hw=T_ee_m))

        T_prop = np.eye(4)
        T_prop[:3, 3] = follow.xyz
        now = time.monotonic()
        lin_now = follow_linear(
            stick["lin"], ignore_z=mode["active"] == "contact")
        quiet = float(np.linalg.norm(lin_now)) <= 1e-12
        tick = drive_field_follow(
            follow, approach_controller, now, motor_from_matrix(T_prop),
            live_xyz=live_tcp_xyz_from_flange(T_flange_viz),
            live_R=T_flange_viz[:3, :3],
            live_ee_xyz=T_ee_m[:3, 3],
            map_target=map_target,
            stick_quiet=quiet,
            tangent=mode["active"] == "contact")
        if not approach_controller._input_active:
            approach_controller.hold_live_pose()
        # Publish the surface normal both loops need, in the *controller* world
        # (map_target has already taken the field pose through field_tcp_to_ee,
        # so its Z column and T_ee's rotation share a frame). The force thread
        # must not query vf itself -- project() is why follow runs in a helper
        # process in the first place.
        if follow.last_target is not None:
            T_cmd = matrix_from_motor(map_target(follow.last_target))
            n_base = T_cmd[:3, 2]
            R_tool = T_ee_m[:3, :3]
            approach_controller.set_contact_normal(n_base, R_tool)
            if contact_controller is not None:
                contact_controller.set_contact_normal(n_base, R_tool)
        if (now - follow_log_t["t"]) >= 1.0:
            follow_log_t["t"] = now
            p_live = live_tcp_xyz_from_flange(T_flange_viz)
            p_cmd = follow.xyz if follow.xyz is not None else p_live
            R_cmd = T_flange_viz[:3, :3]
            if follow.last_target is not None:
                T_cmd = matrix_from_motor(follow.last_target)
                R_cmd = T_cmd[:3, :3]
            lin = np.asarray(stick["lin"], dtype=float).reshape(3)
            age_ms = (
                1e9 if not np.isfinite(stick["t"])
                else (now - float(stick["t"])) * 1000.0)
            print(
                format_follow_lag(
                    p_live, p_cmd, T_flange_viz[:3, :3], R_cmd,
                    approach_controller._input_active,
                    project_ms=tick.project_ms),
                flush=True,
            )
            print(
                "spacemouse: "
                f"|lin|={1000.0 * float(np.linalg.norm(lin)):.2f}mm "
                f"lin=[{lin[0]:.5f},{lin[1]:.5f},{lin[2]:.5f}] "
                f"n={int(stick['n'])} nz={int(stick['nz'])} "
                f"age={age_ms:.0f}ms follow_idle={follow.idle} "
                f"quiet={quiet} pending={1000.0 * follow.pending_m:.2f}mm",
                flush=True,
            )
            stick["n"] = 0
            stick["nz"] = 0
        return tick

    def start_align_keys():
        if args.device == "keyboard":
            return

        def _run():
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while align is not None and not align.finished:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if not ready:
                        continue
                    on_align_key(sys.stdin.read(1))
            except Exception:
                pass
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

        threading.Thread(target=_run, daemon=True, name="align-keys").start()

    def on_align_key(key):
        """Return True when startup align claims the key.

        KeyboardTeleop offers 't'/Enter here first and falls through on a
        falsy return, which is how 't' reaches on_button(0) -- the
        approach<->contact toggle -- once align has finished with it.
        """
        msg = apply_terminal_key(key, align=align, sim_teleop=sim_teleop)
        if msg == "sim_on":
            print(
                "sim teleop ON: SpaceMouse moves the orange ghost. "
                "Press t to snap it back to live TCP. Enter aligns hardware.",
                flush=True,
            )
        elif msg == "sim_off":
            print(
                "sim teleop OFF: orange ghost is the live-TCP snap preview.",
                flush=True,
            )
        elif msg == "align":
            print("startup align: Enter received", flush=True)
        return msg is not None

    def run_startup_align(T_flange, T_ee):
        if align is None or mode["active"] != "approach":
            return None
        if align.phase == "idle":
            align.arm(time.monotonic())
            if args.align_delay is not None:
                print(
                    f"startup align: waiting {args.align_delay:.0f}s (watch viser), then "
                    f"{args.align_duration:.0f}s orientation slerp with the spring on",
                    flush=True,
                )
        R_field, _ = snap_target_flange(T_flange)
        T_flange_hw = chamfer_flange_from_q(interface.get_state().get_position())
        cmd = drive_startup_align(
            align, approach_controller, time.monotonic(), T_flange_hw, R_field, T_ee,
            T_flange_display=T_flange)
        if cmd.phase != align_phase["name"]:
            align_phase["name"] = cmd.phase
            print(f"startup align: {cmd.phase}", flush=True)
            if cmd.finished and sim_probe["p"] is not None:
                sim_probe["p"] = SimTcpProbe(live_tcp_xyz_from_flange(T_flange))
            if cmd.finished:
                p_tcp = live_tcp_xyz_from_flange(T_flange)
                follow.seed(p_tcp, vf.approach_basis_at(p_tcp))
                print(
                    f"field follow: live TCP + stick linear at {args.follow_hz:.0f} Hz → "
                    "slerped field R from live (xyz from stick). Quiet stick freezes.",
                    flush=True,
                )
                if pending_contact["on"]:
                    pending_contact["on"] = False
                    print("startup align done: applying queued contact button", flush=True)
                    toggle_contact_mode()
        return cmd

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

    def _hardware_twist(twist):
        # Hands-off / rest bias: freeze — zero speedJ (approach) and clear speedL (contact).
        if stick_is_idle(twist):
            if mode["active"] == "approach":
                approach_controller.set_input_active(False)
                approach_controller.hold_live_pose()
            elif mode["active"] == "contact":
                # hardware_free_teleop: zero twist still update_command so the
                # pose spring tracks live EE. DualLoop keeps tool Z.
                approach_controller.update_command(twist)
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
        # speedj contact: same pose-spring plant as hardware_free_teleop.
        approach_controller.update_command(twist)

    def on_twist(twist):
        lin = np.asarray(twist.get_linear(), dtype=float).reshape(3)
        now = time.monotonic()
        stick["t"] = now
        stick["lin"] = lin
        stick["n"] = int(stick["n"]) + 1
        approach_controller.set_stick_linear(lin)
        if float(np.linalg.norm(lin)) > 1e-12:
            stick["t_nz"] = now
            stick["nz"] = int(stick["nz"]) + 1
        if handoff_busy["busy"]:
            return
        if align is not None and not align.finished:
            if sim_teleop["on"] and sim_probe["p"] is not None:
                route_spacemouse_twist(
                    twist, sim_probe=sim_probe["p"], on_hardware=lambda _: None)
            return
        if field_follow_in_mode(mode["active"], args.contact_backend):
            lin_f = follow_linear(lin, ignore_z=mode["active"] == "contact")
            if float(np.linalg.norm(lin_f)) > 1e-12:
                follow.add_body_linear(lin_f)
            return
        route_spacemouse_twist(
            twist, sim_probe=None, on_hardware=_hardware_twist)

    def toggle_contact_mode():
        if handoff_busy["busy"]:
            return
        if args.contact_backend == "force_mode":
            if mode["active"] == "approach":
                enter_force_mode_contact()
            else:
                enter_approach_from_force_mode()
            return
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
                f"-> contact mode (speedj): latched until button {args.mode_button}; "
                f"field-follow tangent XY, PID holds Fd ≤2mm/s to {args.target_force:.1f}N",
                flush=True,
            )
            return
        exit_speedj_contact(
            mode=mode,
            contact_controller=contact_controller,
            approach_controller=approach_controller,
            interface=interface,
        )
        print("-> approach mode: holding live pose", flush=True)

    def on_button(index):
        # Handoff is rising-edge button only (SpaceMouseTeleop); twists never toggle mode.
        print(f"spacemouse: button {index}", flush=True)
        if index != args.mode_button:
            return
        if align is not None and not align.finished:
            pending_contact["on"] = not pending_contact["on"]
            queued = "queued" if pending_contact["on"] else "cancelled"
            print(
                f"startup align running: contact {queued} for when align finishes",
                flush=True,
            )
            return
        toggle_contact_mode()

    def start_control_and_device():
        nonlocal device, speedj_trace
        # Start frozen: no speedJ until a non-idle SpaceMouse reading arrives.
        if args.speedj_plot and speedj_trace is None:
            from geodude_teleoperation.speedj_plot import SpeedJTrace
            speedj_trace = SpeedJTrace()
            approach_controller.set_command_trace(speedj_trace.record)
            speedj_trace.start()
        approach_controller.set_input_active(False)
        approach_controller.hold_live_pose()
        interface.control(approach_controller, blocking=False)
        device = KeyboardTeleop(
            step=0.0025) if args.device == "keyboard" else SpaceMouseTeleop(
                # Slow enough that the 20 Hz field carrot stays inside the 2 cm
                # lead and speedJ can catch it (full stick ~7.5 cm/s at 50 Hz).
                linear_scale=STICK_LINEAR_SCALE, angular_scale=STICK_ANGULAR_SCALE,
                deadzone=0.02,
                device_path=args.spacemouse_path,
                device_index=args.spacemouse_index)
        if args.device == "keyboard":
            device.start(on_twist, on_button, on_key=on_align_key)
        else:
            device.start(on_twist, on_button)
        if sim_probe["p"] is not None:
            print(
                "viser: orange ghost is the live-TCP snap preview. Press t to "
                "SpaceMouse-teleop the ghost in sim; t again resets it to live TCP. "
                "Enter aligns hardware. After align, stick linear field-follows "
                "at --follow-hz (project off the RTDE thread); angular ignored.",
                flush=True,
            )
        else:
            print(
                "free-space scale: SpaceMouse lin=0.0015 ang=0.05, "
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
            if align is not None and args.align_delay is None:
                print(
                    "startup align: Press Enter to slerp UR onto the field. "
                    "Press t for viser-only stick (headless has no ghost).",
                    flush=True,
                )
                start_align_keys()
            while True:
                q = q_with_vention()
                T_flange = chamfer_flange_from_q(q)
                T_ee = np.asarray(live_ee_motor().to_transformation_matrix())
                if align is not None and not align.finished:
                    run_startup_align(T_flange, T_ee)
                elif field_follow_in_mode(mode["active"], args.contact_backend):
                    run_field_follow(T_flange, T_ee)
                time.sleep(0.02)
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
            update_ghost = add_ghost_chamfer_visual(viz, visible=True)
            q0 = q_with_vention()
            fk0 = interface.get_system().compute_forward_kinematics(q0)
            T_wrist0 = np.asarray(
                fk0.get_link_pose("right_ur5e/wrist_3_link").to_transformation_matrix())
            T_live0 = chamfer_world_from_wrist3(T_wrist0)
            sim_probe["p"] = SimTcpProbe(live_tcp_xyz_from_flange(T_live0))
            print(
                f"ee viz: hid {n_hidden} gripper meshes; "
                "gray chamfer = held hardware; orange ghost is the snap target "
                "(field orientation at the live TCP, same pose as after pressing t) "
                "until you press Enter. "
                "Hide /right_chamfer_ee_field_ghost in the scene tree if you do not want it.",
                flush=True,
            )

            viz.scene.add_point_cloud(
                "/object", points=object_surface.vertices, colors=(200, 200, 200), point_size=0.002
            )
            viz.scene.add_point_cloud(
                "/field_source", points=vf.source_points, colors=(220, 50, 40),
                point_size=0.004,
            )
            viz.scene.add_point_cloud(
                "/field_sink", points=vf.sink_points, colors=(40, 90, 220),
                point_size=0.004,
            )
            print(
                f"field BCs: {len(vf.source_points)} source (red /field_source) and "
                f"{len(vf.sink_points)} sink (blue /field_sink) end-cap vertices. "
                "Toggle them in the scene tree.",
                flush=True,
            )
            verts = np.asarray(object_surface.vertices, dtype=float)
            bbox_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
            z_len = float(np.clip(0.04 * bbox_diag, 0.015, 0.04))
            print("field Z arrows: sampling approach axis on the object (scene tree /field_z)...",
                  flush=True)
            field_z = vf.field_z_arrows(verts, max_count=256, length=z_len)
            viz.scene.add_arrows(
                "/field_z",
                points=field_z,
                colors=(40, 90, 220),
                shaft_radius=0.08 * z_len,
                head_radius=0.2 * z_len,
                head_length=0.35 * z_len,
                visible=True,
            )
            print(
                f"field Z: {len(field_z)} blue arrows OUT of the top (object +normal). "
                f"Orange EE +Z points into the surface (anti-parallel to arrows). "
                f"Toggle /field_z and /right_chamfer_ee_field_ghost in the viser scene tree.",
                flush=True,
            )
            ws_z = workstation_field_z_arrows(verts, length=z_len)
            viz.scene.add_arrows(
                "/field_z_workstation",
                points=ws_z,
                colors=(150, 200, 255),
                shaft_radius=0.06 * z_len,
                head_radius=0.16 * z_len,
                head_length=0.3 * z_len,
                visible=True,
            )
            print(
                f"workstation field: {len(ws_z)} light-blue +Z arrows on the table plane "
                f"(outer ~6 cm of the object overlaps this frame, full table by ~45 cm). "
                f"Toggle /field_z_workstation in the scene tree.",
                flush=True,
            )

            def update():
                try:
                    _update()
                except Exception as exc:
                    print(f"viser tick skipped: {exc}", flush=True)

            def _update():
                if handoff_busy["busy"] or (
                    mode["active"] == "contact"
                    and args.contact_backend == "force_mode"
                ):
                    # force_mode shuts down gafropy; speedj contact still has live state.
                    return
                q = q_with_vention()
                robot_viz.update(q)
                update_chamfer(q)
                fk = interface.get_system().compute_forward_kinematics(q)
                T_wrist = np.asarray(
                    fk.get_link_pose("right_ur5e/wrist_3_link").to_transformation_matrix())
                T_live_flange = chamfer_world_from_wrist3(T_wrist)
                T_ee = np.asarray(live_ee_motor().to_transformation_matrix())
                if align is not None and not align.finished:
                    run_startup_align(T_live_flange, T_ee)
                    probe = sim_probe["p"]
                    if not sim_teleop["on"]:
                        T_frozen = align.desired_flange()
                        if T_frozen is not None:
                            if probe is not None:
                                probe.set_xyz(live_tcp_xyz_from_flange(T_live_flange))
                            update_ghost(T_frozen)
                            return
                        if probe is not None:
                            probe.set_xyz(live_tcp_xyz_from_flange(T_live_flange))
                    if probe is not None:
                        R_field = field_preview.update_approach_basis(
                            probe.xyz(), now=time.monotonic())
                        p_tcp = (
                            probe.step(R_field) if sim_teleop["on"] else probe.xyz())
                        update_ghost(ghost_flange_world(
                            T_live_flange, R_field, p_tcp, 1.0))
                    return
                if field_follow_in_mode(mode["active"], args.contact_backend):
                    run_field_follow(T_live_flange, T_ee)
                    if follow.last_target is not None:
                        T_tcp = matrix_from_motor(follow.last_target)
                        update_ghost(ghost_flange_world(
                            T_live_flange, T_tcp[:3, :3], T_tcp[:3, 3], 1.0))
                    return
                probe = sim_probe["p"]
                R_field = field_preview.update_approach_basis(
                    probe.xyz(), now=time.monotonic())
                p_tcp = probe.step(R_field)
                T_ghost = ghost_flange_world(T_live_flange, R_field, p_tcp, 1.0)
                update_ghost(T_ghost)

            viz.add_ticker(lambda: update())

            start_control_and_device()
            if align is not None and args.align_delay is None:
                print(
                    "startup align: orange ghost is the live-TCP snap. Press t to "
                    "teleop it in viser; t again resets to live TCP. Enter slerps UR.",
                    flush=True,
                )
                start_align_keys()

            viz.show(message=f"startup align on '{args.task_space}': t=sim ghost, "
                     f"Enter=hardware slerp, then field-follow at {args.follow_hz:.0f} Hz "
                     f"-- {backend_note}. viser at http://localhost:{viz.port} -- ctrl-c to quit.")
    except KeyboardInterrupt:
        pass
    finally:
        if device is not None:
            device.stop()
        if speedj_trace is not None:
            speedj_trace.stop()
        if follow is not None:
            follow.close()
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
