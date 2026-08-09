"""CLI entry for UR stringer keyboard teleoperation (MuJoCo / RTDE backends)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent.parent
GEODUDE_MODEL = ROOT / "assets" / "robots" / "geodude" / "geodude.yaml"
TASK_SPACE_NAME = "right_ee"


def _parse_enable_key(value: str) -> str:
    if value.lower() == "enter":
        return "\n"
    if len(value) != 1:
        raise argparse.ArgumentTypeError(
            "enable-key must be a single character or 'enter'"
        )
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="geodude-ur-stringer-teleop",
        description=(
            "Keyboard teleoperation for Geodude right UR5e against a stringer "
            "orientation field (calibrate-then-enable)."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("mujoco", "rtde"),
        default="mujoco",
        help="Simulation (mujoco) or live robot (rtde)",
    )
    parser.add_argument(
        "--robot-ip",
        default=None,
        help="Robot IP (required for rtde, including dry-run receive-only)",
    )
    parser.add_argument(
        "--arm",
        choices=("right",),
        default="right",
        help="Arm to teleoperate (v1: right only)",
    )
    parser.add_argument(
        "--ply",
        required=True,
        type=Path,
        help="Object point cloud in tag_0 frame",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Voxel downsample size (default from diffused_fields config)",
    )
    parser.add_argument(
        "--normal-orientation",
        type=int,
        choices=(1, -1),
        default=None,
        help="Normal sign for point cloud PCA",
    )
    parser.add_argument(
        "--field-flow",
        choices=("length", "width"),
        default="length",
        help="Scalar field flow direction",
    )
    parser.add_argument(
        "--field-mode",
        choices=("end_lines", "boundary"),
        default="end_lines",
        help="Diffusion boundary condition mode",
    )
    parser.add_argument(
        "--end-line-fraction",
        type=float,
        default=0.06,
        help="Fraction of extent for end-line source/sink bands",
    )
    parser.add_argument(
        "--swap-end-lines",
        action="store_true",
        help="Swap source and sink end-line bands",
    )
    parser.add_argument(
        "--tip-offset",
        type=float,
        nargs=3,
        metavar=("TX", "TY", "TZ"),
        default=(0.0, 0.0, 0.0),
        help="Tip offset in TCP frame (default 0 0 0)",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.001,
        help="Contact distance threshold (m)",
    )
    parser.add_argument(
        "--query-radius",
        type=float,
        default=0.025,
        help="Local surface fit query radius (m)",
    )
    parser.add_argument(
        "--fit-mode",
        choices=("sphere", "plane"),
        default="sphere",
        help="Local surface fit mode",
    )
    parser.add_argument(
        "--wos-batch-size",
        type=int,
        default=256,
        help="Walk-on-Spheres batch size",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=50.0,
        help="Host loop rate for status / VF query (Hz)",
    )
    parser.add_argument(
        "--max-joint-velocity",
        type=float,
        default=1.0,
        help="Joint velocity clamp (rad/s)",
    )
    parser.add_argument(
        "--keyboard-step",
        type=float,
        default=0.0025,
        help="Fixed twist step per keypress",
    )
    parser.add_argument(
        "--rtde-accel",
        type=float,
        default=0.5,
        help="RTDE speedJ acceleration limit (control path only)",
    )
    parser.add_argument(
        "--watchdog-ms",
        type=float,
        default=500.0,
        help="RTDE watchdog: no fresh command → speedJ(0)",
    )
    parser.add_argument(
        "--enable-key",
        type=_parse_enable_key,
        default="\n",
        help="Key to enable motion after calibrate (default Enter)",
    )
    parser.add_argument(
        "--T-tag0-base",
        type=Path,
        default=None,
        help="Fixed SE(3) pose of tag_0 in base (identity default for mujoco)",
    )
    parser.add_argument(
        "--base-frame",
        default=None,
        help="Live TF base frame (rtde)",
    )
    parser.add_argument(
        "--tag0-frame",
        default=None,
        help="Live TF tag_0 frame (rtde)",
    )
    parser.add_argument(
        "--calib-meta",
        type=Path,
        default=None,
        help="ee_object_calib mark meta JSON with T_cam_EE (AprilTag cross-check)",
    )
    parser.add_argument(
        "--camera-rgb-topic",
        default=None,
        help="Wrist RGB image topic for AprilTag cross-check (rtde calibrate)",
    )
    parser.add_argument(
        "--camera-info-topic",
        default=None,
        help="CameraInfo topic matching --camera-rgb-topic",
    )
    parser.add_argument(
        "--tag-id",
        type=int,
        default=0,
        help="Origin AprilTag id for cross-check (default 0)",
    )
    parser.add_argument(
        "--tag-size-m",
        type=float,
        default=0.04,
        help="AprilTag edge length in metres (cross-check)",
    )
    parser.add_argument(
        "--tag-family",
        default="tag36h11",
        help="AprilTag dictionary family (cross-check)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No control interface / no motion commands",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def validate_args(args: argparse.Namespace) -> None:
    if args.arm != "right":
        raise SystemExit(f"v1 supports --arm right only, got {args.arm!r}")
    if args.backend == "rtde" and not args.robot_ip:
        raise SystemExit("--robot-ip is required for --backend rtde")
    if args.enable_key in ("t", "v"):
        raise SystemExit("--enable-key must not be 't' or 'v'")
    if args.control_hz <= 0:
        raise SystemExit("--control-hz must be positive")


def _load_field_bundle(args: argparse.Namespace):
    from geodude_teleoperation.stringer_field import (
        build_wos,
        compute_orientation_field,
        config_defaults,
        load_stringer_pointcloud,
        validate_field_prints,
    )

    defaults = config_defaults()
    voxel_size = (
        defaults["voxel_size"] if args.voxel_size is None else args.voxel_size
    )
    normal_orientation = (
        defaults["normal_orientation"]
        if args.normal_orientation is None
        else args.normal_orientation
    )
    pcloud = load_stringer_pointcloud(args.ply, voxel_size, normal_orientation)
    sd = compute_orientation_field(
        pcloud,
        field_mode=args.field_mode,
        field_flow=args.field_flow,
        end_line_fraction=args.end_line_fraction,
        swap_end_lines=args.swap_end_lines,
    )
    wos = build_wos(pcloud, args.wos_batch_size)
    print(validate_field_prints(pcloud, sd, wos.convergence_threshold))
    return pcloud, wos


def run_mujoco(args: argparse.Namespace) -> int:
    import numpy as np

    try:
        import gafropy
        import gafropy as ga
        from gafropy import robot_interface as ri
    except ImportError as exc:
        print(f"error: gafropy is required for mujoco backend: {exc}", file=sys.stderr)
        return 1

    from geodude_teleoperation.contact_wos_vf import ContactWosOrientationField
    from geodude_teleoperation.controller import TeleoperationController
    from geodude_teleoperation.frames import load_T_tag0_base, warn_tip_offset_xy
    from geodude_teleoperation.keyboard import KeyboardTeleop
    from geodude_teleoperation.teleop_state import TeleopState, format_status_line

    tip_offset = np.asarray(args.tip_offset, dtype=float)
    warning = warn_tip_offset_xy(tip_offset)
    if warning is not None:
        print(warning)

    T_tag0_base = load_T_tag0_base(args.T_tag0_base)
    pcloud, wos = _load_field_bundle(args)

    vf = ContactWosOrientationField(
        pcloud,
        wos,
        tip_offset=tip_offset,
        contact_threshold=args.contact_threshold,
        T_tag0_base=T_tag0_base,
        query_radius=args.query_radius,
        fit_mode=args.fit_mode,
    )

    controller = TeleoperationController(max_joint_velocity=args.max_joint_velocity)
    controller.set_virtual_fixture(vf)

    if not GEODUDE_MODEL.exists():
        print(f"error: robot model not found: {GEODUDE_MODEL}", file=sys.stderr)
        return 1

    config = ri.Configuration()
    config.model_file = str(GEODUDE_MODEL)
    config.type = "mujoco"

    interface = ri.RobotInterface(config)
    interface.initialize()

    task_space = interface.get_task_space(TASK_SPACE_NAME)
    if task_space is None:
        print(
            f"error: task space {TASK_SPACE_NAME!r} not found in {GEODUDE_MODEL}",
            file=sys.stderr,
        )
        interface.shutdown()
        return 1

    controller.set_task_space(task_space)
    controller.set_interface(interface)

    joint_idx = np.asarray(task_space.get_joint_indices())
    live_position = interface.get_state().get_position()[joint_idx]
    home_ee = task_space.compute_ee_motor(live_position)
    initial_target = vf.project(home_ee)
    objective = ga.control.SingleArmMotorControlObjective(task_space, initial_target)
    controller.set_objective(objective)

    state = TeleopState()
    pending_twist = {"twist": None}
    control_started = False
    prev_motion_enabled = False

    def on_enable() -> None:
        state.enable()

    def on_button(index: int) -> None:
        if index == 0:
            state.teleop_active = not state.teleop_active
        elif index == 1:
            state.vf_enabled = not state.vf_enabled
            controller.set_virtual_fixture(vf if state.vf_enabled else None)

    def on_twist(twist) -> None:
        pending_twist["twist"] = twist

    keyboard = KeyboardTeleop(step=args.keyboard_step)
    keyboard.start(
        on_twist,
        on_button=on_button,
        on_enable=on_enable,
        enable_key=args.enable_key,
    )

    dt = 1.0 / args.control_hz
    try:
        while keyboard._running.is_set():
            t0 = time.perf_counter()

            position = interface.get_state().get_position()[joint_idx]
            ee = task_space.compute_ee_motor(position)

            vf.update_from_state(ee, T_tag0_base)

            if state.vf_enabled and control_started and state.motion_enabled:
                projected = vf.project(ee)
                controller.get_objective().set_target(projected)

            if state.motion_enabled and not prev_motion_enabled:
                if not args.dry_run and not control_started:
                    interface.control(controller, blocking=False)
                    control_started = True

            prev_motion_enabled = state.motion_enabled

            twist = pending_twist["twist"]
            if twist is not None:
                pending_twist["twist"] = None
                if state.commands_allowed(
                    dry_run=args.dry_run, fault=False, tf_ok=True
                ):
                    controller.update_command(twist)

            try:
                vel = interface.get_state().get_velocity()[joint_idx]
                qdot_norm = float(np.linalg.norm(vel))
            except Exception:
                qdot_norm = 0.0

            if args.dry_run or not control_started:
                qdot_norm = 0.0

            line = format_status_line(
                state,
                vf.status,
                qdot_norm=qdot_norm,
                tf_age_ms=0.0,
            )
            print(line, flush=True)

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        keyboard.stop()
        if control_started:
            interface.shutdown()

    return 0


def run_rtde(args: argparse.Namespace) -> int:
    import numpy as np

    from geodude_teleoperation.contact_wos_vf import ContactWosOrientationField
    from geodude_teleoperation.controller import TeleoperationController
    from geodude_teleoperation.frames import load_T_tag0_base, warn_tip_offset_xy
    from geodude_teleoperation.rtde_control_loop import run_rtde_loop
    from geodude_teleoperation.teleop_state import TeleopState

    tip_offset = np.asarray(args.tip_offset, dtype=float)
    warning = warn_tip_offset_xy(tip_offset)
    if warning is not None:
        print(warning)

    T_tag0_base = load_T_tag0_base(args.T_tag0_base)
    pcloud, wos = _load_field_bundle(args)

    vf = ContactWosOrientationField(
        pcloud,
        wos,
        tip_offset=tip_offset,
        contact_threshold=args.contact_threshold,
        T_tag0_base=T_tag0_base,
        query_radius=args.query_radius,
        fit_mode=args.fit_mode,
    )

    controller = TeleoperationController(max_joint_velocity=args.max_joint_velocity)
    controller.set_virtual_fixture(vf)

    state = TeleopState()
    return run_rtde_loop(args, (pcloud, wos), vf, controller, state)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)

    if args.backend == "mujoco":
        return run_mujoco(args)
    if args.backend == "rtde":
        return run_rtde(args)
    print(f"error: unknown backend {args.backend!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
