"""CLI entry for UR stringer keyboard teleoperation (MuJoCo / RTDE backends)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

from geodude_teleoperation.paths import default_ply_path, default_T_tag0_base_path
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
        type=Path,
        default=None,
        help=(
            "Object point cloud in tag_0 frame "
            f"(default: {default_ply_path()})"
        ),
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Voxel downsample size (default from local config/pointclouds.yaml)",
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
        "--diffusion-scalar",
        type=float,
        default=5000.0,
        help=(
            "Dimensionless diffusion time τ/h² (default 5000); "
            "dt = diffusion_scalar * mean_edge_length²"
        ),
    )
    parser.add_argument(
        "--r-field-smoothing",
        type=float,
        default=None,
        help="EMA weight on previous R_field (default 0.75; same as Polyscope teleop keys)",
    )
    parser.add_argument(
        "--vf-relock-dist",
        type=float,
        default=None,
        metavar="M",
        help="Re-query/re-lock teleop R_motion after this tip travel (default 0.012 m)",
    )
    parser.add_argument(
        "--vf-relock-period",
        type=float,
        default=None,
        metavar="S",
        help="Re-query/re-lock teleop R_motion after this time (default 0.10 s)",
    )
    parser.add_argument(
        "--vf-motion-track",
        type=float,
        default=None,
        help="Slerp amount of R_motion toward smoothed R_field on relock (default 0.55)",
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
        "--object-offset",
        type=Path,
        default=None,
        help=(
            "JSON with t_vention_m for T_tag0 translation (default: "
            "<T-tag0-base-stem>.object_offset.json if present)"
        ),
    )
    parser.add_argument(
        "--no-object-offset",
        action="store_true",
        help="Ignore sidecar / --object-offset (T_tag0 unmodified)",
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
        help="Host control / VF query loop rate (Hz)",
    )
    parser.add_argument(
        "--status-hz",
        type=float,
        default=5.0,
        help="Status-line print rate (Hz; <=0 prints every control tick)",
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
        help="RTDE watchdog: no active control tick → speedJ(0)",
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
        help=(
            "Fixed SE(3) pose of tag_0 in base "
            f"(default file if present: {default_T_tag0_base_path()}; "
            "else identity for mujoco)"
        ),
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
        help="ee_object_calib mark meta JSON with T_cam_EE "
        "(AprilTag cross-check; also draws wrist camera in --viz)",
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
    parser.add_argument(
        "--viz",
        action="store_true",
        help=(
            "Polyscope field + tip/TCP view. With --backend rtde: live markers "
            "track pendant/teleop until the window is closed (loop continues until q). "
            "With --viz-teleop-keys: offline one-shot preview."
        ),
    )
    parser.add_argument(
        "--viz-frame",
        choices=("tag0", "vention", "rtde", "base"),
        default="vention",
        help=(
            "Frame for --viz: vention (default, crop_meta world), rtde (UR base), "
            "tag0, or base (alias of vention)"
        ),
    )
    parser.add_argument(
        "--viz-frame-stride",
        type=int,
        default=20,
        help="Draw every Nth local frame in --viz (default 20)",
    )
    parser.add_argument(
        "--viz-teleop-keys",
        action="store_true",
        help=(
            "With --viz: interactive Polyscope preview — hold w/s/a/d/x/y to move "
            "the tip/EE along R_field (same linear axes as live teleop); also "
            "shows one-step ghost tips. Implies --viz-hide-object unless "
            "--viz-show-object is set."
        ),
    )
    parser.add_argument(
        "--viz-hide-object",
        action="store_true",
        help="Hide object point cloud in --viz; show R_field frames (+ BCs) only",
    )
    parser.add_argument(
        "--viz-show-object",
        action="store_true",
        help="Keep object cloud visible even with --viz-teleop-keys",
    )
    parser.add_argument(
        "--viz-laser",
        choices=("off", "overlay", "replace"),
        default="off",
        help=(
            "Laser profilometer cloud (left-base merged.ply → vention → tag0): "
            "off (default), overlay (viz only beside --ply field), "
            "replace (field + viz object from laser; still uses --T-tag0-base "
            "+ object offset for alignment)"
        ),
    )
    parser.add_argument(
        "--laser-ply",
        type=Path,
        default=None,
        help=(
            "Laser merged PLY in left UR base (default: "
            "data/processed/laser_profilometer/merged.ply)"
        ),
    )
    parser.add_argument(
        "--laser-voxel",
        type=float,
        default=0.002,
        help="Voxel size (m) when loading --laser-ply (in-memory; default 0.002)",
    )
    parser.add_argument(
        "--T-left-base-vention",
        type=Path,
        default=None,
        help=(
            "4x4 pose of left UR base in vention_frame (.npy/.json). "
            "Default: assets/frames/T_left_base_vention.npy"
        ),
    )
    parser.add_argument(
        "--laser-offset",
        type=Path,
        default=None,
        help=(
            "JSON with t_vention_m for laser cloud (default: "
            "<laser-ply-stem>.laser_offset.json if present)"
        ),
    )
    parser.add_argument(
        "--no-laser-offset",
        action="store_true",
        help="Ignore laser offset sidecar / --laser-offset",
    )
    parser.add_argument(
        "--T-rtde-base-vention",
        type=Path,
        default=None,
        help=(
            "4x4 pose of RTDE right_base in vention_frame (.npy/.json). "
            "Default: assets/frames/T_right_base_vention.npy"
        ),
    )
    parser.add_argument(
        "--no-rtde-base-vention",
        action="store_true",
        help="Do not apply RTDE↔vention mount (legacy; usually wrong on Geodude)",
    )
    parser.add_argument(
        "--log-motion",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "Write motion CSV to this path (standalone). With --record-mcap, "
            "CSV/PNG are written into the MCAP session folder instead"
        ),
    )
    parser.add_argument(
        "--plot-motion",
        action="store_true",
        help=(
            "Write motion PNG beside the CSV. Implied by --record-mcap "
            "(saved next to each rollout .mcap). With --in-contact, uses the "
            "in-contact layout (Fd/Fs/Fe, a_t1/a_t2/a_n, R_motion, keys)."
        ),
    )
    parser.add_argument(
        "--record-mcap",
        action="store_true",
        help=(
            "Record MCAP per Enter→quit session under --record-output-dir; "
            "also writes motion CSV+PNG into that session folder"
        ),
    )
    parser.add_argument(
        "--record-output-dir",
        type=Path,
        default=None,
        help="MCAP session root (default /tmp/ur_stringer_teleop_logs)",
    )
    parser.add_argument(
        "--record-sample-hz",
        type=float,
        default=0.0,
        help="MCAP sample rate (0 = --control-hz)",
    )
    parser.add_argument(
        "--record-save-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Subscribe to default camera topics for MCAP (warn if missing)",
    )
    parser.add_argument(
        "--record-camera-topics",
        nargs="+",
        default=None,
        metavar="TOPIC",
        help="Override default wrist/scene camera topics for --record-mcap",
    )
    parser.add_argument(
        "--record-target-frame",
        default="vention_frame",
        help="frame_id stamped on recorded poses (default vention_frame)",
    )
    parser.add_argument(
        "--diag-hold-zero",
        action="store_true",
        help=(
            "Wobble diag #1: after enable, command speedJ(0) every tick "
            "(no field project, ignore keyboard). Any wobble → RTDE/watchdog/timing"
        ),
    )
    parser.add_argument(
        "--diag-fixed-orient",
        action="store_true",
        help=(
            "Wobble diag #2: bypass contact_wos_vf; lock R_motion to EE orientation "
            "at first update. Smooth here → wobble is field-tracking/relock"
        ),
    )
    # --- Phase 1.5: line tracking + soft stand-off ---
    parser.add_argument(
        "--line-tracking",
        action="store_true",
        help=(
            "Phase 1.5: free tip along R_field normal (kill normal pose error); "
            "tangent + orientation still track. Implies numpy kinematics on RTDE."
        ),
    )
    parser.add_argument(
        "--soft-standoff",
        action="store_true",
        help=(
            "Phase 1.5: geometric stand-off v_n=-k(d-d*) along outward normal "
            "(enables --line-tracking)"
        ),
    )
    parser.add_argument(
        "--standoff-distance",
        type=float,
        default=0.002,
        metavar="M",
        help="Desired signed distance d* for soft stand-off (default 0.002 m)",
    )
    parser.add_argument(
        "--standoff-gain",
        type=float,
        default=1.0,
        help="Soft stand-off gain k (1/s) for v_n=-k(d-d*)",
    )
    parser.add_argument(
        "--standoff-max-speed",
        type=float,
        default=0.05,
        help="Clamp |v_n| from soft stand-off (m/s, default 0.05)",
    )
    parser.add_argument(
        "--no-keyboard-normal-ff",
        action="store_true",
        help="With line tracking: ignore field-z keypresses (no normal FF)",
    )
    # --- Phase 2: DTC / 500 Hz verification ---
    parser.add_argument(
        "--verify-500hz",
        action="store_true",
        help=(
            "Verify Direct Torque at 500 Hz: print readiness + URScript skeleton; "
            "on live RTDE (not --dry-run) also command directTorque([0]*6) for "
            "--dtc-verify-seconds and report measured Hz"
        ),
    )
    parser.add_argument(
        "--dtc-verify-seconds",
        type=float,
        default=2.0,
        help="Duration of live directTorque(0) rate test when --verify-500hz (default 2)",
    )
    parser.add_argument(
        "--verify-500hz-exit",
        action="store_true",
        help="After --verify-500hz (including live DTC test), exit without teleop",
    )
    parser.add_argument(
        "--polyscope-version",
        default=None,
        help="PolyScope version string for DTC check (e.g. 5.23.0 or 10.10.0); "
        "if omitted, DashboardClient is queried on --verify-500hz",
    )
    # --- Phase 3: tactile F/T ---
    parser.add_argument(
        "--tactile",
        action="store_true",
        help=(
            "Phase 3a: use actual_TCP_force for normal-force FF on speedJ "
            "(enables line tracking; DTC τ=-JᵀW comes in Phase 3b)"
        ),
    )
    parser.add_argument(
        "--desired-normal-force",
        type=float,
        default=5.0,
        help="Desired contact force magnitude along approach (N)",
    )
    parser.add_argument(
        "--wrench-kp",
        type=float,
        default=0.02,
        help="Tactile wrench P gain (force axes)",
    )
    parser.add_argument(
        "--wrench-ki",
        type=float,
        default=0.0,
        help="Tactile wrench I gain",
    )
    parser.add_argument(
        "--wrench-kd",
        type=float,
        default=0.0,
        help="Tactile wrench D gain",
    )
    parser.add_argument(
        "--wrench-to-vn",
        type=float,
        default=0.01,
        help="Map force error to normal speed (m/s per N) on speedJ path",
    )
    parser.add_argument(
        "--tactile-max-vn",
        type=float,
        default=0.03,
        help="Clamp tactile normal speed (m/s)",
    )
    parser.add_argument(
        "--ft-bias",
        type=float,
        nargs=6,
        metavar=("FX", "FY", "FZ", "TX", "TY", "TZ"),
        default=None,
        help="Optional wrench bias subtracted from actual_TCP_force",
    )
    # --- In-contact dual-loop teleop ---
    parser.add_argument(
        "--in-contact",
        action="store_true",
        help=(
            "Use DualLoop in-contact control (Align tool-Z→n* first, then Force PID "
            "@ --desired-normal-force; keyboard tangent w/s a/d only). "
            "Requires --backend rtde. Bypasses free-space pose-ref teleop. "
            "With --viz: shows object + tangent/approach overlays before teleop."
        ),
    )
    parser.add_argument(
        "--force-sign",
        type=float,
        default=1.0,
        help=(
            "Sign for Fs=force_sign*(F−bias)·n* (default +1: compression increases "
            "F·n* on this cell). Use -1 if Enter seeks away from the surface."
        ),
    )
    parser.add_argument(
        "--no-align-first",
        dest="align_first",
        action="store_false",
        help="Seek force immediately on Enter (skip tool-Z→n* align gate)",
    )
    parser.set_defaults(align_first=True)
    parser.add_argument(
        "--align-enter-tol-deg",
        type=float,
        default=5.0,
        help="Tool-Z within this of n* (deg) before Force seek (default 5)",
    )
    parser.add_argument(
        "--align-exit-tol-deg",
        type=float,
        default=12.0,
        help="If tool-Z error exceeds this (deg), pause Force and re-align (default 12)",
    )
    parser.add_argument(
        "--approach-v-max",
        type=float,
        default=0.006,
        help="Soft-approach speed cap along n* before light contact (m/s)",
    )
    parser.add_argument(
        "--force-max-v",
        type=float,
        default=0.010,
        help="Max Force PID along-normal speed (m/s)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.ply is None:
        args.ply = default_ply_path()
    if args.T_tag0_base is None:
        candidate = default_T_tag0_base_path()
        if candidate.is_file():
            args.T_tag0_base = candidate
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.arm != "right":
        raise SystemExit(f"v1 supports --arm right only, got {args.arm!r}")
    if args.backend == "rtde" and not args.robot_ip:
        raise SystemExit("--robot-ip is required for --backend rtde")
    if args.enable_key in ("t", "v", "f"):
        raise SystemExit("--enable-key must not be 't', 'v', or 'f'")
    if args.control_hz <= 0:
        raise SystemExit("--control-hz must be positive")
    if args.viz_frame_stride < 1:
        raise SystemExit("--viz-frame-stride must be >= 1")
    if args.viz_teleop_keys and not args.viz:
        raise SystemExit("--viz-teleop-keys requires --viz")
    if getattr(args, "viz_laser", "off") == "overlay" and not args.viz:
        raise SystemExit("--viz-laser overlay requires --viz")
    if getattr(args, "laser_voxel", 0) is not None and float(args.laser_voxel) < 0:
        raise SystemExit("--laser-voxel must be >= 0")
    if args.diag_hold_zero and args.diag_fixed_orient:
        raise SystemExit("use only one of --diag-hold-zero / --diag-fixed-orient")
    if args.plot_motion and args.log_motion is None and not getattr(
        args, "record_mcap", False
    ):
        raise SystemExit(
            "--plot-motion requires --log-motion CSV path "
            "(or use --record-mcap to save CSV/PNG in the session folder)"
        )
    if getattr(args, "record_sample_hz", 0) < 0:
        raise SystemExit("--record-sample-hz must be >= 0")
    if getattr(args, "tactile", False) and args.backend != "rtde":
        raise SystemExit("--tactile requires --backend rtde (ActualTCPForce)")
    if getattr(args, "in_contact", False) and args.backend != "rtde":
        raise SystemExit("--in-contact requires --backend rtde (ActualTCPForce)")
    if getattr(args, "in_contact", False) and getattr(args, "tactile", False):
        raise SystemExit("use only one of --in-contact / --tactile")
    if getattr(args, "verify_500hz_exit", False) and not getattr(
        args, "verify_500hz", False
    ):
        raise SystemExit("--verify-500hz-exit requires --verify-500hz")
    if getattr(args, "dtc_verify_seconds", 0) <= 0:
        raise SystemExit("--dtc-verify-seconds must be positive")
    if getattr(args, "standoff_distance", 0) < 0:
        raise SystemExit("--standoff-distance must be >= 0")
    if getattr(args, "standoff_gain", 0) < 0:
        raise SystemExit("--standoff-gain must be >= 0")
    # Field-only preview by default with teleop keys (less point-cloud noise).
    # --in-contact + --viz keeps the object visible so you can check approach/tangent.
    if getattr(args, "in_contact", False) and args.viz and not args.viz_hide_object:
        args.viz_hide_object = False
    elif args.viz and (
        args.viz_hide_object or (args.viz_teleop_keys and not args.viz_show_object)
    ):
        args.viz_hide_object = True
    elif args.viz_show_object:
        args.viz_hide_object = False


def _vf_tracking_kwargs(args: argparse.Namespace) -> dict:
    """Polyscope teleop-key defaults applied to live ContactWosOrientationField."""
    from geodude_teleoperation.teleop.contact_wos_vf import (
        DEFAULT_MOTION_TRACK,
        DEFAULT_RELOCK_DIST_M,
        DEFAULT_RELOCK_PERIOD_S,
        DEFAULT_R_FIELD_SMOOTHING,
    )

    return {
        "r_field_smoothing": (
            DEFAULT_R_FIELD_SMOOTHING
            if args.r_field_smoothing is None
            else float(args.r_field_smoothing)
        ),
        "relock_dist_m": (
            DEFAULT_RELOCK_DIST_M
            if args.vf_relock_dist is None
            else float(args.vf_relock_dist)
        ),
        "relock_period_s": (
            DEFAULT_RELOCK_PERIOD_S
            if args.vf_relock_period is None
            else float(args.vf_relock_period)
        ),
        "motion_track": (
            DEFAULT_MOTION_TRACK
            if args.vf_motion_track is None
            else float(args.vf_motion_track)
        ),
    }


def _load_field_bundle(args: argparse.Namespace, T_tag0_vention=None):
    from geodude_teleoperation.field.stringer_field import (
        build_wos,
        compute_orientation_field,
        config_defaults,
        load_stringer_pointcloud,
        pointcloud_from_points,
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
    laser_mode = getattr(args, "viz_laser", "off")
    if laser_mode == "replace":
        if T_tag0_vention is None:
            raise SystemExit("--viz-laser replace requires T_tag0_vention")
        from geodude_teleoperation.frames import load_T_left_base_vention
        from geodude_teleoperation.teleop.laser_cloud import (
            load_laser_points_tag0,
            resolve_laser_ply,
        )

        laser_voxel = float(getattr(args, "laser_voxel", 0.002))
        # Prefer --voxel-size for field density when set; else laser voxel.
        field_voxel = (
            float(args.voxel_size)
            if args.voxel_size is not None
            else laser_voxel
        )
        T_left = load_T_left_base_vention(
            getattr(args, "T_left_base_vention", None)
        )
        ply = resolve_laser_ply(getattr(args, "laser_ply", None))
        print(
            f"laser replace: loading {ply} → tag0 "
            f"(voxel={laser_voxel} m, field_voxel={field_voxel} m)",
            flush=True,
        )
        pts_tag0 = load_laser_points_tag0(
            ply,
            T_tag0_vention,
            voxel_m=laser_voxel,
            T_left_base_vention=T_left,
            t_vention_m=getattr(args, "laser_offset_t", None),
        )
        pcloud = pointcloud_from_points(
            pts_tag0, field_voxel, normal_orientation
        )
    else:
        pcloud = load_stringer_pointcloud(args.ply, voxel_size, normal_orientation)
    sd = compute_orientation_field(
        pcloud,
        field_mode=args.field_mode,
        field_flow=args.field_flow,
        end_line_fraction=args.end_line_fraction,
        swap_end_lines=args.swap_end_lines,
        diffusion_scalar=args.diffusion_scalar,
    )
    wos = build_wos(pcloud, args.wos_batch_size)
    print(validate_field_prints(pcloud, sd, wos.convergence_threshold))
    return pcloud, wos, sd


def _peek_rtde_tcp_pose(robot_ip: str):
    """Receive-only snapshot for --viz markers (connect, read, disconnect)."""
    from geodude_teleoperation.rtde.control_loop import open as rtde_open

    session = rtde_open(str(robot_ip), dry_run=True)
    try:
        return session.read_state()["tcp_pose"]
    finally:
        session.close()


def _load_T_rtde_base_vention(args: argparse.Namespace):
    """Return T_rtde_base_vention or None when disabled."""
    if getattr(args, "no_rtde_base_vention", False):
        return None
    from geodude_teleoperation.frames import load_T_rtde_base_vention

    return load_T_rtde_base_vention(getattr(args, "T_rtde_base_vention", None))


def run_mujoco(args: argparse.Namespace) -> int:
    import numpy as np

    try:
        import gafropy
        import gafropy as ga
        from gafropy import robot_interface as ri
    except ImportError as exc:
        print(f"error: gafropy is required for mujoco backend: {exc}", file=sys.stderr)
        return 1

    from geodude_teleoperation.teleop.contact_wos_vf import ContactWosOrientationField
    from geodude_teleoperation.teleop.controller import TeleoperationController
    from geodude_teleoperation.frames import load_T_tag0_base, warn_tip_offset_xy
    from geodude_teleoperation.teleop.keyboard import KeyboardTeleop
    from geodude_teleoperation.teleop.teleop_state import (
        TeleopState,
        format_status_line,
        should_emit_status,
    )

    tip_offset = np.asarray(args.tip_offset, dtype=float)
    warning = warn_tip_offset_xy(tip_offset)
    if warning is not None:
        print(warning)

    from geodude_teleoperation.viz.field_viz import maybe_show_startup_viz
    from geodude_teleoperation.teleop.object_offset import (
        apply_object_t_offset,
        bind_object_offset_to_args,
    )

    bind_object_offset_to_args(args)
    # MuJoCo / gafropy geodude root is vention-like; crop_meta T is already vention.
    T_tag0_vention_raw = load_T_tag0_base(args.T_tag0_base)
    T_tag0_vention = apply_object_t_offset(T_tag0_vention_raw, args.object_offset_t)
    if getattr(args, "viz_laser", "off") != "off":
        from geodude_teleoperation.teleop.laser_offset import bind_laser_offset_to_args

        bind_laser_offset_to_args(args)
    else:
        args.laser_offset_t = np.zeros(3, dtype=float)
        args.laser_offset_step_m = 0.001
        args.laser_offset_path = None
    pcloud, wos, sd = _load_field_bundle(args, T_tag0_vention=T_tag0_vention)

    vf = ContactWosOrientationField(
        pcloud,
        wos,
        tip_offset=tip_offset,
        contact_threshold=args.contact_threshold,
        T_tag0_base=T_tag0_vention,
        query_radius=args.query_radius,
        fit_mode=args.fit_mode,
        **_vf_tracking_kwargs(args),
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

    if args.viz:
        from geodude_teleoperation.se3 import matrix_from_motor, matrix_to_tcp_pose

        maybe_show_startup_viz(
            args,
            pcloud,
            sd,
            T_tag0_vention=T_tag0_vention,
            T_rtde_base_vention=None,
            tip_offset=tip_offset,
            tcp_pose_rtde=matrix_to_tcp_pose(matrix_from_motor(home_ee)),
            wos=wos,
            T_tag0_tcp_base=T_tag0_vention,
            T_tag0_vention_raw=T_tag0_vention_raw,
        )

    initial_target = home_ee
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
    last_status_t = float("-inf")
    status_hz = float(getattr(args, "status_hz", 5.0))
    try:
        while keyboard._running.is_set():
            t0 = time.perf_counter()
            now = time.monotonic()

            position = interface.get_state().get_position()[joint_idx]
            ee = task_space.compute_ee_motor(position)

            vf.update_from_state(ee, T_tag0_base)

            # No per-tick project(ee)→objective: that yanks orientation to live
            # R_field after enable with no keys. Twists project via update_command.

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

            if should_emit_status(last_status_t, now, status_hz=status_hz):
                last_status_t = now
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
    import threading

    import numpy as np

    from geodude_teleoperation.teleop.contact_wos_vf import ContactWosOrientationField
    from geodude_teleoperation.teleop.controller import TeleoperationController
    from geodude_teleoperation.frames import (
        compose_T_tag0_rtde,
        load_T_tag0_base,
        warn_tip_offset_xy,
    )
    from geodude_teleoperation.viz.field_viz import LiveVizState, maybe_show_startup_viz
    from geodude_teleoperation.rtde.control_loop import run_rtde_loop
    from geodude_teleoperation.teleop.teleop_state import TeleopState

    tip_offset = np.asarray(args.tip_offset, dtype=float)
    warning = warn_tip_offset_xy(tip_offset)
    if warning is not None:
        print(warning)

    from geodude_teleoperation.teleop.object_offset import (
        apply_object_t_offset,
        bind_object_offset_to_args,
    )

    bind_object_offset_to_args(args)
    T_tag0_vention_raw = load_T_tag0_base(args.T_tag0_base)
    T_tag0_vention = apply_object_t_offset(T_tag0_vention_raw, args.object_offset_t)
    if getattr(args, "viz_laser", "off") != "off":
        from geodude_teleoperation.teleop.laser_offset import bind_laser_offset_to_args

        bind_laser_offset_to_args(args)
    else:
        args.laser_offset_t = np.zeros(3, dtype=float)
        args.laser_offset_step_m = 0.001
        args.laser_offset_path = None
    try:
        T_rtde_base_vention = _load_T_rtde_base_vention(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if T_rtde_base_vention is not None:
        T_tag0_for_vf = compose_T_tag0_rtde(T_tag0_vention, T_rtde_base_vention)
        print(
            "frames: crop_meta T_tag0 in vention; VF uses T_tag0 in RTDE base "
            f"(mount t={T_rtde_base_vention[:3, 3].tolist()})",
            flush=True,
        )
    else:
        T_tag0_for_vf = T_tag0_vention
        print(
            "warning: --no-rtde-base-vention: treating RTDE TCP as vention "
            "(crop_meta frame) — tip vs cloud will be wrong on Geodude",
            file=sys.stderr,
        )

    pcloud, wos, sd = _load_field_bundle(args, T_tag0_vention=T_tag0_vention)

    # Offline Polyscope preview (--viz-teleop-keys): one-shot before the loop.
    use_live_viz = bool(args.viz) and not bool(getattr(args, "viz_teleop_keys", False))
    if args.viz and not use_live_viz:
        tcp_pose = None
        try:
            tcp_pose = _peek_rtde_tcp_pose(args.robot_ip)
        except Exception as exc:
            print(
                f"warning: could not read RTDE TCP for --viz ({exc}); "
                "showing field without tip/TCP markers",
                file=sys.stderr,
            )
        maybe_show_startup_viz(
            args,
            pcloud,
            sd,
            T_tag0_vention=T_tag0_vention,
            T_rtde_base_vention=T_rtde_base_vention,
            tip_offset=tip_offset,
            tcp_pose_rtde=tcp_pose,
            wos=wos,
            T_tag0_tcp_base=T_tag0_for_vf,
            T_tag0_vention_raw=T_tag0_vention_raw,
        )

    if args.diag_hold_zero:
        print(
            "diag: --diag-hold-zero — speedJ(0) after enable; "
            "no field project / ignore keyboard",
            flush=True,
        )
        from geodude_teleoperation.teleop.contact_wos_vf import FixedOrientationField

        # Still need a status object for the status line; never projected.
        vf = FixedOrientationField()
    elif args.diag_fixed_orient:
        print(
            "diag: --diag-fixed-orient — constant R_motion (EE at first update); "
            "bypasses contact_wos_vf / relock",
            flush=True,
        )
        from geodude_teleoperation.teleop.contact_wos_vf import FixedOrientationField

        vf = FixedOrientationField()
    else:
        track = _vf_tracking_kwargs(args)
        print(
            "vf tracking: "
            f"R_smooth={track['r_field_smoothing']:.2f}  "
            f"relock={track['relock_dist_m']*1000:.0f}mm/"
            f"{track['relock_period_s']*1000:.0f}ms  "
            f"motion_track={track['motion_track']:.2f}",
            flush=True,
        )
        vf = ContactWosOrientationField(
            pcloud,
            wos,
            tip_offset=tip_offset,
            contact_threshold=args.contact_threshold,
            T_tag0_base=T_tag0_for_vf,
            query_radius=args.query_radius,
            fit_mode=args.fit_mode,
            **track,
        )

    controller = TeleoperationController(max_joint_velocity=args.max_joint_velocity)
    if args.diag_hold_zero:
        controller.set_virtual_fixture(None)
    else:
        controller.set_virtual_fixture(vf)

    state = TeleopState()

    if not use_live_viz:
        return run_rtde_loop(args, (pcloud, wos), vf, controller, state)

    # Polyscope on main thread; RTDE loop publishes TCP on a worker thread.
    live_viz = LiveVizState()

    def _loop_worker() -> None:
        try:
            live_viz.loop_exit_code = int(
                run_rtde_loop(
                    args,
                    (pcloud, wos),
                    vf,
                    controller,
                    state,
                    live_viz=live_viz,
                )
            )
        except BaseException as exc:  # noqa: BLE001 — surface to main / join
            live_viz.loop_error = exc
            live_viz.loop_exit_code = 1
            print(f"error: RTDE loop failed: {exc}", file=sys.stderr)
        finally:
            live_viz.loop_done.set()

    worker = threading.Thread(target=_loop_worker, name="rtde-control", daemon=False)
    worker.start()

    # Wait briefly for first TCP publish so the first frame can seed markers.
    seed = None
    deadline = time.time() + 2.0
    while time.time() < deadline and not live_viz.loop_done.is_set():
        _seq, tcp, _R, _d, _c = live_viz.snapshot()
        if tcp is not None:
            seed = tcp
            break
        time.sleep(0.05)

    try:
        maybe_show_startup_viz(
            args,
            pcloud,
            sd,
            T_tag0_vention=T_tag0_vention,
            T_rtde_base_vention=T_rtde_base_vention,
            tip_offset=tip_offset,
            tcp_pose_rtde=seed,
            wos=wos,
            T_tag0_tcp_base=T_tag0_for_vf,
            live_viz=live_viz,
            T_tag0_vention_raw=T_tag0_vention_raw,
        )
    except SystemExit:
        live_viz.mark_viz_closed()
        raise
    finally:
        live_viz.mark_viz_closed()

    if live_viz.loop_done.is_set():
        worker.join(timeout=5.0)
        if live_viz.loop_error is not None:
            return 1
        return int(live_viz.loop_exit_code)

    print(
        "[viz] waiting for RTDE loop (terminal q to quit)…",
        flush=True,
    )
    worker.join()
    if live_viz.loop_error is not None:
        return 1
    return int(live_viz.loop_exit_code)


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
