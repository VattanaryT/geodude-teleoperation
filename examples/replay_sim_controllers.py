"""Check 6 / Test 1: closed-loop MuJoCo replay through speedJ + DualLoop.

Never copies bag ``q`` into DualLoop. SpeedJ/DualLoop take a Cartesian target;
bag joints are only the start pose (and, for Test 1, the FK source of TCP).

Check 6 (default): Test 2 shape. Contact XY is CGA ``a_local`` sandwiched into
the flange. That is the policy path.

``--bag-tcp``: Test 1. Command recorded TCP poses. Contact XY is bag TCP
Δp/Δt in the flange, not ``a_local``. If this walks off, Test 2 cannot succeed.

MuJoCo pose-spring is not the UR. This check does not talk to hardware.

In contact DualLoop owns tool-Z: score XY, not 3D TCP. ``--follow-path`` skips
DualLoop and keeps ``set_target`` through contact.

  PYTHONPATH=src python examples/local_frame_checks.py 6
  PYTHONPATH=src python examples/local_frame_checks.py 7

  PYTHONPATH=src python examples/replay_sim_controllers.py --headless --frames 40
  PYTHONPATH=src python examples/replay_sim_controllers.py --bag-tcp --headless --frames 40
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import gafropy as ga
import numpy as np
from gafropy import robot_interface as ri
from gafropy import viz as gafropy_viz
from scipy.spatial.transform import Rotation

from geodude_teleoperation import AdmittanceTeleoperationController, ContactTeleoperationController
from geodude_teleoperation.local_frames_npz import integrate_desired_xyz, load_cga_episode
from geodude_teleoperation.mesh_fixture import load_points
from geodude_teleoperation.mode_segments import sandwich_local_actions
from geodude_teleoperation.sim_replay import (
    TrackingRow,
    absolute_error_mm,
    apply_mode_edge,
    bag_tcp_flange_velocity,
    body_twist_for_plant,
    enter_speedj_contact,
    exit_speedj_contact,
    format_tracking_table,
    path_follow_contact,
    replay_mode_label,
    score_legend,
    summarize_tracking,
    ur_reaction_fz,
)
from geodude_teleoperation.ee_viz import (
    add_right_chamfer_visual,
    chamfer_tcp_world_from_wrist3,
    chamfer_world_from_wrist3,
    hide_gripper_visuals,
    motor_aligned_to,
    motor_from_matrix,
)
from geodude_teleoperation.laser_frames import (
    right_base_R_to_vention,
    right_base_points_to_vention,
)
from geodude_teleoperation.tcp_kinematics import (
    MODEL,
    bag_tcp_to_yaml_ee_world,
    poses_in_right_base,
    write_geodude_model_with_right_ur6,
)
from geodude_teleoperation.viser_view import (
    AXIS_RGB,
    axis_arrows,
    contact_segment_colors,
    polyline_segments,
    wxyz_from_R,
)
from geodude_teleoperation.wrist_cam import (
    WRIST_CAM_TOPIC,
    annotate_wrist_frame,
    decode_jpeg,
    load_wrist_cam,
    nearest_indices,
)

_DATA = Path("/mnt/data/boeing/local_frames_data/08_16_half_stringer_local_frames")
_DEFAULT_NPZ = _DATA / "bags_08_16_half_stringer_bimanual_0_cga_actual.npz"
_DEFAULT_TRAIN = _DATA / "training" / "bags_08_16_half_stringer_bimanual_0.npz"
_DEFAULT_PLY = _DATA / "merged_rightbase.ply"
_DEFAULT_BAG = Path("/mnt/data/boeing/data/bags_08_16_half_stringer/bimanual_0")
_DEFAULT_CAM = _DATA / "bags_08_16_half_stringer_bimanual_0_wrist_cam.npz"
_DEFAULT_JOINTS = Path(__file__).resolve().parent.parent / (
    "tests/fixtures/bimanual_0_tcp_fk_samples.npz"
)
_START_MODEL = MODEL.parent / ".geodude_start_right_ur6.yaml"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, default=_DEFAULT_NPZ)
    p.add_argument("--training", type=Path, default=_DEFAULT_TRAIN)
    p.add_argument("--frames", type=int, default=0, help="0 = whole episode")
    p.add_argument("--fast", action="store_true",
                   help="sleep 10 ms/step instead of demo dt (sim still integrates)")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--verbose-rows", action="store_true",
                   help="print every sample, not just the segment summary")
    p.add_argument("--ply", type=Path, default=_DEFAULT_PLY)
    p.add_argument("--bag", type=Path, default=_DEFAULT_BAG)
    p.add_argument("--cam", type=Path, default=None)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--bag-tcp", action="store_true",
        help="Test 1: recorded TCP through speedJ+DualLoop; contact XY from bag Δp, not a_local",
    )
    p.add_argument(
        "--follow-path", action="store_true",
        help="keep a_n / never DualLoop; score 3D TCP (does not sign off force loop)",
    )
    p.add_argument(
        "--no-bag-start", action="store_true",
        help="boot yaml home joints instead of bag start (cyan off orange at t=0)",
    )
    p.add_argument(
        "--joints", type=Path, default=_DEFAULT_JOINTS,
        help="NPZ with q_ur6[0] for bag start joints",
    )
    return p


def _make_contact(task_space, interface, home_ee_motor, measured_fz_fn):
    stiffness = np.diag([400.0] * 5 + [0.0])
    z_damping = 200.0
    cfg = ga.control.VelocityAdmittanceControllerConfig()
    cfg.dt = 0.01
    cfg.max_linear_speed = 0.005
    cfg.max_angular_speed = 0.15
    cfg.admittance = ga.control.AdmittanceConfig(
        inertia=1.0 * np.eye(6),
        damping=np.diag([40.0] * 5 + [z_damping]),
        stiffness=stiffness,
    )
    contact = ContactTeleoperationController(
        None, cfg, target_force=15.0,
        kp_force=0.001, ki_force=0.0003,
        v_approach=0.005, max_v_hold=0.002, z_damping=z_damping,
        v_slew_mps2=0.05, lost_s=1.0, sit_s=2.0,
        force_sign=1.0, log_period_s=0.0,
        measured_fz_fn=measured_fz_fn,
    )
    contact.set_task_space(task_space)
    contact.set_interface(interface)
    contact.set_objective(
        ga.control.SingleArmLineControlObjective(
            home_ee_motor.transform_line(ga.Line.Z())))
    return contact


def main() -> None:
    args = build_parser().parse_args()
    train = args.training if args.training.is_file() else None
    ep = load_cga_episode(args.npz, training_path=train)
    n = len(ep.a_local)
    t = np.asarray(ep.timestamps, dtype=float)
    t = t - t[0]
    dt_s = float(np.diff(t).mean()) if n > 1 else 0.1
    R_flange = Rotation.from_quat(np.asarray(ep.poses[: n + 1, 3:7], dtype=float)).as_matrix()
    raw = np.asarray(ep.in_contact[:n], dtype=bool)
    fn = None if ep.f_n is None else np.asarray(ep.f_n, dtype=float).reshape(-1)[:n]
    n_run = n if args.frames <= 0 else min(n, int(args.frames))
    bag_xyz = np.asarray(ep.action_xyz[: n_run + 1], dtype=float)
    if args.bag_tcp:
        v_raw = bag_tcp_flange_velocity(
            bag_xyz, R_flange[: len(bag_xyz)], dt_s)
        v_stick = np.zeros((n_run, 3), dtype=float)
        n_v = min(n_run, len(v_raw))
        v_stick[:n_v] = v_raw[:n_v]
    else:
        sandwich_contact = path_follow_contact(raw, follow_path=args.follow_path)
        _, v_stick, _, _, _ = sandwich_local_actions(
            R_field=ep.R_field, R_ee=R_flange, a_local=ep.a_local,
            in_contact=sandwich_contact, timestamps=t, f_n=fn, lost_s=0.0)

    check_n = 7 if args.bag_tcp else 6
    replay_kind = "bag-tcp" if args.bag_tcp else ("path" if args.follow_path else "controllers")
    print(
        f"check {check_n}  episode={ep.meta.get('episode')}  {ep.hz} Hz  "
        f"frames={n_run}/{n}  plant=yaml right_ee  replay={replay_kind}  "
        f"error=chamfer vs bag (absolute)  target=bag TCP pose-spring",
        flush=True,
    )
    if args.bag_tcp:
        print(
            "test 1: recorded TCP through speedJ+DualLoop. "
            "Contact XY = bag Δp/Δt in flange (not a_local). "
            "Joints only set the start pose. Not hardware.",
            flush=True,
        )
    print(score_legend(follow_path=args.follow_path), flush=True)
    print(
        "mode=raw in_contact  Fd=|f_n| at enter  "
        "Fz=bag playback (not MuJoCo contact)  not hardware",
        flush=True,
    )
    if args.fast:
        print(
            "--fast sleeps 10 ms/step; mm error is not realtime tracking. "
            "Drop --fast to measure the spring.",
            flush=True,
        )

    cam_pre = None
    if not args.headless:
        cam_src = args.cam if args.cam is not None else (
            _DEFAULT_CAM if _DEFAULT_CAM.is_file() else args.bag)
        print(f"loading wrist cam from {cam_src} ({WRIST_CAM_TOPIC})", flush=True)
        cam_pre = load_wrist_cam(cam_src, WRIST_CAM_TOPIC)

    model_file = MODEL
    if not args.no_bag_start:
        if not args.joints.is_file():
            raise FileNotFoundError(f"bag start joints missing: {args.joints}")
        q6 = np.asarray(np.load(args.joints)["q_ur6"][0], dtype=float).reshape(6)
        model_file = write_geodude_model_with_right_ur6(q6, _START_MODEL)
        print(f"start joints from {args.joints.name} -> {model_file.name}", flush=True)

    config = ri.Configuration()
    config.model_file = str(model_file)
    config.type = "mujoco"
    interface = ri.RobotInterface(config)
    interface.initialize()
    system = interface.get_system()
    task_space = interface.get_task_space("right_ee")
    if task_space is None:
        interface.shutdown()
        raise RuntimeError("task space 'right_ee' missing")
    joint_idx = np.array(task_space.get_joint_indices())
    home = task_space.compute_ee_motor(interface.get_state().get_position()[joint_idx])
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
    approach.set_objective(ga.control.SingleArmMotorControlObjective(home))
    approach._max_slide_mps = 0.08
    _speedj = approach.compute_control_command

    def _speedj_lock_rail(robot_state, objective_state, task_space_state):
        qdot = np.asarray(
            _speedj(robot_state, objective_state, task_space_state), dtype=float).copy()
        qdot[0] = 0.0
        return qdot

    approach.compute_control_command = _speedj_lock_rail

    fz_state = {"i": 0, "contact": False, "fn": 0.0}

    def measured_fz():
        return ur_reaction_fz(fz_state["fn"], in_contact=fz_state["contact"])

    contact = _make_contact(task_space, interface, home, measured_fz)
    mode = {"active": "path" if args.follow_path else "approach"}
    rows: list[TrackingRow] = []
    bag0 = bag_xyz[0]
    view = {"draw": None}

    def live_ee_motor():
        return task_space.compute_ee_motor(
            interface.get_state().get_position()[joint_idx])

    def live_frames():
        q = np.asarray(interface.get_state().get_position(), dtype=float)
        return q, poses_in_right_base(system, q)

    def on_enter():
        i = fz_state["i"]
        fd = 15.0 if fn is None else abs(float(fn[i]))
        contact._target_force = fd
        contact._contact_engage_n = max(0.4, 0.25 * fd)
        enter_speedj_contact(
            mode=mode, contact_controller=contact, approach_controller=approach)

    def on_exit():
        exit_speedj_contact(
            mode=mode, contact_controller=contact, approach_controller=approach)

    def step(i: int) -> None:
        fz_state["i"] = i
        fz_state["contact"] = bool(raw[i])
        fz_state["fn"] = 0.0 if fn is None else float(fn[i])
        if args.follow_path:
            mode["active"] = "path"
        else:
            apply_mode_edge(
                mode, bool(raw[i]), on_enter=on_enter, on_exit=on_exit)
        _q, frames = live_frames()
        pose_i = ep.poses[min(i, len(ep.poses) - 1)]
        m_des = motor_aligned_to(
            motor_from_matrix(bag_tcp_to_yaml_ee_world(system, _q, pose_i)),
            live_ee_motor())
        if mode["active"] == "contact":
            vel = body_twist_for_plant(
                v_stick[i], dt_s=dt_s, contact=True,
                R_flange=R_flange[i], R_plant=frames["yaml_ee"][:3, :3])
            approach.update_command(ga.Twist.from_array(vel))
        else:
            approach.get_objective().set_target(m_des)
            approach.set_input_active(True)
        xy, z, err = absolute_error_mm(
            live_xyz=frames["chamfer"][:3, 3], bag_xyz=bag_xyz[i])
        rows.append(TrackingRow(
            i=i, t=float(t[i]),
            mode=replay_mode_label(bool(raw[i]), follow_path=args.follow_path),
            err_xy_mm=xy, err_z_mm=z, err_mm=err,
            abs_fn=None if fn is None else abs(float(fn[i])),
            plant_vz=float(approach._tool_z_velocity),
        ))
        if view["draw"] is not None:
            view["draw"](i, frames["chamfer"][:3, 3], frames["flange"][:3, :3], err, _q)

    def run_clock() -> None:
        last_mode = None
        try:
            for i in range(n_run):
                step(i)
                if rows and rows[-1].mode != last_mode:
                    last_mode = rows[-1].mode
                    r = rows[-1]
                    print(
                        f"t={r.t:.2f}s -> {r.mode}  "
                        f"xy={r.err_xy_mm:.1f}mm  z={r.err_z_mm:.1f}mm  "
                        f"err={r.err_mm:.1f}mm",
                        flush=True,
                    )
                time.sleep(0.01 if args.fast else dt_s)
        finally:
            if mode["active"] == "contact":
                on_exit()
            _print_table()

    def _print_table() -> None:
        if not rows:
            print("no samples", flush=True)
            return
        print(flush=True)
        print(score_legend(follow_path=args.follow_path), flush=True)
        print(summarize_tracking(rows), flush=True)
        contact_rows = [r for r in rows if r.mode == "contact"]
        if contact_rows:
            xys = np.array([r.err_xy_mm for r in contact_rows], dtype=float)
            zs = np.array([r.err_z_mm for r in contact_rows], dtype=float)
            print(
                f"contact score mean_xy={float(xys.mean()):.2f} mm  "
                f"(DualLoop Z mean_z={float(zs.mean()):.2f} mm)",
                flush=True,
            )
        errs = np.array([r.err_mm for r in rows], dtype=float)
        print(
            f"overall mean {float(errs.mean()):.2f} mm  max {float(errs.max()):.2f} mm  "
            f"n={len(rows)}",
            flush=True,
        )
        if args.verbose_rows:
            print(flush=True)
            print(format_tracking_table(rows), flush=True)

    _q0, fr0 = live_frames()
    start_mm = float(np.linalg.norm(fr0["chamfer"][:3, 3] - bag0) * 1e3)
    start_note = (
        "yaml home, not bag joints" if args.no_bag_start
        else "bag start joints"
    )
    print(
        f"sim chamfer vs bag TCP at t=0: {start_mm:.1f} mm ({start_note}). "
        f"Each tick tracks bag TCP on the pose-spring (live Motor hemisphere). "
        f"Stringer is T_right overlay; rails stay at yaml 0.25 m.",
        flush=True,
    )

    def start_plant() -> None:
        interface.control(approach, blocking=False)
        approach.hold_live_pose()
        approach.set_input_active(True)

    def _build_wrist_view():
        n_pose = len(ep.poses)
        actual = np.asarray(ep.poses[:, :3], dtype=float)
        mode_flags = np.zeros(n_pose, dtype=bool)
        mode_flags[: min(n_pose, len(raw))] = raw[:n_pose]
        if cam_pre is None:
            cam_src = args.cam if args.cam is not None else (
                _DEFAULT_CAM if _DEFAULT_CAM.is_file() else args.bag)
            print(f"loading wrist cam from {cam_src} ({WRIST_CAM_TOPIC})", flush=True)
            cam_t, jpegs = load_wrist_cam(cam_src, WRIST_CAM_TOPIC)
        else:
            cam_t, jpegs = cam_pre
        stamps = np.asarray(ep.timestamps, dtype=float)
        idx = nearest_indices(stamps, cam_t)
        q0 = np.asarray(interface.get_state().get_position(), dtype=float)
        cloud = load_points(args.ply) if args.ply.is_file() else None
        cloud_w = None if cloud is None else right_base_points_to_vention(cloud)
        actual_w = right_base_points_to_vention(actual)
        replay_w = None
        if not args.bag_tcp:
            replay_xyz = integrate_desired_xyz(ep.action_xyz[0], ep.R_field, ep.a_local)
            replay_w = right_base_points_to_vention(replay_xyz)
        if cloud is None:
            print(f"stringer ply missing: {args.ply}", flush=True)
        else:
            print(
                f"stringer {args.ply.name} in vention via T_right_base_vention "
                f"(worktop overlay, yaml rail 0.25 m, not 0.50 m)",
                flush=True,
            )

        viz = gafropy_viz.Visualizer(port=args.port)
        server = viz.server
        server.gui.configure_theme(control_width="large")
        server.gui.main_panel.dock_right()
        server.gui.main_panel.set_width(560)
        robot_viz = viz.add_robot(interface.get_system(), name="geodude", joint_sliders=False)
        robot_viz.update(q0)
        n_hid = hide_gripper_visuals(robot_viz)
        update_chamfer = add_right_chamfer_visual(viz, interface.get_system(), q=q0)
        print(f"ee viz: hid {n_hid} gripper meshes; gray chamfer = live sim tool", flush=True)
        if cloud_w is not None:
            viz.scene.add_point_cloud(
                "/object", points=cloud_w, colors=(200, 200, 200), point_size=0.002)
        if replay_w is not None:
            viz.scene.add_line_segments(
                "/replay", points=polyline_segments(replay_w),
                colors=contact_segment_colors(mode_flags, n_pose - 1),
                thickness=0.0025,
            )
        viz.scene.add_line_segments(
            "/actual", points=polyline_segments(actual_w),
            colors=(255, 170, 40), thickness=0.0015,
        )
        first = annotate_wrist_frame(
            decode_jpeg(jpegs[int(idx[0])]), mode="free", t_s=0.0, f_n=None)
        caption = server.gui.add_markdown(f"## check {check_n}")
        img = server.gui.add_image(first, label="wrist_cam_0", format="jpeg")

        def draw(i: int, live_xyz: np.ndarray, live_R: np.ndarray, _err: float,
                 q=None) -> None:
            del live_R
            i = int(np.clip(i, 0, n_pose - 1))
            qv = q0 if q is None else np.asarray(q, dtype=float)
            robot_viz.update(qv)
            update_chamfer(qv)
            fk = system.compute_forward_kinematics(qv)
            T_wrist = np.asarray(
                fk.get_link_pose("right_ur5e/wrist_3_link").to_transformation_matrix())
            T_flange = chamfer_world_from_wrist3(T_wrist)
            T_tcp = chamfer_tcp_world_from_wrist3(T_wrist)
            p = actual[i]
            p_w = right_base_points_to_vention(p)[0]
            R_f = right_base_R_to_vention(
                ep.R_field[min(i, len(ep.R_field) - 1)])
            contact_i = bool(mode_flags[i])
            mode = "contact" if contact_i else "free"
            fn_i = None if fn is None else float(fn[min(i, len(fn) - 1)])
            img.image = annotate_wrist_frame(
                decode_jpeg(jpegs[int(idx[i])]), mode=mode, t_s=float(t[i]), f_n=fn_i)
            xy_mm = float(np.hypot(live_xyz[0] - p[0], live_xyz[1] - p[1]) * 1e3)
            abs_mm = float(np.linalg.norm(live_xyz - p) * 1e3)
            fn_txt = f"{fn_i:.1f}N" if fn_i is not None else "n/a"
            caption.content = (
                f"## {mode}  t={t[i]:.2f}s  fn={fn_txt}  "
                f"xy={xy_mm:.0f}mm  abs={abs_mm:.0f}mm"
            )
            server.scene.add_frame(
                "/flange", wxyz=wxyz_from_R(T_flange[:3, :3]), position=T_flange[:3, 3],
                axes_length=0.05, axes_radius=0.0025)
            server.scene.add_arrows(
                "/R_field", points=axis_arrows(p_w, R_f, 0.06), colors=AXIS_RGB,
                shaft_radius=0.0015, head_radius=0.0035, head_length=0.008)
            server.scene.add_icosphere(
                "/tcp_bag", radius=0.006, color=(255, 170, 40), position=p_w)
            server.scene.add_icosphere(
                "/tcp_sim", radius=0.007, color=(80, 180, 255), position=T_tcp[:3, 3])

        print(
            f"viser http://localhost:{args.port}  "
            f"T_right overlay  rails=0.25m  orange=bag  cyan=sim TCP",
            flush=True,
        )
        return draw

    try:
        if args.headless:
            start_plant()
            run_clock()
        else:
            view["draw"] = _build_wrist_view()
            start_plant()
            worker = threading.Thread(
                target=run_clock, daemon=True, name=f"check{check_n}-clock")
            worker.start()
            try:
                while worker.is_alive():
                    time.sleep(0.05)
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            worker.join(timeout=1.0)
    finally:
        contact.stop()
        try:
            interface.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
