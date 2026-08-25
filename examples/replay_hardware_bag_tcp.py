"""Hardware Test 1: recorded bag TCP through speedJ + DualLoop.

Does not send bag joints. Does not send local actions. Replay is slow
(``--speed 0.2``). Default opens viser (stringer + bag TCP) and does **not**
connect. ``--live`` is the UR.

  PYTHONPATH=src python examples/replay_hardware_bag_tcp.py
  PYTHONPATH=src python examples/replay_hardware_bag_tcp.py --live --follow-path --frames 40

Jog is optional: ``--live`` first ramps **only the right UR** to bag TCP[0]
at 20 mm/s (left arm and rails stay). Then Enter starts the replay.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from geodude_teleoperation.laser_frames import (
    right_base_R_to_vention,
    right_base_points_to_vention,
)
from geodude_teleoperation.local_frames_npz import load_cga_episode
from geodude_teleoperation.mesh_fixture import load_points
from geodude_teleoperation.sim_replay import (
    DEFAULT_CMD_HZ,
    DEFAULT_REPLAY_SPEED,
    MAX_START_GAP_MM,
    TrackingRow,
    absolute_error_mm,
    apply_mode_edge,
    bag_tcp_flange_velocity,
    body_twist_for_plant,
    enter_speedj_contact,
    exit_speedj_contact,
    interpolate_tcp_xyzw,
    refuse_start_gap,
    replay_mode_label,
    ramp_duration_s,
    blend_pose_matrix,
    score_legend,
    slow_replay_clock,
    start_gap_mm,
    summarize_tracking,
)
from geodude_teleoperation.tcp_kinematics import (
    MODEL,
    bag_tcp_to_yaml_ee_world,
    poses_in_right_base,
)
from geodude_teleoperation.viser_view import (
    apply_stringer_view,
    contact_segment_colors,
    polyline_segments,
    wxyz_from_R,
)

_DATA = Path("/mnt/data/boeing/local_frames_data/08_16_half_stringer_local_frames")
_DEFAULT_NPZ = _DATA / "bags_08_16_half_stringer_bimanual_0_cga_actual.npz"
_DEFAULT_TRAIN = _DATA / "training" / "bags_08_16_half_stringer_bimanual_0.npz"
_DEFAULT_PLY = _DATA / "merged_rightbase.ply"
_DEFAULT_HW = Path(__file__).resolve().parent.parent / (
    "assets/robots/geodude/geodude_hardware.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, default=_DEFAULT_NPZ)
    p.add_argument("--training", type=Path, default=_DEFAULT_TRAIN)
    p.add_argument("--frames", type=int, default=0, help="0 = whole episode (original samples)")
    p.add_argument(
        "--speed", type=float, default=DEFAULT_REPLAY_SPEED,
        help="1.0 = demo rate. Default 0.2 (5x slower). Must be in (0, 1].",
    )
    p.add_argument("--max-start-gap-mm", type=float, default=MAX_START_GAP_MM)
    p.add_argument(
        "--ramp-mps", type=float, default=0.02,
        help="Cartesian crawl to bag start (m/s). Right UR only. Default 20 mm/s.",
    )
    p.add_argument(
        "--max-ramp-mm", type=float, default=500.0,
        help="refuse to ramp if live TCP is farther than this from bag start",
    )
    p.add_argument(
        "--no-ramp", action="store_true",
        help="do not move to start; require live TCP within --max-start-gap-mm",
    )
    p.add_argument(
        "--follow-path", action="store_true",
        help="never DualLoop; set_target the whole path (safer first pass)",
    )
    p.add_argument("--live", action="store_true", help="connect to the right UR")
    p.add_argument(
        "--bring-up", action="store_true",
        help="dashboard power-on / brake-release (off by default)",
    )
    p.add_argument("--hardware-config", type=Path, default=_DEFAULT_HW)
    p.add_argument(
        "--max-joint-velocity", type=float, default=0.2,
        help="speedJ |qdot| cap (teleop uses 0.8; this script stays slower)",
    )
    p.add_argument("--ply", type=Path, default=_DEFAULT_PLY)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--no-viz", action="store_true", help="terminal only; no viser")
    return p


def _nearest_idx(source: np.ndarray, query: np.ndarray) -> np.ndarray:
    s = np.asarray(source, dtype=float).reshape(-1)
    q = np.asarray(query, dtype=float).reshape(-1)
    i = np.clip(np.searchsorted(s, q), 0, len(s) - 1)
    prev = np.clip(i - 1, 0, len(s) - 1)
    pick_prev = np.abs(q - s[prev]) <= np.abs(q - s[i])
    return np.where(pick_prev, prev, i)


def _plan(args):
    if not 0.0 < float(args.speed) <= 1.0:
        raise SystemExit("--speed must be in (0, 1] (1.0 = demo rate)")
    train = args.training if args.training.is_file() else None
    ep = load_cga_episode(args.npz, training_path=train)
    n = min(len(ep.poses), len(ep.timestamps), len(ep.in_contact))
    if ep.f_n is not None:
        n = min(n, int(np.asarray(ep.f_n).reshape(-1).size))
    n_run = n if args.frames <= 0 else min(n, int(args.frames))
    t = np.asarray(ep.timestamps[:n_run], dtype=float)
    t = t - t[0]
    poses = np.asarray(ep.poses[:n_run], dtype=float)
    raw = np.asarray(ep.in_contact[:n_run], dtype=bool)
    fn = None if ep.f_n is None else np.asarray(ep.f_n, dtype=float).reshape(-1)[:n_run]
    t_wall, t_src = slow_replay_clock(t, args.speed, DEFAULT_CMD_HZ)
    poses_c = interpolate_tcp_xyzw(poses, t, t_src)
    idx = np.clip(_nearest_idx(t, t_src), 0, n_run - 1)
    raw_c = raw[idx]
    fn_c = None if fn is None else fn[idx]
    dt_s = float(np.diff(t_wall).mean()) if len(t_wall) > 1 else 1.0 / DEFAULT_CMD_HZ
    R_flange = Rotation.from_quat(poses_c[:, 3:7]).as_matrix()
    v_raw = bag_tcp_flange_velocity(poses_c[:, :3], R_flange, dt_s)
    v_stick = np.zeros((len(poses_c), 3), dtype=float)
    n_v = min(len(v_stick), len(v_raw))
    v_stick[:n_v] = v_raw[:n_v]
    contact_i = int(np.argmax(raw)) if raw.any() else -1
    print(
        f"test 1 hardware  episode={ep.meta.get('episode')}  "
        f"src={n_run}/{n} samples  cmd={len(poses_c)} @ {DEFAULT_CMD_HZ:.0f} Hz  "
        f"speed={args.speed:g}x  wall={t_wall[-1]:.1f}s (demo {t[-1]:.1f}s)  "
        f"live={args.live}",
        flush=True,
    )
    tcp0 = poses[0, :3].round(4).tolist()
    if contact_i >= 0:
        print(
            f"bag TCP[0]={tcp0}  first contact t_demo={t[contact_i]:.2f}s",
            flush=True,
        )
    else:
        print(f"bag TCP[0]={tcp0}  no contact in slice", flush=True)
    print(score_legend(follow_path=args.follow_path), flush=True)
    if not args.live:
        print("preview: viser only, not the UR. Pass --live after you like the path.", flush=True)
    return {
        "t_wall": t_wall, "poses": poses_c, "raw": raw_c, "fn": fn_c,
        "v_stick": v_stick, "dt_s": dt_s, "bag0": poses[0, :3],
        "R_flange": R_flange,
    }


def _open_preview(plan, args):
    import viser

    poses = plan["poses"]
    raw = plan["raw"]
    actual = np.asarray(poses[:, :3], dtype=float)
    cloud = load_points(args.ply) if args.ply.is_file() else None
    cloud_w = None if cloud is None else right_base_points_to_vention(cloud)
    actual_w = right_base_points_to_vention(actual)
    look = actual_w if cloud_w is None else cloud_w
    server = viser.ViserServer(port=args.port)
    apply_stringer_view(server, look)
    if cloud_w is not None:
        server.scene.add_point_cloud(
            "/object", points=cloud_w, colors=(200, 200, 200), point_size=0.002)
    else:
        print(f"stringer ply missing: {args.ply}", flush=True)
    n_seg = max(1, len(actual_w) - 1)
    server.scene.add_line_segments(
        "/bag_tcp", points=polyline_segments(actual_w),
        colors=contact_segment_colors(raw, n_seg),
        thickness=0.0025,
    )
    print(
        f"viser http://localhost:{args.port}  "
        f"orange path = bag TCP  green=approach  red=contact  "
        f"orange ball = command  cyan = live UR (only with --live)",
        flush=True,
    )

    def draw(i: int, live_xyz=None) -> None:
        i = int(np.clip(i, 0, len(poses) - 1))
        p_w = actual_w[i]
        R_w = right_base_R_to_vention(plan["R_flange"][i])
        server.scene.add_frame(
            "/cmd_tcp", wxyz=wxyz_from_R(R_w), position=p_w,
            axes_length=0.04, axes_radius=0.002)
        server.scene.add_icosphere(
            "/cmd_ball", radius=0.006, color=(255, 170, 40), position=p_w)
        if live_xyz is not None:
            live_w = right_base_points_to_vention(
                np.asarray(live_xyz, dtype=float).reshape(3))[0]
            server.scene.add_icosphere(
                "/live_tcp", radius=0.007, color=(80, 180, 255), position=live_w)

    draw(0)
    return draw


def _play_preview(draw, plan) -> None:
    print(
        f"playing {len(plan['poses'])} cmds at {plan['dt_s']*1e3:.0f} ms/step "
        f"({plan['t_wall'][-1]:.1f}s). Ctrl+C when done looking.",
        flush=True,
    )
    try:
        for i in range(len(plan["poses"])):
            draw(i)
            time.sleep(plan["dt_s"])
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass


def _wait_enter(msg: str) -> None:
    print(msg, flush=True)
    try:
        input()
    except EOFError as exc:
        raise RuntimeError("need a terminal for Enter") from exc


def _dashboard_bring_up(ip: str) -> None:
    import socket

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
        if "RUNNING" in command("robotmode"):
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"{ip}: robot never reached RUNNING mode")
    sock.close()
    print(f"{ip}: RUNNING", flush=True)


def _make_approach(interface, task_space, home_ee_motor, max_joint_velocity: float):
    import gafropy as ga
    from geodude_teleoperation import AdmittanceTeleoperationController

    approach = AdmittanceTeleoperationController(
        inertia=np.diag([1.0] * 6),
        damping=np.diag([20.0] * 6),
        stiffness=np.diag([2000.0] * 6),
        dt=0.001,
        max_joint_velocity=float(max_joint_velocity),
        twist_linear_scale=1.0,
        max_qddot=5.0,
    )
    approach.set_task_space(task_space)
    approach.set_interface(interface)
    approach.set_objective(ga.control.SingleArmMotorControlObjective(home_ee_motor))
    return approach


def _ramp_right_to_start(
    *, args, plan, approach, system, live_q, live_ee_motor, draw,
) -> None:
    from geodude_teleoperation.ee_viz import motor_aligned_to, motor_from_matrix

    frames = poses_in_right_base(system, live_q())
    gap = start_gap_mm(frames["chamfer"][:3, 3], plan["bag0"])
    print(f"live chamfer vs bag TCP[0]: {gap:.1f} mm", flush=True)
    if draw is not None:
        draw(0, live_xyz=frames["chamfer"][:3, 3])
    if args.no_ramp:
        refuse_start_gap(gap, limit_mm=args.max_start_gap_mm)
        return
    if gap > float(args.max_ramp_mm):
        raise RuntimeError(
            f"live TCP is {gap:.1f} mm from bag start "
            f"(ramp limit {float(args.max_ramp_mm):.0f} mm). "
            "Move the right arm into free space nearer the stringer."
        )
    if gap <= float(args.max_start_gap_mm):
        print("already at bag start; skip ramp", flush=True)
        return
    dur = ramp_duration_s(gap * 1e-3, speed_mps=float(args.ramp_mps))
    dt_s = 0.1
    n = max(2, int(round(dur / dt_s)))
    t0 = np.asarray(live_ee_motor().to_transformation_matrix(), dtype=float)
    t1 = bag_tcp_to_yaml_ee_world(system, live_q(), plan["poses"][0])
    _wait_enter(
        f"right UR only: Enter to ramp {gap:.0f} mm to bag start "
        f"at {float(args.ramp_mps) * 1e3:.0f} mm/s (~{dur:.0f}s). Ctrl+C holds."
    )
    for k in range(n):
        T = blend_pose_matrix(t0, t1, k / (n - 1))
        approach.get_objective().set_target(
            motor_aligned_to(motor_from_matrix(T), live_ee_motor()))
        approach.set_input_active(True)
        if draw is not None:
            draw(0, live_xyz=poses_in_right_base(system, live_q())["chamfer"][:3, 3])
        time.sleep(dt_s)
    for _ in range(10):
        approach.get_objective().set_target(
            motor_aligned_to(motor_from_matrix(t1), live_ee_motor()))
        approach.set_input_active(True)
        if draw is not None:
            draw(0, live_xyz=poses_in_right_base(system, live_q())["chamfer"][:3, 3])
        time.sleep(dt_s)
    gap2 = start_gap_mm(
        poses_in_right_base(system, live_q())["chamfer"][:3, 3], plan["bag0"])
    print(f"after ramp: {gap2:.1f} mm from bag start", flush=True)
    refuse_start_gap(gap2, limit_mm=args.max_start_gap_mm)


def _make_contact(task_space, interface, home_ee_motor, measured_fz_fn, target_force: float):
    import gafropy as ga
    from geodude_teleoperation import ContactTeleoperationController

    z_damping = 200.0
    cfg = ga.control.VelocityAdmittanceControllerConfig()
    cfg.dt = 0.01
    cfg.max_linear_speed = 0.005
    cfg.max_angular_speed = 0.15
    cfg.admittance = ga.control.AdmittanceConfig(
        inertia=1.0 * np.eye(6),
        damping=np.diag([40.0] * 5 + [z_damping]),
        stiffness=np.diag([400.0] * 5 + [0.0]),
    )
    contact = ContactTeleoperationController(
        None, cfg, target_force=target_force,
        kp_force=0.001, ki_force=0.0003,
        v_approach=0.005, max_v_hold=0.002, z_damping=z_damping,
        v_slew_mps2=0.05, lost_s=1.0, sit_s=2.0,
        force_sign=-1.0, log_period_s=0.0,
        measured_fz_fn=measured_fz_fn,
    )
    contact.set_task_space(task_space)
    contact.set_interface(interface)
    contact.set_objective(
        ga.control.SingleArmLineControlObjective(
            home_ee_motor.transform_line(ga.Line.Z())))
    return contact


def _run_live(args, plan, draw=None) -> None:
    import yaml
    import gafropy as ga
    from gafropy import robot_interface as ri

    from geodude_teleoperation.ee_viz import motor_aligned_to, motor_from_matrix
    from geodude_teleoperation.tcp_force import connect_tcp_force_reader, fz_from_tcp_force

    with open(args.hardware_config) as f:
        hardware_config = yaml.safe_load(f)
    robot_ip = hardware_config["right_ur5e"]["robot_ip"]
    if args.bring_up:
        for entry in hardware_config.values():
            _dashboard_bring_up(entry["robot_ip"])

    interface = ri.RobotInterface(
        ri.Configuration(str(MODEL), "hardware", str(args.hardware_config)))
    interface.initialize()
    task_space = interface.get_task_space("right_ee")
    if task_space is None:
        interface.shutdown()
        raise RuntimeError("task space 'right_ee' missing")
    joint_idx = np.array(task_space.get_joint_indices())
    system = interface.get_system()

    def live_q():
        return np.asarray(interface.get_state().get_position(), dtype=float)

    def live_ee_motor():
        return task_space.compute_ee_motor(live_q()[joint_idx])

    home = live_ee_motor()
    approach = _make_approach(interface, task_space, home, args.max_joint_velocity)
    approach.set_input_active(False)
    approach.hold_live_pose()
    interface.control(approach, blocking=False)

    frames0 = poses_in_right_base(system, live_q())
    if draw is not None:
        draw(0, live_xyz=frames0["chamfer"][:3, 3])
    try:
        _ramp_right_to_start(
            args=args, plan=plan, approach=approach, system=system,
            live_q=live_q, live_ee_motor=live_ee_motor, draw=draw)
    except Exception:
        try:
            interface.shutdown()
        except Exception:
            pass
        raise

    measured_fz_fn = None
    close_tcp_force = None
    contact = None
    if not args.follow_path:
        try:
            read_tcp_force, close_tcp_force = connect_tcp_force_reader(robot_ip)
            measured_fz_fn = lambda: fz_from_tcp_force(read_tcp_force())
            print(f"contact FT: RTDE ActualTCPForce from {robot_ip}", flush=True)
        except Exception as exc:
            try:
                interface.shutdown()
            except Exception:
                pass
            raise RuntimeError(
                f"no ActualTCPForce reader ({exc}). Use --follow-path or fix RTDE."
            ) from exc
        contact = _make_contact(task_space, interface, home, measured_fz_fn, 15.0)

    _wait_enter(
        f"at start. Enter to run {args.speed:g}x replay "
        f"({len(plan['poses'])} cmds, {plan['t_wall'][-1]:.1f}s wall). Ctrl+C holds."
    )

    mode = {"active": "path" if args.follow_path else "approach"}
    rows: list[TrackingRow] = []
    raw = plan["raw"]
    fn = plan["fn"]
    poses_c = plan["poses"]
    v_stick = plan["v_stick"]
    R_flange = plan["R_flange"]
    dt_s = plan["dt_s"]
    t_wall = plan["t_wall"]

    def on_enter():
        i = fz_state["i"]
        fd = 15.0 if fn is None else abs(float(fn[i]))
        contact._target_force = fd
        contact._contact_engage_n = max(0.4, 0.25 * fd)
        enter_speedj_contact(
            mode=mode, contact_controller=contact, approach_controller=approach)
        print(f"-> contact  Fd={fd:.1f}N", flush=True)

    def on_exit():
        exit_speedj_contact(
            mode=mode, contact_controller=contact, approach_controller=approach)
        print("-> approach", flush=True)

    fz_state = {"i": 0}

    try:
        last_mode = None
        for i in range(len(poses_c)):
            fz_state["i"] = i
            if args.follow_path:
                mode["active"] = "path"
            else:
                apply_mode_edge(
                    mode, bool(raw[i]), on_enter=on_enter, on_exit=on_exit)
            q = live_q()
            frames = poses_in_right_base(system, q)
            m_des = motor_aligned_to(
                motor_from_matrix(bag_tcp_to_yaml_ee_world(system, q, poses_c[i])),
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
                live_xyz=frames["chamfer"][:3, 3], bag_xyz=poses_c[i, :3])
            if draw is not None:
                draw(i, live_xyz=frames["chamfer"][:3, 3])
            label = replay_mode_label(bool(raw[i]), follow_path=args.follow_path)
            rows.append(TrackingRow(
                i=i, t=float(t_wall[i]), mode=label,
                err_xy_mm=xy, err_z_mm=z, err_mm=err,
                abs_fn=None if fn is None else abs(float(fn[i])),
                plant_vz=float(approach._tool_z_velocity),
            ))
            if label != last_mode:
                last_mode = label
                print(
                    f"t={t_wall[i]:.2f}s -> {label}  "
                    f"xy={xy:.1f}mm  z={z:.1f}mm  err={err:.1f}mm",
                    flush=True,
                )
            time.sleep(dt_s)
        print(summarize_tracking(rows), flush=True)
    finally:
        if mode["active"] == "contact" and contact is not None:
            on_exit()
        approach.set_input_active(False)
        approach.hold_live_pose()
        if contact is not None:
            contact.stop()
        if close_tcp_force is not None:
            close_tcp_force()
        try:
            interface.shutdown()
        except Exception:
            pass


def main() -> None:
    args = build_parser().parse_args()
    plan = _plan(args)
    draw = None if args.no_viz else _open_preview(plan, args)
    if not args.live:
        if draw is None:
            return
        _play_preview(draw, plan)
        return
    _run_live(args, plan, draw=draw)


if __name__ == "__main__":
    main()
