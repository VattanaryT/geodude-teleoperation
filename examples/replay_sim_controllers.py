"""Check 6: closed-loop MuJoCo replay through speedJ + DualLoop (not hardware).

Kinematics are checks 3/5. This clocks the check-4 sandwich into the same
plants as ``hardware_object_contact_teleop.py`` and prints TCP tracking error.

The gafro plant is yaml ``right_ee``. Error is chamfer FK vs bag TCP, relative
to each path's start (MuJoCo home joints are not the bag). DualLoop Fz is the
bag force label played back as a UR reaction, not simulated contact physics.

  PYTHONPATH=src python examples/local_frame_checks.py 6

  PYTHONPATH=src python examples/replay_sim_controllers.py --headless --fast --frames 40
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import gafropy as ga
import numpy as np
from gafropy import robot_interface as ri
from scipy.spatial.transform import Rotation

from geodude_teleoperation import AdmittanceTeleoperationController, ContactTeleoperationController
from geodude_teleoperation.local_frames_npz import load_cga_episode
from geodude_teleoperation.mode_segments import sandwich_local_actions
from geodude_teleoperation.sim_replay import (
    TrackingRow,
    apply_mode_edge,
    body_twist_for_plant,
    enter_speedj_contact,
    exit_speedj_contact,
    format_tracking_table,
    relative_error_mm,
    summarize_tracking,
    ur_reaction_fz,
)
from geodude_teleoperation.tcp_kinematics import MODEL, poses_in_right_base

_DATA = Path("/mnt/data/boeing/local_frames_data/08_16_half_stringer_local_frames")
_DEFAULT_NPZ = _DATA / "bags_08_16_half_stringer_bimanual_0_cga_actual.npz"
_DEFAULT_TRAIN = _DATA / "training" / "bags_08_16_half_stringer_bimanual_0.npz"


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
    _, v_body, _, _, _ = sandwich_local_actions(
        R_field=ep.R_field, R_ee=R_flange, a_local=ep.a_local,
        in_contact=raw, timestamps=t, f_n=fn, lost_s=0.0)
    n_run = n if args.frames <= 0 else min(n, int(args.frames))
    bag_xyz = np.asarray(ep.action_xyz[: n_run + 1], dtype=float)

    print(
        f"check 6  episode={ep.meta.get('episode')}  {ep.hz} Hz  "
        f"frames={n_run}/{n}  plant=yaml right_ee  error=chamfer vs bag (relative t=0)",
        flush=True,
    )
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

    config = ri.Configuration()
    config.model_file = str(MODEL)
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

    fz_state = {"i": 0, "contact": False, "fn": 0.0}

    def measured_fz():
        return ur_reaction_fz(fz_state["fn"], in_contact=fz_state["contact"])

    contact = _make_contact(task_space, interface, home, measured_fz)
    mode = {"active": "approach"}
    rows: list[TrackingRow] = []
    live0 = {"p": None}
    bag0 = bag_xyz[0]
    approach_target = {"m": None}

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
        approach_target["m"] = None

    def step(i: int) -> None:
        fz_state["i"] = i
        fz_state["contact"] = bool(raw[i])
        fz_state["fn"] = 0.0 if fn is None else float(fn[i])
        apply_mode_edge(
            mode, bool(raw[i]), on_enter=on_enter, on_exit=on_exit)
        _q, frames = live_frames()
        tw = body_twist_for_plant(
            v_body[i], dt_s=dt_s, contact=mode["active"] == "contact",
            R_flange=R_flange[i], R_plant=frames["yaml_ee"][:3, :3])
        if mode["active"] == "contact":
            approach.update_command(ga.Twist.from_array(tw))
        else:
            # Increment a chained target, not the lagged live EE, so the spring
            # can catch up. SpaceMouse does the catch-up with the operator's eye.
            if approach_target["m"] is None:
                approach_target["m"] = live_ee_motor()
            w, v = tw[:3], tw[3:]
            approach_target["m"] = approach_target["m"].multiply(
                ga.Motor.exp(w[0], w[1], w[2], v[0], v[1], v[2]))
            approach.get_objective().set_target(approach_target["m"])
            approach.set_input_active(True)
        if live0["p"] is None:
            live0["p"] = frames["chamfer"][:3, 3].copy()
        xy, z, err = relative_error_mm(
            live_xyz=frames["chamfer"][:3, 3], bag_xyz=bag_xyz[i],
            live0=live0["p"], bag0=bag0)
        rows.append(TrackingRow(
            i=i, t=float(t[i]), mode=mode["active"],
            err_xy_mm=xy, err_z_mm=z, err_mm=err,
            abs_fn=None if fn is None else abs(float(fn[i])),
            plant_vz=float(approach._tool_z_velocity),
        ))

    def run_clock() -> None:
        last_mode = None
        try:
            for i in range(n_run):
                step(i)
                if rows and rows[-1].mode != last_mode:
                    last_mode = rows[-1].mode
                    r = rows[-1]
                    print(
                        f"t={r.t:.2f}s -> {r.mode}  err={r.err_mm:.1f}mm",
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
        print(summarize_tracking(rows), flush=True)
        errs = np.array([r.err_mm for r in rows], dtype=float)
        print(
            f"overall mean {float(errs.mean()):.2f} mm  max {float(errs.max()):.2f} mm  "
            f"n={len(rows)}",
            flush=True,
        )
        if args.verbose_rows:
            print(flush=True)
            print(format_tracking_table(rows), flush=True)

    approach.set_input_active(False)
    approach.hold_live_pose()
    interface.control(approach, blocking=False)

    try:
        if args.headless:
            run_clock()
        else:
            from gafropy import viz as gafropy_viz
            from geodude_teleoperation.ee_viz import (
                add_right_chamfer_visual,
                hide_gripper_visuals,
            )

            viz = gafropy_viz.Visualizer()
            robot_viz = viz.add_robot(interface.get_system(), joint_sliders=False)
            hide_gripper_visuals(robot_viz)
            q0 = np.asarray(interface.get_state().get_position(), dtype=float)
            update_chamfer = add_right_chamfer_visual(viz, system, q=q0)

            def update():
                q = np.asarray(interface.get_state().get_position(), dtype=float)
                robot_viz.update(q)
                update_chamfer(q)

            viz.add_ticker(lambda: update())
            worker = threading.Thread(target=run_clock, daemon=True, name="check6-clock")
            worker.start()
            try:
                viz.show(
                    message=(
                        f"check 6 mujoco replay (not hardware). "
                        f"viser http://localhost:{viz.port} -- ctrl-c to quit."
                    ))
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
