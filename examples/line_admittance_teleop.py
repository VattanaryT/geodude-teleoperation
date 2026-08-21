"""Line admittance + force-PID control, teleoperated.

Combines examples/line_admittance_controller.py's line-tracking admittance
controller and gufic_line.py's force-PID push (see
geodude_teleoperation.controller.LineAdmittanceTeleoperationController) with
teleoperation.py's device handling: the target line stays normal to the table
(straight down) for the whole run -- the operator only drives its X/Y position,
never its tilt -- while depth along it is left force-controlled, a PID on the
measured contact wrench keeping the tool pressed onto the table at a constant
force. The two DOF a line target can't see (roll about it, translation along
it) are exactly the ones force control governs, so teleoperating the line's
X/Y position and pushing along it never fight over the same axis.
"""

import argparse
import pathlib
import time

import gafropy as ga
import numpy as np
from gafropy import robot_interface as ri
from gafropy import viz as gafropy_viz

from geodude_teleoperation import (
    KeyboardTeleop,
    LineAdmittanceTeleoperationController,
    SpaceMouseTeleop,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent

ROBOTS = {
    "geodude": {
        "model": str(ROOT / "assets" / "robots" / "geodude" / "geodude_cylinder_ee.yaml"),
        "task_space": "right_ee",
        "has_vention": True,
    }
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--robot", choices=list(ROBOTS), default="geodude")
    parser.add_argument(
        "--device", choices=["keyboard", "spacemouse"], default="spacemouse")
    parser.add_argument(
        "--target-force", type=float, default=1.0,
        help="desired contact force (N) pressing the tool into the table")
    parser.add_argument("--no-viz", action="store_true",
                        help="disable the viser visualization")
    args = parser.parse_args()

    robot_cfg = ROBOTS[args.robot]

    config = ri.Configuration()
    config.model_file = robot_cfg["model"]
    config.type = "mujoco"

    interface = ri.RobotInterface(config)
    interface.initialize()

    task_space = interface.get_task_space(robot_cfg["task_space"])
    if task_space is None:
        raise RuntimeError(
            f"task space '{robot_cfg['task_space']}' not found in {robot_cfg['model']}")

    joint_idx = np.array(task_space.get_joint_indices())
    live_position = interface.get_state().get_position()[joint_idx]
    home_ee_motor = task_space.compute_ee_motor(live_position)
    home_translator = home_ee_motor.get_translator()
    home_xyz = np.array(
        [home_translator.x(), home_translator.y(), home_translator.z()])

    if robot_cfg["has_vention"]:
        DEFAULT_VENTION_HEIGHT = 0.25
        vention_offset = interface.get_state().get_position()[0] - DEFAULT_VENTION_HEIGHT
        table_z = home_xyz[2] - vention_offset
    else:
        table_z = 0.0

    # Initial target line: straight down (normal to the table) through a point a
    # little in front of the arm's home position and just above the table
    # surface. Direction stays fixed at normal-to-table for the whole run (see
    # LineAdmittanceTeleoperationController); only this (x, y) is teleoperated.
    # Depth along the line (how far down the tool presses into the table) is
    # left to the force PID.
    start_xy = home_xyz[:2] + np.array([0.15, 0.0])
    contact_height = table_z + 0.05
    target_line = ga.Line(
        ga.Point(start_xy[0], start_xy[1], contact_height),
        ga.Point(start_xy[0], start_xy[1], contact_height - 1.0))

    stiffness = np.eye(6) * 10000.0
    stiffness[5, 5] = 0.0  # linear Z (depth along the line) is force-controlled

    admittance_config = ga.control.VelocityAdmittanceControllerConfig()
    admittance_config.dt = 0.01
    admittance_config.max_linear_speed = 0.15
    admittance_config.max_angular_speed = 0.75
    admittance_config.admittance = ga.control.AdmittanceConfig(
        inertia=1.0 * np.eye(6), damping=20.0 * np.eye(6), stiffness=stiffness)

    controller = LineAdmittanceTeleoperationController(
        admittance_config, target_force=args.target_force,
        kp_force=5.0, ki_force=1.0, kd_force=0.0, force_feedforward=0.25)

    objective = ga.control.SingleArmLineControlObjective(target_line)
    controller.set_objective(objective)
    controller.set_task_space(task_space)
    controller.set_interface(interface)

    interface.control(controller, blocking=False)
    controller.start()

    device = KeyboardTeleop(
        step=0.0001) if args.device == "keyboard" else SpaceMouseTeleop(linear_scale=0.001)
    device.start(lambda twist: controller.update_command(twist))

    try:
        if args.no_viz:
            print(
                f"Teleoperating the line on '{robot_cfg['task_space']}' (headless). Ctrl+C to quit.")
            while True:
                time.sleep(0.1)
        else:
            viz = gafropy_viz.Visualizer()

            robot_viz = viz.add_robot(interface.get_system(),
                                      name=args.robot, joint_sliders=False)

            # Distinct color from add_line's default orange (which /ee_line below
            # also uses) -- once the controller converges the two lines nearly
            # coincide, and same-colored 100-unit lines are indistinguishable.
            target_line_viz = viz.add_line(
                target_line, name="/target_line", color=(220, 20, 60))

            position = interface.get_state().get_position()[joint_idx]
            ee_line = task_space.compute_ee_motor(position).transform_line(ga.Line.Z())
            ee_line_viz = viz.add_line(ee_line, name="/ee_line")

            def update():
                q = interface.get_state().get_position()
                robot_viz.update(q)

                position = q[joint_idx]
                ee_motor = task_space.compute_ee_motor(position)
                ee_line_viz.update(ee_motor.transform_line(ga.Line.Z()))
                target_line_viz.update(objective.get_target())

            viz.add_ticker(lambda: update())

            viz.show(
                message=f"Teleoperating the line on '{robot_cfg['task_space']}'. viser running at http://localhost:{viz.port} -- ctrl-c to quit.")
    except KeyboardInterrupt:
        pass
    finally:
        device.stop()
        controller.stop()
        interface.shutdown()


if __name__ == "__main__":
    main()
