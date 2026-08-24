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

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OBJECT = str(ROOT / "assets" / "laser_profilometer" / "merged.ply")
MODEL = str(ROOT / "assets" / "robots" / "geodude" / "geodude.yaml")
DEFAULT_HARDWARE_CONFIG = str(ROOT / "assets" / "robots" / "geodude" / "geodude_hardware.yaml")

DEFAULT_VENTION_HEIGHT = 0.60


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


def main():
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
        "--target-force", type=float, default=5.0,
        help="desired contact force (N) pressing the tool into the object in contact mode")
    parser.add_argument(
        "--mode-button", type=int, default=0,
        help="spacemouse button index that toggles between approach and contact mode")
    parser.add_argument("--no-viz", action="store_true",
                        help="disable the viser visualization")
    parser.add_argument(
        "--vention-height", type=float, default=DEFAULT_VENTION_HEIGHT,
        help="physical vention rail height in meters for both arms (hardware does "
             "not sense these joints; used for visualization FK). Default matches "
             "geodude.yaml keyframe 'ready'.")
    parser.add_argument(
        "--left-vention-height", type=float, default=None,
        help="override --vention-height for the left rail only")
    parser.add_argument(
        "--right-vention-height", type=float, default=None,
        help="override --vention-height for the right rail only")
    args = parser.parse_args()

    if args.bring_up:
        with open(args.hardware_config) as f:
            hardware_config = yaml.safe_load(f)
        for control_group, entry in hardware_config.items():
            _dashboard_bring_up(entry["robot_ip"])

    config = ri.Configuration(MODEL, "hardware", args.hardware_config)
    interface = ri.RobotInterface(config)
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

    # Object / table pose in the same frame viser uses (rails overlaid). The
    # hold-pose objective stays on unpatched hardware FK so the C++ loop does
    # not see a 0.25 m z-error and command the UR joints up.
    viz_position = q_with_vention()[joint_idx]
    viz_ee_motor = task_space.compute_ee_motor(viz_position)
    viz_translator = viz_ee_motor.get_translator()
    home_xyz = np.array(
        [viz_translator.x(), viz_translator.y(), viz_translator.z()])
    rail_height = left_vention if args.task_space == "left_ee" else right_vention
    vention_offset = float(viz_position[0]) - rail_height
    table_reference_xyz = home_xyz - np.array([0.0, 0.0, vention_offset])

    object_points_raw = load_points(args.object)
    rotation = compute_resting_orientation(object_points_raw)
    translation = table_placement_translation(
        object_points_raw, rotation, table_reference_xyz)

    object_surface = load_object_mesh(
        args.object, rotation=rotation, translation=translation,
        normal_orientation=args.normal_orientation)

    key_direction = np.asarray(home_ee_motor.get_rotor().to_rotation_matrix())[:, 0]

    vf = OrientationField(object_surface, key_direction=key_direction)

    # Hold the live pose. vf.project() replaces orientation with the field's
    # local approach basis; using that as the first objective would immediately
    # servo the arm, before viser is even up.
    approach_controller = AdmittanceTeleoperationController()
    approach_controller.set_virtual_fixture(vf)
    approach_controller.set_task_space(task_space)
    approach_controller.set_interface(interface)
    approach_controller.set_objective(
        ga.control.SingleArmMotorControlObjective(home_ee_motor))

    contact_stiffness = np.diag([3000.0] * 5 + [0.0])
    contact_damping = np.diag([20.0] * 5 + [20.0])
    contact_config = ga.control.VelocityAdmittanceControllerConfig()
    contact_config.dt = 0.01
    contact_config.max_linear_speed = 0.1
    contact_config.max_angular_speed = 0.5
    contact_config.admittance = ga.control.AdmittanceConfig(
        inertia=1.0 * np.eye(6), damping=contact_damping, stiffness=contact_stiffness)

    contact_controller = ContactTeleoperationController(
        vf, contact_config, target_force=args.target_force)
    contact_controller.set_task_space(task_space)
    contact_controller.set_interface(interface)
    contact_controller.set_objective(
        ga.control.SingleArmLineControlObjective(
            vf.project(home_ee_motor).transform_line(ga.Line.Z())))

    mode = {"active": "approach"}
    device = None
    viz = None

    def live_ee_motor():
        position = interface.get_state().get_position()[joint_idx]
        return task_space.compute_ee_motor(position)

    def on_twist(twist):
        if is_idle_twist(twist):
            return
        active = approach_controller if mode["active"] == "approach" else contact_controller
        active.update_command(twist)

    def on_button(index):
        if index != args.mode_button:
            return
        if mode["active"] == "approach":
            mode["active"] = "contact"
            contact_controller.get_objective().set_target(
                vf.project(live_ee_motor()).transform_line(ga.Line.Z()))
            interface.control(contact_controller, blocking=False)
            contact_controller.start()
            print(f"-> contact mode: pressing at {args.target_force:.1f}N "
                  "along the field's local approach direction")
        else:
            contact_controller.stop()
            mode["active"] = "approach"
            approach_controller.get_objective().set_target(live_ee_motor())
            interface.control(approach_controller, blocking=False)
            print("-> approach mode: holding live pose, field orientation on spacemouse input")

    def start_control_and_device():
        nonlocal device
        interface.control(approach_controller, blocking=False)
        device = KeyboardTeleop(
            step=0.0025) if args.device == "keyboard" else SpaceMouseTeleop()
        device.start(on_twist, on_button)

    try:
        if args.no_viz:
            print(f"Teleoperating '{args.task_space}' on real geodude (headless) "
                  f"-- holding live pose until spacemouse input; "
                  f"button {args.mode_button} toggles contact. Ctrl+C to quit.")
            start_control_and_device()
            while True:
                time.sleep(0.1)
        else:
            viz = gafropy_viz.Visualizer()
            print(f"viser running at http://localhost:{viz.port} -- "
                  "holding pose; open the page before moving the spacemouse.")

            robot = viz.add_robot(interface.get_system(),
                                  name="geodude", joint_sliders=False)
            robot.update(q_with_vention())

            viz.scene.add_point_cloud(
                "/object", points=object_surface.vertices, colors=(200, 200, 200), point_size=0.002
            )

            eeviz = viz.add_motor(viz_ee_motor)

            def update():
                q = q_with_vention()
                robot.update(q)
                eeviz.update(task_space.compute_ee_motor(q[joint_idx]))

            viz.add_ticker(lambda: update())

            start_control_and_device()

            viz.show(message=f"Teleoperating '{args.task_space}' on real geodude "
                     f"-- holding live pose until spacemouse input; "
                     f"button {args.mode_button} toggles contact. "
                     f"viser at http://localhost:{viz.port} -- ctrl-c to quit.")
    except KeyboardInterrupt:
        pass
    finally:
        if device is not None:
            device.stop()
        contact_controller.stop()
        interface.shutdown()


if __name__ == "__main__":
    main()
