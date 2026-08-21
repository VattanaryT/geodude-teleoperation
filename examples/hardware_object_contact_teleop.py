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
    compute_resting_orientation,
    load_object_mesh,
    load_points,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OBJECT = str(ROOT / "assets" / "objects" / "banana_half.ply")
MODEL = str(ROOT / "assets" / "robots" / "geodude" / "geodude.yaml")
DEFAULT_HARDWARE_CONFIG = str(ROOT / "assets" / "robots" / "geodude" / "geodude_hardware.yaml")

DEFAULT_VENTION_HEIGHT = 0.25


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
    args = parser.parse_args()

    if args.bring_up:
        with open(args.hardware_config) as f:
            hardware_config = yaml.safe_load(f)
        for control_group, entry in hardware_config.items():
            _dashboard_bring_up(entry["robot_ip"])

    config = ri.Configuration(MODEL, "hardware", args.hardware_config)
    interface = ri.RobotInterface(config)
    interface.initialize()

    task_space = interface.get_task_space(args.task_space)
    if task_space is None:
        raise RuntimeError(f"task space '{args.task_space}' not found in {MODEL}")

    joint_idx = np.array(task_space.get_joint_indices())
    live_position = interface.get_state().get_position()[joint_idx]
    home_ee_motor = task_space.compute_ee_motor(live_position)
    home_translator = home_ee_motor.get_translator()
    home_xyz = np.array(
        [home_translator.x(), home_translator.y(), home_translator.z()])

    vention_offset = interface.get_state().get_position()[0] - DEFAULT_VENTION_HEIGHT
    table_reference_xyz = home_xyz - np.array([0.0, 0.0, vention_offset])
    plane_z = table_reference_xyz[2]

    object_points_raw = load_points(args.object)
    object_centroid_raw = object_points_raw.mean(axis=0)

    rotation = compute_resting_orientation(object_points_raw)
    rotated_centered = (object_points_raw - object_centroid_raw) @ rotation.T

    translation = np.zeros(3)
    translation[:2] = (table_reference_xyz[:2] + np.array([0.15, 0.0])
                       - object_centroid_raw[:2])
    translation[2] = plane_z - \
        (rotated_centered[:, 2].min() + object_centroid_raw[2])

    object_surface = load_object_mesh(
        args.object, rotation=rotation, translation=translation,
        normal_orientation=args.normal_orientation)

    key_direction = np.asarray(home_ee_motor.get_rotor().to_rotation_matrix())[:, 0]

    vf = OrientationField(object_surface, key_direction=key_direction)

    initial_target = vf.project(home_ee_motor)

    approach_controller = AdmittanceTeleoperationController()
    approach_controller.set_virtual_fixture(vf)
    approach_controller.set_task_space(task_space)
    approach_controller.set_interface(interface)
    approach_controller.set_objective(
        ga.control.SingleArmMotorControlObjective(initial_target))

    contact_stiffness = np.diag([3000.0] * 5 + [0.0])
    contact_damping = np.diag([20.0] * 5 + [20.0])
    contact_config = ga.control.VelocityAdmittanceControllerConfig()
    contact_config.dt = 0.01
    contact_config.max_linear_speed = 0.1
    contact_config.max_angular_speed = 0.5
    contact_config.admittance = ga.control.AdmittanceConfig(
        inertia=1.0 * np.eye(6), damping=contact_damping, stiffness=contact_stiffness)

    initial_line = initial_target.transform_line(ga.Line.Z())

    contact_controller = ContactTeleoperationController(
        vf, contact_config, target_force=args.target_force)
    contact_controller.set_task_space(task_space)
    contact_controller.set_interface(interface)
    contact_controller.set_objective(
        ga.control.SingleArmLineControlObjective(initial_line))

    interface.control(approach_controller, blocking=False)

    mode = {"active": "approach"}

    def on_twist(twist):
        active = approach_controller if mode["active"] == "approach" else contact_controller
        active.update_command(twist)

    def on_button(index):
        if index != args.mode_button:
            return
        if mode["active"] == "approach":
            mode["active"] = "contact"
            interface.control(contact_controller, blocking=False)
            contact_controller.start()
            print(f"-> contact mode: pressing at {args.target_force:.1f}N "
                  "along the field's local approach direction")
        else:
            contact_controller.stop()
            mode["active"] = "approach"
            interface.control(approach_controller, blocking=False)
            print("-> approach mode: free 6-DOF admittance, no contact force")

    device = KeyboardTeleop(
        step=0.0025) if args.device == "keyboard" else SpaceMouseTeleop()
    device.start(on_twist, on_button)

    try:
        if args.no_viz:
            print(f"Teleoperating '{args.task_space}' on real geodude (headless) "
                  f"-- starting in approach mode, button {args.mode_button} toggles contact. "
                  "Ctrl+C to quit.")
            while True:
                time.sleep(0.1)
        else:
            viz = gafropy_viz.Visualizer()

            robot = viz.add_robot(interface.get_system(),
                                  name="geodude", joint_sliders=False)

            viz.scene.add_point_cloud(
                "/object", points=object_surface.vertices, colors=(200, 200, 200), point_size=0.002
            )

            position = interface.get_state().get_position()[joint_idx]
            ee = task_space.compute_ee_motor(position)
            eeviz = viz.add_motor(ee)

            def update():
                q = interface.get_state().get_position()
                robot.update(q)
                position = q[joint_idx]
                ee = task_space.compute_ee_motor(position)
                eeviz.update(ee)

            viz.add_ticker(lambda: update())

            viz.show(message=f"Teleoperating '{args.task_space}' on real geodude "
                     f"-- starting in approach mode, button {args.mode_button} toggles "
                     f"contact. viser running at http://localhost:{viz.port} -- "
                     "ctrl-c to quit.")
    except KeyboardInterrupt:
        pass
    finally:
        device.stop()
        contact_controller.stop()
        interface.shutdown()


if __name__ == "__main__":
    main()
