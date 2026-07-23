import argparse
import pathlib
import time

import gafropy.numpy as gn
import gafropy as ga
import numpy as np
from diffused_fields.manifold._rotor_interop import as_scipy_rotation
from gafropy import robot_interface as ri
from gafropy import viz as gafropy_viz

from geodude_teleoperation import (
    AdmittanceTeleoperationController,
    TeleoperationController,
    KeyboardTeleop,
    OrientationField,
    SpaceMouseTeleop,
    compute_resting_orientation,
    load_object_pointcloud,
    load_points,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OBJECT = str(ROOT / "assets" / "objects" / "banana_half.ply")

ROBOTS = {
    "geodude": {
        "model": str(ROOT / "assets" / "robots" / "geodude" / "geodude.yaml"),
        "task_space": "left_ee",
        "has_vention": True,
    },
    "lite6": {
        "model": str(ROOT / "assets" / "robots" / "lite6" / "lite6.yaml"),
        "task_space": "ee",
        "has_vention": False,
    },
}


def main():

    print(ROOT)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--robot", choices=list(ROBOTS), default="geodude")
    parser.add_argument(
        "--device", choices=["keyboard", "spacemouse"], default="keyboard")
    parser.add_argument(
        "--controller", choices=["velocity", "admittance"], default="velocity")
    parser.add_argument(
        "--object", default=DEFAULT_OBJECT,
        help="mesh or point cloud file to place on the table (see module docstring)")
    parser.add_argument(
        "--normal-orientation", type=int, choices=[1, -1], default=1,
        help="sign correction for point-cloud objects' PCA-estimated normals (see module docstring)")
    parser.add_argument("--no-viz", action="store_true",
                        help="disable the viser visualization")
    args = parser.parse_args()

    robot_cfg = ROBOTS[args.robot]

    controller = (
        TeleoperationController(damping=0.05) if args.controller == "velocity"
        else AdmittanceTeleoperationController()
    )

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
        vention_offset = live_position[0] - DEFAULT_VENTION_HEIGHT
        table_reference_xyz = home_xyz - np.array([0.0, 0.0, vention_offset])
    else:
        table_reference_xyz = np.array([home_xyz[0], home_xyz[1], 0.0])

    plane_z = table_reference_xyz[2]

    object_points_raw = load_points(args.object)
    object_centroid_raw = object_points_raw.mean(axis=0)

    rotation = (compute_resting_orientation(object_points_raw))
    rotated_centered = (object_points_raw - object_centroid_raw) @ rotation.T

    translation = np.zeros(3)
    translation[:2] = (table_reference_xyz[:2] + np.array([0.15, 0.0])
                       - object_centroid_raw[:2])
    translation[2] = plane_z - \
        (rotated_centered[:, 2].min() + object_centroid_raw[2])

    object_surface = load_object_pointcloud(
        args.object, rotation=rotation, translation=translation,
        normal_orientation=args.normal_orientation)

    key_direction = as_scipy_rotation(
        home_ee_motor.get_rotor()).as_matrix()[:, 0]

    vf = OrientationField(
        object_surface, key_direction=key_direction,
        batch_size=256, max_iterations=48, convergence_threshold=0.01)

    controller.set_virtual_fixture(vf)
    controller.set_task_space(task_space)
    controller.set_interface(interface)

    initial_target = vf.project(home_ee_motor)
    objective = ga.control.SingleArmMotorControlObjective(
        task_space, initial_target)
    controller.set_objective(objective)

    interface.control(controller, blocking=False)

    def on_twist(twist):
        controller.update_command(twist)

    def on_button(index):
        print("no button")

    device = KeyboardTeleop(
        step=0.0025) if args.device == "keyboard" else SpaceMouseTeleop()
    device.start(on_twist, on_button)

    try:
        if args.no_viz:
            print(f"Teleoperating '{
                  robot_cfg['task_space']}' (headless). Ctrl+C to quit.")
            while True:
                time.sleep(0.1)
        else:
            viz = gafropy_viz.Visualizer()

            robot = viz.add_robot(interface.get_system(),
                                  name=args.robot, joint_sliders=False)

            object_world_points = object_surface.vertices
            viz.scene.add_point_cloud(
                "/object", points=object_world_points, colors=(200, 200, 200), point_size=0.002
            )

            joint_indices = np.asarray(task_space.get_joint_indices())
            position = interface.get_state().get_position()[
                joint_indices]
            ee = task_space.compute_ee_motor(position)

            eeviz = viz.add_motor(ee)

            def update():
                q = interface.get_state().get_position()
                robot.update(q)
                joint_indices = np.asarray(task_space.get_joint_indices())
                position = q[joint_indices]
                ee = task_space.compute_ee_motor(position)
                eeviz.update(ee)

            viz.add_ticker(lambda: update())

            viz.show(message=f"Teleoperating '{
                     robot_cfg['task_space']}'. viser running at http://localhost:{viz.port} -- ctrl-c to quit.")
    except KeyboardInterrupt:
        pass
    finally:
        device.stop()
        interface.shutdown()


if __name__ == "__main__":
    main()
