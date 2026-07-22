import argparse
import pathlib
import time

import gafropy.numpy as gn
import gafropy as ga
import numpy as np
from diffused_fields.diffusion.walk_on_spheres import WalkOnSpheresDiffusion
from gafropy import robot_interface as ri
from gafropy import viz as gafropy_viz

from geodude_teleoperation import (
    AdmittanceTeleoperationController,
    TeleoperationController,
    KeyboardTeleop,
    SpaceMouseTeleop,
    WalkOnSpheresVirtualFixture,
    build_wos_boundary,
    compute_resting_orientation,
    load_object_surface,
    load_points,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets" / "robots" / "geodude"
MODEL = str(ASSETS / "geodude.yaml")
DEFAULT_OBJECT = str(ROOT / "assets" / "objects" / "debug_box.stl")
TASK_SPACE_NAME = "left_ee"


def main():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    parser.add_argument(
        "--auto-orient", action="store_true",
        help="rotate the object to rest on its flattest axis (see module docstring)")
    parser.add_argument("--no-viz", action="store_true",
                        help="disable the viser visualization")
    args = parser.parse_args()

    controller = (
        TeleoperationController(damping=0.05) if args.controller == "velocity"
        else AdmittanceTeleoperationController()
    )

    config = ri.Configuration()
    config.model_file = MODEL
    config.type = "mujoco"

    interface = ri.RobotInterface(config)
    interface.initialize()

    task_space = interface.get_task_space(TASK_SPACE_NAME)
    if task_space is None:
        raise RuntimeError(
            f"task space '{TASK_SPACE_NAME}' not found in {MODEL}")

    joint_idx = np.array(task_space.get_joint_indices())
    live_position = interface.get_state().get_position()[joint_idx]
    home_ee_motor = task_space.compute_ee_motor(live_position)
    home_translator = home_ee_motor.get_translator()
    home_xyz = np.array(
        [home_translator.x(), home_translator.y(), 0.0])

    # joint_idx[0] is left_arm_linear_vention, a vertical prismatic rail (see
    # geodude.yaml) -- its raw joint value is a direct additive Z offset on every
    # downstream link, including the end effector. Table height below is derived from
    # home_xyz, so raising/lowering the vention before running this demo (e.g. to get
    # more headroom, see controller.py's DLS fix) would otherwise raise/lower the
    # object right along with it instead of leaving it resting on a fixed physical
    # table. Subtracting the live vention offset and adding back the "ready" keyframe's
    # default (0.25, see geodude.yaml keyframes) keeps the table at the same fixed
    # height regardless of where the vention happens to be parked at startup.
    # DEFAULT_VENTION_HEIGHT = 0.0
    # vention_offset = live_position[0] - DEFAULT_VENTION_HEIGHT
    # table_reference_xyz = home_xyz - np.array([0.0, 0.0, vention_offset])

    object_points_raw = load_points(args.object)

    table_pose = np.eye(4)
    if args.auto_orient:
        table_pose[:3, :3] = compute_resting_orientation(object_points_raw)
    rotated_points = object_points_raw @ table_pose[:3, :3].T
    object_centroid = (rotated_points.max(axis=0) +
                       rotated_points.min(axis=0)) / 2

    table_pose[:3, 3] = home_xyz + \
        np.array([0.05, 0.0, 0.8]) - object_centroid

    object_surface = load_object_surface(
        args.object, table_pose=table_pose, normal_orientation=args.normal_orientation)

    object_wos_boundary = build_wos_boundary(object_surface)

    object_wos = WalkOnSpheresDiffusion(
        boundaries=[object_wos_boundary], batch_size=768, max_iterations=24,
        convergence_threshold=0.005)
    vf = WalkOnSpheresVirtualFixture(object_wos)

    # controller.set_virtual_fixture(vf)
    controller.set_task_space(task_space)
    controller.set_interface(interface)

    initial_target = home_ee_motor
    initial_target = vf.project(home_ee_motor)
    objective = ga.control.SingleArmMotorControlObjective(
        task_space, initial_target)
    controller.set_objective(objective)

    interface.control(controller, blocking=False)

    def on_twist(twist):
        controller.update_command(twist)

    def on_button(index):
        print("no button")
        # if index == 0:
        # controller.toggle_teleoperation_enabled()
        # print("teleoperation:",
        #       "ON" if controller.is_teleoperation_enabled() else "OFF")
        # elif index == 1:
        #     controller.toggle_virtual_fixtures_enabled()
        #     print("virtual fixture:",
        #           "ON" if controller.is_virtual_fixtures_enabled() else "OFF")

    device = KeyboardTeleop(
        step=0.01) if args.device == "keyboard" else SpaceMouseTeleop()
    device.start(on_twist, on_button)

    # print("virtual fixture:",
    #       "ON" if controller.is_virtual_fixtures_enabled() else "OFF")

    try:
        if args.no_viz:
            print(f"Teleoperating '{
                  TASK_SPACE_NAME}' (headless). Ctrl+C to quit.")
            while True:
                time.sleep(0.1)
        else:
            viz = gafropy_viz.Visualizer()

            robot = viz.add_robot(interface.get_system(),
                                  name="geodude", joint_sliders=False)

            object_world_points = object_surface.vertices  # already placed by table_pose
            viz.scene.add_point_cloud(
                "/object", points=object_world_points, colors=(200, 200, 200), point_size=0.002
            )

            joint_indices = np.asarray(task_space.get_joint_indices())
            position = interface.get_state().get_position()[
                joint_indices]
            ee = task_space.compute_ee_motor(position)

            eeviz = viz.add_motor(ee)
            mviz = viz.add_motor(vf.project(ee))
            oviz = viz.add_motor(objective.get_target())

            def update():
                q = interface.get_state().get_position()
                robot.update(q)
                joint_indices = np.asarray(task_space.get_joint_indices())
                position = q[joint_indices]
                ee = task_space.compute_ee_motor(position)
                mviz.update(vf.project(ee))
                oviz.update(objective.get_target())
                eeviz.update(ee)

            viz.add_ticker(lambda: update())

            viz.show(message=f"Teleoperating '{
                     TASK_SPACE_NAME}'. viser running at http://localhost:{viz.port} -- ctrl-c to quit.")
    except KeyboardInterrupt:
        pass
    finally:
        device.stop()
        interface.shutdown()


if __name__ == "__main__":
    main()
