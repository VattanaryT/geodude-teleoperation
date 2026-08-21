import argparse
import pathlib
import time

import gafropy.numpy as gn
import gafropy as ga
import numpy as np
from gafropy import robot_interface as ri
from gafropy import viz as gafropy_viz

from geodude_teleoperation import (
    AdmittanceTeleoperationController,
    TeleoperationController,
    KeyboardTeleop,
    OrientationField,
    SpaceMouseTeleop,
    compute_resting_orientation,
    load_object_mesh,
    load_points,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OBJECT = str(ROOT / "assets" / "objects" / "banana_half.ply")

ROBOTS = {
    "geodude": {
        "model": str(ROOT / "assets" / "robots" / "geodude" / "geodude.yaml"),
        "task_space": "left_ee",
        "has_vention": True,
    }
}


def main():

    print(ROOT)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "--object", default=DEFAULT_OBJECT,
        help="mesh or point cloud file to place on the table (see module docstring)")

    args = parser.parse_args()

    robot_cfg = ROBOTS["geodude"]

    system = ga.SystemSerialization.load(robot_cfg["model"])

    task_space = system.get_task_space(robot_cfg["task_space"])
    if task_space is None:
        raise RuntimeError(
            f"task space '{robot_cfg['task_space']}' not found in {robot_cfg['model']}")

    joint_idx = np.array(task_space.get_joint_indices())
    q = system.get_default_configuration()[joint_idx]
    home_ee_motor = task_space.compute_ee_motor(q)

    object_surface = load_object_mesh(args.object)

    key_direction = np.asarray(home_ee_motor.get_rotor().to_rotation_matrix())[:, 0]

    vf = OrientationField(object_surface, key_direction=key_direction)

    # assuming you have transformation matrices
    # this needs to be the actual end effector poses
    poses = np.tile(np.eye(4), (100, 1, 1))
    # this needs to be the actual end effector twists
    global_actions = np.tile(np.zeros(6), (100, 1))

    local_actions = []

    for i in range(100):
        local_frame = vf.project(ga.Motor.from_matrix(poses[i, :, :]))

        local_actions.append(ga.Twist.from_array(global_actions[i, :]).transform(
            local_frame.reverse()).to_array())

    local_actions = np.asarray(local_actions)

    print(local_actions)


if __name__ == "__main__":
    main()
