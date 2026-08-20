import os
import time

import numpy as np
import gafropy as ga
from gafropy import robot_interface as ri
from gafropy import viz as gafropy_viz


def main():

    config = ri.Configuration()
    config.model_file = os.path.join(os.path.dirname(__file__), "..",
                                     "assets", "robots", "geodude", "geodude.yaml")
    config.type = "mujoco"

    robot = ri.RobotInterface(config)
    task_space = robot.get_task_space("left_ee")

    controller = ga.control.VelocityAdmittanceController(config=ga.control.AdmittanceConfig(stiffness=np.eye(6), damping=np.eye(6), inertia=np.eye(6)), dt=0.01)

    target = ga.Line.Random()
    objective = ga.control.SingleArmLineControlObjective(target)

    controller.set_objective(objective)
    controller.set_task_space(task_space)

    robot.initialize()

    robot.control(controller)

    try:
        viz = gafropy_viz.Visualizer()

        robot_viz = viz.add_robot(robot.get_system())

        position = task_space.extract_configuration(robot.get_state().get_position())
        ee_line = task_space.compute_ee_motor(position).transform_line(ga.Line.Z())

        viz.add_line(target)
        ee_viz = viz.add_line(ee_line)

        def update():
            q = robot.get_state().get_position()
            robot_viz.update(q)

            position = task_space.extract_configuration(robot.get_state().get_position())
            ee_line = task_space.compute_ee_motor(position).transform_line(ga.Line.Z())
            ee_viz.update(ee_line)

        viz.add_ticker(lambda: update())

        viz.show()
    except KeyboardInterrupt:
        pass
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
