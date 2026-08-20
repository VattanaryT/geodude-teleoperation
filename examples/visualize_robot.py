import os
import threading
import time

import numpy as np
import gafropy as ga
from gafropy import control
from gafropy import robot_interface as ri
from gafropy import viz as gafropy_viz


def control_groups_driving(system, task_space):
    """The system's control groups whose joints this task space occupies.

    A controller writes its command to the joints of the control groups it
    carries, so it has to be told them explicitly. ``task_space.get_control_groups()``
    is only populated when the task space was *constructed* naming groups; geodude's
    left_ee task space does name one (left_ur5e), so this is mostly redundant here,
    but it's kept general -- match by joints instead of relying on that.
    """
    joints = set(task_space.get_joint_indices())
    return [group for group in system.get_control_groups()
            if joints & set(group.joint_indices)]


def estimate_wrench(task_space, robot_state):
    """Least-squares 6D wrench estimate at the EE, from the residual joint torque.

    6 x n Jacobian at the EE point, world frame, rows [torque(3); force(3)] (gafro's
    Twist/Wrench pairing is angular-then-linear -- see gafro/physics/Twist.hpp and
    gafro/control/abstract/AdmittanceController.hxx, which combines this exact space
    additively with stiffness/damping terms). tau = J^T w, solved by least squares.

    Caveat: robot_state.get_torque() is qfrc_applied - qfrc_bias (MujocoBackend::
    sampleState); real contact forces land in MuJoCo's qfrc_constraint, which this
    never sees. So this estimate is blind to genuine contact with the box -- it
    reads the model's own gravity/Coriolis dynamics term (roughly constant at rest,
    hence the caller calibrating a baseline), not a real F/T sensor.
    """
    indices = task_space.get_joint_indices()
    q = robot_state.get_position()[indices]
    v = robot_state.get_velocity()[indices]
    tau = robot_state.get_torque()[indices]

    state = task_space.compute_state(q, v)
    jacobian = state.get_jacobian()

    wrench, *_ = np.linalg.lstsq(jacobian.T, tau, rcond=None)
    return wrench


def line_direction(line):
    """Unit direction of a gafropy.Line, as a plain (3,) array."""
    d = line.get_direction()
    v = np.array([d.x(), d.y(), d.z()])
    return v / np.linalg.norm(v)


class LineForcePIDController(control.VelocityAdmittanceController):

    def __init__(self, task_space, config, target_force: float,
                 kp: float = 2.0, ki: float = 0.5, kd: float = 0.0,
                 update_period: float = 0.005):
        super().__init__(config)
        self.task_space = task_space
        self.target_force = target_force
        self.target_wrench = ga.Wrench(tx=0, ty=0, tz=0, fx=0, fy=0, fz=target_force).to_array()
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.update_period = update_period

        self._integral = 0.0
        self._prev_error = None
        self._baseline = None
        self._stop = threading.Event()
        self._thread = None

    def start(self, robot):
        self._thread = threading.Thread(target=self._run, args=(robot,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self, robot):
        prev_time = time.time()

        while not self._stop.is_set():
            state = robot.get_state()

            # raw_wrench = estimate_wrench(self.task_space, state)
            # if self._baseline is None:
            #     self._baseline = raw_wrench
            # wrench = raw_wrench - self._baseline
            wrench = state.get_wrench().to_array()

            now = time.time()
            dt = max(now - prev_time, 1e-6)
            prev_time = now

            error = self.target_wrench - wrench

            self._integral += error * dt
            derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
            self._prev_error = error

            desired = self.kp * error + self.ki * self._integral + self.kd * derivative
            self.set_desired_wrench(np.array([0, 0, 0, 0, 0, error[3]]))

            time.sleep(self.update_period)


def main():

    config = ri.Configuration()
    config.model_file = os.path.join(os.path.dirname(__file__), "..",
                                     "assets", "robots", "geodude", "geodude_cylinder_ee.yaml")
    config.type = "mujoco"

    robot = ri.RobotInterface(config)

    robot.initialize()

    task_space = robot.get_task_space("right_ee")
    print(task_space.get_joint_indices())
    print(task_space.get_controlled_joint_indices())
    print(task_space.get_task_space_joint_indices())

    default_configuration = robot.get_system().get_default_configuration()
    default_task_configuration = task_space.extract_configuration(default_configuration)
    ee_motor = task_space.compute_ee_motor(default_task_configuration)
    ee_position = ee_motor.get_translator().to_translation_vector()
    approach_direction = np.array([0, 0, -1.0])

    r = ga.Rotor.exp(0.05, 0.0, 0.0)
    t = ga.Translator.exp(*ee_position-np.array([0.1, 0.0, 0.0]))
    m = ga.Motor(t, r) * ga.Motor(t.reverse(), ga.Rotor())

    target_line = ga.Line(ga.Point(*ee_position), ga.Point(*(ee_position + approach_direction)))

    config = ga.control.VelocityAdmittanceControllerConfig()
    config.dt = 0.01
    config.max_linear_speed = 0.1
    config.max_angular_speed = 0.5
    config.admittance = control.AdmittanceConfig(
        inertia=1.0 * np.eye(6),
        damping=50.0 * np.eye(6),
        stiffness=50000.0 * np.eye(6),
    )
    # controller = control.VelocityAdmittanceController(config)
    controller = LineForcePIDController(task_space, config,
                                        target_force=10.0, kp=2.0, ki=0.75, kd=0.)

    objective = control.SingleArmLineControlObjective(target_line)
    controller.set_objective(objective)
    controller.set_task_space(task_space)

    robot.control(controller, blocking=False)
    controller.start(robot)

    try:
        viz = gafropy_viz.Visualizer()

        robot_viz = viz.add_robot(robot.get_system())

        target_line_viz = viz.add_line(target_line, name="/target_line")
        position = task_space.extract_configuration(robot.get_state().get_position())
        ee_line = task_space.compute_ee_motor(position).transform_line(ga.Line.Z())
        ee_line_viz = viz.add_line(ee_line, name="/ee_line")

        def update():
            nonlocal target_line  # Tell Python we're using outer target_line

            state = robot.get_state()
            robot_viz.update(state.get_position())

            position = task_space.extract_configuration(state.get_position())
            ee_line_viz.update(task_space.compute_ee_motor(position).transform_line(ga.Line.Z()))

            target_line = m.transform_line(target_line)
            objective.set_target(target_line)
            objective.set_target_twist(ga.Twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            target_line_viz.update(target_line)

        viz.add_ticker(lambda: update())

        viz.show()
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        robot.shutdown()


if __name__ == "__main__":
    main()
