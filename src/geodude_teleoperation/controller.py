import numpy as np

try:
    import gafropy
    _VelocityControllerBase = gafropy.control.VelocityController
except ImportError:
    gafropy = None

    class _VelocityControllerBase:
        pass


def qdot_from_error(error, jacobian, *, gain, damping, max_joint_velocity):
    error = np.asarray(error)
    jacobian = np.asarray(jacobian)
    j_inverse = np.linalg.inv(
        jacobian.T @ jacobian + damping**2 * np.eye(jacobian.shape[1]))
    qdot = -gain * j_inverse @ jacobian.T @ error
    speed = np.linalg.norm(qdot)
    if speed > max_joint_velocity:
        qdot = qdot * (max_joint_velocity / speed)
    return qdot


def _adaptive_linear_scale(distance, min_factor, max_factor, offset=0.01, scale_range=0.075):
    """ufactory_demos' TeleoperationController::updateCommand adaptive linear-twist scaling
    factor, verbatim (gafro_teleoperation/TeleoperationController.hxx): grows linearly with
    distance from the surface, from min_factor right at the surface up to max_factor once
    distance clears offset + scale_range, clamped at both ends. offset/scale_range (0.01,
    0.075) are hardcoded constants in that source, not configurable there either; min/max
    default to ufactory_demos' own teleoperation_config.yaml values for this exact demo
    (1.0, 10.0) -- notably *not* a "slow down near the surface" factor <1 the way the name
    might suggest: even the minimum leaves the linear twist at full raw magnitude, scaling
    *up* to 10x that as distance grows. A fixed, much smaller scale (an earlier version of
    both controllers here used a constant 0.1) is 10-100x smaller than this across the
    whole range -- confirmed as the main source of "sticky, unresponsive" teleoperation
    feel relative to the original, especially moving away from the surface where the
    original speeds up and a fixed small scale can't.
    """
    factor = min_factor + (distance - offset) * \
        (max_factor - min_factor) / scale_range
    return min(max_factor, max(min_factor, factor))


class TeleoperationController(_VelocityControllerBase):
    """Direct proportional-velocity teleoperation toward a composed, fixture-corrected
    reference pose -- shares AdmittanceTeleoperationController's target-composition pattern
    exactly (see its docstring for the full rationale), differing only in the control law
    applied to the resulting pose error: plain proportional velocity control here
    (qdot = gain * J_pinv * error) instead of mass-spring-damper dynamics.

    An earlier version of this class composed differently and, in hindsight, wrongly:
    instead of composing the twist onto the EE's own body frame and then having the
    virtual fixture correct the resulting reference's orientation (what both this class
    and ufactory_demos' original TeleoperationController::updateCommand do), it rotated the
    raw input twist itself into "the fixture's local frame" before mapping it straight to
    joint velocities, with no persisted reference at all. That meant each twist axis's
    meaning silently changed depending on the object's local surface orientation at the
    current position, rather than staying fixed relative to the tool the way normal
    body-frame teleoperation does -- confirmed against ufactory_demos' actual source
    (gafro_teleoperation/TeleoperationController.hxx): `reference = ee_motor *
    Motor::exp(factor * twist.getLinear() + twist.getAngular())` composes in EE frame
    *unconditionally*, whether or not virtual fixtures are enabled; the fixture only
    replaces the orientation of that already-composed reference afterward, never the frame
    the twist itself is interpreted in.
    """

    def __init__(self, gain=16.0, damping=0.05, max_joint_velocity=1.0,
                 min_twist_scaling_factor=1.0, max_twist_scaling_factor=10.0):
        super().__init__()
        self.vf = None
        self._gain = gain
        self._damping = damping
        self._max_joint_velocity = max_joint_velocity
        self._min_twist_scaling_factor = min_twist_scaling_factor
        self._max_twist_scaling_factor = max_twist_scaling_factor
        self._distance = 0.0
        self._interface = None

    def compute_control_command(self, task_space_state, state):
        error = np.asarray(task_space_state.get_error())
        jacobian = np.asarray(task_space_state.get_jacobian())
        return qdot_from_error(
            error,
            jacobian,
            gain=self._gain,
            damping=self._damping,
            max_joint_velocity=self._max_joint_velocity,
        )

    def set_virtual_fixture(self, vf):
        self.vf = vf

    def set_interface(self, interface):
        self._interface = interface

    def update_command(self, twist):
        task_space = self.get_task_space()
        joint_indices = np.asarray(task_space.get_joint_indices())
        position = self._interface.get_state().get_position()[joint_indices]
        ee = task_space.compute_ee_motor(position)

        factor = _adaptive_linear_scale(
            self._distance, self._min_twist_scaling_factor, self._max_twist_scaling_factor)

        if self.vf is not None:
            translator = ee.get_translator()
            position = translator.to_translation_vector()
            reference = self.vf.project(ee)
            self._distance = self.vf.get_distance(position)

            twist = twist.transform(reference).transform(ee.reverse())
            angular = twist.get_angular()
            linear = 0.25 * factor * twist.get_linear()
            increment = gafropy.Motor.exp(
                angular[0], angular[1], angular[2], linear[0], linear[1], linear[2])
            reference = ee.multiply(increment)
            reference = self.vf.project(reference)

        # if ee.to_array()[0] > 0.0 and reference.to_array()[0] < 0.0:
        #     reference = reference.negate()
        # elif ee.to_array()[0] < 0.0 and reference.to_array()[0] > 0.0:
        #     reference = reference.negate()

        self.get_objective().set_target(reference)


class AdmittanceTeleoperationController(_VelocityControllerBase):

    def __init__(self, inertia=None, damping=None, stiffness=None, dt=0.001,
                 max_joint_velocity=1.0, min_twist_scaling_factor=1.0,
                 max_twist_scaling_factor=10.0):
        super().__init__()
        self._inertia = np.diag([1.0] * 6) if inertia is None else inertia
        self._damping = np.diag([20.0] * 6) if damping is None else damping
        self._stiffness = np.diag(
            [3000.0] * 6) if stiffness is None else stiffness
        self._dt = dt
        self._max_joint_velocity = max_joint_velocity
        self._min_twist_scaling_factor = min_twist_scaling_factor
        self._max_twist_scaling_factor = max_twist_scaling_factor
        self._distance = 0.0

        self._interface = None
        self.vf = None

    def compute_control_command(self, task_space_state, state):
        error = np.asarray(task_space_state.get_error())
        error_derivative = np.asarray(task_space_state.get_error_derivative())
        jacobian_derivative = np.asarray(
            task_space_state.get_jacobian_derivative())
        velocity = np.asarray(state.get_velocity())

        wrench_net = -self._stiffness @ error - self._damping @ error_derivative
        task_acceleration = np.linalg.solve(self._inertia, wrench_net)

        bias = jacobian_derivative @ velocity[1:7]

        jacobian = np.asarray(task_space_state.get_jacobian())
        # joint_acceleration = np.linalg.lstsq(
        #     jacobian, task_acceleration - bias, rcond=None)[0]

        j_inverse = np.linalg.inv(
            jacobian.T @ jacobian + 1e-6 * np.eye(jacobian.shape[1]))

        qdot = self._dt * \
            j_inverse @ jacobian.T @ (task_acceleration - bias)

        speed = np.linalg.norm(qdot)
        if speed > self._max_joint_velocity:
            qdot = qdot * (self._max_joint_velocity / speed)

        return qdot

    def set_virtual_fixture(self, vf):
        self.vf = vf

    def set_interface(self, interface):
        self._interface = interface

    def update_command(self, twist):
        task_space = self.get_task_space()
        joint_indices = np.asarray(task_space.get_joint_indices())
        position = self._interface.get_state().get_position()[joint_indices]
        ee = task_space.compute_ee_motor(position)

        angular = twist.get_angular()
        linear = 2.0 * twist.get_linear()
        increment = gafropy.Motor.exp(
            angular[0], angular[1], angular[2], linear[0], linear[1], linear[2])
        reference = ee.multiply(increment)

        if self.vf is not None:
            reference = self.vf.project(reference)

        # if ee.to_array()[0] > 0.0 and reference.to_array()[0] < 0.0:
        #     reference = reference.negate()
        # elif ee.to_array()[0] < 0.0 and reference.to_array()[0] > 0.0:
        #     reference = reference.negate()

        self.get_objective().set_target(reference)
