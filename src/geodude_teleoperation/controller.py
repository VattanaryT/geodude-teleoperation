import threading
import time

import gafropy
import numpy as np


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


class TeleoperationController(gafropy.control.VelocityController):
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

    def compute_control_command(self, robot_state, objective_state, task_space_state):
        error = np.asarray(objective_state.get_error())
        jacobian = np.asarray(task_space_state.get_jacobian())

        j_inverse = np.linalg.inv(
            jacobian.T @ jacobian + self._damping**2 * np.eye(jacobian.shape[1]))

        qdot = -self._gain * j_inverse @ jacobian.T @ error

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
        else:
            # No virtual fixture: compose the raw twist straight onto the live EE pose,
            # unconditionally-in-EE-frame, exactly the ufactory_demos formula quoted in this
            # class's own docstring (`reference = ee_motor * Motor::exp(...)`) minus the
            # fixture-only orientation replacement that formula's second half describes.
            angular = twist.get_angular()
            linear = 0.25 * factor * twist.get_linear()
            increment = gafropy.Motor.exp(
                angular[0], angular[1], angular[2], linear[0], linear[1], linear[2])
            reference = ee.multiply(increment)

        # if ee.to_array()[0] > 0.0 and reference.to_array()[0] < 0.0:
        #     reference = reference.negate()
        # elif ee.to_array()[0] < 0.0 and reference.to_array()[0] > 0.0:
        #     reference = reference.negate()

        self.get_objective().set_target(reference)


class AdmittanceTeleoperationController(gafropy.control.VelocityController):

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

    def compute_control_command(self, robot_state, objective_state, task_space_state):
        error = np.asarray(objective_state.get_error())
        error_derivative = np.asarray(objective_state.get_error_derivative())
        jacobian_derivative = np.asarray(
            task_space_state.get_jacobian_derivative())
        velocity = np.asarray(robot_state.get_velocity())

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


class LineAdmittanceTeleoperationController(gafropy.control.VelocityAdmittanceController):
    """gafropy's C++ VelocityAdmittanceController tracking a *line* target (rather
    than a full pose), wrapped with an outer force-PID loop and teleoperated line
    motion -- the line-control analogue of AdmittanceTeleoperationController above.

    A line target only constrains the EE body-frame Z-axis's direction and its
    lateral offset from the target line; translation along the line and rotation
    about it (roll) leave the line unchanged, so those two DOF are free for whatever
    the admittance config's stiffness doesn't otherwise pin down. Zeroing stiffness
    index 5 (linear Z, see AdmittanceConfig/spacemouse.py's twist-blade-order note)
    hands depth-along-the-line to `set_desired_wrench` instead of position control;
    `_run`'s background thread drives that wrench from a PID on the measured contact
    force (`state.get_wrench()`, index 5 = force along body Z) against
    `target_force`, mirroring gufic_line.py's GUFICLineVelocityController.

    The target line's *direction* (`direction`, normal to the table by default,
    i.e. straight down) is fixed for the controller's whole lifetime -- teleop never
    rotates it, only translates it. `update_command` (wired to a teleop device's
    on_twist callback) accumulates only the twist's linear X/Y onto a *persisted*
    (x, y) point; the angular components and linear Z are ignored outright, and the
    line is rebuilt each call from that (x, y) plus the fixed `direction`. This is
    deliberately unlike TeleoperationController/AdmittanceTeleoperationController
    above, which recompose their (full-pose) target from the live EE pose every
    call -- right for them (idle input means "hold here," since target == current
    pose), wrong for a line target, where it would continuously re-lock the target
    line onto the current EE line (and, worse, onto the EE's live *orientation*,
    silently drifting the line off-normal from whatever this controller is meant to
    stay perpendicular to). It's also a trap with SpaceMouseTeleop specifically:
    its on_twist fires continuously, zero twist included, so live-pose
    recomposition would drive a *held-still* target line to visually track the EE
    line move-for-move -- "second EE line" symptom rather than a fixed target the
    operator repositions. Persisting (x, y) instead of a full motor means a zero
    twist truly leaves the target unchanged at any callback rate, and there is no
    orientation state left to drift in the first place. Depth along `direction` and
    roll about it -- the two DOF a line target can't see -- are exactly the axes
    force control governs (see above), so teleoperating (x, y) and running force
    control along the line never fight over the same DOF.
    """

    def __init__(self, config, direction=(0.0, 0.0, -1.0), target_force=10.0,
                 kp_force=5.0, ki_force=1.0, kd_force=0.0, force_feedforward=0.0,
                 force_limit=50.0, integral_limit=20.0, update_period=0.005,
                 linear_scale=2.0):
        super().__init__(config)
        direction = np.asarray(direction, dtype=float)
        self._direction = direction / np.linalg.norm(direction)
        self._target_force = target_force
        self._kp_force = kp_force
        self._ki_force = ki_force
        self._kd_force = kd_force
        self._force_feedforward = force_feedforward
        self._force_limit = force_limit
        self._integral_limit = integral_limit
        self._update_period = update_period
        self._linear_scale = linear_scale

        self._interface = None
        self._integral = 0.0
        self._prev_error = None
        self._stop = threading.Event()
        self._thread = None
        self._target_xy = None

    def set_interface(self, interface):
        self._interface = interface

    def _compute_force_command(self):
        wrench = self._interface.get_state().get_wrench().to_array()
        force_along_z = wrench[5]
        error = self._target_force - force_along_z

        self._integral = np.clip(
            self._integral + error * self._update_period,
            -self._integral_limit, self._integral_limit)
        derivative = 0.0 if self._prev_error is None else \
            (error - self._prev_error) / self._update_period
        self._prev_error = error

        magnitude = np.clip(
            self._kp_force * error + self._ki_force * self._integral +
            self._kd_force * derivative + self._force_feedforward,
            -self._force_limit, self._force_limit)

        wrench_cmd = np.zeros(6)
        wrench_cmd[5] = magnitude
        return wrench_cmd

    def _run(self):
        while not self._stop.is_set():
            self.set_desired_wrench(self._compute_force_command())
            time.sleep(self._update_period)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def update_command(self, twist):
        if self._target_xy is None:
            # Lazy-initialized from the objective's own initial target (set by the
            # caller before teleop starts) rather than the live EE pose -- so the
            # starting point is whatever the caller placed it at, not wherever the
            # arm happened to be resting, and `direction` (not the arm's current
            # orientation) is the only thing that ever determines the line's
            # direction.
            start = self.get_objective().get_target().project(gafropy.Point(0.0, 0.0, 0.0))
            self._target_xy = np.array([start.x(), start.y()])

        self._target_xy = self._target_xy + \
            self._linear_scale * np.asarray(twist.get_linear())[:2]

        x, y = self._target_xy
        a = gafropy.Point(x, y, 0.0)
        b = gafropy.Point(x + self._direction[0], y + self._direction[1], self._direction[2])
        self.get_objective().set_target(gafropy.Line(a, b))


class ContactTeleoperationController(gafropy.control.VelocityAdmittanceController):
    """LineAdmittanceTeleoperationController, generalized from a fixed table-normal line to a
    curved, object-centric one: same SingleArmLineControlObjective / index-5-zero-stiffness /
    force-PID-along-the-line design (see that class's docstring for why a Line target's
    "linear Z" is the axis to zero, and why depth along it therefore never needs to be
    teleoperated), except the target line's *direction* isn't fixed for the controller's whole
    lifetime -- it's re-read every tick from `vf` (an OrientationField), evaluated at wherever
    the operator is currently pointing.

    `update_command` composes the teleoperated twist onto the live EE pose (so idle input
    holds still, same as AdmittanceTeleoperationController), zeroes the composed increment's
    own approach-axis component first (index 2 of its linear part, using the EE's *current*
    frame as the approach-axis proxy -- teleop input along it is ignored outright, exactly
    LineAdmittanceTeleoperationController's "linear Z ignored", rather than relying on it being
    projected away later), and reprojects the result through `vf.project(...)`, which snaps
    the resulting pose's orientation to the field's locally diffused approach basis (see
    OrientationField.project / _orient_approach_into_surface).

    Only that projected pose's line -- `reference.transform_line(gafropy.Line.Z())`, its
    position and the world-frame direction of its own local Z axis, packaged as a Line the
    same way `Motor.transform_line` is already used elsewhere (e.g.
    line_admittance_teleop.py's `ee_motor.transform_line(ga.Line.Z())` for the EE's own line
    in its viz) -- reaches the objective, never the full pose. A Line target's Plücker
    representation is invariant to which point along the infinite line is used to construct
    it (LineAdmittanceTeleoperationController's own "any point on the line" comment), so which
    exact position vf.project(...) returned along that direction is irrelevant -- depth along
    the field's local approach axis genuinely never reaches this controller's target at all.

    `_run`'s background thread PIDs `set_desired_wrench` off the measured contact force
    (`state.get_wrench()`, index 5) against `target_force`, identical to
    LineAdmittanceTeleoperationController's own force loop -- see that class's docstring for
    why index 5 is the right one: SingleArmLineControlObjective computes its error against the
    target line expressed in the EE's *own* current frame (`ee.reverse().transform(target)`,
    see SingleArmLineControlObjective.cpp), so once orientation tracking has converged, the
    EE's body-frame Z axis *is* the line's direction, whatever that direction happens to be.
    `config` (built by the caller, mirroring LineAdmittanceTeleoperationController) needs its
    stiffness/damping zeroed at index 5 for this to hold: gafropy's AdmittanceConfig has no
    Python-side setters, so which axis is force-controlled can't be changed after construction
    -- swapping between an AdmittanceTeleoperationController (free 6-DOF admittance) and this
    class via `interface.control(...)` is how a caller offers two distinct modes from one
    teleop session (see examples/hardware_object_contact_teleop.py).
    """

    def __init__(self, vf, config, target_force=5.0, kp_force=5.0, ki_force=1.0, kd_force=0.0,
                 force_feedforward=0.0, force_limit=30.0, integral_limit=15.0,
                 update_period=0.005, linear_scale=1.0):
        super().__init__(config)
        self.vf = vf
        self._target_force = target_force
        self._kp_force = kp_force
        self._ki_force = ki_force
        self._kd_force = kd_force
        self._force_feedforward = force_feedforward
        self._force_limit = force_limit
        self._integral_limit = integral_limit
        self._update_period = update_period
        self._linear_scale = linear_scale

        self._interface = None
        self._integral = 0.0
        self._prev_error = None
        self._stop = threading.Event()
        self._thread = None

    def set_interface(self, interface):
        self._interface = interface

    def _compute_force_command(self):
        wrench = self._interface.get_state().get_wrench().to_array()
        force_along_z = wrench[5]
        error = self._target_force - force_along_z

        self._integral = np.clip(
            self._integral + error * self._update_period,
            -self._integral_limit, self._integral_limit)
        derivative = 0.0 if self._prev_error is None else \
            (error - self._prev_error) / self._update_period
        self._prev_error = error

        magnitude = np.clip(
            self._kp_force * error + self._ki_force * self._integral +
            self._kd_force * derivative + self._force_feedforward,
            -self._force_limit, self._force_limit)

        wrench_cmd = np.zeros(6)
        wrench_cmd[5] = magnitude
        return wrench_cmd

    def _run(self):
        while not self._stop.is_set():
            self.set_desired_wrench(self._compute_force_command())
            time.sleep(self._update_period)

    def start(self):
        # Unlike LineAdmittanceTeleoperationController's start/stop (only ever called once per
        # process), this controller's PID thread is meant to be started and stopped repeatedly
        # as the operator toggles modes -- clear `_stop` and the PID's integrator/derivative
        # state each time, or a second start() after a stop() would see `_stop` already set and
        # the thread would exit immediately, and stale windup would carry over from the
        # previous contact session.
        self._integral = 0.0
        self._prev_error = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def update_command(self, twist):
        task_space = self.get_task_space()
        joint_indices = np.asarray(task_space.get_joint_indices())
        position = self._interface.get_state().get_position()[joint_indices]
        ee = task_space.compute_ee_motor(position)

        angular = twist.get_angular()
        linear = np.asarray(twist.get_linear())
        increment = gafropy.Motor.exp(
            angular[0], angular[1], angular[2],
            self._linear_scale * linear[0], self._linear_scale * linear[1], 0.0)
        reference = self.vf.project(ee.multiply(increment))

        self.get_objective().set_target(reference.transform_line(gafropy.Line.Z()))
