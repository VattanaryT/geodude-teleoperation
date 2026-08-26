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


def slew_qdot(prev, desired, dt, max_qddot, max_dt=0.004):
    """Limit |Δq̇| / dt so idle-zero and target jumps cannot command an accel spike."""
    desired = np.asarray(desired, dtype=float).reshape(-1).copy()
    if prev is None:
        return desired
    prev = np.asarray(prev, dtype=float).reshape(-1)
    if prev.shape != desired.shape:
        return desired
    dt = min(max(float(dt), 0.0), float(max_dt))
    max_step = float(max_qddot) * dt
    if max_step <= 0.0:
        return prev.copy()
    delta = desired - prev
    n = float(np.linalg.norm(delta))
    if n <= max_step:
        return desired
    return prev + delta * (max_step / n)


def lowpass_qdot(prev, desired, dt, tau=0.03, max_dt=0.004):
    """Smooth the pre-slew command so a flipping wrist desire cannot bang-bang at max_qddot."""
    desired = np.asarray(desired, dtype=float).reshape(-1).copy()
    if prev is None:
        return desired
    prev = np.asarray(prev, dtype=float).reshape(-1)
    if prev.shape != desired.shape:
        return desired
    tau = max(float(tau), 1e-6)
    dt = min(max(float(dt), 0.0), float(max_dt))
    alpha = min(1.0, dt / tau)
    return prev + alpha * (desired - prev)


def contact_pid_force_limit(target_force, *, headroom=5.0):
    """PID output clip. Must be >= |Fd| or the host loop cannot reach the target."""
    return abs(float(target_force)) + abs(float(headroom))


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
                 max_twist_scaling_factor=10.0, twist_linear_scale=2.0,
                 max_qddot=5.0):
        super().__init__()
        self._inertia = np.diag([1.0] * 6) if inertia is None else inertia
        self._damping = np.diag([20.0] * 6) if damping is None else damping
        self._stiffness = np.diag(
            [3000.0] * 6) if stiffness is None else stiffness
        self._dt = dt
        self._max_joint_velocity = max_joint_velocity
        self._min_twist_scaling_factor = min_twist_scaling_factor
        self._max_twist_scaling_factor = max_twist_scaling_factor
        self._twist_linear_scale = float(twist_linear_scale)
        self._distance = 0.0

        self._interface = None
        self.vf = None
        # When False (hands-off SpaceMouse), stream zero joint velocity — otherwise
        # residual pose error / wrong-arm bias keeps speedJ moving with no stick input.
        self._input_active = False
        self._tool_z_velocity = 0.0
        self._stiffness = np.array(self._stiffness, dtype=float, copy=True)
        self._kz_pose = float(self._stiffness[5, 5])
        self._z_compliant = False
        self._slide_twist = np.zeros(6)
        self._max_slide_mps = 0.015
        self._max_slide_radps = 0.12
        self._command_trace = None
        self._stick_linear = np.zeros(3)
        self._max_qddot = float(max_qddot)
        self._qdot_cmd = None
        self._qdot_des = None
        self._qdot_cmd_t = None
        self._force_owns_z = False

    def set_command_trace(self, fn):
        """Optional callback(qdot, measured=, spring=, stick=) after each speedJ command."""
        self._command_trace = fn

    def set_stick_linear(self, linear):
        """Latest SpaceMouse linear (m) for the command plot. Does not drive speedJ."""
        self._stick_linear = np.asarray(linear, dtype=float).reshape(3).copy()

    def set_input_active(self, active: bool) -> None:
        self._input_active = bool(active)
        if not self._input_active:
            self._slide_twist = np.zeros(6)

    def contact_slide_twist(self):
        return np.array(self._slide_twist, copy=True)

    def merged_contact_twist(self, vz):
        """DualLoop merge: slide never presses. Retract while sliding is still allowed."""
        twist = np.array(self._slide_twist, dtype=float)
        vz = float(vz)
        sliding = np.linalg.norm(twist[:5]) > 1e-4
        twist[5] = min(0.0, vz) if sliding else vz
        return twist

    def hold_live_pose(self) -> None:
        """Snap objective to the current EE so the next teleop step has no pose jump."""
        if self._interface is None:
            return
        task_space = self.get_task_space()
        joint_indices = np.asarray(task_space.get_joint_indices())
        position = self._interface.get_state().get_position()[joint_indices]
        ee = task_space.compute_ee_motor(position)
        self.get_objective().set_target(ee)

    def set_tool_z_velocity(self, vz):
        """Overlay tool-Z speed (m/s) on top of the pose spring. DualLoop press axis."""
        self._tool_z_velocity = float(vz)

    def set_z_compliant(self, compliant):
        """Zero pose-spring Kz so a Z overlay is not fought. Stick Z is ignored.

        Also switches the plant to velocity-slide (no XY spring). Replay that
        must stay on a Cartesian path should use ``set_force_owns_z`` instead.
        """
        self._z_compliant = bool(compliant)
        if self._z_compliant:
            self._stiffness[5, 5] = 0.0
        else:
            self._stiffness[5, 5] = (
                0.0 if self._force_owns_z else self._kz_pose)
            self._slide_twist = np.zeros(6)

    def set_force_owns_z(self, owns: bool) -> None:
        """Zero Kz so DualLoop vz is not fought. XY pose-spring stays on."""
        self._force_owns_z = bool(owns)
        if self._z_compliant:
            return
        self._stiffness[5, 5] = 0.0 if self._force_owns_z else self._kz_pose

    def compute_control_command(self, robot_state, objective_state, task_space_state):
        jacobian = np.asarray(task_space_state.get_jacobian())
        n_joints = int(jacobian.shape[1])
        vz = float(self._tool_z_velocity)
        velocity = np.asarray(robot_state.get_velocity(), dtype=float).reshape(-1)
        # robot_state here is the *control-group* slice (len 7 for one arm+vention),
        # not the full bimanual q. get_joint_indices() are system-global and must not
        # be used to index this vector (that crashed with index 15 on size 7).
        if velocity.size == n_joints:
            qdot_meas = velocity
        elif velocity.size > n_joints:
            qdot_meas = velocity[-n_joints:]
        else:
            qdot_meas = None

        if self._z_compliant:
            # DualLoop merge: stick ⊥n + ω as a capped Cartesian velocity, force owns Z.
            # Do not run the K=2000 pose spring while loaded (that is the slide C153).
            twist = self.merged_contact_twist(vz)
            j_inverse = np.linalg.inv(
                jacobian.T @ jacobian + 1e-6 * np.eye(jacobian.shape[1]))
            qdot = j_inverse @ jacobian.T @ twist
        elif not self._input_active:
            qdot = np.zeros(n_joints)
        else:
            if qdot_meas is None:
                raise RuntimeError(
                    f"robot_state velocity length {velocity.size} < task joints {n_joints}")
            qdot_task = qdot_meas
            error = np.asarray(objective_state.get_error())
            error_derivative = np.asarray(objective_state.get_error_derivative())
            jacobian_derivative = np.asarray(
                task_space_state.get_jacobian_derivative())

            wrench_net = -self._stiffness @ error - self._damping @ error_derivative
            task_acceleration = np.linalg.solve(self._inertia, wrench_net)

            bias = jacobian_derivative @ qdot_task

            j_inverse = np.linalg.inv(
                jacobian.T @ jacobian + 1e-6 * np.eye(jacobian.shape[1]))

            qdot = self._dt * \
                j_inverse @ jacobian.T @ (task_acceleration - bias)

            if abs(vz) > 1e-12:
                twist = np.zeros(6)
                twist[5] = vz
                qdot = qdot + j_inverse @ jacobian.T @ twist

        speed = np.linalg.norm(qdot)
        if speed > self._max_joint_velocity:
            qdot = qdot * (self._max_joint_velocity / speed)

        now = time.perf_counter()
        dt_slew = self._dt if self._qdot_cmd_t is None else now - self._qdot_cmd_t
        self._qdot_des = lowpass_qdot(self._qdot_des, qdot, dt_slew)
        qdot = slew_qdot(
            self._qdot_cmd, self._qdot_des, dt_slew, self._max_qddot)
        self._qdot_cmd = np.asarray(qdot, dtype=float).reshape(-1).copy()
        self._qdot_cmd_t = now

        if self._command_trace is not None:
            self._command_trace(
                qdot, measured=qdot_meas, spring=self._input_active,
                stick=self._stick_linear)
        return qdot

    def set_virtual_fixture(self, vf):
        self.vf = vf

    def set_interface(self, interface):
        self._interface = interface

    def update_command(self, twist):
        angular = np.asarray(twist.get_angular(), dtype=float).reshape(3)
        linear = self._twist_linear_scale * np.asarray(twist.get_linear(), dtype=float).reshape(3)
        if self._z_compliant:
            linear[2] = 0.0
            lin_xy = np.linalg.norm(linear[:2])
            if lin_xy > self._max_slide_mps:
                linear[:2] *= self._max_slide_mps / lin_xy
            ang = np.linalg.norm(angular)
            if ang > self._max_slide_radps:
                angular = angular * (self._max_slide_radps / ang)
            self._slide_twist = np.array(
                [angular[0], angular[1], angular[2], linear[0], linear[1], 0.0])
            self._input_active = True
            return

        task_space = self.get_task_space()
        joint_indices = np.asarray(task_space.get_joint_indices())
        position = self._interface.get_state().get_position()[joint_indices]
        ee = task_space.compute_ee_motor(position)

        increment = gafropy.Motor.exp(
            angular[0], angular[1], angular[2], linear[0], linear[1], linear[2])
        reference = ee.multiply(increment)

        if self.vf is not None:
            reference = self.vf.project(reference)

        self.get_objective().set_target(reference)
        self._input_active = True


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
    force (`state.get_wrench().to_array()[3]` = fz; Wrench blade order is not the same
    as Twist/AdmittanceConfig) against `target_force`. The PID writes
    `set_desired_wrench` at index 5 (Twist linear Z), matching the zeroed
    stiffness axis.

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
        # Wrench.to_array() blade order != Twist/AdmittanceConfig:
        #   Wrench: [fx, fy, tz, fz, ty, tx]  → fz at index 3
        #   Twist:  [tx, ty, tz, fx, fy, fz]  → fz at index 5 (set_desired_wrench)
        wrench = self._interface.get_state().get_wrench().to_array()
        force_along_z = wrench[3]
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
    """Hybrid contact: line spring on XY/rotation, DualLoop-style force PID on tool Z.

    SpaceMouse slides the target line (Z stick ignored). The C++ VelocityAdmittance
    plant tracks that line with Kz=0. A host thread owns press: APPROACH is a constant
    tool-Z feed along CLI ``force_sign``; CONTACT is DualLoop ``Fe = Fd − Fs`` with
    ``Fs = force_sign · fz`` on the same-tick wrench and ``Fext`` left at 0
    (mapped through Z damping: ``set_desired_wrench[5] = v_z * z_damping``).
    CONTACT engages when compression ``Fs ≥ max(0.4 N, 0.25·|Fd|)`` with
    ``Fs = −force_sign · fz`` (UR ``ActualTCPForce`` is the reaction, so a +Z
    press reads ``Fz < 0``). Unload needs ``Fs < 0.4 N`` for ``lost_s``.
    ``force_sign`` is never latched from Fz polarity. Over target, press velocity
    is 0 or retract — speedJ must not keep inching into a rigid stop (C153).
    """

    def __init__(self, vf, config, target_force=5.0, kp_force=0.001, ki_force=0.0, kd_force=0.0,
                 force_feedforward=0.0, force_limit=30.0, integral_limit=20.0,
                 update_period=0.005, linear_scale=1.0, force_sign=1.0, force_ramp_s=0.0,
                 log_period_s=0.5, measured_fz_fn=None, dead_wrench_abort_s=0.4,
                 dead_wrench_eps_n=0.05, contact_engage_n=None, contact_release_n=0.4,
                 fz_tau_s=0.03, v_approach=0.005, max_v_hold=0.005, lost_s=1.0,
                 z_damping=200.0, v_slew_mps2=0.05, contact_sign_n=None, sit_s=2.0):
        super().__init__(config)
        self.vf = vf
        self._target_force = abs(float(target_force))
        self._kp_force = float(kp_force)
        self._ki_force = float(ki_force)
        self._kd_force = float(kd_force)
        self._force_feedforward = force_feedforward
        self._force_limit = force_limit
        self._integral_limit = integral_limit
        self._update_period = update_period
        self._linear_scale = linear_scale
        self._force_sign = float(force_sign)
        # DualLoop: |Fs| ≥ max(0.4 N, 0.25·|Fd|) on the same-tick tared wrench.
        self._contact_engage_n = max(0.4, 0.25 * self._target_force)
        if contact_engage_n is not None:
            self._contact_engage_n = max(0.0, float(contact_engage_n))
        if contact_sign_n is not None:
            self._contact_engage_n = max(self._contact_engage_n, float(contact_sign_n))
        self._contact_release_n = max(0.0, float(contact_release_n))
        self._sit_s = max(0.0, float(sit_s))
        self._fz_tau_s = max(0.0, float(fz_tau_s))
        self._v_approach = abs(float(v_approach))
        self._max_v_hold = abs(float(max_v_hold))
        self._lost_s = max(0.0, float(lost_s))
        self._z_damping = float(z_damping)
        self._v_slew_mps2 = max(0.0, float(v_slew_mps2))
        self._force_ramp_s = max(0.0, float(force_ramp_s))
        self._log_period_s = max(0.0, float(log_period_s))
        self._measured_fz_fn = measured_fz_fn
        self._dead_wrench_abort_s = max(0.0, float(dead_wrench_abort_s))
        self._dead_wrench_eps_n = max(0.0, float(dead_wrench_eps_n))

        self._interface = None
        self._integral = 0.0
        self._prev_error = None
        self._stop = threading.Event()
        self._thread = None
        self._t_start = None
        self._t_last_log = None
        self._force_aborted = False
        self._force_feedback_seen = False
        self._press_sign = 1.0 if self._force_sign >= 0.0 else -1.0
        self._fz_filt = None
        self._z_mode = "approach"
        self._unload_since = None
        self._contact_since = None
        self._prev_v_z = None
        self._external_wrench = np.zeros(6)
        self._on_force_abort = None
        self._on_vz = None

    def set_interface(self, interface):
        self._interface = interface

    def set_measured_fz_fn(self, measured_fz_fn):
        """Callable () -> float tool-frame fz in Newtons, or None to use get_wrench()[3]."""
        self._measured_fz_fn = measured_fz_fn

    def set_on_force_abort(self, callback):
        """Optional callback() invoked once if force feedback looks dead (open-loop)."""
        self._on_force_abort = callback

    def set_on_vz(self, callback):
        """Optional callback(v_z) each force tick. Overlay press on the pose-spring plant."""
        self._on_vz = callback

    def _fs(self, fz_raw):
        # UR ActualTCPForce is environment-on-TCP (reaction). Press along +Z reads Fz < 0.
        # DualLoop wants Fs > 0 in compression: Fs = −force_sign · fz.
        return -self._press_sign * float(fz_raw)

    def _read_fz_raw(self):
        if self._measured_fz_fn is not None:
            return float(self._measured_fz_fn())
        wrench = np.asarray(
            self._interface.get_state().get_wrench().to_array(), dtype=float).reshape(6)
        return float(wrench[3])

    def _press_velocity(self, fs, now):
        """Positive = press into the part (tool Z = press_sign * this)."""
        if self._z_mode == "approach":
            if fs >= self._contact_engage_n:
                self._z_mode = "contact"
                self._unload_since = None
                self._contact_since = now
                self._integral = 0.0
                self._prev_error = None
                # Kill the approach feed immediately. Slew from +5 mm/s would keep ramming.
                self._prev_v_z = 0.0
            else:
                return self._v_approach

        if abs(fs) < self._contact_release_n:
            if self._unload_since is None:
                self._unload_since = now
            elif self._lost_s > 0.0 and (now - self._unload_since) >= self._lost_s:
                self._z_mode = "approach"
                self._integral = 0.0
                self._prev_error = None
                self._unload_since = None
                self._contact_since = None
                return self._v_approach
        else:
            self._unload_since = None

        if self._z_mode == "approach":
            return self._v_approach

        fe = self._target_force - fs
        if fs > 0.0:
            self._integral = np.clip(
                self._integral + fe * self._update_period,
                -self._integral_limit, self._integral_limit)
        else:
            self._integral = 0.0
        deriv = 0.0 if self._prev_error is None else \
            (fe - self._prev_error) / self._update_period
        self._prev_error = fe
        v_pid = (self._kp_force * fe + self._ki_force * self._integral +
                 self._kd_force * deriv + self._force_feedforward)
        v_pid = float(np.clip(v_pid, -self._max_v_hold, self._max_v_hold))
        sitting = (
            self._sit_s > 0.0
            and self._contact_since is not None
            and (now - self._contact_since) < self._sit_s
        )
        if sitting and fs >= 0.5 * self._target_force:
            # DualLoop path waits ~2 s at real load. speedJ cannot yield, so do not press.
            return min(0.0, v_pid)
        if fs >= self._target_force:
            # Over target: hold or retract. Never keep inching into a rigid stop (C153).
            return min(0.0, v_pid)
        if fs < 0.0:
            # FT sign flip / not compressed: DualLoop still presses, but slowly on UR.
            return v_pid
        if fs < 0.5 * self._target_force:
            # Light contact: keep the approach feed so we actually reach Fd.
            return self._v_approach
        return v_pid

    def _slew_vz(self, v_z):
        if self._v_slew_mps2 <= 0.0 or self._prev_v_z is None:
            self._prev_v_z = v_z
            return v_z
        max_dv = self._v_slew_mps2 * self._update_period
        v_z = float(np.clip(v_z, self._prev_v_z - max_dv, self._prev_v_z + max_dv))
        self._prev_v_z = v_z
        return v_z

    def _compute_force_command(self):
        fz_raw = self._read_fz_raw()
        fs = self._fs(fz_raw)
        now = time.monotonic()
        elapsed = 0.0 if self._t_start is None else now - self._t_start

        if abs(fz_raw) > self._dead_wrench_eps_n:
            self._force_feedback_seen = True
        if (
            not self._force_aborted
            and not self._force_feedback_seen
            and self._dead_wrench_abort_s > 0.0
            and elapsed >= self._dead_wrench_abort_s
            and self._target_force > self._dead_wrench_eps_n
        ):
            self._force_aborted = True
            src = "rtde" if self._measured_fz_fn is not None else "get_wrench"
            reason = (
                "RTDE ActualTCPForce stayed near zero; refusing open-loop press "
                "(this is what caused C153A1)."
                if self._measured_fz_fn is not None else
                "gafropy URRobot does not populate get_wrench() from actual_TCP_force; "
                "refusing open-loop press (this is what caused C153A1)."
            )
            print(
                "\033[91mERROR: contact force feedback is dead "
                f"({src} fz≈{fz_raw:+.3f}N for {elapsed:.2f}s). {reason}\033[0m",
                flush=True,
            )
            if self._on_force_abort is not None:
                try:
                    self._on_force_abort()
                except Exception as exc:
                    print(f"on_force_abort failed: {exc}", flush=True)

        self._external_wrench = np.zeros(6)
        if self._force_aborted:
            self._prev_v_z = 0.0
            if self._on_vz is not None:
                try:
                    self._on_vz(0.0)
                except Exception as exc:
                    print(f"on_vz failed: {exc}", flush=True)
            return np.zeros(6)

        v_press = self._press_velocity(fs, now)
        v_z = self._slew_vz(self._press_sign * v_press)

        if self._log_period_s > 0.0:
            if self._t_last_log is None or (now - self._t_last_log) >= self._log_period_s:
                self._t_last_log = now
                src = "rtde" if self._measured_fz_fn is not None else "get_wrench"
                sit = (
                    self._z_mode == "contact"
                    and self._sit_s > 0.0
                    and self._contact_since is not None
                    and (now - self._contact_since) < self._sit_s
                )
                print(
                    f"contact force[{src}]: mode={self._z_mode}  fz_raw={fz_raw:+.2f}N  "
                    f"Fs={fs:+.2f}N  Fd={self._target_force:.1f}N  "
                    f"v_z={v_z:+.4f}m/s  Fext=0  sign={self._press_sign:+.0f}"
                    f"{'  sit' if sit else ''}",
                    flush=True,
                )

        wrench_cmd = np.zeros(6)
        wrench_cmd[5] = v_z * self._z_damping
        if self._on_vz is not None:
            try:
                self._on_vz(v_z)
            except Exception as exc:
                print(f"on_vz failed: {exc}", flush=True)
        return wrench_cmd

    def _run(self):
        while not self._stop.is_set():
            desired = self._compute_force_command()
            if self._on_vz is None:
                self.set_external_wrench(np.zeros(6))
                self.set_desired_wrench(desired)
            time.sleep(self._update_period)

    def start(self):
        self._integral = 0.0
        self._prev_error = None
        self._force_aborted = False
        self._force_feedback_seen = False
        self._press_sign = 1.0 if self._force_sign >= 0.0 else -1.0
        self._fz_filt = None
        self._z_mode = "approach"
        self._unload_since = None
        self._contact_since = None
        self._prev_v_z = None
        self._external_wrench = np.zeros(6)
        self._t_start = time.monotonic()
        self._t_last_log = None
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
