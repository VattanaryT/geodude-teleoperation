import csv
import pathlib
import threading
import time

import gafropy
import numpy as np

from geodude_teleoperation.in_contact_control.force_pid import (
    ForcePIDConfig,
    ForcePIDController,
)
from geodude_teleoperation.tcp_force import is_safe_to_command, safety_mode_name


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
                 max_qddot=5.0, stick_hz=50.0, free_along_normal=True,
                 dls_damping=0.05, per_joint_velocity=True,
                 err_clip_lin=0.01, err_clip_rot=0.2,
                 rearm_wrench_s=0.15):
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
        self._stick_hz = float(stick_hz)
        self._distance = 0.0

        self._interface = None
        self.vf = None
        # When False (hands-off SpaceMouse), stream zero joint velocity — otherwise
        # residual pose error / wrong-arm bias keeps speedJ moving with no stick input.
        self._input_active = False
        self._tool_z_velocity = 0.0
        self._stiffness = np.array(self._stiffness, dtype=float, copy=True)
        self._kz_pose = float(self._stiffness[5, 5])
        self._kang_pose = np.diag(self._stiffness)[:3].copy()
        self._z_compliant = False
        self._ori_compliant = False
        # Body-frame surface normal for the free-along-normal projection.
        # set_z_compliant() only zeroes stiffness[5,5], which is the body Z of
        # the *pose error* -- not the surface normal. field_follow slerps tool Z
        # behind the field normal on purpose, so mid-slide the two differ and
        # the K on the remaining axes gets a component along the true normal,
        # which outruns the press loop and lifts the tool off. Project it out.
        self._free_along_normal = bool(free_along_normal)
        # Wrench law only: whether the press has reached the surface. Until it
        # has, the pose spring keeps the normal so it can do the approaching.
        # Default True so nothing outside the contact hand-off changes, and the
        # velocity path never reads it -- speedJ stays bit-identical.
        self._press_engaged = True
        # Damped-least-squares regularizer for the jacobian inverse. This was a
        # hardcoded 1e-6 (lambda = 1e-3), inherited from ufactory_demos via the
        # root commit and never validated here, while the sibling
        # TeleoperationController uses damping=0.05 -- 2500x more, and
        # configurable. DLS gain on a singular value is sigma/(sigma^2+lambda^2),
        # peaking at 1/(2*lambda): 500 at lambda=1e-3 against 10 at 0.05. Near a
        # shoulder singularity that turns a few mm/s of task command into rad/s
        # of joint command, the norm clamp pins it, and the arm windmills one
        # joint for almost no TCP motion until the UR protective-stops on
        # position deviation (C153A1 "position deviates from path", shoulder).
        # Well-conditioned poses barely notice: at sigma = 0.2 the gain goes
        # 5.00 -> 4.71, a 6% change inside the pose spring's own time constant.
        self._dls_damping = float(dls_damping)
        # The magnitude clamp is on norm(qdot), so one joint could carry the
        # entire budget. Also clamp per joint.
        self._per_joint_velocity = bool(per_joint_velocity)
        self._last_sigma_min = float("nan")
        self._last_clamp_scale = 1.0
        self._last_qdot_max = 0.0
        # Linear pose error per axis, so a log can answer which direction is
        # prone to the fault instead of leaving it to the operator's memory.
        self._last_err_lin = np.zeros(3)
        # ...and the angular half. Nothing recorded it, so "the tool rotates
        # instead of sliding" was an operator observation with no number behind
        # it. In contact the stick only ever writes xyz; every bit of commanded
        # rotation comes from the field, so this is the channel to measure when
        # the motion does not match the input.
        self._last_err_ang = np.zeros(3)
        self._last_vtask_lin = 0.0
        self._n_joints_seen = None
        self._t_cond_warn = None
        self._n_body = None
        # Commanded EE orientation for the idle hold. None = the plain live-pose
        # snap, which erases the angular error along with the linear one.
        self._hold_rotation = None
        self._slide_twist = np.zeros(6)
        self._max_slide_mps = 0.008
        self._max_slide_radps = 0.12
        self._slide_accel_mps2 = 0.04
        self._slide_slew_t = None
        self._command_trace = None
        self._stick_linear = np.zeros(3)
        # Latest SpaceMouse Twist blade order [angular(3), linear(3)], for the
        # dtc per-tick CSV. Independent of whether field-follow drops angular.
        self._raw_twist = np.zeros(6)
        self._max_qddot = float(max_qddot)
        self._qdot_cmd = None
        self._qdot_des = None
        self._qdot_cmd_t = None
        # --- wrench law (compute_control_wrench / tau = J.T @ W) ---
        # Bound on the pose error *before* it becomes force -- this is the
        # wrench law's shove cap, and it replaces field-follow's max_step_m,
        # which bounded the speedJ shove by capping the target step instead.
        # Without it a large error (a re-target, a stall, clearing a protective
        # stop against a stale target) commands a large force.
        #
        # It must be sized against K, not copied: the cap in newtons is
        # K_lin * err_clip_lin. At K_lin=3000 N/m, 0.01 m is 30 N -- above the
        # ~15 N a slide needs (5 mm of field-follow lead x 3000) so ordinary
        # sliding stays proportional, and still backstopped by tau_max. The
        # reference UR OSC loop uses 0.05 m, but with K=1000 N/m and a
        # different task; at our K that would be 150 N.
        # Angular: K_ang=50 N.m/rad x 0.2 rad = 10 N.m, at the wrist tau_max.
        self._err_clip_lin = float(err_clip_lin)
        self._err_clip_rot = float(err_clip_rot)
        # Press force in the *body* frame, as a 3-vector. Deliberately not a
        # scalar on blade 5: free_along_normal has already removed the field
        # normal from the spring's linear block, and tool Z is not that normal
        # (field-follow slerps tool Z behind it on purpose), so writing blade 5
        # would mix axes. The caller owns the sign -- see
        # ContactTeleoperationController._press_axis_base, where the
        # --force-sign hemisphere convention lives.
        self._press_wrench_body = np.zeros(3)
        self._last_wrench_lin = 0.0
        self._last_wrench_ang = 0.0
        # Re-arm ramp for the wrench law. ``rearm_slerp_s`` ramps the commanded
        # *rotation*; nothing ramped the wrench itself, so the ``input_active``
        # 0->1 edge was a one-tick step in both blades. Measured on
        # dtc_20260830_115039.csv (9 edges): |W_lin| 9.25 -> 38.30 N and
        # |W_ang| 1.09 -> 10.35 N.m between two 8 ms ticks, because while quiet
        # the idle hold keeps the target on the live pose (err_lin ~ 0) and the
        # first active tick hands back the full field-follow carrot.
        #
        # Deliberately an *edge* ramp, not a slew-rate limit on W. The two are
        # not separable by rate on this rig: ordinary sliding already runs
        # |dW_lin|/dt of 1005 N/s at p95 and 3268 N/s at p99.9, against 613 to
        # 3641 N/s at the edges. A rate limit tight enough to catch the edges
        # would fire constantly and add phase lag to a loop whose stability
        # budget is already spent on the delayed damper (see resolve_wrench_zeta
        # and the 24.5 Hz limit cycle). This blend is armed only by the edge,
        # so outside it the law is byte-identical to before.
        #
        # The blend starts from the wrench that was already holding the arm, so
        # the hold is never dropped -- and because the press is added before the
        # blend and is unchanged across the edge, it passes through untouched:
        # ``W0 + a*(W - W0)`` leaves any component common to both alone.
        self._rearm_wrench_s = max(0.0, float(rearm_wrench_s))
        self._rearm_t0 = None
        self._rearm_w0 = None
        self._prev_input_active = False
        self._last_wrench = np.zeros(6)

    def set_command_trace(self, fn):
        """Optional callback(qdot, measured=, spring=, stick=) after each speedJ command."""
        self._command_trace = fn

    def set_stick_linear(self, linear):
        """Latest SpaceMouse linear (m) for the command plot. Does not drive speedJ."""
        self._stick_linear = np.asarray(linear, dtype=float).reshape(3).copy()

    @property
    def stick_linear(self):
        """Read side of ``set_stick_linear``, for the direct-torque plot trace.

        The speedJ trace reads ``_stick_linear`` from inside
        ``compute_control_command``; the wrench law is traced from the backend,
        which is outside this class and needs a public accessor."""
        return np.array(self._stick_linear, copy=True)

    def set_raw_twist(self, twist):
        """Latest SpaceMouse Twist [ang(3), lin(3)]. Log only; does not drive."""
        self._raw_twist = np.asarray(twist, dtype=float).reshape(6).copy()

    @property
    def raw_twist(self):
        """Read side of ``set_raw_twist``, for the dtc per-tick CSV."""
        return np.array(self._raw_twist, copy=True)

    def set_input_active(self, active: bool) -> None:
        self._input_active = bool(active)

    def contact_slide_twist(self):
        return np.array(self._slide_twist, copy=True)

    def slew_slide(self, dt=None, max_dt=0.004):
        """Ramp contact XY slide to 0. Idle must not dump twist in one tick."""
        now = time.perf_counter()
        if dt is None:
            dt = self._dt if self._slide_slew_t is None else now - self._slide_slew_t
        self._slide_slew_t = now
        if self._input_active:
            return
        dt = min(max(float(dt), 0.0), float(max_dt))
        max_step = float(self._slide_accel_mps2) * dt
        if max_step <= 0.0:
            return
        lin = self._slide_twist[3:5]
        n = float(np.linalg.norm(lin))
        if n <= max_step:
            self._slide_twist = np.zeros(6)
            return
        self._slide_twist[3:5] = lin * ((n - max_step) / n)

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
        self.hold_pose_at(position)

    def hold_pose_at(self, q_control_group) -> None:
        """``hold_live_pose`` for a caller that already has the control-group q.

        Same snap, no state read. This exists because *how often* the snap runs
        is a control gain, not bookkeeping: the target becomes ``pose(t - T)``,
        so the spring contributes ``K * T/2`` of delay-based damping on top of
        the explicit ``D``. Called from run_field_follow at --follow-hz 20 that
        is 3000 * 0.025 = 75 N.s/m of *uncounted* damping -- 70% again on top of
        the 109.5 the damping ratio asks for, and delayed by 50 ms rather than
        one control period, which is the half that destabilizes. The dtc loop
        calls this every tick instead so the term stays small and current.
        """
        ee = self.get_task_space().compute_ee_motor(
            np.asarray(q_control_group, dtype=float))
        R_hold = self._hold_rotation
        if R_hold is not None:
            # Hold the *commanded field orientation* with the live position.
            #
            # The plain snap sets target := live pose, which erases the angular
            # error as well as the linear one. Measured on
            # dtc_20260829_214019.csv: with this running at 125 Hz whenever the
            # stick was quiet, |W| was exactly zero on 42% of ticks and the
            # wrists moved 0.06 deg over the whole run while j0..j2 moved 8 deg.
            # The loop could not converge onto the field -- quiet erased the
            # error, the next stick sample handed back the full field pose in
            # one tick, and the wrist tau cap threw the correction away. Only
            # the position is re-based here; the orientation stays commanded, so
            # the wrist keeps working toward the field between stick motions and
            # there is no step to make when the stick moves again.
            #
            # set_hold_rotation publishes field-follow's *commanded* R (already
            # ramped by rearm_slerp_s), not the raw field goal, so this and the
            # follow target are the same rotation by construction.
            T = np.asarray(ee.to_transformation_matrix(), dtype=float).copy()
            T[:3, :3] = R_hold
            ee = gafropy.Motor.from_matrix(T)
        self.get_objective().set_target(ee)

    def set_hold_rotation(self, R_base):
        """Commanded EE orientation (base frame) for the idle hold, or None.

        Published from the field-follow tick, same call site and same frame as
        ``set_contact_normal`` -- ``R_base`` is the mapped commanded EE rotation
        in the *controller* world, so it composes with ``compute_ee_motor``
        without a further transform. ``None`` restores the plain live-pose snap.
        """
        if R_base is None:
            self._hold_rotation = None
            return
        R = np.asarray(R_base, dtype=float).reshape(3, 3)
        self._hold_rotation = None if not np.all(np.isfinite(R)) else R.copy()

    @property
    def input_active(self) -> bool:
        """Whether a command source is driving the target (stick or follow)."""
        return bool(self._input_active)

    def set_tool_z_velocity(self, vz):
        """Overlay tool-Z speed (m/s) on top of the pose spring. DualLoop press axis."""
        self._tool_z_velocity = float(vz)

    def set_press_engaged(self, engaged):
        """Has the press reached the surface? Wrench law only.

        ``False`` keeps the pose spring on the contact normal so it can do the
        approaching -- the press is force-limited and the gap is friction-
        limited, and the spring has ~2x the authority. ``True`` frees the normal
        so the force loop owns it alone, which is the invariant everything else
        here depends on. Read by ``compute_control_wrench``;
        ``compute_control_command`` never looks at it.
        """
        self._press_engaged = bool(engaged)


    def set_press_wrench(self, w_body):
        """Press force (N) in the *body* frame, for the wrench law.

        The counterpart of ``set_tool_z_velocity`` on the ``tau = J.T @ W``
        path, and a 3-vector rather than a scalar for a reason: the press is
        along the *contact normal*, which is not tool Z (field-follow leads tool
        Z behind the field normal by design), and ``free_along_normal`` has
        already emptied that axis of spring. Pass the already-signed axis times
        the commanded magnitude -- ``ContactTeleoperationController`` owns the
        ``--force-sign`` hemisphere convention and is the only place that should
        resolve it. ``None`` or zeros means no press.
        """
        if w_body is None:
            self._press_wrench_body = np.zeros(3)
            return
        self._press_wrench_body = np.asarray(
            w_body, dtype=float).reshape(3).copy()

    def set_z_compliant(self, compliant):
        """Zero pose-spring Kz so a Z overlay is not fought. Stick Z is ignored."""
        self._z_compliant = bool(compliant)
        self._stiffness[5, 5] = 0.0 if self._z_compliant else self._kz_pose
        if not self._z_compliant:
            self._slide_twist = np.zeros(6)
            self._n_body = None

    def set_orientation_compliant(self, compliant):
        """Zero angular K so the pose spring tracks xyz only (field follow)."""
        self._ori_compliant = bool(compliant)
        for i in range(3):
            self._stiffness[i, i] = (
                0.0 if self._ori_compliant else float(self._kang_pose[i]))

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

        jtj = jacobian.T @ jacobian
        j_inverse = np.linalg.inv(
            jtj + self._dls_damping ** 2 * np.eye(jacobian.shape[1]))
        self._note_conditioning(jacobian)

        # Same plant as hardware_free_teleop. Kz=0 in contact; DualLoop overlays vz
        # even when the stick is idle. Do not replace this with J# * slide.
        if not self._input_active:
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

            v_task = self._dt * (task_acceleration - bias)
            # Blade order is [angular, linear], so 3:6 is the linear block.
            err_flat = np.asarray(error, dtype=float).reshape(-1)
            self._last_err_lin = err_flat[3:6].copy()
            self._last_err_ang = err_flat[0:3].copy()
            self._last_vtask_lin = float(np.linalg.norm(v_task[3:6]))
            if (self._z_compliant and self._free_along_normal
                    and self._n_body is not None):
                # Align never owns the unbound axis -- force does. Same
                # invariant as in_contact_control's alignment_admittance
                # (`v_lin -= (v_lin.n) n`, free_along_normal=True). Twist blade
                # order puts linear in 3:6, matching `twist[5] = vz` below.
                n = self._n_body
                v_task[3:6] = v_task[3:6] - float(v_task[3:6] @ n) * n
            qdot = j_inverse @ jacobian.T @ v_task

        if abs(vz) > 1e-12:
            twist = np.zeros(6)
            twist[5] = vz
            qdot = qdot + j_inverse @ jacobian.T @ twist

        speed = np.linalg.norm(qdot)
        scale = 1.0
        if speed > self._max_joint_velocity:
            scale = self._max_joint_velocity / speed
            qdot = qdot * scale
        if self._per_joint_velocity:
            # norm(qdot) alone lets one joint carry the whole budget, which is
            # exactly what a near-singular pose produces: a single shoulder
            # joint at the full limit delivering almost no TCP motion. Note the
            # vz overlay is added *above* this, so the press shares the budget
            # and is scaled by `scale` too -- the contact log prints the
            # pre-clamp v_z, which is why the force loop can read healthy while
            # the joint path is saturated.
            qdot = np.clip(qdot, -self._max_joint_velocity, self._max_joint_velocity)
        self._last_clamp_scale = float(scale)
        self._last_qdot_max = float(np.max(np.abs(qdot))) if np.size(qdot) else 0.0

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

    def _control_group_qdot(self, robot_state, n_joints):
        """The measured joint velocity for this task space, from the *control-group*
        slice. See compute_control_command: get_joint_indices() are system-global
        and must not index this vector."""
        velocity = np.asarray(robot_state.get_velocity(), dtype=float).reshape(-1)
        if velocity.size == n_joints:
            return velocity
        if velocity.size > n_joints:
            return velocity[-n_joints:]
        raise RuntimeError(
            f"robot_state velocity length {velocity.size} < task joints {n_joints}")

    def compute_control_wrench(self, robot_state, objective_state,
                               task_space_state, inertia=None):
        """Task-space wrench for the direct-torque law: ``tau = J.T @ W``.

        The velocity sibling above computes this *same spring* and then spends
        it: ``/inertia``, ``*dt``, DLS inverse, norm and per-joint clamps, LPF,
        slew -- all of which exist to keep a position servo from over-shoving.
        Under direct torque they are exactly what starves the arm: a 4 mm error
        becomes ~8 mm/s becomes ~0.02 rad/s becomes ~1 N.m of ``kv * qdot_err``
        against a UR5e breakaway of several N.m. Here ``K`` is a real stiffness
        (N/m on the linear blades, N.m/rad on the angular ones) and the wrench
        is the commanded force, so lateral authority is set on purpose rather
        than arrived at through five conversions.

        **Reduction to the working case.** Both laws read the same
        ``objective_state.get_error()`` and the same
        ``task_space_state.get_jacobian()``, and the velocity law's own output
        is ``(J.T J + lambda^2 I)^-1 J.T * (dt/M) * wrench_net``. The frames and
        signs are therefore shared by construction -- this method stops at
        ``wrench_net`` instead of inverting it, and introduces no new
        convention.

        Two deliberate differences from ``compute_control_command``'s
        ``wrench_net``, neither of which may leak back into it (speedJ must stay
        bit-identical):

        - **Damping on the measured EE velocity ``J @ qdot``**, not on
          ``get_error_derivative()``. Both are ``d(error)/dt`` and carry the
          same sign, but the latter also contains *target* motion, so it fights
          the SpaceMouse; a pure damper does not kick when the target moves.
          Same choice as the reference UR OSC loop.
        - **The pose error is clipped before it becomes force.** ``max_step_m``
          bounded the shove on the speedJ path by capping the target step;
          nothing bounds it here.
        """
        jacobian = np.asarray(task_space_state.get_jacobian())
        n_joints = int(jacobian.shape[1])
        qdot_meas = self._control_group_qdot(robot_state, n_joints)
        self._note_conditioning(jacobian)

        # Blade order is [angular, linear]: 0:3 angular, 3:6 linear.
        error = np.asarray(
            objective_state.get_error(), dtype=float).reshape(-1).copy()
        self._last_err_ang = error[0:3].copy()
        self._last_err_lin = error[3:6].copy()
        # Note there is no `_input_active` gate here. Under speedJ, hands-off
        # streams zero qdot or residual pose error keeps the arm creeping.
        # Under torque the spring *is* the hold, and zeroing it would let the
        # arm sag -- hold_live_pose keeps the target on the live pose, so the
        # error is already near zero and there is nothing to gate.
        error[0:3] = np.clip(error[0:3], -self._err_clip_rot, self._err_clip_rot)
        error[3:6] = np.clip(error[3:6], -self._err_clip_lin, self._err_clip_lin)

        ee_velocity = jacobian @ qdot_meas
        # With ``inertia=None`` this IS the wrench and the law is unchanged --
        # that is the reduction to the previously-working case, and it is exact,
        # because Lambda = I is the only difference between the two branches.
        #
        # With Lambda given, the spring output is an *acceleration* command and
        # Lambda = (J M^-1 J.T)^-1 converts it to the force that actually
        # produces that acceleration at the EE. Without it, tau = J.T @ W hands
        # the same wrench to joints whose mobility spans 78x on this arm
        # (examples/analyze_dtc_mobility.py), so the wrist takes the motion and
        # the shoulder takes the force. Lambda is what makes the EE behave like
        # a unit mass in every direction, which is also what makes one gain
        # correct at more than one pose.
        #
        # Units change with it: K is N/m without Lambda and 1/s^2 with it, so
        # the caller owns re-deriving the gains (omega_n = sqrt(K)). Nothing
        # here can detect a stale N/m gain, hence --dtc-lambda is opt-in.
        wrench = -self._stiffness @ error - self._damping @ ee_velocity
        if inertia is not None:
            wrench = np.asarray(inertia, dtype=float).reshape(6, 6) @ wrench

        if (self._z_compliant and self._free_along_normal
                and self._press_engaged and self._n_body is not None):
            # Align never owns the unbound axis -- force does. Identical to the
            # projection the velocity law applies to v_task[3:6]; the wrench is
            # the same vector one conversion earlier.
            #
            # ...but only once the press is actually on the surface. Freeing the
            # normal the instant contact mode is entered leaves the *press*
            # alone to cross the gap, and the press is force-limited while the
            # gap is friction-limited: dtc_20260903_150932.csv commanded a
            # steady |W_lin| of 23.0 N along the normal for 43 s with |qdot|
            # median 0.0000 rad/s and Fs never above 0.73 N. The pose spring is
            # the thing with enough authority to approach (Lambda*K*max_lead,
            # ~46 N here) and it is what the operator is steering, so it keeps
            # the normal until Fs says the tool has arrived. This is the
            # standard hybrid hand-off -- position control to the surface, force
            # control on it -- and it is what the DualLoop velocity seek was
            # doing implicitly.
            #
            # This runs *after* Lambda deliberately: Lambda is not diagonal, so
            # projecting the acceleration command first would let the coupling
            # write a normal component straight back into the force. Project
            # what is actually commanded, then add the press -- the other order
            # deletes the press.
            n = self._n_body
            lin = wrench[3:6]
            wrench[3:6] = lin - float(lin @ n) * n

        wrench[3:6] = wrench[3:6] + self._press_wrench_body

        wrench = self._apply_rearm_ramp(wrench)

        self._last_wrench = wrench.copy()
        self._last_wrench_lin = float(np.linalg.norm(wrench[3:6]))
        self._last_wrench_ang = float(np.linalg.norm(wrench[0:3]))
        return wrench

    def _apply_rearm_ramp(self, wrench):
        """Blend the spring's first active tick out of the idle hold wrench.

        Wrench law only, and armed only by the ``input_active`` 0->1 edge --
        see the note in ``__init__``. ``rearm_wrench_s = 0`` disables it, which
        is the reduction to the previously-working case and is exact: with the
        ramp inactive this returns its argument unchanged.
        """
        active = bool(self._input_active)
        if not active:
            self._prev_input_active = False
            self._rearm_t0 = None
            self._rearm_w0 = None
            return wrench
        if not self._prev_input_active:
            self._prev_input_active = True
            if self._rearm_wrench_s > 0.0:
                self._rearm_t0 = time.perf_counter()
                # The wrench that was holding the arm one tick ago, not zero:
                # ramping from zero would drop the hold (and the press with it)
                # at exactly the moment the operator asked the arm to move.
                self._rearm_w0 = self._last_wrench.copy()
        if self._rearm_t0 is None or self._rearm_w0 is None:
            return wrench
        alpha = (time.perf_counter() - self._rearm_t0) / self._rearm_wrench_s
        if not np.isfinite(alpha) or alpha >= 1.0:
            self._rearm_t0 = None
            self._rearm_w0 = None
            return wrench
        alpha = max(0.0, float(alpha))
        return self._rearm_w0 + alpha * (wrench - self._rearm_w0)

    def reset_wrench_rearm(self) -> None:
        """Forget the re-arm blend and the held wrench.

        Paired with every ``law.reset()`` in ``DirectTorqueBackend``: a pendant
        stop must not queue up a ramp that opens from a pre-stop wrench when
        someone clears it. Same rule as the idle latch and the joint hold.
        """
        self._rearm_t0 = None
        self._rearm_w0 = None
        self._prev_input_active = False
        self._last_wrench = np.zeros(6)

    def _note_conditioning(self, jacobian):
        """Record how close this pose is to singular, and shout once if it is.

        sigma_min is the smallest singular value of the jacobian, and it decides
        whether the DLS inverse behaves or amplifies. Nothing recorded it
        before, so a C153 protective stop left no evidence of the configuration
        that produced it.

        Use `J @ J.T` (6x6, one eigenvalue per task DOF), not the `J.T @ J`
        (7x7) the inverse is built from: with 7 joints and 6 task DOF the
        latter is rank-deficient by construction, so its smallest eigenvalue is
        always ~0 and would report every pose as singular. The redundancy is a
        nullspace, not ill-conditioning.
        """
        try:
            ev = np.linalg.eigvalsh(jacobian @ jacobian.T)
            self._last_sigma_min = float(np.sqrt(max(0.0, float(ev[0]))))
        except np.linalg.LinAlgError:
            self._last_sigma_min = float("nan")
        if self._n_joints_seen is None:
            self._n_joints_seen = tuple(jacobian.shape)
            # sigma_min still matters under the wrench law -- a near-singular
            # pose makes J.T tiny in some direction, so the same wrench buys
            # almost no torque there. dls_lambda does not: it belongs to the
            # inverse, which only the velocity law takes. It is printed anyway
            # rather than branched on, because the controller does not know
            # which law the backend will call.
            print(f"jacobian: {jacobian.shape[0]}x{jacobian.shape[1]}"
                  f"  sigma_min={self._last_sigma_min:.4f}"
                  f"  dls_lambda={self._dls_damping:g}", flush=True)
        if self._last_sigma_min < 0.02:
            now = time.perf_counter()
            if self._t_cond_warn is None or (now - self._t_cond_warn) > 1.0:
                self._t_cond_warn = now
                print(f"\033[93mnear-singular pose: sigma_min="
                      f"{self._last_sigma_min:.4f} (DLS gain up to "
                      f"{1.0 / (2.0 * self._dls_damping):.0f}x)\033[0m", flush=True)

    def conditioning(self):
        """Last tick's kinematic health, for the contact log."""
        return {
            "sigma_min": self._last_sigma_min,
            "clamp_scale": self._last_clamp_scale,
            "qdot_max": self._last_qdot_max,
            "err_x": float(self._last_err_lin[0]),
            "err_y": float(self._last_err_lin[1]),
            "err_z": float(self._last_err_lin[2]),
            "vtask_lin": self._last_vtask_lin,
            "w_lin": self._last_wrench_lin,
            "w_ang": self._last_wrench_ang,
            "err_rx": float(self._last_err_ang[0]),
            "err_ry": float(self._last_err_ang[1]),
            "err_rz": float(self._last_err_ang[2]),
            # Constant for the run, but per-row: these are the two levers on
            # how the pose spring splits its authority between sliding and
            # turning, and a log that does not carry them cannot be compared
            # against another run's.
            "k_xy": float(self._stiffness[3, 3]),
            "k_ang": float(self._stiffness[0, 0]),
        }

    def set_contact_normal(self, n_base, R_tool_to_base):
        """Surface normal (base frame) to free the pose spring along in contact.

        Stored in the *body* frame, because the jacobian and the objective error
        this controller works in are body-frame. Pass ``None`` to clear, which
        falls back to the pre-existing behavior of relying on stiffness[5,5]=0
        alone. Only has an effect while ``_z_compliant`` is set, so free space is
        untouched.
        """
        if n_base is None or R_tool_to_base is None:
            self._n_body = None
            return
        n = np.asarray(n_base, dtype=float).reshape(3)
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            self._n_body = None
            return
        R = np.asarray(R_tool_to_base, dtype=float).reshape(3, 3)
        n_body = R.T @ (n / norm)
        n_norm = float(np.linalg.norm(n_body))
        self._n_body = None if n_norm < 1e-12 else n_body / n_norm

    @property
    def contact_normal_body(self):
        """Last commanded field normal, in the live body frame (or None).

        Read side of ``set_contact_normal``. The angle between this and body +Z
        is the number that says whether the tool is actually conforming to the
        object field: ``n_base`` is the commanded target's Z column and
        ``R_tool_to_base`` is the *live* EE rotation, so the two differ by
        exactly the orientation the arm has failed to track. Logged per tick by
        ``DirectTorqueBackend`` -- the 1 Hz console ``R_err`` samples it, and a
        re-arm transient lives entirely between two samples.
        """
        return None if self._n_body is None else np.array(self._n_body, copy=True)

    def set_virtual_fixture(self, vf):
        self.vf = vf

    def set_interface(self, interface):
        self._interface = interface

    def update_command(self, twist):
        angular = np.asarray(twist.get_angular(), dtype=float).reshape(3)
        linear = self._twist_linear_scale * np.asarray(twist.get_linear(), dtype=float).reshape(3)
        if self._z_compliant:
            # Same pose increment as hardware_free_teleop. Force owns tool Z;
            # wrist angular is held (loaded yaw was a C153 lever).
            linear[2] = 0.0
            angular[:] = 0.0

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


# Smallest |press axis . tool Z| the wrench-law press will aim with. Below this
# the hemisphere that _press_axis_base picks is decided by numerical noise and
# flips tick to tick. Same threshold as the velocity path's near-perpendicular
# guard, but the answer differs: that one falls back to tool Z (millimetres per
# second, survivable), this one refuses to press at all (newtons, not).
# Fs thresholds, as fractions of Fd, for latching out of the approach regime
# into the hold regime. Hysteresis, because Fs noise is ~0.3 N and chattering
# between a breakaway-sized command and a hold-sized one at 200 Hz would be a
# square wave into the surface.
PRESS_ENGAGE_FRAC = 0.5
PRESS_RELEASE_FRAC = 0.15


PRESS_AXIS_MIN_DOT_TOOL_Z = 0.2


class ContactTeleoperationController(gafropy.control.VelocityAdmittanceController):
    """Hybrid contact: force PID along the surface normal, latched for the session.

    ``start()`` (button 0) is contact until ``stop()`` (button 0 again). There is
    no approach / free-space fallback: unload must not re-arm a 5 mm/s feed.

    **This controller owns Fd, and nothing else in the speedj path does.** It is
    never handed to ``interface.control(...)`` in that backend (see
    ``enter_speedj_contact``) and ``_run`` skips ``set_desired_wrench`` whenever
    ``_on_vz`` is set, so the gafropy VelocityAdmittance plant underneath is
    never stepped -- ``config`` / ``z_damping`` survive only for the legacy
    wrench path and the tests that read ``v_z`` back out of it. Two loops
    authoritative on the same axis is the fight that produces drift, so the
    pose spring must stay freed along the normal (see
    ``AdmittanceTeleoperationController.set_contact_normal``).

    ``press_loop`` selects the seek law:

    ``"dtc"`` (default) delegates to ``ForcePIDController`` ported from
    ``exp/in-contact-dtc`` -- 5 Hz LPF and deadband on the measurement plus
    oscillation detector, spike guard, overshoot retract, soft-approach cap and
    force-rate damping. It needs the full force *vector* and the surface normal,
    so it uses ``measured_force_fn`` and the normal pushed in by
    ``set_contact_normal``.

    ``"legacy"`` is the previous bare PID ``v = clip(kp*(Fd − Fs), ±max_v_hold)``
    on ``fz`` alone, kept verbatim so ``--press-loop legacy`` reproduces the old
    behavior on hardware. Over target, press is 0 or retract -- speedJ must not
    keep inching into a rigid stop (C153).

    ``Fs = −force_sign · fz`` (UR ``ActualTCPForce`` is the reaction, so a +Z
    press reads ``Fz < 0``).
    """

    def __init__(self, vf, config, target_force=5.0, kp_force=0.001, ki_force=0.0, kd_force=0.0,
                 force_feedforward=0.0, force_limit=30.0, integral_limit=20.0,
                 update_period=0.005, linear_scale=1.0, force_sign=1.0, force_ramp_s=0.0,
                 log_period_s=0.5, measured_fz_fn=None, dead_wrench_abort_s=0.4,
                 dead_wrench_eps_n=0.05, contact_engage_n=None, contact_release_n=0.4,
                 fz_tau_s=0.03, v_approach=0.005, max_v_hold=0.005, lost_s=1.0,
                 z_damping=200.0, v_slew_mps2=0.4, contact_sign_n=None, sit_s=2.0,
                 press_loop="dtc", measured_force_fn=None, force_tare=True,
                 force_tare_samples=20, force_tare_period=0.002,
                 wrong_sign_fs_n=-2.0, wrong_sign_fd_frac=0.5,
                 unload_v_m_s=0.004, spike_dF_dt_N_s=None,
                 safety_mode_fn=None,
                 press_kp_n_per_n=0.5, press_max_n=None,
                 press_slew_n_s=200.0, press_kp_hold=0.5):
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
        # --- wrench-law press (set_on_press_wrench). Newtons, not m/s. ---
        # Proportional trim on a feedforward Fd. Under torque the commanded
        # force *is* the applied force, so the steady-state answer is simply Fd
        # and the trim only has to cover sensor bias and the surface moving; a
        # velocity seek had to integrate its way to Fd through a position servo
        # that could not yield, which is what every guard in _press_velocity is
        # defending against. Deliberately no integrator: an I term on a stiction
        # nonlinearity is a stick-slip amplifier, same reason DEFAULT_KI is 0 on
        # the joint-velocity torque law.
        self._press_kp_n_per_n = abs(float(press_kp_n_per_n))
        # Hard ceiling on the commanded press. Defaults to 3x Fd: enough trim
        # authority to cover a few N of bias, small enough that a runaway trim
        # cannot become the 140 N the shoulder tau_max would now allow.
        self._press_max_n = (3.0 * self._target_force if press_max_n is None
                             else abs(float(press_max_n)))
        # N/s. The wrench-law counterpart of v_slew_mps2: a step in commanded
        # force is a step in tau, and the arm is a lightly damped structure.
        self._press_slew_n_s = abs(float(press_slew_n_s))
        self._prev_press_n = None
        # Approach and hold are two regimes, latched with hysteresis on Fs.
        # Approach needs enough command to break static friction; hold needs the
        # opposite, because that same command keeps accelerating the tool into
        # the part once it IS free. Measured on contact_20260903_140311.csv with
        # one gain for both: F_press median 20.6 N got the tool moving and then
        # drove Fs to a p95 of 19.6 and a max of 33.2 against Fd = 5.
        self._press_kp_hold = abs(float(press_kp_hold))
        self._press_engaged = False
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
        # v_slew_mps2 was 0.05, which needs 0.16 s to walk +8 mm/s down to zero
        # and 0.32 s to reach full retract. Against a rigid part that is ~1.3 mm
        # of continued press *after* the loop has already decided to back off,
        # and it is why v_z printed +0.0023 m/s at Fs=21.7 N with Fd=10 N: the
        # command was negative, the slew was still descending. The joint-space
        # ramp is separately bounded by max_qddot, so this only ever needed to
        # be gentle, not slow.
        self._v_slew_mps2 = max(0.0, float(v_slew_mps2))
        # Same wrong-sign rule as the dtc loop's ForcePIDConfig, so both seek
        # laws answer a flipped/bouncing reading the same way.
        self._wrong_sign_fs_n = float(wrong_sign_fs_n)
        self._wrong_sign_fd_frac = float(wrong_sign_fd_frac)
        self._unload_v_m_s = abs(float(unload_v_m_s))
        self._force_ramp_s = max(0.0, float(force_ramp_s))
        self._log_period_s = max(0.0, float(log_period_s))
        self._measured_fz_fn = measured_fz_fn
        self._measured_force_fn = measured_force_fn
        self._dead_wrench_abort_s = max(0.0, float(dead_wrench_abort_s))
        self._dead_wrench_eps_n = max(0.0, float(dead_wrench_eps_n))

        # Free-space force offset, tool frame. UR ActualTCPForce is not gravity
        # compensated unless the payload is configured, and the speedj backend
        # never issues zero_ftsensor() (only the force_mode URScript does), so a
        # perfectly live sensor reads several N of tool weight with nothing
        # touching. Both seek laws subtract this; the dead-feedback abort does
        # not, because it has to keep telling "live sensor, constant offset"
        # apart from "get_wrench() is dead" (C153A1).
        self._force_bias = np.zeros(3)
        self._force_bias_base = None
        self._force_bias_valid = False
        self._force_tare = bool(force_tare)
        self._force_tare_samples = max(1, int(force_tare_samples))
        self._force_tare_period = max(0.0, float(force_tare_period))

        if press_loop not in ("dtc", "legacy"):
            raise ValueError(f"press_loop must be 'dtc' or 'legacy', got {press_loop!r}")
        self._press_loop = press_loop
        self._pid = None
        if press_loop == "dtc":
            # max_v_along caps the *hold* seek; approach_v_max caps the soft
            # feed while barely touching, so it must not exceed it.
            # DTC's Fs = force_sign * (R_ee @ f) . n_base. UR ActualTCPForce is
            # the reaction, so compression reads negative along the press axis:
            # force_sign = -1 lands on the same Fs > 0 convention as _fs().
            self._pid = ForcePIDController(ForcePIDConfig(
                Fd_N=self._target_force,
                force_sign=-1.0,
                max_v_along_m_s=self._max_v_hold,
                approach_v_max_m_s=min(self._v_approach, self._max_v_hold),
                integral_limit_N_s=self._integral_limit,
                kp=self._kp_force,
                ki=self._ki_force,
                kd=self._kd_force,
                wrench_in_tool_frame=True,
            ))
            if spike_dF_dt_N_s is not None:
                self._pid.config.spike_dF_dt_N_s = float(spike_dF_dt_N_s)
        # Surface normal (base frame) and tool->base rotation for the dtc loop,
        # pushed in by the field-follow tick. Falls back to the live tool Z.
        self._n_star = None
        self._R_tool_to_base = None

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
        self._z_mode = "contact"
        self._had_load = False
        self._unload_since = None
        self._contact_since = None
        self._prev_v_z = None
        self._external_wrench = np.zeros(6)
        self._on_force_abort = None
        self._on_vz = None
        self._on_press_wrench = None
        self._on_press_engaged = None
        self._last_diag = {}
        self._t_tick = None
        # Optional () -> UR safety mode. After a protective stop the controller
        # discards our speedJ stream, but this loop went on computing press
        # velocity against a robot that was not moving -- so the force error
        # stayed large and the command it was holding would have been applied
        # the instant someone cleared the stop on the pendant.
        self._safety_mode_fn = safety_mode_fn
        self._safety_halted = False
        # Optional () -> dict from AdmittanceTeleoperationController.conditioning().
        # The joint path is where the C153A1 shoulder stop lives, and the force
        # loop is blind to it, so carry it into the same durable CSV.
        self._conditioning_fn = None
        self._log_path = None
        self._log_file = None
        self._log_writer = None
        self._log_rows = 0

    def set_interface(self, interface):
        self._interface = interface

    def set_measured_fz_fn(self, measured_fz_fn):
        """Callable () -> float tool-frame fz in Newtons, or None to use get_wrench()[3]."""
        self._measured_fz_fn = measured_fz_fn

    def set_measured_force_fn(self, measured_force_fn):
        """Callable () -> (fx, fy, fz) tool-frame force in N, for the dtc loop."""
        self._measured_force_fn = measured_force_fn

    def set_contact_normal(self, n_base, R_tool_to_base=None):
        """Surface normal in base frame (and tool->base R) for the dtc press loop.

        Pushed in from the field-follow tick rather than queried here: the force
        thread runs at 1/update_period and must not block on vf.
        """
        # Write R first, then n: _press_axis_base falls back to the legacy
        # tool-Z axis when n is None, so the transient pair is the safe one.
        if R_tool_to_base is not None:
            self._R_tool_to_base = np.asarray(
                R_tool_to_base, dtype=float).reshape(3, 3).copy()
        if n_base is None:
            self._n_star = None
        else:
            n = np.asarray(n_base, dtype=float).reshape(3)
            norm = float(np.linalg.norm(n))
            self._n_star = None if norm < 1e-12 else n / norm

    def set_contact_log(self, path):
        """Write one CSV row per force tick to ``path`` (None disables).

        The speedj trace is joint space and only saved on a clean exit -- the
        last debug session lost its .npz to a second Ctrl+C landing inside
        savez_compressed. This streams the force loop's own view to disk as it
        runs, so an aborted or protective-stopped run still leaves the record
        that explains it. Opened on start(), closed on stop().
        """
        self._log_path = None if path is None else str(path)

    _LOG_FIELDS = (
        "t", "elapsed", "dt", "mode", "press_loop", "fz_raw", "bias_x",
        "bias_y", "bias_z", "Fs", "Fd", "v_press", "v_z", "aborted",
        "Fs_n", "Fs_n_raw", "Fe_n", "dFs_dt", "gain", "v_along", "spike",
        "osc", "wrong_sign", "fault_fs", "F_press",
        "sigma_min", "clamp_scale", "qdot_max",
        "err_x", "err_y", "err_z", "vtask_lin",
        "err_rx", "err_ry", "err_rz", "k_xy", "k_ang",
    )

    def _open_contact_log(self):
        self._close_contact_log()
        if not self._log_path:
            return
        try:
            path = pathlib.Path(self._log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = path.open("w", newline="")
            self._log_writer = csv.DictWriter(
                self._log_file, fieldnames=list(self._LOG_FIELDS),
                extrasaction="ignore")
            self._log_writer.writeheader()
            self._log_rows = 0
            print(f"contact log: {path.resolve()}", flush=True)
        except Exception as exc:
            print(f"contact log: cannot open {self._log_path} ({exc})", flush=True)
            self._log_file = None
            self._log_writer = None

    def _close_contact_log(self):
        if self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass
        self._log_file = None
        self._log_writer = None

    def _write_contact_log(self, row):
        if self._log_writer is None:
            return
        try:
            self._log_writer.writerow(row)
            self._log_rows += 1
            # Flush often: the point of this file is to survive an abort, and a
            # protective stop is exactly when the buffered tail matters most.
            if self._log_rows % 50 == 0:
                self._log_file.flush()
        except Exception as exc:
            print(f"contact log: write failed ({exc}); disabling", flush=True)
            self._close_contact_log()

    def set_conditioning_fn(self, conditioning_fn):
        """Callable () -> {sigma_min, clamp_scale, qdot_max} for the contact log."""
        self._conditioning_fn = conditioning_fn

    def _conditioning(self):
        if self._conditioning_fn is None:
            return {}
        try:
            return self._conditioning_fn() or {}
        except Exception:
            return {}

    def set_safety_mode_fn(self, safety_mode_fn):
        """Callable () -> UR safety mode int (or None). See _safety_mode_fn."""
        self._safety_mode_fn = safety_mode_fn

    def _safety_blocks_motion(self):
        """True when the robot is stopped and must not be commanded."""
        if self._safety_mode_fn is None:
            return False
        try:
            mode = self._safety_mode_fn()
        except Exception:
            return False        # a failed read is not evidence of a stop
        if is_safe_to_command(mode):
            if self._safety_halted:
                self._safety_halted = False
                print(f"\033[92mrobot safety cleared ({safety_mode_name(mode)}); "
                      f"press resumes\033[0m", flush=True)
            return False
        if not self._safety_halted:
            self._safety_halted = True
            print(f"\033[91mrobot is in {safety_mode_name(mode)}: holding press "
                  f"at zero until it is cleared\033[0m", flush=True)
        return True

    def set_on_force_abort(self, callback):
        """Optional callback() invoked once if force feedback looks dead (open-loop)."""
        self._on_force_abort = callback

    def set_on_vz(self, callback):
        """Optional callback(v_z) each force tick. Overlay press on the pose-spring plant."""
        self._on_vz = callback

    def set_on_press_wrench(self, callback):
        """Optional callback(w_body_3) each force tick, in **newtons**.

        The wrench-law counterpart of ``set_on_vz``. Wire it to
        ``AdmittanceTeleoperationController.set_press_wrench`` and the press
        becomes a commanded force instead of a commanded velocity:

            W = Lambda (K e - D J qdot)      # free space, unchanged
            W[lin] -= (W[lin] . n) n         # free_along_normal, already there
            W[lin] += F_press * n            # this callback
            tau = clip(J.T W, +-tau_max)

        **No gafropy wrench type is involved on this path.** ``_run`` already
        skips ``set_desired_wrench`` whenever an overlay callback is set, so the
        VelocityAdmittance plant beneath this controller is never stepped; the
        vector handed over is a plain numpy 3-vector and
        ``compute_control_wrench`` adds it to its own numpy wrench. That is why
        the dead ``get_wrench()`` / ``Wrench.to_array()`` blade-order question
        cannot reach this code.

        Frame: **body**, matching ``set_contact_normal``'s ``_n_body``. This
        controller owns the conversion because it also owns the ``--force-sign``
        hemisphere convention (``_press_axis_base``), and it rotates with the
        same ``R_tool_to_base`` that the same field-follow tick published to
        both controllers -- so the projection and the press cannot disagree by
        a tick.
        """
        self._on_press_wrench = callback

    def _press_force_n(self, fs, now, dt):
        """Commanded press magnitude, newtons. Positive = into the part.

        ``F = Fd + kp * (Fd - Fs)``, clipped to ``[0, press_max_n]``.

        Why this is so much simpler than ``_press_velocity``: under the wrench
        law the commanded force *is* the applied force, so the steady-state
        answer is Fd and the trim only covers bias and a surface that moves.
        The speedJ seek had to *find* Fd by integrating a velocity into a
        position servo that could not yield -- "never keep inching into a rigid
        stop" (C153), the unload floor, the overshoot/mag_overload levels -- and
        none of that applies to an actuator that yields by construction.

        Two guards do survive, because they are about the *signal*, not the
        plant:

        - **Wrong sign.** ``Fs`` at or below ``min(-2, -0.5 Fd)`` is a bounce or
          a flipped reading, not more force owed. Both seek laws answer it the
          same way and so does this one: command nothing.
        - **A ceiling.** ``press_max_n`` bounds the trim, because the wrist and
          shoulder ``tau_max`` under --dtc-compliance-hold are now high enough
          that a runaway trim would be delivered in full.
        """
        del now
        fd = float(self._target_force)
        if fs <= min(self._wrong_sign_fs_n, self._wrong_sign_fd_frac * -fd):
            self._press_engaged = False
            return self._slew_press(0.0, dt)
        # Latch with hysteresis so a noisy Fs near the threshold cannot chatter
        # between a 23 N command and a 5 N one at 200 Hz.
        if fs >= PRESS_ENGAGE_FRAC * fd:
            self._press_engaged = True
        elif fs < PRESS_RELEASE_FRAC * fd:
            self._press_engaged = False
        kp = self._press_kp_hold if self._press_engaged else self._press_kp_n_per_n
        f = fd + kp * (fd - fs)
        f = float(np.clip(f, 0.0, self._press_max_n))
        return self._slew_press(f, dt)

    def _slew_press(self, f_n, dt):
        """Rate-limit the commanded press. **Rising only** -- falling is instant.

        A symmetric limit is wrong here and it cost a run. Backing off is always
        the safe direction, and at 200 N/s a drop from the 23 N breakaway
        command to 5 N takes 90 ms -- during which the tool, having just broken
        free, is still being driven into the part. On
        contact_20260903_140311.csv that is how Fs reached 33.2 N against a
        target of 5. Rising stays limited because a step in commanded force is a
        step in tau into a lightly damped structure.
        """
        if self._press_slew_n_s <= 0.0 or self._prev_press_n is None:
            self._prev_press_n = f_n
            return f_n
        max_df = self._press_slew_n_s * float(dt)
        f_n = float(min(f_n, self._prev_press_n + max_df))
        self._prev_press_n = f_n
        return f_n

    def _press_wrench_body(self, f_n):
        """``f_n`` newtons along the contact normal, in the body frame.

        ``None`` when there is no trustworthy axis, and the caller must then
        command **no press** rather than guess one. Unlike the velocity overlay
        -- which fell back to the legacy tool-Z axis at millimetres per second
        -- a mis-aimed *force* is delivered at full magnitude on the next tick.

        Two cases return None:

        - **No pose yet** (``_R_tool_to_base`` unset, before the first
          field-follow tick).
        - **The normal is near-perpendicular to tool Z.** ``_press_axis_base``
          picks the hemisphere by ``sign(n . press_sign*tool_z)``, so at 90 deg
          that dot product is ~0 and the sign is decided by floating-point
          noise -- the axis flips between ``+n`` and ``-n`` tick to tick. The
          velocity path guards the same degeneracy with the same 0.2 threshold
          and answers it by falling back to tool Z; at 5 mm/s that is
          survivable and at 30 N it is not, so this refuses instead. In normal
          operation field-follow keeps tool Z within a few degrees of the
          normal, so this never fires; if it does, the tool is not conforming
          and pressing is the wrong thing to do anyway.
        """
        axis, axis_dot_tool_z = self._press_axis_base()
        R = self._R_tool_to_base
        if axis is None or R is None:
            return None
        if abs(float(axis_dot_tool_z)) < PRESS_AXIS_MIN_DOT_TOOL_Z:
            return None
        return np.asarray(R, dtype=float).reshape(3, 3).T @ (
            np.asarray(axis, dtype=float).reshape(3) * float(f_n))

    def set_on_press_engaged(self, callback):
        """Optional ``callback(bool)`` each force tick: is the tool on the part?

        Wire it to ``AdmittanceTeleoperationController.set_press_engaged`` so the
        pose spring keeps the contact normal until the press has arrived. Same
        latch and hysteresis the press gains use, so the spring hands over on
        exactly the tick the press stops needing breakaway authority.
        """
        self._on_press_engaged = callback

    def _emit_press_wrench(self, f_n):
        """Hand the press to the wrench law. Never raises into the force loop."""
        if self._on_press_engaged is not None:
            try:
                self._on_press_engaged(bool(self._press_engaged))
            except Exception as exc:
                print(f"on_press_engaged failed: {exc}", flush=True)
        if self._on_press_wrench is None:
            return
        w = np.zeros(3) if f_n == 0.0 else self._press_wrench_body(f_n)
        try:
            self._on_press_wrench(np.zeros(3) if w is None else w)
        except Exception as exc:
            print(f"on_press_wrench failed: {exc}", flush=True)

    def _fs(self, fz_raw):
        # UR ActualTCPForce is environment-on-TCP (reaction). Press along +Z reads Fz < 0.
        # DualLoop wants Fs > 0 in compression: Fs = −force_sign · fz.
        return -self._press_sign * float(fz_raw)

    def _read_fz_raw(self):
        if self._measured_fz_fn is not None:
            return float(self._measured_fz_fn())
        if self._measured_force_fn is not None:
            # Vector reader only: take its Z rather than falling through to
            # get_wrench(), which is always zero on hardware and would trip the
            # dead-feedback abort on a perfectly live sensor.
            return float(np.asarray(
                self._measured_force_fn(), dtype=float).reshape(-1)[2])
        wrench = np.asarray(
            self._interface.get_state().get_wrench().to_array(), dtype=float).reshape(6)
        return float(wrench[3])

    def _read_force_tool_raw(self):
        """Untared tool-frame (fx, fy, fz) in N; falls back to (0, 0, fz)."""
        if self._measured_force_fn is not None:
            return np.asarray(
                self._measured_force_fn(), dtype=float).reshape(-1)[:3].copy()
        return np.array([0.0, 0.0, self._read_fz_raw()], dtype=float)

    def _bias_tool_now(self):
        """The tare, expressed in the *current* tool frame.

        The offset is the tool's own weight, which is fixed in base and only
        looks constant in the tool frame while the wrist holds still. Two runs
        minutes apart on the same hardware tared at fz=+3.74 N and fz=+10.42 N
        for exactly that reason. Storing it in base and rotating it back each
        tick removes the orientation dependence; the tool-frame snapshot is the
        fallback for a tare taken before any pose was known.
        """
        R = self._R_tool_to_base
        if self._force_bias_base is None or R is None:
            return self._force_bias
        return np.asarray(R, dtype=float).reshape(3, 3).T @ self._force_bias_base

    def _read_force_tool(self):
        """Tare-corrected tool-frame (fx, fy, fz) in N for the dtc loop."""
        return self._read_force_tool_raw() - self._bias_tool_now()

    def tare_force(self, samples=None, period=None):
        """Sample the free-space force offset. Tool must be clear of the part.

        Without this the dtc loop trips on the raw payload weight the instant
        contact latches: at Fd=5 N, ``wrong_sign`` fires at Fs <= -2.5 N and
        ``mag_overload`` at |Fs| > 6 N, and both command ``-unload_v_m_s``, so
        the arm retracts at 4 mm/s and never approaches. The legacy loop does
        not trip on it but regulates the offset instead of the contact force --
        the same bug with a quieter symptom.

        The bias is a tool-frame constant while the tool's weight is fixed in
        base, so it is only valid near the orientation it was taken at. Take it
        close to the press pose, and re-take it after a large wrist move.
        Returns the bias, or None if there is no force reader.
        """
        if self._measured_force_fn is None and self._measured_fz_fn is None:
            return None
        n = self._force_tare_samples if samples is None else max(1, int(samples))
        dt = self._force_tare_period if period is None else max(0.0, float(period))
        acc = []
        for i in range(n):
            try:
                acc.append(self._read_force_tool_raw())
            except Exception as exc:
                print(f"force tare read failed: {exc}", flush=True)
                return None
            if dt > 0.0 and i + 1 < n:
                time.sleep(dt)
        self._force_bias = np.mean(np.asarray(acc, dtype=float), axis=0)
        R = self._R_tool_to_base
        self._force_bias_base = (
            None if R is None
            else np.asarray(R, dtype=float).reshape(3, 3) @ self._force_bias)
        self._force_bias_valid = True
        return self._force_bias.copy()

    def clear_force_tare(self):
        """Forget the tare so the next ``start()`` takes a fresh one."""
        self._force_bias = np.zeros(3)
        self._force_bias_base = None
        self._force_bias_valid = False

    def _press_axis_base(self):
        """Press direction in base frame, and its component along tool Z.

        Reduces exactly to the legacy ``press_sign * tool_Z`` when no field
        normal has been pushed in. When one has, it is used for *direction*
        only -- the hemisphere stays the one ``--force-sign`` selected and the
        operator has already validated, so a field sign convention can never
        silently invert the press.
        """
        # Snapshot both together: set_contact_normal writes them from the
        # field-follow thread, and a torn (new R, old n) pair would aim the
        # press one tick stale.
        R, n = self._R_tool_to_base, self._n_star
        if R is None:
            return None, self._press_sign
        tool_z = np.asarray(R, dtype=float).reshape(3, 3)[:, 2]
        axis = self._press_sign * tool_z
        if n is not None:
            axis = n if float(n @ axis) > 0.0 else -n
        return axis, float(axis @ tool_z)

    def _press_velocity_dtc(self, now, dt):
        """Delegate the seek to the ported DTC ForcePIDController.

        Returns press velocity in the same units/sign as ``_press_velocity``:
        positive = into the part, and the caller maps it to tool Z.
        """
        self._z_mode = "contact"
        if self._contact_since is None:
            self._contact_since = now

        axis, axis_dot_tool_z = self._press_axis_base()
        R_ee = self._R_tool_to_base
        if axis is None or R_ee is None:
            # No pose yet: press along tool Z with the legacy sign, using fz only.
            axis = np.array([0.0, 0.0, self._press_sign], dtype=float)
            axis_dot_tool_z = self._press_sign
            R_ee = np.eye(3)

        _cmd, diag = self._pid.step(
            n_base=axis,
            force_base=self._read_force_tool(),
            dt=dt,
            R_ee=R_ee,
        )
        self._last_diag = diag
        v_along = float(diag["v_along_m_s"])
        # v_along is along the base-frame press axis, but the overlay downstream
        # is a tool-Z speed, so take the component: the delivered normal speed
        # picks up a cos^2 of the axis/tool-Z angle. That is a bounded positive
        # gain inside a closed force loop, so it costs response, not steady
        # state -- field-follow keeps the angle small anyway. Near-perpendicular
        # is the one case that would silently stall the press, so fall back to
        # the legacy tool-Z axis there rather than command ~0.
        if abs(axis_dot_tool_z) < 0.2:
            axis_dot_tool_z = self._press_sign
        return v_along * float(axis_dot_tool_z) * self._press_sign

    def _press_velocity(self, fs, now, dt):
        """Positive = press into the part (tool Z = press_sign * this)."""
        # Latched for this DualLoop session (button 0 → start, button 0 → stop).
        # Unload must not drop to approach: that is free-space 5 mm/s and C153.
        self._z_mode = "contact"
        if self._contact_since is None:
            self._contact_since = now
        self._unload_since = None

        fe = self._target_force - fs
        # Wrong sign: Fs below -max(2, 0.5*Fd) is not "even more force owed" --
        # it is a bounce, a flipped reading, or the tool being pulled. fe grows
        # without bound as Fs goes negative, so the bare PID answered a -11.58 N
        # read with kp*21.58 = +21.6 mm/s, clipped to the +8 mm/s ceiling and
        # driven straight into the part; the C153 protective stop followed on
        # the next tick. The over-target guard below never saw it, because Fs
        # was nowhere near Fd. Mirrors ForcePIDController's wrong_sign trip.
        if fs <= self._wrong_sign_threshold():
            self._integral = 0.0
            self._prev_error = fe
            return -self._unload_v_m_s
        if fs > 0.0:
            self._integral = np.clip(
                self._integral + fe * dt,
                -self._integral_limit, self._integral_limit)
        else:
            self._integral = 0.0
        deriv = 0.0 if self._prev_error is None else (fe - self._prev_error) / dt
        self._prev_error = fe
        v_pid = (self._kp_force * fe + self._ki_force * self._integral +
                 self._kd_force * deriv + self._force_feedforward)
        v_pid = float(np.clip(v_pid, -self._max_v_hold, self._max_v_hold))
        if fs >= self._target_force:
            # Over target: hold or retract. Never keep inching into a rigid stop (C153).
            return min(0.0, v_pid)
        # Under Fd (air, light load, or signed dip): keep PID. Retract-on-Fs<0
        # and lost_s→approach both walk the TCP off into free space.
        return v_pid

    def _tick_dt(self, now):
        """Real seconds since the previous tick, clamped.

        ``_run`` sleeps ``update_period`` *after* the work, and the work
        includes a blocking RTDE read, so the true period is always longer than
        nominal. Handing the nominal value to the seek laws rescales the
        integral and the LPF pole, and -- worst -- the spike guard, which then
        divides a real ~32 ms change by a nominal 20 ms baseline and inflates
        every rate by that ratio. That manufactures trips: on hardware SPIKE
        latched on almost every tick once Fs was past contact_threshold_N, so
        v_along sat at 0 and the loop could not close the last 2 N to Fd.
        """
        prev = self._t_tick
        self._t_tick = now
        if prev is None:
            return self._update_period
        return float(np.clip(now - prev,
                             0.2 * self._update_period,
                             10.0 * self._update_period))

    def _wrong_sign_threshold(self):
        """Fs at or below this is a bad reading, not a deeper press."""
        return min(self._wrong_sign_fs_n,
                   -abs(self._wrong_sign_fd_frac * self._target_force))

    def _slew_vz(self, v_z, dt):
        if self._v_slew_mps2 <= 0.0 or self._prev_v_z is None:
            self._prev_v_z = v_z
            return v_z
        max_dv = self._v_slew_mps2 * dt
        v_z = float(np.clip(v_z, self._prev_v_z - max_dv, self._prev_v_z + max_dv))
        self._prev_v_z = v_z
        return v_z

    def _compute_force_command(self):
        fz_raw = self._read_fz_raw()
        # Legacy seek regulates the tared reading; the abort below still looks
        # at fz_raw, since a live sensor sitting on a constant offset must not
        # read as dead feedback.
        bias_tool = self._bias_tool_now()
        fs = self._fs(fz_raw - float(bias_tool[2]))
        now = time.monotonic()
        dt = self._tick_dt(now)
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
        if self._safety_blocks_motion():
            # Zero the overlay and reset the slew, so clearing the stop does not
            # resume mid-ramp into the part.
            self._prev_v_z = 0.0
            self._prev_press_n = 0.0
            if self._on_vz is not None:
                try:
                    self._on_vz(0.0)
                except Exception as exc:
                    print(f"on_vz failed: {exc}", flush=True)
            # A protective stop must stop the press, not queue one up -- and
            # under the wrench law "not pressing" is 0 N, not a stale vector.
            self._emit_press_wrench(0.0)
            # Keep logging through the stop. Returning early left the CSV blank
            # for exactly the window worth reading -- what the force was doing
            # while the robot was halted, and whether it settled before the
            # operator cleared the pendant. mode marks the rows as not-commanded.
            if self._log_writer is not None:
                self._write_contact_log({
                    "t": f"{now:.6f}", "elapsed": f"{elapsed:.4f}",
                    "dt": f"{dt:.6f}", "mode": "safety_halt",
                    "press_loop": self._press_loop,
                    "fz_raw": f"{fz_raw:.4f}",
                    "bias_x": f"{bias_tool[0]:.4f}",
                    "bias_y": f"{bias_tool[1]:.4f}",
                    "bias_z": f"{bias_tool[2]:.4f}",
                    "Fs": f"{fs:.4f}", "Fd": f"{self._target_force:.3f}",
                    "v_press": "0.0", "v_z": "0.0", "F_press": "0.0",
                    "aborted": int(self._force_aborted),
                })
            return np.zeros(6)
        if self._force_aborted:
            self._prev_v_z = 0.0
            self._prev_press_n = 0.0
            if self._on_vz is not None:
                try:
                    self._on_vz(0.0)
                except Exception as exc:
                    print(f"on_vz failed: {exc}", flush=True)
            self._emit_press_wrench(0.0)
            return np.zeros(6)

        if self._on_press_wrench is not None:
            # Wrench law: the press is a force, so the velocity seek does not
            # run at all. Two loops authoritative on the same axis is the fight
            # that produces drift -- the same reason the outer PID owns Fd alone
            # on the speedJ path.
            f_press = self._press_force_n(fs, now, dt)
            v_press = 0.0
            v_z = 0.0
        elif self._press_loop == "dtc":
            f_press = float("nan")
            v_press = self._press_velocity_dtc(now, dt)
            v_z = self._slew_vz(self._press_sign * v_press, dt)
        else:
            f_press = float("nan")
            v_press = self._press_velocity(fs, now, dt)
            v_z = self._slew_vz(self._press_sign * v_press, dt)

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
                extra = ""
                if self._press_loop == "dtc" and self._last_diag:
                    d = self._last_diag
                    # Fs here is the dtc loop's own along-normal measurement,
                    # which is what it actually regulates; `fs` above is the
                    # tool-Z-only legacy figure and can differ once the field
                    # normal is off tool Z.
                    extra = (
                        f"  Fs_n={d.get('Fs_N', 0.0):+.2f}N"
                        f"  gain={d.get('gain_scale', 1.0):.2f}"
                        f"{'  SPIKE' if d.get('spike') else ''}"
                        f"{'  OSC' if d.get('osc') else ''}"
                        f"{'  WRONGSIGN' if d.get('wrong_sign') else ''}"
                        f"{'  FAULT' if d.get('fault_fs') else ''}"
                    )
                print(
                    f"contact force[{src}/{self._press_loop}]: mode={self._z_mode}  "
                    f"fz_raw={fz_raw:+.2f}N  bias_z={bias_tool[2]:+.2f}N"
                    f"{'' if self._force_bias_valid else '(untared)'}  "
                    f"Fs={fs:+.2f}N  Fd={self._target_force:.1f}N  "
                    + (f"F_press={f_press:+.2f}N  "
                       if self._on_press_wrench is not None
                       else f"v_z={v_z:+.4f}m/s  ")
                    + f"Fext=0  sign={self._press_sign:+.0f}"
                    f"{'  sit' if sit else ''}{extra}",
                    flush=True,
                )

        if self._log_writer is not None:
            d = self._last_diag or {}
            self._write_contact_log({
                "t": f"{now:.6f}", "elapsed": f"{elapsed:.4f}",
                "dt": f"{dt:.6f}",
                "mode": self._z_mode, "press_loop": self._press_loop,
                "fz_raw": f"{fz_raw:.4f}",
                "bias_x": f"{bias_tool[0]:.4f}", "bias_y": f"{bias_tool[1]:.4f}",
                "bias_z": f"{bias_tool[2]:.4f}",
                "Fs": f"{fs:.4f}", "Fd": f"{self._target_force:.3f}",
                "v_press": f"{v_press:.6f}", "v_z": f"{v_z:.6f}",
                "F_press": f"{f_press:.4f}",
                "aborted": int(self._force_aborted),
                "Fs_n": f"{d.get('Fs_N', float('nan')):.4f}",
                "Fs_n_raw": f"{d.get('Fs_raw_N', float('nan')):.4f}",
                "Fe_n": f"{d.get('Fe_N', float('nan')):.4f}",
                "dFs_dt": f"{d.get('dFs_dt_N_s', float('nan')):.3f}",
                "gain": f"{d.get('gain_scale', float('nan')):.3f}",
                "v_along": f"{d.get('v_along_m_s', float('nan')):.6f}",
                "spike": int(d.get("spike", 0)), "osc": int(d.get("osc", 0)),
                "wrong_sign": int(d.get("wrong_sign", 0)),
                "fault_fs": int(d.get("fault_fs", 0)),
                **{k: f"{v:.5f}" for k, v in self._conditioning().items()},
            })

        if self._on_press_wrench is not None:
            self._emit_press_wrench(f_press)
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
            if self._on_vz is None and self._on_press_wrench is None:
                self.set_external_wrench(np.zeros(6))
                self.set_desired_wrench(desired)
            time.sleep(self._update_period)

    def _reset_press_loop(self):
        """Clear every piece of seek state a contact session accumulates.

        Split out of ``start()`` so it can be asserted on without spawning the
        200 Hz thread -- that thread used to re-accumulate the integral before
        the assertion ran, so the reset looked broken when it was not.
        """
        self._integral = 0.0
        self._prev_error = None
        self._force_aborted = False
        self._force_feedback_seen = False
        self._press_sign = 1.0 if self._force_sign >= 0.0 else -1.0
        self._fz_filt = None
        self._z_mode = "contact"
        self._had_load = False
        self._unload_since = None
        self._contact_since = None
        self._prev_v_z = None
        self._prev_press_n = None
        self._press_engaged = False
        self._external_wrench = np.zeros(6)
        self._last_diag = {}
        self._t_last_log = None
        self._t_tick = None
        self._safety_halted = False
        if self._pid is not None:
            # Same reason as the legacy integrator reset: a second start() must
            # not inherit windup, LPF state or a cut gain_scale from the
            # previous contact session.
            self._pid.reset()

    def start(self):
        self._reset_press_loop()
        # Tare once per session, not per entry: the first contact entry is from
        # free space by construction (the loop has to approach), while a later
        # re-entry can happen with the tool already loaded, where taring would
        # zero out real contact force and press Fd on top of it. Call
        # clear_force_tare() to force a fresh one.
        if self._force_tare and not self._force_bias_valid:
            bias = self.tare_force()
            if bias is None:
                print(
                    "\033[93mforce tare skipped: no force reader\033[0m",
                    flush=True,
                )
            else:
                frame = ("base-tracked" if self._force_bias_base is not None
                         else "tool-frame only (no pose yet)")
                print(
                    f"force tare (free space, {frame}): "
                    f"fx={bias[0]:+.2f} fy={bias[1]:+.2f} fz={bias[2]:+.2f} N",
                    flush=True,
                )
        self._t_start = time.monotonic()
        self._open_contact_log()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        # After the join, so the last tick's row is in the file.
        self._close_contact_log()

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
