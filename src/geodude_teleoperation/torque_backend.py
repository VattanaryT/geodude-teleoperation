"""Actuate the existing control law with UR Direct Torque Control.

The speedJ path sends a joint *velocity*; the UR integrates it into a position
setpoint and servos to that setpoint with its own stiff internal gain. Our pose
spring only shapes the setpoint we ask for -- it cannot soften the servo
underneath it, which is downstream of us. A friction grab therefore reads to the
controller as "position deviates from path" (C153A1). Torque commands carry no
setpoint, so that fault cannot fire from this cause and arm stiffness is exactly
what ``JointVelocityTorqueLaw`` says it is.

``AdmittanceTeleoperationController.compute_control_command`` is unchanged and
still produces ``qdot``. This module is only what consumes it.

Nothing here imports gafropy or ur_rtde at module scope, and every seam that
touches them -- the driverless kinematics, the RTDE clients, the tau streamer --
is injectable, so ``tests/test_torque_backend.py`` runs the law, the
control-group splice, the staleness bound and the safety gate with no hardware
packages installed at all.
"""
from __future__ import annotations

import pathlib
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np

# Both are stdlib+numpy at module scope (ur_rtde is reached through getattr on
# the control object, never imported), so this costs the tests nothing.
from .direct_torque import (
    HostDirectTorqueStreamer,
    install_dtc_latch_control_script,
)
from .tcp_force import is_safe_to_command, safety_mode_name

# UR5e joint torque limits are [150, 150, 150, 28, 28, 28] N.m. Gravity is
# compensated on-controller, so tau only has to overcome inertia and friction --
# these defaults sit far below the hardware limit on purpose. This cap *is* the
# compliance: a friction grab saturates here instead of protective-stopping, so
# raising it trades safety for tracking. Tune it on hardware, low first.
DEFAULT_TAU_MAX = (30.0, 30.0, 30.0, 10.0, 10.0, 10.0)

# N.m per rad/s of velocity error. Same 3:1 shoulder/wrist split as tau_max, so
# every joint saturates at the same velocity error (0.5 rad/s at these values).
DEFAULT_KV = (60.0, 60.0, 60.0, 20.0, 20.0, 20.0)

# N.m per rad of drift while idle. See JointVelocityTorqueLaw for why a hold
# term is needed at all.
DEFAULT_KP_HOLD = (200.0, 200.0, 200.0, 60.0, 60.0, 60.0)

# N.m per (rad/s * s) of accumulated velocity error, and the cap on what that
# accumulation may contribute. Off by default: the integral changes what the
# arm does under load, so it is opt-in per session like every other contact fix
# in this repo. Measured starting point on this rig -- the pose spring commands
# ~0.02 rad/s, breakaway friction is several N.m, and kv=60 delivers only
# ~1.5-2.6 N.m -- is ki=(800,800,800,250,250,250) with i_max=(8,8,8,3,3,3),
# which reaches breakaway in ~0.3 s of persistent error.
DEFAULT_KI = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
DEFAULT_I_MAX = (8.0, 8.0, 8.0, 3.0, 3.0, 3.0)

# --- the joints tau = J.T @ W structurally cannot reach ---------------------
#
# Wrist 3's jacobian column is pure tool-Z angular: the TCP sits on that axis,
# and the measurement says so far better than the CAD does -- on
# dtc_20260829_233332.csv, |W_lin| reached 98.8 N while |tau5| never exceeded
# 4e-4 N.m, a moment arm under 4 um. The orientation field constrains only the
# *direction* of tool Z, never the roll about it, so the pose spring's wrench is
# orthogonal to that column and tau5 was 0 on all 3056 ticks of that run, and on
# all 3545 of the one before it. q5 moved 0.004 deg end to end.
#
# Under speedJ that joint was position-servoed by the UR. Under torque it is a
# free, gravity-compensated joint held by nothing but its own friction -- which
# held on those two runs and is not a control law. This PD is the hold the task
# wrench cannot supply, and it is deliberately small: it shapes a joint nothing
# else is asking for, and must never out-argue a roll the field does command.
DEFAULT_JOINT_HOLD_KP = 2.0     # N.m per rad
DEFAULT_JOINT_HOLD_KD = 0.1     # N.m per rad/s
# Its own cap, well under any tau_max, so this can never be the term that rails.
DEFAULT_JOINT_HOLD_TAU_MAX = 2.0
# The task wrench is considered to own a joint above this, and the hold target
# follows the joint instead of fighting it. 0.05 N.m is ~100x the largest tau5
# either run produced and ~2% of the 2.5 N.m the wrists carry when working.
DEFAULT_JOINT_HOLD_YIELD_NM = 0.05


def _as_vec(value, n: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.full(n, float(arr[0]))
    if arr.size != n:
        raise ValueError(f"{name} must be scalar or length {n}, got {arr.size}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


class JointVelocityTorqueLaw:
    """``qdot_cmd`` -> ``tau``, closing on velocity rather than position.

    ``tau = kv * (qdot_cmd - qdot_meas)``, clipped to ``tau_max``.

    This is deliberately *not* the task-space OSC law that
    ``rtde_interpolation_controller.py`` uses. That one is
    ``tau = J.T @ (Kp @ e - Kd @ v)`` -- a pure position spring with no velocity
    command anywhere, so under load it sags by ``e = F / Kp`` by construction.
    Closing on velocity has no steady-state position offset, and it leaves every
    gain above the actuation seam (pose spring, field follow, contact vz overlay)
    with the meaning and the tuning it already has.

    **The idle hold.** Pure velocity feedback has no restoring term: at
    ``qdot_cmd = 0`` the law reduces to ``-kv * qdot_meas``, which damps motion
    but does not undo drift, so a long stationary hold creeps. speedJ never had
    this problem because ``speedj(0)`` is a hold on the UR side. So when the
    command is idle across every joint the law latches ``q`` and adds
    ``kp_hold * (q_hold - q)``; any non-idle command releases the latch. This is
    a hold, not a target the arm chases, so it does not reintroduce the
    displacement behavior -- while moving, ``kp_hold`` contributes nothing.

    **The integral.** ``kv`` alone cannot break static friction here. The pose
    spring tops out near 8 mm/s, which is ~0.02 rad/s of joint command, so
    ``kv * err`` is ~1-2 N.m against a UR5e breakaway of several N.m -- and
    raising ``kv`` to cover the gap destabilizes the loop (``qdot_meas`` returns
    over RTDE with latency; measured on hardware, 5x kv shook the arm violently
    without moving it any better). ``ki`` accumulates the *same* error into
    torque that does not scale with loop gain: it ramps until the joint breaks
    loose, then settles holding exactly the torque that sustains motion against
    kinetic friction, and unwinds only when the joint overruns the command
    (``qdot_meas > qdot_cmd``, so the error goes negative). Bounded by ``i_max``
    well under ``tau_max`` so a genuinely stuck joint pushes hard but not
    without limit, frozen while the output is saturated (anti-windup), and
    dropped whenever the command is idle so the hold latch owns standstill
    alone.
    """

    def __init__(self, *, n_joints: int = 6, kv=DEFAULT_KV,
                 tau_max=DEFAULT_TAU_MAX, kp_hold=DEFAULT_KP_HOLD,
                 ki=DEFAULT_KI, i_max=DEFAULT_I_MAX,
                 idle_radps: float = 1e-4, hold_max_rad: float = 0.05):
        self._n = int(n_joints)
        self._kv = _as_vec(kv, self._n, "kv")
        self._ki = np.abs(_as_vec(ki, self._n, "ki"))
        self._i_max = np.abs(_as_vec(i_max, self._n, "i_max"))
        self._integ = np.zeros(self._n, dtype=float)
        self._tau_max = np.abs(_as_vec(tau_max, self._n, "tau_max"))
        self._kp_hold = _as_vec(kp_hold, self._n, "kp_hold")
        self._idle_radps = float(idle_radps)
        # Bound the hold spring's authority the same way tau_max bounds the
        # whole command: a large latched error must not become a large lunge
        # when the operator leans the arm somewhere and lets go.
        self._hold_max_rad = float(hold_max_rad)
        self._q_hold: np.ndarray | None = None
        self._last_saturated = np.zeros(self._n, dtype=bool)

    @property
    def n_joints(self) -> int:
        return self._n

    @property
    def tau_max(self) -> np.ndarray:
        return self._tau_max.copy()

    @property
    def holding(self) -> bool:
        return self._q_hold is not None

    @property
    def last_saturated(self) -> np.ndarray:
        return self._last_saturated.copy()

    @property
    def integral(self) -> np.ndarray:
        return self._integ.copy()

    def reset(self) -> None:
        self._q_hold = None
        self._integ = np.zeros(self._n, dtype=float)
        self._last_saturated = np.zeros(self._n, dtype=bool)

    def step(self, qdot_cmd, qdot_meas, q=None, dt: float | None = None) -> np.ndarray:
        """Return the torque for one tick. ``q`` may be omitted to skip the hold.

        ``dt`` is the measured tick period; omitted, the nominal host period is
        used. It only scales the integral -- the same reason the contact loop
        integrates on ``_tick_dt()`` rather than on ``update_period``.
        """
        cmd = _as_vec(qdot_cmd, self._n, "qdot_cmd")
        meas = _as_vec(qdot_meas, self._n, "qdot_meas")

        idle = bool(np.all(np.abs(cmd) <= self._idle_radps))
        err = cmd - meas
        tau = self._kv * err

        if idle:
            # Standstill belongs to the hold latch alone. Leaving the integral
            # charged here would hand it a shove the moment the latch forms.
            self._integ = np.zeros(self._n, dtype=float)
        elif np.any(self._ki > 0.0):
            step_dt = 1.0 / DEFAULT_DTC_HZ if dt is None else max(float(dt), 0.0)
            candidate = np.clip(
                self._integ + self._ki * err * step_dt, -self._i_max, self._i_max)
            # Anti-windup: while the output is already saturated, only let the
            # integral shrink. Otherwise a joint held by friction charges it to
            # the cap and keeps pushing after the obstruction is gone.
            winding = np.abs(candidate) > np.abs(self._integ)
            frozen = winding & self._last_saturated
            self._integ = np.where(frozen, self._integ, candidate)
            tau = tau + self._integ

        if q is None:
            self._q_hold = None
        elif not idle:
            self._q_hold = None
        else:
            q_now = _as_vec(q, self._n, "q")
            if self._q_hold is None:
                self._q_hold = q_now.copy()
            err = np.clip(
                self._q_hold - q_now, -self._hold_max_rad, self._hold_max_rad)
            tau = tau + self._kp_hold * err

        clipped = np.clip(tau, -self._tau_max, self._tau_max)
        self._last_saturated = np.abs(tau) > self._tau_max + 1e-12
        return clipped


# Below this: the actuation seam itself. The law above is *what* to send; this
# is what sends it, in place of gafropy's ``interface.control(...)`` speedJ loop.

# Host publish / control-law rate. The on-robot latch runs at the UR's own
# 500 Hz, so this is only how often we refresh the value it latches.
DEFAULT_DTC_HZ = 125.0

# A state read older than this means the loop is flying blind, so tau drops to
# zero (gravity-compensated float) rather than acting on a stale qdot_meas.
# 0.1 s is ~12 ticks at DEFAULT_DTC_HZ -- long enough to ride out one slow RTDE
# read, short enough that a dead socket is caught before the arm drifts far.
DEFAULT_STALE_AFTER_S = 0.1

# Floor on the publish-side staleness bound, independent of --dtc-hz. Below the
# period the host loop can actually hold, this bound does not protect against a
# stalled loop, it chops a live one. See DirectTorqueBackend.__init__.
# Physical bounds on Lambda's eigenvalues for this arm with its tool. Measured
# on geodude.yaml at well-conditioned poses: 4.3-8.7 kg linear, 0.0008-0.57
# kg.m^2 angular (examples/analyze_dtc_mobility.py section 5).
#
# These are CLAMPS, not a pass/reject test, and they are tight rather than
# generous. The first version of this gate allowed 200 kg / 50 kg.m^2 "on
# purpose", reasoning that a sanity check should not double as a tuning knob.
# That reasoning was wrong and it put the arm into an uncommanded high-speed
# motion: Lambda multiplies the commanded force directly, so a ceiling 46x above
# the true value is a licence to command 46x the intended force, and every value
# under the ceiling passed through untouched. A bound that a runaway fits inside
# is not a bound.
#
# Pass/reject was the second half of that failure. Rejecting a tick drops
# straight to the un-shaped law, so a pose flickering across the threshold
# alternates between a heavily shaped command and an unshaped one at 124 Hz --
# an impulse train, the same shape as the tau-chop that caused C283A2. Clamping
# eigenvalues degrades continuously instead: at the limit Lambda simply stops
# growing.
LAMBDA_MIN_LIN_KG = 1.0
LAMBDA_MAX_LIN_KG = 15.0
LAMBDA_MIN_ANG_KGM2 = 0.0005
LAMBDA_MAX_ANG_KGM2 = 1.0
# Below this smallest singular value of J the pose is near-singular and
# (J M^-1 J.T) is losing rank -- which is where the 4576 kg came from. The
# controller already computes sigma_min for the same reason (_note_conditioning
# shouts under 0.02). Shaping ramps out to identity across this band rather
# than switching off at a threshold.
LAMBDA_SIGMA_FULL = 0.05
LAMBDA_SIGMA_NONE = 0.02
DEFAULT_STALE_TAU_MIN_S = 0.04


# --- control-group splice -------------------------------------------------
#
# The control group this backend drives is 7-DoF: the undriven vention rail plus
# the 6 UR joints, and it is a *slice* of the 30-DoF bimanual system. RTDE only
# ever reports the 6. Getting this wrong is the failure recorded in CLAUDE.md --
# a system-global joint index used against the control-group slice, which
# crashed with "index 15 on size 7" -- so the slots are resolved from the model
# at runtime and cross-checked, never assumed.

def resolve_control_group_slots(joint_indices, rail_indices, controlled_indices=None,
                                n_ur: int = 6):
    """Split a task space's global joint indices into UR slots and rail slots.

    ``joint_indices`` is ``task_space.get_joint_indices()`` (system-global).
    ``rail_indices`` is the vention DoF indices from
    ``vention.vention_dof_indices``. The returned slots are *positions within*
    ``joint_indices`` -- the only thing that may index a control-group slice.

    ``controlled_indices`` (``task_space.get_controlled_joint_indices()``) is an
    independent cross-check when given: gafropy already knows which joints the
    jacobian's columns belong to, and if the two derivations disagree then the
    model is not what this code assumes and we must stop rather than mis-slice.
    """
    ji = [int(v) for v in joint_indices]
    rails = {int(v) for v in rail_indices}
    rail_slots = [i for i, g in enumerate(ji) if g in rails]
    ur_slots = [i for i, g in enumerate(ji) if g not in rails]
    if len(ur_slots) != int(n_ur):
        raise RuntimeError(
            f"control group has {len(ur_slots)} non-rail joints, expected {n_ur}; "
            f"joint_indices={ji} rail_indices={sorted(rails)}"
        )
    if controlled_indices is not None:
        expected = [int(v) for v in controlled_indices]
        got = [ji[i] for i in ur_slots]
        if got != expected:
            raise RuntimeError(
                "control-group splice disagrees with the task space: "
                f"non-rail joints {got} vs controlled joints {expected}"
            )
    return ur_slots, rail_slots


def _clamp_spd(ang, ang_lo, ang_hi, lin, lin_lo, lin_hi, full):
    """Rebuild a block-diagonal Lambda with each blade's eigenvalues clamped.

    The cross blocks are dropped deliberately. They are what couple a linear
    command into an angular force, and near a singularity they are exactly the
    terms that grow without bound while the diagonal still looks plausible --
    so clamping the diagonal alone would not have stopped the runaway.
    """
    out = np.zeros((6, 6))
    for blk, lo, hi, sl in ((ang, ang_lo, ang_hi, slice(0, 3)),
                            (lin, lin_lo, lin_hi, slice(3, 6))):
        w, v = np.linalg.eigh(0.5 * (blk + blk.T))
        out[sl, sl] = v @ np.diag(np.clip(w, lo, hi)) @ v.T
    return out


def splice_ur(base, ur_values, ur_slots):
    """Copy of ``base`` with ``ur_values`` written into ``ur_slots``.

    Everything outside ``ur_slots`` -- the rail, and in the full-system vector
    the whole other arm -- keeps the value it already had. That is deliberate:
    see ``DirectTorqueBackend`` on why the rail must carry its pre-handoff value
    rather than ``--vention-height``.
    """
    out = np.asarray(base, dtype=float).copy()
    slots = np.asarray(ur_slots, dtype=int).reshape(-1)
    vals = np.asarray(ur_values, dtype=float).reshape(-1)
    if vals.size != slots.size:
        raise ValueError(
            f"expected {slots.size} UR values for slots {list(slots)}, got {vals.size}")
    if slots.size and (slots.min() < 0 or slots.max() >= out.size):
        raise ValueError(
            f"UR slots {list(slots)} do not fit a vector of length {out.size}")
    out[slots] = vals
    return out


class DirectTorqueBackend:
    """Runs the existing control law in Python and actuates it with ``directTorque``.

    gafropy's C++ loop is what calls ``compute_control_command`` and issues
    speedJ today. ``directTorque`` needs urcl to let go of the RTDE control
    client first, so this class re-creates that loop on the host::

        q, qd (RTDE)  ->  control-group splice  ->  SystemState / TaskSpaceState
                          / ObjectiveState      ->  compute_control_command
                          ->  JointVelocityTorqueLaw  ->  on-robot tau latch

    ``compute_control_command`` is untouched, so the pose spring, field-follow,
    free-along-normal and the contact vz overlay all keep their current meaning
    and their current tuning. Only the consumer of ``qdot`` changed.

    It also duck-types the slice of ``gafropy.robot_interface.RobotInterface``
    the rest of the session reads -- ``get_state()``, ``get_system()``,
    ``get_task_space()``, ``shutdown()`` -- so ``hold_live_pose``,
    ``update_command``, ``run_field_follow`` and the viser tick keep working
    through the calls they already make, with no branch on the backend.

    **Kinematics must outlive the handoff.** ``shutdown_gafropy_for_handoff``
    calls ``interface.shutdown()``, which takes the system with it. This class
    therefore loads its own System straight from the model file with
    ``gafropy.SystemSerialization.load``, which builds no driver and opens no
    RTDE client at all. That is the construction to use: a hardware-mode
    ``RobotInterface`` would grab the control client we are trying to take, and a
    mujoco-mode one would spin up a physics sim.

    **The rail keeps its pre-handoff value; it is not ``--vention-height``.**
    Under speedJ, gafropy hands the controller ``q_full[joint_indices]`` straight
    from hardware, and hardware reports the unsensed vention rail as 0. The
    example depends on that split: ``run_field_follow`` builds ``T_flange_hw``
    from the *raw* state and ``T_flange_viz`` from the vention-overlaid one, then
    maps between the two worlds. Substituting the true rail height here would
    shift the controller world by 0.25 m and break exactly the field conformance
    this backend exists to preserve. ``rail_position`` overrides it only for
    someone who deliberately wants the shifted world.

    **Losing speedJ means losing the UR's path supervisor, so the gate is ours.**
    Outside NORMAL/REDUCED -- and on a state read older than ``stale_after_s`` --
    tau goes to zero *and* ``law.reset()`` is called. Zeroing alone would repeat
    the mistake CLAUDE.md records as "A UR protective stop must stop the press,
    not queue one up": the idle latch would still hold a ``q_hold`` captured
    before the stop, so the first tick after the pendant is cleared would command
    a lunge back to that stale pose. Resetting means the latch re-forms wherever
    the arm actually is. A *failed* safety read is not a stop
    (``is_safe_to_command(None)`` is True) -- refusing to command on a missing
    status signal is the worse failure.

    Everything that needs gafropy, ur_rtde or a robot is injectable
    (``system``, ``control``, ``receive``, ``state_fn``, ``qdot_cmd_fn``,
    ``streamer_factory``, ``install_latch``) so the splice, the staleness bound
    and the gate can all be tested with no hardware packages present.
    """

    def __init__(
        self,
        *,
        task_space_name: str,
        model_file=None,
        system: Any = None,
        robot_ip: str | None = None,
        kv=DEFAULT_KV,
        tau_max=DEFAULT_TAU_MAX,
        kp_hold=DEFAULT_KP_HOLD,
        ki=DEFAULT_KI,
        i_max=DEFAULT_I_MAX,
        law: "JointVelocityTorqueLaw | None" = None,
        law_kind: str = "velocity",
        joint_hold_kp: float = DEFAULT_JOINT_HOLD_KP,
        joint_hold_kd: float = DEFAULT_JOINT_HOLD_KD,
        joint_hold_joints: Sequence[int] | None = None,
        joint_hold_tau_max: float = DEFAULT_JOINT_HOLD_TAU_MAX,
        hz: float = DEFAULT_DTC_HZ,
        hold_zero: bool = False,
        hold_at_tick: bool = True,
        friction_comp: bool = True,
        use_lambda: bool = False,
        safety_mode_fn: Callable[[], Any] | None = None,
        stale_after_s: float = DEFAULT_STALE_AFTER_S,
        stale_tau_s: float | None = None,
        rail_position: float | None = None,
        control: Any = None,
        receive: Any = None,
        state_fn: Callable[[], Any] | None = None,
        qdot_cmd_fn: Callable[[], Any] | None = None,
        payload_mass: float | None = None,
        payload_cog: Sequence[float] | None = None,
        install_latch: bool = True,
        log_path=None,
        latch_out_dir=None,
        stock_script=None,
        streamer_factory: Callable[..., Any] | None = None,
    ):
        from .vention import vention_dof_indices

        self._task_space_name = str(task_space_name)
        self._robot_ip = None if robot_ip is None else str(robot_ip)
        self._hz = float(hz)
        self._hold_zero = bool(hold_zero)
        # "velocity": tau = kv * (qdot_cmd - qdot_meas), the first DTC port --
        # it swaps only the actuator and keeps the whole speedJ velocity
        # pipeline (inertia, dt, DLS inverse, joint clamps) in front of it.
        # "wrench": tau = J.T @ W straight off the pose spring. The velocity
        # path's caps exist to protect a position servo and are what starve the
        # torque law; see AdmittanceTeleoperationController.compute_control_wrench.
        law_kind = str(law_kind)
        if law_kind not in ("velocity", "wrench"):
            raise ValueError(
                f"law_kind must be 'velocity' or 'wrench', got {law_kind!r}")
        self._law_kind = law_kind
        # Wrench law only. Re-snap the objective onto the live pose here, at the
        # control rate, instead of leaving it to run_field_follow at --follow-hz.
        # The snap period is a control gain -- see
        # AdmittanceTeleoperationController.hold_pose_at -- and at 20 Hz it was
        # worth 75 N.s/m of delayed damping nobody had counted, on top of the
        # 109.5 the damping ratio asks for. False restores the old behavior.
        self._hold_at_tick = bool(hold_at_tick)
        self._friction_comp = bool(friction_comp)
        # Wrench law only. Joint-space PD on the joints tau = J.T @ W cannot
        # reach -- see DEFAULT_JOINT_HOLD_KP. The joint list is resolved after
        # the control group is sized, because the default is "the last one".
        self._joint_hold_kp = max(0.0, float(joint_hold_kp))
        self._joint_hold_kd = max(0.0, float(joint_hold_kd))
        self._joint_hold_tau_max = abs(float(joint_hold_tau_max))
        self._joint_hold_request = joint_hold_joints
        self._joint_hold_q = None
        self._n_joint_hold = 0
        # Inertia shaping: tau = J.T @ (Lambda @ a) instead of J.T @ W.
        # Opt-in because it changes what K means (N/m -> 1/s^2) and nothing
        # here can detect a gain left over from the un-shaped law.
        self._use_lambda = bool(use_lambda)
        self._n_lambda_reject = 0
        self._lambda_logged = False
        self._last_lambda_alpha = 1.0
        self._safety_mode_fn = safety_mode_fn
        self._stale_after_s = float(stale_after_s)
        # The on-robot latch holds the last tau forever if this host loop stalls,
        # which is the safe failure for one dropped packet but the wrong one for
        # a dead loop. See HostDirectTorqueStreamer(stale_tau_s=...).
        #
        # This bound must sit *above* the period the loop actually holds, not
        # above the nominal one. Two control periods was 16 ms at 125 Hz against
        # a measured dt_max of 25 ms, so the publisher chopped tau several times
        # a second on a perfectly healthy loop -- and the chop, not the stall,
        # is what moved the arm. Four periods with a 40 ms floor clears the
        # measured jitter; the streamer decays rather than steps, and full float
        # still arrives at stale_after_s.
        self._stale_tau_s = (
            max(4.0 / max(self._hz, 1.0), DEFAULT_STALE_TAU_MIN_S)
            if stale_tau_s is None else float(stale_tau_s))
        self._rail_position = (
            None if rail_position is None else float(rail_position))
        self._payload_mass = payload_mass
        self._payload_cog = payload_cog
        self._payload_note = "not connected yet"
        self._install_latch = bool(install_latch)
        self._latch_out_dir = latch_out_dir
        self._stock_script = stock_script
        self._streamer_factory = streamer_factory

        if system is None:
            if model_file is None:
                raise ValueError("give either system= or model_file=")
            import gafropy  # deferred: importing this module needs no gafropy

            system = gafropy.SystemSerialization.load(str(model_file))
        self._system = system
        task_space = system.get_task_space(self._task_space_name)
        if task_space is None:
            raise RuntimeError(
                f"task space '{self._task_space_name}' not found in {model_file}")
        self._task_space = task_space

        self._joint_indices = np.asarray(task_space.get_joint_indices(), dtype=int)
        controlled = getattr(task_space, "get_controlled_joint_indices", None)
        self._ur_slots, self._rail_slots = resolve_control_group_slots(
            self._joint_indices,
            vention_dof_indices(system).values(),
            controlled_indices=(controlled() if callable(controlled) else None),
        )
        self._n = len(self._ur_slots)
        self._ur_global = self._joint_indices[np.asarray(self._ur_slots, dtype=int)]

        # Default: the last driven joint (wrist 3). Named by control-group slot,
        # not by global index, because that is the order tau is in.
        req = ([self._n - 1] if self._joint_hold_request is None
               else [int(j) for j in self._joint_hold_request])
        bad = [j for j in req if not 0 <= j < self._n]
        if bad:
            raise ValueError(
                f"joint_hold_joints {bad} outside the {self._n} driven joints")
        self._joint_hold_idx = np.asarray(sorted(set(req)), dtype=int)

        self._law = law if law is not None else JointVelocityTorqueLaw(
            n_joints=self._n, kv=kv, tau_max=tau_max, kp_hold=kp_hold,
            ki=ki, i_max=i_max)
        if self._law.n_joints != self._n:
            raise ValueError(
                f"law has {self._law.n_joints} joints, the control group drives "
                f"{self._n}")

        dof = int(system.get_dof())
        self._q_seed = np.asarray(
            system.get_default_configuration(), dtype=float).reshape(dof).copy()

        self._control = control
        self._owns_control = control is None
        self._receive = receive
        self._owns_receive = receive is None
        self._state_fn = state_fn
        self._qdot_cmd_fn = qdot_cmd_fn
        self._controller = None
        self._streamer: Any = None
        self._latch_path = None

        self._lock = threading.Lock()
        self._q6: np.ndarray | None = None
        self._qd6: np.ndarray | None = None
        self._t_state: float | None = None
        self._last_tau = np.zeros(self._n, dtype=float)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._n_ticks = 0
        self._n_stale = 0
        self._n_gated = 0
        self._t_tick: float | None = None
        self._n_read_fail = 0
        self._last_error: str | None = None
        self._gated_now = False
        self._cmd_fail_logged = False
        self._n_tau_clipped = 0
        # Which joint railed, not just that one did. A wrist at tau_max and a
        # shoulder at tau_max are different faults with the same scalar count:
        # the wrist is the light, lightly-damped blade that shook the arm into
        # C283A2, the shoulder is a friction grab doing its job.
        self._n_clip_joint = np.zeros(self._n, dtype=int)
        self._last_wrench = np.zeros(6, dtype=float)
        # Live plot feed for --dtc-law wrench. The speedJ trace hangs off
        # AdmittanceTeleoperationController and fires at the end of
        # compute_control_command, which the wrench law never calls -- so the
        # plot window was empty for every wrench-law run. Traced from here
        # instead, where both the wrench and the clipped tau exist.
        self._command_trace = None
        self._trace_fail_logged = False
        # One CSV row per control tick, flushed as it goes -- the 1 Hz console
        # line has now hidden the failure three times (|W_lin| against a railed
        # tau, the period the loop was holding, and a publish-side stale count
        # printed as 0). A protective-stopped session must leave the record that
        # explains it, the same way --contact-log does.
        self._log_path = None if log_path in (None, "", "none") else str(log_path)
        self._log_fh = None
        self._log_t0 = None
        self._log_fail_logged = False
        # Measured loop period. The damping term's stability product is
        # (D / M) * T, so a loop that quietly runs at a third of its nominal
        # rate triples the number that decides whether the arm shakes. Nothing
        # reported T before; the stale-tau warnings were the only hint.
        self._dt_ewma: float | None = None
        self._dt_max = 0.0

    # -- RobotInterface duck-type ------------------------------------------

    def get_system(self):
        return self._system

    def get_task_space(self, name=None):
        if name is None or str(name) == self._task_space_name:
            return self._task_space
        return self._system.get_task_space(str(name))

    def get_position(self) -> np.ndarray:
        """Full-system ``q``, the same shape and convention gafropy reports.

        Only the driven joints are live. Everything else -- the other arm, the
        grippers, both rails -- keeps whatever gafropy last said, captured by
        :meth:`seed_from_interface` before the handoff. That is what lets the
        viser tick draw the whole bimanual robot and lets ``q_with_vention``
        overlay the rails exactly as it did under speedJ.
        """
        with self._lock:
            q = self._q_seed.copy()
            q6 = None if self._q6 is None else self._q6.copy()
        return q if q6 is None else splice_ur(q, q6, self._ur_global)

    def get_velocity(self) -> np.ndarray:
        """Full-system ``qdot``. Undriven DoF read zero, as hardware reports them."""
        with self._lock:
            qd6 = None if self._qd6 is None else self._qd6.copy()
        qd = np.zeros_like(self._q_seed)
        return qd if qd6 is None else splice_ur(qd, qd6, self._ur_global)

    def get_state(self):
        """A real ``gafropy.control.SystemState`` carrying the live q / qdot.

        Callers reach it as ``interface.get_state().get_position()``. Building a
        genuine SystemState rather than a stand-in also means ``get_wrench()``
        answers the way gafropy's hardware state does -- zero, per the note at
        the top of ``tcp_force.py`` -- instead of raising at some caller we did
        not think of.
        """
        import gafropy

        q = self.get_position()
        state = gafropy.control.SystemState(int(q.size))
        state.set_position(q)
        state.set_velocity(self.get_velocity())
        return state

    def shutdown(self) -> None:
        """``RobotInterface.shutdown()`` shim so the example's teardown is uniform."""
        self.stop()

    # -- accessors ----------------------------------------------------------

    @property
    def law(self) -> JointVelocityTorqueLaw:
        return self._law

    @property
    def hold_zero(self) -> bool:
        return self._hold_zero

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def latch_path(self):
        return self._latch_path

    @property
    def payload_note(self) -> str:
        """Which payload gravity compensation is running on, and where it came from."""
        return self._payload_note

    @property
    def n_joints(self) -> int:
        return self._n

    @property
    def ur_slots(self) -> list:
        """Positions of the driven joints inside the control-group slice."""
        return list(self._ur_slots)

    def q(self):
        """Last good driven-joint position, or ``None`` before the first read."""
        with self._lock:
            return None if self._q6 is None else self._q6.copy()

    def qdot(self):
        """Last good driven-joint velocity, or ``None`` before the first read."""
        with self._lock:
            return None if self._qd6 is None else self._qd6.copy()

    def last_tau(self) -> np.ndarray:
        with self._lock:
            return self._last_tau.copy()

    # -- lifecycle ----------------------------------------------------------

    def seed_from_interface(self, interface) -> np.ndarray:
        """Snapshot the full-system ``q`` while gafropy still has it.

        Must run *before* ``shutdown_gafropy_for_handoff``: RTDE reports the six
        driven joints and nothing else, so every other DoF in
        :meth:`get_position` comes from here.
        """
        q = np.asarray(
            interface.get_state().get_position(), dtype=float).reshape(-1).copy()
        if q.size != self._q_seed.size:
            raise RuntimeError(
                f"gafropy reports {q.size} DoF but the driverless model has "
                f"{self._q_seed.size}; the model file is not the running system")
        if self._rail_position is not None:
            q[self._joint_indices[np.asarray(self._rail_slots, dtype=int)]] = (
                self._rail_position)
        with self._lock:
            self._q_seed = q
        return q

    def connect(self) -> None:
        """Open both RTDE clients and install the on-robot tau latch.

        Only valid after ``shutdown_gafropy_for_handoff`` -- urcl holds the
        control client (30003) until then. The receive client (30004) is a
        separate socket, which is why ``connect_tcp_force_reader``'s interface
        can be handed in as ``receive`` and reused rather than opening a third.
        """
        from .direct_torque import UR_SYSTEM_HZ

        if self._control is None:
            from rtde_control import RTDEControlInterface

            if self._robot_ip is None:
                raise ValueError("give robot_ip= or an already-open control=")
            try:
                self._control = RTDEControlInterface(self._robot_ip, UR_SYSTEM_HZ)
            except TypeError:  # older ur_rtde without the frequency argument
                self._control = RTDEControlInterface(self._robot_ip)
        # tau=0 only means "float" if the controller's gravity compensation knows
        # what is bolted to the flange -- that is the whole bring-up gate. The
        # pendant's value is the trustworthy default here (CLAUDE.md: TCP and
        # payload on this rig are calibrated), but silent inheritance is exactly
        # the failure the F/T tare taught us about, so say which one is in force.
        if self._payload_mass is not None:
            cog = ([0.0, 0.0, 0.0] if self._payload_cog is None
                   else [float(v) for v in self._payload_cog])
            self._control.setPayload(float(self._payload_mass), cog)
            self._payload_note = (
                f"{float(self._payload_mass):.3f} kg cog={cog} "
                "(--dtc-payload-mass)")
        else:
            self._payload_note = "pendant installation (not overridden)"
        print(f"dtc: payload: {self._payload_note}", flush=True)
        if self._receive is None and self._state_fn is None:
            from rtde_receive import RTDEReceiveInterface

            self._receive = RTDEReceiveInterface(self._robot_ip)

        # Stock ur_rtde applies one direct_torque per host call, which cannot
        # sustain 500 Hz; the patched script latches tau in an on-robot thread so
        # the host only republishes. Installed here rather than in start() so
        # verify_rate() runs against a robot already in its final configuration.
        if self._install_latch:
            self._latch_path = install_dtc_latch_control_script(
                self._control,
                stock_script=self._stock_script,
                out_dir=self._latch_out_dir,
            )
        self._ensure_streamer()
        self._read_state_raw()  # fail here, not inside the control thread

    def _ensure_streamer(self):
        if self._streamer is not None:
            return self._streamer
        if self._control is None:
            raise RuntimeError("connect() first: no RTDE control client")
        if self._streamer_factory is not None:
            self._streamer = self._streamer_factory(
                self._control,
                friction_comp=self._friction_comp,
                target_hz=self._hz,
                robot_latch=True,
            )
        else:
            self._streamer = HostDirectTorqueStreamer(
                self._control,
                friction_comp=self._friction_comp,
                target_hz=self._hz,
                robot_latch=True,
                stale_tau_s=self._stale_tau_s,
                # Decay from the bound to full float at the same age a dead
                # state read floats the arm. One number, both paths.
                stale_zero_s=max(self._stale_after_s, 2.0 * self._stale_tau_s),
            )
        return self._streamer

    def verify_rate(self, *, duration_s: float = 2.0, **kwargs):
        """``verify_direct_torque_rate`` on the control client.

        Only valid before :meth:`start`: this and the tau streamer share one
        ``directTorque`` channel and would pace each other into nonsense.
        """
        from .direct_torque import verify_direct_torque_rate

        if self._control is None:
            raise RuntimeError("connect() first")
        if self.active:
            raise RuntimeError("stop the control loop before measuring its rate")
        kwargs.setdefault("friction_comp", self._friction_comp)
        return verify_direct_torque_rate(self._control, duration_s=duration_s, **kwargs)

    def set_controller(self, controller) -> None:
        """The controller whose ``compute_control_command`` this loop drives."""
        self._controller = controller

    def set_safety_mode_fn(self, safety_mode_fn) -> None:
        """Callable () -> UR safety mode int (or None). See the class docstring."""
        self._safety_mode_fn = safety_mode_fn

    def set_receive(self, receive) -> None:
        """Reuse an already-open RTDE *receive* client instead of opening one.

        Only one RTDE client is allowed on most controllers, and
        ``connect_tcp_force_reader`` already holds one on 30004. Hand its
        interface in (``read_tcp_force.receive``) so DTC reads
        ``getActualQ``/``getActualQd`` off that same socket. Ownership passes
        with it: ``stop()`` will not disconnect a socket it did not open.
        ``None`` is ignored, so a caller can pass a missing reader unconditionally.
        """
        if receive is None:
            return
        self._receive = receive
        self._owns_receive = False

    def start(self, controller=None) -> None:
        if controller is not None:
            self._controller = controller
        if self._controller is None and self._qdot_cmd_fn is None:
            raise RuntimeError("no controller: start(approach_controller)")
        if self.active:
            return
        streamer = self._ensure_streamer()
        self._log_open()
        self._cmd_fail_logged = False
        self._law.reset()
        self._joint_hold_reset()
        # Enter torque mode already holding zero, not whatever tau a previous
        # session left latched on the robot.
        streamer.set_tau(np.zeros(self._n))
        streamer.start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="dtc-control-law", daemon=True)
        self._thread.start()
        print(
            f"\033[93mdtc: {self._law_kind} law at {self._hz:g} Hz, tau_max="
            f"{np.max(self._law.tau_max):g} N.m, payload={self._payload_note}, "
            f"latch={self._latch_path}"
            f"{'  [HOLD-ZERO: no actuation -- the arm must FLOAT]' if self._hold_zero else ''}"
            f"\033[0m",
            flush=True,
        )

    def stop(self) -> None:
        """Zero tau, leave torque mode, drop the script, close what we opened.

        Ordering follows the known-good teardown in
        ``rtde_interpolation_controller.py``: torque to zero first, then a brief
        servoJ hold, so the arm is not handed from torque mode straight to
        nothing while it still has momentum.
        """
        self._stop.set()
        thr = self._thread
        if thr is not None and thr.is_alive():
            thr.join(timeout=1.0)
        self._thread = None
        self._log_close()
        streamer, self._streamer = self._streamer, None
        if streamer is not None:
            try:
                streamer.stop(zero=True)
            except Exception:
                pass
        self._law.reset()
        self._joint_hold_reset()

        control, self._control = self._control, None
        receive, self._receive = self._receive, None
        if control is not None:
            steps = [lambda: control.directTorque([0.0] * self._n, False)]
            if receive is not None:
                steps.append(
                    lambda: control.servoJ(
                        list(receive.getActualQ()), 0.5, 0.5, 0.1, 0.1, 300))
                steps.append(lambda: control.servoStop())
            if self._owns_control:
                steps.append(lambda: control.stopScript())
                steps.append(lambda: control.disconnect())
            for step in steps:
                try:
                    step()
                except Exception:
                    pass
        if receive is not None and self._owns_receive:
            try:
                receive.disconnect()
            except Exception:
                pass

    def __enter__(self) -> "DirectTorqueBackend":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- state and command --------------------------------------------------

    def _read_state_raw(self):
        """``(q, qdot)`` for the driven joints. RTDE by default, injectable."""
        if self._state_fn is not None:
            q6, qd6 = self._state_fn()
        else:
            q6 = self._receive.getActualQ()
            qd6 = self._receive.getActualQd()
        q6 = _as_vec(q6, self._n, "q")
        qd6 = _as_vec(qd6, self._n, "qdot")
        with self._lock:
            self._q6 = q6
            self._qd6 = qd6
        return q6, qd6

    def control_group_state(self):
        """``(q, qdot)`` over the whole control group: rail + driven joints.

        The rail is neither driven nor sensed, so its position comes from the
        seed and its velocity is zero. This is the 7-vector gafropy's C++ loop
        used to hand the controller, and the reason the slots are resolved from
        the model instead of assumed.
        """
        with self._lock:
            q6 = None if self._q6 is None else self._q6.copy()
            qd6 = None if self._qd6 is None else self._qd6.copy()
            base = self._q_seed[self._joint_indices].copy()
        if q6 is None or qd6 is None:
            return None
        return (splice_ur(base, q6, self._ur_slots),
                splice_ur(np.zeros(base.size), qd6, self._ur_slots))

    def _control_states(self):
        """``(controller, robot_state, objective_state, task_space_state)``.

        The state gafropy's C++ loop would have built before calling the control
        law, rebuilt here because we took its RTDE client. Shared by both laws
        so they cannot drift apart in how they see the arm.
        """
        if self._controller is None:
            return None      # no command source yet: a pure damped hold
        state = self.control_group_state()
        if state is None:
            return None
        import gafropy

        q_ctrl, qd_ctrl = state
        controller = self._controller
        robot_state = gafropy.control.SystemState(int(q_ctrl.size))
        robot_state.set_position(q_ctrl)
        robot_state.set_velocity(qd_ctrl)
        task_space = controller.get_task_space()
        self._maybe_hold(controller, q_ctrl)
        tss = task_space.compute_state(q_ctrl, qd_ctrl)
        objective_state = controller.get_objective().compute_state(robot_state, tss)
        return controller, robot_state, objective_state, tss

    def _maybe_hold(self, controller, q_ctrl) -> bool:
        """Re-snap the idle pose target onto the live q, at the control rate.

        Called from ``_control_states`` *before* the objective state is built,
        or the tick's error is still measured against the previous target.

        Idle only: while a command source owns the target -- the stick, or
        field-follow -- moving it here would erase the command. Wrench law only:
        the velocity law has its own idle hold latch inside
        ``JointVelocityTorqueLaw``, and two things holding one axis is the fight
        that produces drift.
        """
        if not (self._hold_at_tick and self._law_kind == "wrench"):
            return False
        if getattr(controller, "input_active", False):
            return False
        hold = getattr(controller, "hold_pose_at", None)
        if hold is None:
            return False
        hold(q_ctrl)
        return True

    def set_command_trace(self, fn):
        """Optional ``callback(tau, measured=, spring=, stick=, wrench=)`` per tick.

        Wrench law only. Under ``--dtc-law velocity`` the controller's own
        ``set_command_trace`` still fires from inside
        ``compute_control_command``, unchanged.
        """
        self._command_trace = fn

    def _trace_wrench(self, controller, robot_state, tau, wrench):
        trace = self._command_trace
        if trace is None:
            return
        try:
            qd = np.asarray(
                robot_state.get_velocity(), dtype=float).reshape(-1)
            measured = qd[self._ur_slots] if qd.size != self._n else qd
            trace(tau, measured=measured,
                  spring=bool(getattr(controller, "input_active", False)),
                  stick=getattr(controller, "stick_linear", None),
                  wrench=wrench)
        except Exception as exc:
            # A plot must never take down the loop that is holding the robot's
            # 500 Hz clock. Say it once and keep streaming tau.
            if not self._trace_fail_logged:
                self._trace_fail_logged = True
                print(f"\033[93mdtc: command trace disabled ({exc!r})\033[0m",
                      flush=True)
            self._command_trace = None

    def _task_inertia(self, jacobian):
        """``Lambda = (J M^-1 J.T)^-1``, eigenvalue-clamped and singularity-faded.

        Hand-rolled from ``System.compute_mass_matrix`` rather than
        ``TaskSpace.compute_task_space_inertia_matrix``: the built-in was
        measured returning ~1e14 on the diagonal on this model.

        Three things stand between this and the robot, because the first version
        of this method had none of them and drove the arm:

        1. **Singularity fade.** Lambda's blow-up is not a numerical curiosity,
           it is what a near-singular pose *means* -- the EE cannot accelerate in
           some direction, so the mass in that direction goes to infinity.
           Shaping fades to identity between ``LAMBDA_SIGMA_FULL`` and
           ``LAMBDA_SIGMA_NONE`` instead of switching, so crossing the band is
           not a step change in commanded force.
        2. **Eigenvalue clamp,** in the two blades separately, to the range this
           arm actually exhibits. Lambda multiplies the command, so its ceiling
           IS the force ceiling.
        3. **Blended, never rejected.** A per-tick accept/reject alternates
           between two different laws at the control rate.

        Returns None only when Lambda cannot be built at all, which is the
        un-shaped law -- correct, and merely un-shaped.
        """
        state = self.control_group_state()
        if state is None:
            return None
        try:
            sigma = float(np.sqrt(max(0.0, float(
                np.linalg.eigvalsh(jacobian @ jacobian.T)[0]))))
            q_full = self._q_seed.copy()
            q_full[self._joint_indices] = state[0]
            mass = np.asarray(
                self._system.compute_mass_matrix(q_full), dtype=float)
            idx = np.asarray(self._ur_global, dtype=int)
            m_inv = np.linalg.inv(mass[np.ix_(idx, idx)])
            lam = np.linalg.inv(jacobian @ m_inv @ jacobian.T)
        except Exception as exc:
            self._n_lambda_reject += 1
            self._last_error = f"lambda: {type(exc).__name__}: {exc}"
            return None
        if not np.all(np.isfinite(lam)):
            self._n_lambda_reject += 1
            return None

        lam = 0.5 * (lam + lam.T)
        lam = _clamp_spd(lam[:3, :3], LAMBDA_MIN_ANG_KGM2, LAMBDA_MAX_ANG_KGM2,
                         lam[3:, 3:], LAMBDA_MIN_LIN_KG, LAMBDA_MAX_LIN_KG, lam)

        # Fade to identity near a singularity. alpha=1 is fully shaped.
        span = LAMBDA_SIGMA_FULL - LAMBDA_SIGMA_NONE
        alpha = float(np.clip((sigma - LAMBDA_SIGMA_NONE) / span, 0.0, 1.0))
        if alpha <= 0.0:
            self._n_lambda_reject += 1
            return None
        lam = alpha * lam + (1.0 - alpha) * np.eye(6)
        self._last_lambda_alpha = alpha
        if not self._lambda_logged:
            self._lambda_logged = True
            d = np.abs(np.diag(lam))
            print(f"dtc: inertia shaping on -- apparent mass "
                  f"lin={np.round(d[3:], 2)} kg  ang={np.round(d[:3], 4)} kg.m^2  "
                  f"sigma_min={sigma:.4f} alpha={alpha:.2f}", flush=True)
        return lam

    def _joint_hold_tau(self, tau_task):
        """Joint-space PD on the joints ``tau = J.T @ W`` structurally cannot reach.

        Wrist 3 is the case this exists for, and the reasoning is at
        ``DEFAULT_JOINT_HOLD_KP``: its jacobian column is pure tool-Z angular,
        the orientation field constrains only the *direction* of tool Z, so the
        pose spring's wrench is orthogonal to that column and the commanded
        torque there was 0 on every tick of both hardware runs. Under speedJ the
        UR's own position servo held that joint; under torque nothing does.

        Three things keep this from becoming a second authority on an axis, the
        fight that produces drift:

        - **It yields.** Above ``DEFAULT_JOINT_HOLD_YIELD_NM`` of task torque on
          a joint the wrench owns it, the target follows the live angle, and
          only the damper is left. The condition that makes wrist 3 unreachable
          is a property of the current pose and field, not a theorem.
        - **The target snaps while the stick is quiet**, the same discipline as
          ``_maybe_hold``, so it cannot wind up against a slow commanded turn.
        - **Its contribution is capped** at ``joint_hold_tau_max``, well under
          any ``tau_max``, so it shapes a joint nothing else is asking for
          rather than owning one.

        Damping needs no target and is always safe; the spring is what actually
        undoes a drift, so both are on. ``--dtc-joint-hold-kp 0`` leaves the
        damper alone and reduces this to the previously-running behavior.
        """
        idx = self._joint_hold_idx
        zero = np.zeros(self._n)
        if idx.size == 0 or (self._joint_hold_kp <= 0.0
                             and self._joint_hold_kd <= 0.0):
            return zero
        with self._lock:
            q6 = None if self._q6 is None else self._q6.copy()
            qd6 = None if self._qd6 is None else self._qd6.copy()
        if q6 is None or qd6 is None:
            return zero

        target = self._joint_hold_q
        if target is None:
            target = q6.copy()
        # Re-snap where the wrench owns the joint, and everywhere while the
        # stick is quiet -- idle is when "hold here" is a true statement.
        idle = not bool(getattr(self._controller, "input_active", False))
        owned = np.abs(np.asarray(tau_task, dtype=float).reshape(-1)) \
            > DEFAULT_JOINT_HOLD_YIELD_NM
        follow = owned.copy()
        if idle:
            follow[:] = True
        target = np.where(follow, q6, target)
        self._joint_hold_q = target

        tau = zero.copy()
        tau[idx] = (-self._joint_hold_kp * (q6[idx] - target[idx])
                    - self._joint_hold_kd * qd6[idx])
        np.clip(tau, -self._joint_hold_tau_max, self._joint_hold_tau_max,
                out=tau)
        if np.any(np.abs(tau) > 1e-9):
            self._n_joint_hold += 1
        return tau

    def _joint_hold_reset(self):
        """Forget the held angles. Paired with every ``law.reset()``.

        A pendant stop must not queue up a lunge back to a target from before
        it -- the same reason the velocity law's idle latch is reset there.
        """
        self._joint_hold_q = None

    def _compute_wrench(self):
        """One tick of the wrench law: ``tau = clip(J.T @ W, +/-tau_max)``.

        ``JointVelocityTorqueLaw`` is bypassed entirely on this path -- no
        ``kv``, no integral, no idle hold latch. The pose spring *is* the hold
        under torque, and two things holding one axis is the fight that
        produces drift. Only ``tau_max`` is reused from the law object, and only
        as a configured vector, so the clamp cannot diverge between the two
        paths.
        """
        states = self._control_states()
        if states is None:
            return None
        controller, robot_state, objective_state, tss = states
        jacobian = np.asarray(tss.get_jacobian(), dtype=float)
        # Read the jacobian before the law, because Lambda is built from it.
        lam = self._task_inertia(jacobian) if self._use_lambda else None
        # The kwarg is passed ONLY when shaping is on. Passing inertia=None
        # would still change the call signature every control law has to accept,
        # and the un-shaped path is required to stay byte-identical -- that is
        # the whole basis for trusting it against the speedJ A/B.
        wrench = np.asarray(
            controller.compute_control_wrench(robot_state, objective_state, tss)
            if lam is None else
            controller.compute_control_wrench(
                robot_state, objective_state, tss, inertia=lam),
            dtype=float).reshape(-1)
        if wrench.size != 6:
            raise RuntimeError(
                f"control law returned a {wrench.size}-wrench, expected 6")
        if jacobian.shape[1] != self._n:
            # 6x6 on this rig: the vention rail is in get_joint_indices() but
            # not in the jacobian, which is why the slots are resolved from the
            # model rather than assumed.
            raise RuntimeError(
                f"jacobian has {jacobian.shape[1]} columns, the control group "
                f"drives {self._n}")
        self._last_wrench = wrench.copy()
        tau_max = self._law.tau_max
        tau = jacobian.T @ wrench
        # Added *before* the clip, so the hold is bounded by the same tau_max as
        # everything else and cannot smuggle torque past it. Its own cap keeps
        # it small enough that it never changes which joint rails.
        tau = tau + self._joint_hold_tau(tau)
        clipped = np.clip(tau, -tau_max, tau_max)
        railed = np.abs(tau) > tau_max + 1e-12
        if np.any(railed):
            self._n_tau_clipped += 1
            self._n_clip_joint += railed
        self._trace_wrench(controller, robot_state, clipped, wrench)
        return clipped

    def _compute_qdot(self):
        """One tick of the *unchanged* control law, off gafropy's C++ loop."""
        if self._qdot_cmd_fn is not None:
            return self._qdot_cmd_fn()
        states = self._control_states()
        if states is None:
            return None
        controller, robot_state, objective_state, tss = states
        qdot = np.asarray(
            controller.compute_control_command(robot_state, objective_state, tss),
            dtype=float).reshape(-1)
        if qdot.size != self._n:
            # Swallowed into last_error this would just look like "the arm does
            # not move". Say it once, loudly.
            if not self._cmd_fail_logged:
                self._cmd_fail_logged = True
                print(
                    f"\033[91mdtc: control law returned {qdot.size} joint "
                    f"velocities, expected {self._n}; tau held at zero\033[0m",
                    flush=True,
                )
            raise RuntimeError(
                f"control law returned {qdot.size} joint velocities, "
                f"expected {self._n}")
        return qdot

    def _read_state(self):
        try:
            return self._read_state_raw()
        except Exception as exc:
            self._n_read_fail += 1
            self._last_error = f"state_fn: {type(exc).__name__}: {exc}"
            return None

    def _read_qdot_cmd(self) -> np.ndarray:
        try:
            cmd = self._compute_qdot()
        except Exception as exc:
            self._last_error = f"qdot_cmd: {type(exc).__name__}: {exc}"
            return np.zeros(self._n, dtype=float)
        if cmd is None:
            return np.zeros(self._n, dtype=float)
        try:
            return _as_vec(cmd, self._n, "qdot_cmd")
        except ValueError as exc:
            self._last_error = f"qdot_cmd: {exc}"
            return np.zeros(self._n, dtype=float)

    def _read_tau_wrench(self) -> np.ndarray:
        try:
            tau = self._compute_wrench()
        except Exception as exc:
            self._last_error = f"wrench: {type(exc).__name__}: {exc}"
            if not self._cmd_fail_logged:
                self._cmd_fail_logged = True
                print(
                    f"\033[91mdtc: wrench law failed ({exc}); tau held at "
                    f"zero\033[0m", flush=True)
            return np.zeros(self._n, dtype=float)
        if tau is None:
            return np.zeros(self._n, dtype=float)
        return _as_vec(tau, self._n, "tau")

    # -- the loop -----------------------------------------------------------

    def tick(self, now: float | None = None) -> np.ndarray:
        """One control step. Public so a test can drive it without a thread."""
        now = time.perf_counter() if now is None else float(now)
        self._n_ticks += 1

        if self._read_state() is not None:
            with self._lock:
                self._t_state = now
        with self._lock:
            q_last, qdot_last, t_state = self._q6, self._qd6, self._t_state

        stale = t_state is None or (now - t_state) > self._stale_after_s
        mode = None
        if self._safety_mode_fn is not None:
            try:
                mode = self._safety_mode_fn()
            except Exception:
                mode = None  # a failed read is not evidence of a stop
        gated = not is_safe_to_command(mode)

        phase = "live"
        if gated or stale:
            phase = "gated" if gated else "stale"
            if gated:
                self._n_gated += 1
                if not self._gated_now:
                    print(
                        f"\033[91mrobot is in {safety_mode_name(mode)}: DTC "
                        f"holding tau at zero until it is cleared\033[0m",
                        flush=True,
                    )
            else:
                self._n_stale += 1
            self._gated_now = gated
            # Zero *and* forget the latch -- see the class docstring.
            self._law.reset()
            self._joint_hold_reset()
            # Same rule for the wrench law's re-arm blend: it opens from the
            # last wrench, and a pre-stop wrench is exactly what must not be
            # handed back when someone clears the pendant.
            rearm_reset = getattr(self._controller, "reset_wrench_rearm", None)
            if callable(rearm_reset):
                rearm_reset()
            tau = np.zeros(self._n, dtype=float)
        else:
            if self._gated_now:
                self._gated_now = False
                print(
                    f"\033[92mrobot safety cleared ({safety_mode_name(mode)}); "
                    f"DTC resumes\033[0m", flush=True)
            # Both laws run every tick even under hold_zero -- the speedJ
            # trace, the field-follow path and the viser EE frame stay live,
            # which is the whole point of watching the arm float in viser
            # rather than reading a console report.
            if self._law_kind == "wrench":
                tau_wrench = self._read_tau_wrench()
                cmd = None
            else:
                tau_wrench = None
                cmd = self._read_qdot_cmd()
            if self._hold_zero:
                phase = "hold_zero"
                # Bring-up gate. The law above the seam still runs every tick --
                # the speedJ trace, the field-follow path and the viser EE frame
                # all stay live, which is the whole point of watching the arm
                # float in viser rather than reading a console report -- but
                # nothing reaches the robot, and the torque law is not stepped so
                # it cannot latch an idle hold either.
                tau = np.zeros(self._n, dtype=float)
            elif tau_wrench is not None:
                # tau = J.T @ W, already clamped. The velocity law is bypassed
                # entirely here: no kv, no integral, no idle hold latch.
                tau = tau_wrench
            else:
                # Measured period, not nominal: the tick does a blocking RTDE
                # read, so the real dt runs longer than 1/hz and a nominal one
                # would under-integrate by that ratio.
                dt_tick = None if self._t_tick is None else now - self._t_tick
                tau = self._law.step(cmd, qdot_last, q_last, dt=dt_tick)

        dt_meas = None
        if self._t_tick is not None:
            dt_meas = now - self._t_tick
            self._dt_ewma = (
                dt_meas if self._dt_ewma is None
                else 0.98 * self._dt_ewma + 0.02 * dt_meas)
            self._dt_max = max(self._dt_max, dt_meas)
        self._t_tick = now
        self._log_row(now, dt_meas, phase, safety_mode_name(mode), tau,
                      q_last, qdot_last)
        with self._lock:
            self._last_tau = np.asarray(tau, dtype=float).reshape(-1).copy()
        if self._streamer is not None:
            self._streamer.set_tau(tau)
        return tau

    def _run(self) -> None:
        dt = 1.0 / max(self._hz, 1.0)
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self.tick(t0)
            except Exception as exc:  # never let the loop die silently
                self._last_error = f"tick: {type(exc).__name__}: {exc}"
                if self._streamer is not None:
                    self._streamer.set_tau(np.zeros(self._n))
            remain = dt - (time.perf_counter() - t0)
            if remain > 0:
                self._stop.wait(remain)

    # -- per-tick log -------------------------------------------------------

    def _log_open(self) -> None:
        if self._log_path is None or self._log_fh is not None:
            return
        try:
            path = pathlib.Path(self._log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = path.open("w", buffering=1)
        except Exception as exc:
            self._log_path = None
            print(f"\033[93mdtc: could not open --dtc-log ({exc}); "
                  f"running without the per-tick record\033[0m", flush=True)
            return
        meta = self._gain_note()
        if meta:
            # One '#' line before the header, so a run's gains travel with its
            # ticks. Three runs on 2026-08-30 used three different gain sets and
            # nothing in the CSV distinguished them -- including the zeta the
            # damping cap actually applied, which is not the zeta on the command
            # line (at --dtc-fn-lin 4.0 the cap took 0.30 down to 0.239, and the
            # only record of that was a startup warning in the scrollback).
            self._log_fh.write(f"# {meta}\n")
        n = self._n
        cols = ["t", "dt_ms", "phase", "safety"]
        cols += [f"w{i}" for i in range(6)]
        # SpaceMouse Twist blade order [ang(3), lin(3)]; see controller.raw_twist.
        cols += [f"sm{i}" for i in range(6)]
        cols += [f"tau{i}" for i in range(n)]
        cols += [f"q{i}" for i in range(n)]
        cols += [f"qd{i}" for i in range(n)]
        cols += [f"clip{i}" for i in range(n)]
        # Tracking. The wrench columns say what was *commanded*; these say what
        # the arm was actually asked to correct, which is the only way to tell a
        # loop that is following the field from one that is fighting a stale
        # target. err* is the raw (pre-clip) objective error in blade order,
        # n_deg is the angle between live tool +Z and the commanded field
        # normal (see TeleoperationController.contact_normal_body), and act is
        # the spring gate -- a 0->1 edge with a large err is a re-arm step, and
        # that step is what rails tau.
        cols += ["erx", "ery", "erz", "ex", "ey", "ez", "n_deg", "act"]
        self._log_fh.write(",".join(cols) + "\n")
        self._log_t0 = None
        print(f"dtc log: {self._log_path}", flush=True)

    def _gain_note(self) -> str:
        """The gains actually in force, for the log header.

        Read off the live matrices rather than off the CLI args on purpose:
        every resolver and cap sits between the two, and the cap is the half
        that has bitten. ``K`` is 1/s^2 under Lambda and N/m without it, which
        is why the flag is recorded next to the numbers.
        """
        bits = [f"law={self._law_kind}", f"hz={self._hz:g}",
                f"lambda={'on' if self._use_lambda else 'off'}"]
        ctrl = self._controller
        for name, attr in (("K", "_stiffness"), ("D", "_damping")):
            m = getattr(ctrl, attr, None)
            if m is None:
                continue
            try:
                d = np.asarray(m, dtype=float)
                d = np.diag(d) if d.ndim == 2 else d.reshape(-1)
            except Exception:
                continue
            bits.append(f"{name}_ang={d[0]:.4g},{d[1]:.4g},{d[2]:.4g}")
            bits.append(f"{name}_lin={d[3]:.4g},{d[4]:.4g},{d[5]:.4g}")
        for name, attr in (("err_clip_lin", "_err_clip_lin"),
                           ("err_clip_rot", "_err_clip_rot"),
                           ("rearm_wrench_s", "_rearm_wrench_s")):
            v = getattr(ctrl, attr, None)
            if v is not None:
                bits.append(f"{name}={float(v):g}")
        try:
            bits.append("tau_max=" + ",".join(
                f"{v:g}" for v in np.asarray(self._law.tau_max, dtype=float)))
        except Exception:
            pass
        return "  ".join(bits)

    def _log_close(self) -> None:
        fh, self._log_fh = self._log_fh, None
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass

    def _raw_twist_for_log(self):
        """Latest spacemouse Twist from the controller, or zeros if unset."""
        ctrl = self._controller
        if ctrl is None:
            return np.zeros(6)
        tw = getattr(ctrl, "raw_twist", None)
        if tw is None:
            return np.zeros(6)
        return np.asarray(tw, dtype=float).reshape(-1)[:6]

    def _tracking_for_log(self):
        """(err_ang(3), err_lin(3), normal_deg, input_active) from the controller.

        Pulled here rather than passed down because the wrench law already
        recorded it on the controller during this tick; nothing new is computed
        on the 125 Hz path except one arccos.
        """
        ctrl = self._controller
        err_ang = np.zeros(3)
        err_lin = np.zeros(3)
        n_deg = float("nan")
        act = 0
        if ctrl is None:
            return err_ang, err_lin, n_deg, act
        cond = getattr(ctrl, "conditioning", None)
        if callable(cond):
            try:
                c = cond()
                err_ang = np.array(
                    [c["err_rx"], c["err_ry"], c["err_rz"]], dtype=float)
                err_lin = np.array(
                    [c["err_x"], c["err_y"], c["err_z"]], dtype=float)
            except Exception:
                pass
        n_body = getattr(ctrl, "contact_normal_body", None)
        if n_body is not None:
            n_deg = float(np.degrees(np.arccos(
                np.clip(float(np.asarray(n_body, dtype=float).reshape(3)[2]),
                        -1.0, 1.0))))
        act = 1 if getattr(ctrl, "input_active", False) else 0
        return err_ang, err_lin, n_deg, act

    def _log_row(self, now, dt, phase, safety, tau, q, qd) -> None:
        fh = self._log_fh
        if fh is None:
            return
        if self._log_t0 is None:
            self._log_t0 = now
        n = self._n
        tau = np.asarray(tau, dtype=float).reshape(-1)
        zeros = np.zeros(n)
        q = zeros if q is None else np.asarray(q, dtype=float).reshape(-1)
        qd = zeros if qd is None else np.asarray(qd, dtype=float).reshape(-1)
        clip = np.abs(tau) >= self._law.tau_max - 1e-9
        sm = self._raw_twist_for_log()
        if sm.size < 6:
            sm = np.pad(sm, (0, 6 - sm.size))
        fields = [f"{now - self._log_t0:.4f}",
                  "" if dt is None else f"{1e3 * dt:.2f}", phase, safety]
        fields += [f"{v:.4f}" for v in self._last_wrench]
        fields += [f"{v:.5f}" for v in sm[:6]]
        fields += [f"{v:.4f}" for v in tau]
        fields += [f"{v:.5f}" for v in q[:n]]
        fields += [f"{v:.5f}" for v in qd[:n]]
        fields += ["1" if c else "0" for c in clip]
        err_ang, err_lin, n_deg, act = self._tracking_for_log()
        fields += [f"{v:.6f}" for v in err_ang]
        fields += [f"{v:.6f}" for v in err_lin]
        fields += ["" if not np.isfinite(n_deg) else f"{n_deg:.2f}", str(act)]
        try:
            fh.write(",".join(fields) + "\n")
        except Exception as exc:
            self._log_close()
            if not self._log_fail_logged:
                self._log_fail_logged = True
                print(f"\033[93mdtc: per-tick log write failed ({exc}); "
                      f"continuing without it\033[0m", flush=True)

    def stats(self) -> dict:
        out = {
            "active": self.active,
            "hold_zero": self._hold_zero,
            "n_ticks": self._n_ticks,
            "n_stale": self._n_stale,
            "n_gated": self._n_gated,
            "n_read_fail": self._n_read_fail,
            "holding": self._law.holding,
            "n_saturated": int(np.count_nonzero(self._law.last_saturated)),
            # The single number that says whether the seam is doing anything.
            # Joint breakaway friction on a UR5e is several N.m, so a command
            # that reads healthy above the seam and a peak tau of ~1 N.m means
            # the arm is simply not being pushed hard enough to move.
            "tau_peak": float(np.max(np.abs(self._last_tau)))
                        if self._last_tau.size else 0.0,
            "i_peak": float(np.max(np.abs(self._law.integral))),
            "law": self._law_kind,
            "n_lambda_reject": self._n_lambda_reject,
            "lambda_alpha": self._last_lambda_alpha,
            # Ticks on which the joint-space hold contributed anything. Its
            # torque is already in the per-joint tau columns of --dtc-log (on
            # wrist 3 it is the *only* thing there); this says whether it is
            # running at all, which a flag typo would otherwise hide.
            "n_joint_hold": self._n_joint_hold,
            # Wrench law only. |W_lin| is the commanded lateral force in N, and
            # it is the number to read against tau_peak: a healthy slide is a
            # sane |W_lin| with tau_peak well under tau_max. n_tau_clipped
            # rising means the wrench is asking for more than tau_max allows,
            # i.e. the clamp -- not the spring -- is deciding the motion.
            "w_lin": float(np.linalg.norm(self._last_wrench[3:6])),
            "w_ang": float(np.linalg.norm(self._last_wrench[0:3])),
            "n_tau_clipped": self._n_tau_clipped,
            "n_clip_joint": self._n_clip_joint.tolist(),
            # The joint that railed most. -1 when nothing has.
            "clip_joint_worst": (
                int(np.argmax(self._n_clip_joint))
                if int(self._n_clip_joint.sum()) else -1),
            # Nominal vs achieved. hz_meas well under hz means the damping term
            # is being fed back through a longer delay than it was tuned for,
            # which is the failure this seam actually has.
            "hz": self._hz,
            "hz_meas": (0.0 if not self._dt_ewma else 1.0 / self._dt_ewma),
            "dt_max_ms": 1000.0 * self._dt_max,
            "last_error": self._last_error,
            "latch_path": (
                None if self._latch_path is None else str(self._latch_path)),
        }
        if self._streamer is not None:
            out["streamer"] = self._streamer.stats()
        return out


__all__ = [
    "DEFAULT_DTC_HZ",
    "DEFAULT_KP_HOLD",
    "DEFAULT_KI",
    "DEFAULT_I_MAX",
    "DEFAULT_JOINT_HOLD_KD",
    "DEFAULT_JOINT_HOLD_KP",
    "DEFAULT_JOINT_HOLD_TAU_MAX",
    "DEFAULT_JOINT_HOLD_YIELD_NM",
    "DEFAULT_KV",
    "DEFAULT_STALE_AFTER_S",
    "DEFAULT_STALE_TAU_MIN_S",
    "DEFAULT_TAU_MAX",
    "DirectTorqueBackend",
    "JointVelocityTorqueLaw",
    "resolve_control_group_slots",
    "splice_ur",
]
