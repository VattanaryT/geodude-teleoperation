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
    """

    def __init__(self, *, n_joints: int = 6, kv=DEFAULT_KV,
                 tau_max=DEFAULT_TAU_MAX, kp_hold=DEFAULT_KP_HOLD,
                 idle_radps: float = 1e-4, hold_max_rad: float = 0.05):
        self._n = int(n_joints)
        self._kv = _as_vec(kv, self._n, "kv")
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

    def reset(self) -> None:
        self._q_hold = None
        self._last_saturated = np.zeros(self._n, dtype=bool)

    def step(self, qdot_cmd, qdot_meas, q=None) -> np.ndarray:
        """Return the torque for one tick. ``q`` may be omitted to skip the hold."""
        cmd = _as_vec(qdot_cmd, self._n, "qdot_cmd")
        meas = _as_vec(qdot_meas, self._n, "qdot_meas")

        idle = bool(np.all(np.abs(cmd) <= self._idle_radps))
        tau = self._kv * (cmd - meas)

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
        law: "JointVelocityTorqueLaw | None" = None,
        hz: float = DEFAULT_DTC_HZ,
        hold_zero: bool = False,
        friction_comp: bool = True,
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
        latch_out_dir=None,
        stock_script=None,
        streamer_factory: Callable[..., Any] | None = None,
    ):
        from .vention import vention_dof_indices

        self._task_space_name = str(task_space_name)
        self._robot_ip = None if robot_ip is None else str(robot_ip)
        self._hz = float(hz)
        self._hold_zero = bool(hold_zero)
        self._friction_comp = bool(friction_comp)
        self._safety_mode_fn = safety_mode_fn
        self._stale_after_s = float(stale_after_s)
        # The on-robot latch holds the last tau forever if this host loop stalls,
        # which is the safe failure for one dropped packet but the wrong one for
        # a dead loop. Default: two control periods, so ordinary jitter never
        # trips it. See HostDirectTorqueStreamer(stale_tau_s=...).
        self._stale_tau_s = (
            2.0 / max(self._hz, 1.0) if stale_tau_s is None else float(stale_tau_s))
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

        self._law = law if law is not None else JointVelocityTorqueLaw(
            n_joints=self._n, kv=kv, tau_max=tau_max, kp_hold=kp_hold)
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
        self._n_read_fail = 0
        self._last_error: str | None = None
        self._gated_now = False
        self._cmd_fail_logged = False

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
        self._cmd_fail_logged = False
        self._law.reset()
        # Enter torque mode already holding zero, not whatever tau a previous
        # session left latched on the robot.
        streamer.set_tau(np.zeros(self._n))
        streamer.start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="dtc-control-law", daemon=True)
        self._thread.start()
        print(
            f"\033[93mdtc: control law at {self._hz:g} Hz, tau_max="
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
        streamer, self._streamer = self._streamer, None
        if streamer is not None:
            try:
                streamer.stop(zero=True)
            except Exception:
                pass
        self._law.reset()

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

    def _compute_qdot(self):
        """One tick of the *unchanged* control law, off gafropy's C++ loop."""
        if self._qdot_cmd_fn is not None:
            return self._qdot_cmd_fn()
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
        tss = task_space.compute_state(q_ctrl, qd_ctrl)
        objective_state = controller.get_objective().compute_state(robot_state, tss)
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

        if gated or stale:
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
            tau = np.zeros(self._n, dtype=float)
        else:
            if self._gated_now:
                self._gated_now = False
                print(
                    f"\033[92mrobot safety cleared ({safety_mode_name(mode)}); "
                    f"DTC resumes\033[0m", flush=True)
            cmd = self._read_qdot_cmd()
            if self._hold_zero:
                # Bring-up gate. The law above the seam still runs every tick --
                # the speedJ trace, the field-follow path and the viser EE frame
                # all stay live, which is the whole point of watching the arm
                # float in viser rather than reading a console report -- but
                # nothing reaches the robot, and the torque law is not stepped so
                # it cannot latch an idle hold either.
                tau = np.zeros(self._n, dtype=float)
            else:
                tau = self._law.step(cmd, qdot_last, q_last)

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
    "DEFAULT_KV",
    "DEFAULT_STALE_AFTER_S",
    "DEFAULT_TAU_MAX",
    "DirectTorqueBackend",
    "JointVelocityTorqueLaw",
    "resolve_control_group_slots",
    "splice_ur",
]
