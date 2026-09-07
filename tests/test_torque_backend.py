"""The velocity->torque law and the DTC seam.

Pure numpy: no gafropy, no ur_rtde, no hardware. The backend's model, RTDE
clients and tau streamer are all injected, so the control-group splice, the
staleness bound and the safety gate are exercised here without a robot.
"""
from __future__ import annotations

import math
import pathlib
import time
import unittest

import numpy as np
import pytest

from geodude_teleoperation.torque_backend import (
    RailOscillationGuard,
    DEFAULT_KV,
    DEFAULT_STALE_TAU_MIN_S,
    DEFAULT_TAU_MAX,
    DirectTorqueBackend,
    JointVelocityTorqueLaw,
    resolve_control_group_slots,
    splice_ur,
)


def test_zero_command_and_zero_motion_is_zero_torque():
    law = JointVelocityTorqueLaw()
    assert np.allclose(law.step(np.zeros(6), np.zeros(6)), 0.0)


def test_torque_opposes_velocity_error():
    """Commanding faster than measured must push the joint the same way."""
    law = JointVelocityTorqueLaw()
    tau = law.step([0.1] * 6, [0.0] * 6)
    assert np.all(tau > 0)
    assert np.allclose(tau, np.asarray(DEFAULT_KV) * 0.1)
    assert np.allclose(law.step([-0.1] * 6, [0.0] * 6), -tau)


def test_pure_damping_when_idle_and_drifting():
    """qdot_cmd=0 with the arm moving must brake, not accelerate."""
    law = JointVelocityTorqueLaw()
    tau = law.step(np.zeros(6), [0.05] * 6)
    assert np.all(tau < 0)


def test_saturation_clamps_and_reports():
    law = JointVelocityTorqueLaw()
    tau = law.step([10.0] * 6, np.zeros(6))
    assert np.allclose(tau, DEFAULT_TAU_MAX)
    assert law.last_saturated.all()
    assert np.all(law.step([0.01] * 6, np.zeros(6)) != 0.0)
    assert not law.last_saturated.any()


def test_saturation_is_the_compliance_not_a_bigger_command():
    """A friction grab (commanded motion, zero measured) tops out at tau_max.

    This is the whole point of the backend: speedJ answered the same situation
    by growing a position deviation until the UR protective-stopped.
    """
    law = JointVelocityTorqueLaw()
    small = law.step([1.0] * 6, np.zeros(6))
    huge = law.step([1000.0] * 6, np.zeros(6))
    assert np.allclose(small, huge)
    assert np.allclose(huge, DEFAULT_TAU_MAX)


def test_idle_hold_latches_and_restores():
    """Drift while idle must be pushed back; speedj(0) held, -kv*qdot does not."""
    law = JointVelocityTorqueLaw()
    q0 = np.zeros(6)
    law.step(np.zeros(6), np.zeros(6), q=q0)
    assert law.holding
    drifted = np.full(6, 0.01)
    tau = law.step(np.zeros(6), np.zeros(6), q=drifted)
    assert np.all(tau < 0)  # pushed back toward q0


def test_hold_releases_on_any_command_and_does_not_pull_back():
    law = JointVelocityTorqueLaw()
    law.step(np.zeros(6), np.zeros(6), q=np.zeros(6))
    assert law.holding
    moved = np.full(6, 0.2)
    tau = law.step([0.1] * 6, [0.1] * 6, q=moved)
    assert not law.holding
    # Velocity is tracked exactly, so no torque -- no sag toward the old latch.
    assert np.allclose(tau, 0.0)


def test_hold_error_is_bounded():
    """A far-away latch must not become a lunge when the operator lets go."""
    law = JointVelocityTorqueLaw(hold_max_rad=0.05)
    law.step(np.zeros(6), np.zeros(6), q=np.zeros(6))
    near = law.step(np.zeros(6), np.zeros(6), q=np.full(6, 0.05))
    far = law.step(np.zeros(6), np.zeros(6), q=np.full(6, 5.0))
    assert np.allclose(near, far)


def test_omitting_q_disables_the_hold():
    law = JointVelocityTorqueLaw()
    law.step(np.zeros(6), np.zeros(6))
    assert not law.holding


def test_gains_reject_wrong_length_and_non_finite():
    with pytest.raises(ValueError):
        JointVelocityTorqueLaw(kv=[1.0, 2.0])
    with pytest.raises(ValueError):
        JointVelocityTorqueLaw(tau_max=[np.nan] * 6)
    law = JointVelocityTorqueLaw()
    with pytest.raises(ValueError):
        law.step(np.zeros(5), np.zeros(6))


def test_scalar_gains_broadcast():
    law = JointVelocityTorqueLaw(kv=100.0, tau_max=50.0)
    assert np.allclose(law.step([0.1] * 6, np.zeros(6)), 10.0)


# --------------------------------------------------------------------------
# The control-group splice. The task space is a 7-DoF slice of the 30-DoF
# bimanual system -- the undriven vention rail plus the 6 UR joints -- and RTDE
# reports only the 6. CLAUDE.md records what happens when this is guessed: a
# system-global index used against the slice, crashing with "index 15 on size 7".
# --------------------------------------------------------------------------

# geodude.yaml, verified against the real model: right_ee occupies actuated DoF
# 15..21 with the rail first, left_ee occupies 0..6 the same way.
RIGHT_EE_JOINTS = [15, 16, 17, 18, 19, 20, 21]
RIGHT_EE_CONTROLLED = [16, 17, 18, 19, 20, 21]
LEFT_EE_JOINTS = [0, 1, 2, 3, 4, 5, 6]
LEFT_EE_CONTROLLED = [1, 2, 3, 4, 5, 6]
VENTION_DOF = [0, 15]


def test_slots_are_positions_in_the_slice_not_global_indices():
    ur, rail = resolve_control_group_slots(
        RIGHT_EE_JOINTS, VENTION_DOF, RIGHT_EE_CONTROLLED)
    assert ur == [1, 2, 3, 4, 5, 6]
    assert rail == [0]
    # The bug this guards: 15 is a legal global index and an illegal slot.
    assert max(ur) < len(RIGHT_EE_JOINTS)


def test_both_arms_resolve():
    assert resolve_control_group_slots(
        LEFT_EE_JOINTS, VENTION_DOF, LEFT_EE_CONTROLLED) == ([1, 2, 3, 4, 5, 6], [0])


def test_wrong_joint_count_refuses_rather_than_mis_slicing():
    with pytest.raises(RuntimeError, match="expected 6"):
        resolve_control_group_slots([15, 16, 17], VENTION_DOF)


def test_disagreement_with_the_task_space_is_fatal():
    """gafropy already knows which joints the jacobian columns belong to."""
    with pytest.raises(RuntimeError, match="disagrees"):
        resolve_control_group_slots(
            RIGHT_EE_JOINTS, VENTION_DOF, [15, 16, 17, 18, 19, 20])


def test_splice_writes_only_the_driven_slots():
    base = np.arange(7, dtype=float)
    out = splice_ur(base, np.full(6, 9.0), [1, 2, 3, 4, 5, 6])
    assert out[0] == 0.0, "the rail must keep its pre-handoff value"
    assert np.allclose(out[1:], 9.0)
    assert np.allclose(base, np.arange(7)), "must not modify in place"


def test_splice_into_the_full_system_vector():
    """get_position() splices at global indices, not at slice positions."""
    full = np.zeros(30)
    out = splice_ur(full, np.arange(1.0, 7.0), RIGHT_EE_CONTROLLED)
    assert np.allclose(out[16:22], np.arange(1.0, 7.0))
    assert out[15] == 0.0 and np.allclose(out[:15], 0.0)


def test_splice_rejects_a_mismatched_or_out_of_range_write():
    with pytest.raises(ValueError):
        splice_ur(np.zeros(7), np.zeros(5), [1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError):
        splice_ur(np.zeros(7), np.zeros(6), RIGHT_EE_CONTROLLED)  # global, not slots


# --------------------------------------------------------------------------
# DirectTorqueBackend. Still no hardware and no gafropy: the model, the RTDE
# clients and the tau streamer are all injected.
# --------------------------------------------------------------------------


class FakeJoint:
    def __init__(self, index, name, actuated=True):
        self._i, self._n, self._a = index, name, actuated

    def is_actuated(self):
        return self._a

    def get_name(self):
        return self._n

    def get_index(self):
        return self._i


class FakeTaskSpace:
    def __init__(self, joints, controlled):
        self._j, self._c = list(joints), list(controlled)

    def get_joint_indices(self):
        return list(self._j)

    def get_controlled_joint_indices(self):
        return list(self._c)


class FakeSystem:
    """The slice of gafropy.System the backend reads at construction time."""

    def __init__(self, dof=30):
        self._dof = dof
        self._ts = FakeTaskSpace(RIGHT_EE_JOINTS, RIGHT_EE_CONTROLLED)

    def get_task_space(self, name):
        return self._ts if name == "right_ee" else None

    def get_dof(self):
        return self._dof

    def get_default_configuration(self):
        return np.zeros(self._dof)

    def get_joints(self):
        return [
            FakeJoint(0, "left_arm_linear_vention"),
            FakeJoint(15, "right_arm_linear_vention"),
            FakeJoint(16, "right_ur5e/shoulder_pan_joint"),
        ]


class FakeStreamer:
    """Stands in for HostDirectTorqueStreamer. Records what it was handed."""

    def __init__(self, control, *, friction_comp=True, target_hz=500.0,
                 robot_latch=True, stale_tau_s=None):
        self.control = control
        self.friction_comp = friction_comp
        self.target_hz = target_hz
        self.robot_latch = robot_latch
        self.started = False
        self.stopped_zero = None
        self.taus = []

    def set_tau(self, tau):
        self.taus.append(np.asarray(tau, dtype=float).copy())

    def start(self):
        self.started = True

    def stop(self, *, zero=True):
        self.started = False
        self.stopped_zero = zero

    def stats(self):
        return {"active": self.started, "n_ok": len(self.taus)}


class FakeState:
    """Mutable (q, qdot) source that can be made to fail on demand."""

    def __init__(self, q=None, qdot=None):
        self.q = np.zeros(6) if q is None else np.asarray(q, dtype=float)
        self.qdot = np.zeros(6) if qdot is None else np.asarray(qdot, dtype=float)
        self.fail = False
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("rtde read failed")
        return self.q, self.qdot


def _csv_lines(path):
    """The log's data lines. The `#` gain header is metadata, not a row."""
    return [ln for ln in path.read_text().strip().splitlines()
            if not ln.startswith("#")]


def _backend(**kwargs):
    """A backend on fakes: fake model, injected state, no latch upload."""
    state = kwargs.pop("state_fn", None) or FakeState()
    held = {}

    def factory(control, **kw):
        held["s"] = FakeStreamer(control, **kw)
        return held["s"]

    backend = DirectTorqueBackend(
        task_space_name="right_ee",
        system=FakeSystem(),
        control=object(),
        state_fn=state,
        install_latch=False,
        streamer_factory=factory,
        **kwargs,
    )
    return backend, state, held


def test_backend_resolves_the_splice_from_the_model():
    backend, _s, _h = _backend()
    assert backend.n_joints == 6
    assert backend.ur_slots == [1, 2, 3, 4, 5, 6]


def test_control_group_state_puts_the_rail_where_the_model_says():
    """The 7-vector gafropy's C++ loop used to hand compute_control_command."""
    state = FakeState(q=np.arange(1.0, 7.0), qdot=np.full(6, 0.2))
    backend, _s, _h = _backend(state_fn=state)
    backend.tick(1.0)
    q_ctrl, qd_ctrl = backend.control_group_state()
    assert q_ctrl.size == 7 and qd_ctrl.size == 7
    assert np.allclose(q_ctrl[1:], np.arange(1.0, 7.0))
    assert q_ctrl[0] == 0.0, "rail carries its pre-handoff value"
    assert qd_ctrl[0] == 0.0, "rail is undriven, so its velocity is zero"
    assert np.allclose(qd_ctrl[1:], 0.2)


def test_get_position_is_the_full_system_vector():
    """The viser loop reads this exactly as it reads gafropy's state today."""
    state = FakeState(q=np.arange(1.0, 7.0))
    backend, _s, _h = _backend(state_fn=state)
    backend.tick(1.0)
    q = backend.get_position()
    assert q.shape == (30,)
    assert np.allclose(q[16:22], np.arange(1.0, 7.0))
    assert np.allclose(q[:16], 0.0), "the other arm and the rails are untouched"


def test_seed_carries_the_rest_of_the_robot_across_the_handoff():
    backend, _s, _h = _backend()

    class Iface:
        def get_state(self):
            return type("S", (), {"get_position": lambda _self: np.arange(30.0)})()

    backend.seed_from_interface(Iface())
    assert np.allclose(backend.get_position()[:16], np.arange(16.0))


def test_seed_refuses_a_model_that_is_not_the_running_system():
    backend, _s, _h = _backend()

    class Iface:
        def get_state(self):
            return type("S", (), {"get_position": lambda _self: np.zeros(7)})()

    with pytest.raises(RuntimeError, match="not the running system"):
        backend.seed_from_interface(Iface())


def test_backend_sends_what_the_law_makes_of_the_qdot_command():
    backend, state, held = _backend(qdot_cmd_fn=lambda: np.full(6, 0.1))
    backend._ensure_streamer()
    tau = backend.tick(1.0)
    assert np.allclose(tau, np.asarray(DEFAULT_KV) * 0.1)
    assert np.allclose(held["s"].taus[-1], tau)
    assert state.calls == 1


def test_hold_zero_computes_the_command_but_actuates_nothing():
    """--dtc-hold-zero is the bring-up gate: the arm floats, the law stays live.

    The command still has to be computed every tick or the speedJ trace and the
    field-follow path go dark, and watching the viser EE frame track a floating
    arm is the evidence the gate is actually looking for.
    """
    asked = {"n": 0}

    def qdot_cmd():
        asked["n"] += 1
        return np.full(6, 5.0)

    backend, _s, held = _backend(qdot_cmd_fn=qdot_cmd, hold_zero=True)
    backend._ensure_streamer()
    assert np.allclose(backend.tick(1.0), 0.0)
    assert asked["n"] == 1, "the law above the seam must keep running"
    assert np.allclose(held["s"].taus[-1], 0.0)
    assert not backend.law.holding, "hold_zero must not latch an idle hold"
    assert backend.hold_zero


def test_q_accessor_holds_the_last_good_read():
    state = FakeState(q=np.arange(6, dtype=float))
    backend, _s, _h = _backend(state_fn=state)
    assert backend.q() is None and backend.qdot() is None
    backend.tick(1.0)
    assert np.allclose(backend.q(), np.arange(6))
    state.fail = True
    backend.tick(1.01)
    assert np.allclose(backend.q(), np.arange(6))


def test_stale_state_zeroes_torque():
    """A dead RTDE socket must float the arm, not act on a stale qdot_meas."""
    state = FakeState(qdot=np.full(6, 0.5))
    backend, _s, _h = _backend(
        state_fn=state, qdot_cmd_fn=lambda: np.full(6, 0.1), stale_after_s=0.1)
    backend.tick(1.0)
    state.fail = True
    assert not np.allclose(backend.tick(1.05), 0.0), "0.05 s is inside the window"
    assert np.allclose(backend.tick(1.5), 0.0)
    assert backend.stats()["n_stale"] >= 1


def test_read_failure_is_counted_but_not_fatal():
    state = FakeState()
    backend, _s, _h = _backend(state_fn=state)
    backend.tick(1.0)
    state.fail = True
    backend.tick(1.01)
    stats = backend.stats()
    assert stats["n_read_fail"] == 1
    assert "rtde read failed" in stats["last_error"]


def test_safety_gate_zeroes_torque_outside_normal():
    mode = {"m": 1}
    backend, _s, _h = _backend(
        qdot_cmd_fn=lambda: np.full(6, 0.1), safety_mode_fn=lambda: mode["m"])
    assert not np.allclose(backend.tick(1.0), 0.0)
    mode["m"] = 3  # PROTECTIVE_STOP
    assert np.allclose(backend.tick(1.001), 0.0)
    mode["m"] = 2  # REDUCED is still commandable
    assert not np.allclose(backend.tick(1.002), 0.0)
    assert backend.stats()["n_gated"] == 1


def test_unreadable_safety_mode_is_not_a_stop():
    """Same convention as the contact loop: a missing signal must not refuse."""
    def boom():
        raise RuntimeError("no getSafetyMode on this firmware")

    backend, _s, _h = _backend(
        qdot_cmd_fn=lambda: np.full(6, 0.1), safety_mode_fn=boom)
    assert not np.allclose(backend.tick(1.0), 0.0)
    backend2, _s2, _h2 = _backend(
        qdot_cmd_fn=lambda: np.full(6, 0.1), safety_mode_fn=lambda: None)
    assert not np.allclose(backend2.tick(1.0), 0.0)


def test_gate_resets_the_latch_so_recovery_is_not_a_lunge():
    """Clearing a protective stop must not apply a q_hold from before it.

    Same shape as the CLAUDE.md invariant "A UR protective stop must stop the
    press, not queue one up": zeroing tau without resetting the law would leave
    the idle latch holding the pre-stop pose, so the first tick after the pendant
    is cleared commands a lunge back to it.
    """
    state = FakeState(q=np.zeros(6))
    mode = {"m": 1}
    backend, _s, _h = _backend(
        state_fn=state, qdot_cmd_fn=lambda: np.zeros(6),
        safety_mode_fn=lambda: mode["m"])
    backend.tick(1.0)
    assert backend.law.holding, "an idle command latches a hold"
    mode["m"] = 3
    backend.tick(1.001)
    assert not backend.law.holding, "the gate must forget the latch"
    # Someone shoves the arm 0.3 rad while it is stopped, then clears the pendant.
    state.q = np.full(6, 0.3)
    mode["m"] = 1
    tau = backend.tick(1.002)
    assert np.allclose(tau, 0.0), "re-latch where the arm is, no pull back"


def test_bad_qdot_command_reads_as_zero_not_as_a_crash():
    backend, _s, _h = _backend(qdot_cmd_fn=lambda: np.full(6, np.nan))
    assert np.allclose(backend.tick(1.0), 0.0)
    assert "non-finite" in backend.stats()["last_error"]
    b2, _s2, _h2 = _backend(qdot_cmd_fn=lambda: None)
    assert np.allclose(b2.tick(1.0), 0.0)


def test_missing_command_source_is_a_pure_hold():
    backend, _s, _h = _backend()
    assert np.allclose(backend.tick(1.0), 0.0)
    assert backend.law.holding


def test_start_enters_torque_mode_holding_zero():
    backend, _s, held = _backend(qdot_cmd_fn=lambda: np.full(6, 0.1), hz=1000.0)
    backend.start()
    try:
        assert held["s"].started
        assert np.allclose(held["s"].taus[0], 0.0)
        assert held["s"].robot_latch is True
        assert backend.stats()["streamer"]["active"] is True
    finally:
        backend.stop()
    assert held["s"].stopped_zero is True
    assert not backend.active


def test_law_joint_count_must_match_the_control_group():
    with pytest.raises(ValueError):
        DirectTorqueBackend(
            task_space_name="right_ee", system=FakeSystem(), control=object(),
            state_fn=lambda: (np.zeros(6), np.zeros(6)),
            law=JointVelocityTorqueLaw(n_joints=7),
            install_latch=False, streamer_factory=FakeStreamer)


def test_verify_rate_refuses_while_the_loop_is_running():
    backend, _s, _h = _backend(qdot_cmd_fn=lambda: np.zeros(6), hz=1000.0)
    backend.start()
    try:
        with pytest.raises(RuntimeError, match="stop the control loop"):
            backend.verify_rate()
    finally:
        backend.stop()


# --- the shared RTDE receive client ----------------------------------------
#
# Only one RTDE client is allowed on most controllers and
# connect_tcp_force_reader already holds one on 30004, so DTC reads its joint
# state off that same socket rather than opening a second. Ownership is the part
# that bites: teardown must not disconnect a socket the force reader is still
# reading, or contact mode loses Fz the moment the arm shuts down.


class FakeReceive:
    """Stands in for the RTDEReceiveInterface that the force reader owns."""

    def __init__(self, q=None, qdot=None):
        self.q = np.arange(6, dtype=float) if q is None else np.asarray(q, dtype=float)
        self.qdot = np.zeros(6) if qdot is None else np.asarray(qdot, dtype=float)
        self.disconnected = False

    def getActualQ(self):
        return list(self.q)

    def getActualQd(self):
        return list(self.qdot)

    def disconnect(self):
        self.disconnected = True


class FakeControl:
    """Records setPayload and the teardown sequence; nothing else is called."""

    def __init__(self):
        self.payload = None
        self.calls = []

    def setPayload(self, mass, cog):
        self.payload = (float(mass), [float(v) for v in cog])
        self.calls.append("setPayload")

    def directTorque(self, tau, friction_comp):
        self.calls.append("directTorque")

    def servoJ(self, *args):
        self.calls.append("servoJ")

    def servoStop(self):
        self.calls.append("servoStop")


def _connected(**kwargs):
    """A backend whose ``connect()`` runs end to end on fakes."""
    control = FakeControl()
    backend = DirectTorqueBackend(
        task_space_name="right_ee", system=FakeSystem(), control=control,
        install_latch=False, streamer_factory=FakeStreamer, **kwargs)
    return backend, control


def test_state_is_read_off_the_handed_in_receive_client():
    receive = FakeReceive(q=np.arange(6.0) * 0.1, qdot=np.full(6, 0.02))
    backend, _control = _connected()
    backend.set_receive(receive)
    backend.connect()
    np.testing.assert_allclose(backend.q(), receive.q)
    np.testing.assert_allclose(backend.qdot(), receive.qdot)


def test_a_shared_receive_client_survives_teardown():
    receive = FakeReceive()
    backend, control = _connected()
    backend.set_receive(receive)
    backend.connect()
    backend.stop()
    assert receive.disconnected is False, (
        "stop() disconnected the force reader's socket")
    # It is still used for the servoJ hold on the way out -- shared, not unused.
    assert "servoJ" in control.calls


def test_set_receive_ignores_a_missing_reader():
    # The example passes getattr(read_tcp_force, "receive", None) unconditionally,
    # so a session with no force reader must not blow up here.
    backend, _control = _connected(state_fn=FakeState())
    backend.set_receive(None)
    backend.connect()
    assert backend.q() is not None


# --- payload provenance ----------------------------------------------------
#
# tau=0 only means "float" if the controller's gravity compensation knows what
# is on the flange, and that is the whole --dtc-hold-zero bring-up gate. The
# pendant's calibrated value is the right default; inheriting it *silently* is
# the failure the F/T tare already taught us about.


def test_payload_defaults_to_the_pendant_and_says_so():
    backend, control = _connected(state_fn=FakeState())
    backend.connect()
    assert control.payload is None, "setPayload overrode the pendant unasked"
    assert "pendant" in backend.payload_note


def test_payload_override_is_applied_and_named():
    backend, control = _connected(
        state_fn=FakeState(), payload_mass=1.25, payload_cog=(0.0, 0.0, 0.06))
    backend.connect()
    assert control.payload == (1.25, [0.0, 0.0, 0.06])
    assert "1.250 kg" in backend.payload_note
    assert "--dtc-payload-mass" in backend.payload_note


def test_payload_override_without_a_cog_is_the_flange_origin():
    backend, control = _connected(state_fn=FakeState(), payload_mass=1.25)
    backend.connect()
    assert control.payload == (1.25, [0.0, 0.0, 0.0])


# --- the wrench law: tau = clip(J.T @ W, +/-tau_max) -------------------------
#
# `law_kind="wrench"` bypasses JointVelocityTorqueLaw entirely -- no kv, no
# integral, no idle hold latch -- because under torque the pose spring *is* the
# hold, and two things holding one axis is the fight that produces drift. Only
# `tau_max` is reused, and only as a configured vector, so the clamp cannot
# diverge between the two paths.


class FakeWrenchController:
    """Stands in for AdmittanceTeleoperationController on the seam.

    The backend only ever asks it for a wrench and asks the task-space state
    for a jacobian; everything else lives above the seam and is covered in
    tests/test_wrench_law.py.
    """

    def __init__(self, wrench, jacobian=None, size=6):
        self.wrench = np.asarray(wrench, dtype=float)
        self.jacobian = np.eye(6) if jacobian is None else np.asarray(
            jacobian, dtype=float)
        self.size = size
        self.calls = 0

    def get_jacobian(self):
        return self.jacobian

    def compute_control_wrench(self, robot_state, objective_state, tss):
        self.calls += 1
        return self.wrench


def _wrench_backend(controller, **kwargs):
    """A wrench-law backend whose gafropy state building is stubbed out."""
    backend, state, held = _backend(law_kind="wrench", **kwargs)
    backend._control_states = lambda: (controller, None, None, controller)
    return backend, state, held


def test_law_kind_is_validated():
    with pytest.raises(ValueError, match="law_kind"):
        _backend(law_kind="admittance")


def test_wrench_becomes_torque_through_the_jacobian_transpose():
    # Identity jacobian: tau is the wrench itself, so the mapping is readable.
    # Stay under DEFAULT_TAU_MAX's wrist entry (10 N.m) so this reads the
    # mapping and not the clamp -- the clamp has its own test below.
    ctrl = FakeWrenchController([0.0, 0.0, 0.0, 8.0, 0.0, 0.0])
    backend, _s, _h = _wrench_backend(ctrl)
    assert np.allclose(backend._compute_wrench(), [0, 0, 0, 8, 0, 0])


def test_the_jacobian_transpose_is_not_the_jacobian():
    """A transposed-J bug is silent on the identity fixture and wrong on hardware."""
    j = np.zeros((6, 6))
    j[0, 1] = 2.0          # task blade 0 driven by joint 1
    ctrl = FakeWrenchController([4.0, 0, 0, 0, 0, 0], jacobian=j)
    backend, _s, _h = _wrench_backend(ctrl)
    tau = backend._compute_wrench()
    assert np.allclose(tau, [0.0, 8.0, 0.0, 0.0, 0.0, 0.0])


def test_torque_is_clamped_to_tau_max_and_counted():
    ctrl = FakeWrenchController([0.0, 0.0, 0.0, 500.0, 0.0, 0.0])
    backend, _s, _h = _wrench_backend(ctrl, tau_max=[30.0] * 6)
    assert np.allclose(backend._compute_wrench(), [0, 0, 0, 30, 0, 0])
    assert backend.stats()["n_tau_clipped"] == 1


def test_a_wrench_inside_tau_max_is_not_counted_as_clipped():
    ctrl = FakeWrenchController([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    backend, _s, _h = _wrench_backend(ctrl, tau_max=[30.0] * 6)
    backend._compute_wrench()
    assert backend.stats()["n_tau_clipped"] == 0


def test_the_velocity_law_is_never_stepped_on_the_wrench_path():
    """No kv, no integral, no idle hold latch -- the spring is the hold."""
    class Tripwire(JointVelocityTorqueLaw):
        def step(self, *a, **kw):        # pragma: no cover - must not run
            raise AssertionError("wrench path stepped the velocity law")

    ctrl = FakeWrenchController([0.0, 0.0, 0.0, 4.0, 0.0, 0.0])
    backend, _s, _h = _wrench_backend(ctrl, law=Tripwire(n_joints=6))
    tau = backend.tick()
    assert np.allclose(tau, [0, 0, 0, 4, 0, 0])
    assert not backend.law.holding


def test_hold_zero_runs_the_wrench_law_but_sends_nothing():
    """The bring-up gate: the field path and the viser frame stay live while
    the arm floats, which is the point of watching it in viser."""
    ctrl = FakeWrenchController([0.0, 0.0, 0.0, 9.0, 0.0, 0.0])
    backend, _s, _h = _wrench_backend(ctrl, hold_zero=True)

    assert np.allclose(backend.tick(), 0.0)
    assert ctrl.calls == 1
    assert backend.stats()["w_lin"] == pytest.approx(9.0)


def test_a_safety_stop_zeroes_the_wrench_torque():
    ctrl = FakeWrenchController([0.0, 0.0, 0.0, 6.0, 0.0, 0.0])
    mode = {"m": 1}
    backend, _s, _h = _wrench_backend(
        ctrl, safety_mode_fn=lambda: mode["m"])
    assert np.allclose(backend.tick(), [0, 0, 0, 6, 0, 0])
    mode["m"] = 3          # PROTECTIVE_STOP
    assert np.allclose(backend.tick(), 0.0)
    mode["m"] = 1          # cleared: the spring resumes, it does not lunge
    assert np.allclose(backend.tick(), [0, 0, 0, 6, 0, 0])


def test_a_failing_wrench_law_holds_tau_at_zero_rather_than_the_last_value():
    class Boom(FakeWrenchController):
        def compute_control_wrench(self, *a):
            raise RuntimeError("objective not set")

    backend, _s, _h = _wrench_backend(Boom([0.0] * 6))
    assert np.allclose(backend.tick(), 0.0)
    assert "objective not set" in backend.stats()["last_error"]


def test_a_wrong_sized_jacobian_refuses_rather_than_broadcasting():
    """6 task DOF, 6 driven joints: the vention rail is in get_joint_indices()
    but not in the jacobian. Silently broadcasting here is the "index 15 on
    size 7" crash wearing a different hat."""
    ctrl = FakeWrenchController([0.0] * 6, jacobian=np.eye(6)[:, :5])
    backend, _s, _h = _wrench_backend(ctrl)
    assert np.allclose(backend.tick(), 0.0)
    assert "jacobian has 5 columns" in backend.stats()["last_error"]


def test_stats_name_the_active_law():
    backend, _s, _h = _backend()
    assert backend.stats()["law"] == "velocity"
    ctrl = FakeWrenchController([0.0] * 6)
    wb, _s2, _h2 = _wrench_backend(ctrl)
    assert wb.stats()["law"] == "wrench"


# --- the idle pose snap is a control gain, not bookkeeping --------------------
#
# The objective target becomes pose(t - T), so the spring contributes K * T/2 of
# delay-based damping on top of the explicit D. Left to run_field_follow at
# --follow-hz 20 that was 3000 * 0.025 = 75 N.s/m of uncounted damping, 70%
# again on top of the 109.5 the damping ratio asks for, and delayed by 50 ms
# rather than one control period -- which is the half that destabilizes.


class FakeHoldController:
    """Just the two attributes ``_maybe_hold`` reads, plus a call log."""

    def __init__(self, input_active=False, has_hold=True):
        self.input_active = input_active
        self.held = []
        if not has_hold:
            self.hold_pose_at = None

    def hold_pose_at(self, q):
        self.held.append(np.asarray(q, dtype=float).copy())


def test_the_idle_snap_runs_every_tick_under_the_wrench_law():
    backend, _s, _h = _backend(law_kind="wrench")
    ctrl = FakeHoldController()
    q = np.arange(7.0)
    assert backend._maybe_hold(ctrl, q) is True
    assert np.allclose(ctrl.held[0], q)


def test_a_command_source_owning_the_target_is_never_overwritten():
    """Snapping while the stick or field-follow drives would erase the command."""
    backend, _s, _h = _backend(law_kind="wrench")
    ctrl = FakeHoldController(input_active=True)
    assert backend._maybe_hold(ctrl, np.zeros(7)) is False
    assert ctrl.held == []


def test_the_velocity_law_keeps_its_own_idle_latch():
    """JointVelocityTorqueLaw already holds; two holds on one axis is drift."""
    backend, _s, _h = _backend(law_kind="velocity")
    ctrl = FakeHoldController()
    assert backend._maybe_hold(ctrl, np.zeros(7)) is False
    assert ctrl.held == []


def test_hold_at_tick_can_be_turned_off_to_reproduce_the_old_behavior():
    backend, _s, _h = _backend(law_kind="wrench", hold_at_tick=False)
    ctrl = FakeHoldController()
    assert backend._maybe_hold(ctrl, np.zeros(7)) is False


def test_the_release_snap_fires_once_and_then_leaves_the_spring_alone():
    """The per-tick snap has the target chase the arm, so a push meets nothing.

    ``hold_on_release`` is the whole idle stiffness: snap on the 1->0 edge, then
    freeze, so a push on a released arm builds real pose error. Under the
    per-tick policy the same push is erased at the control rate and the wrench
    is identically zero -- which is what a hand push on a released arm was
    actually measuring in every teleop-mode comparison against
    rtde_interpolation_controller.py.
    """
    backend, _s, _h = _backend(
        law_kind="wrench", hold_at_tick=False, hold_on_release=True)
    ctrl = FakeHoldController(input_active=True)
    # Driving: the command source owns the target.
    assert backend._maybe_hold(ctrl, np.zeros(7)) is False
    # Release: exactly one snap, onto the pose at the moment of release.
    ctrl.input_active = False
    q_rel = np.arange(7.0)
    assert backend._maybe_hold(ctrl, q_rel) is True
    # Now the operator pushes the arm somewhere else. The target must NOT move.
    for _ in range(5):
        assert backend._maybe_hold(ctrl, q_rel + 0.01) is False
    assert len(ctrl.held) == 1
    assert np.allclose(ctrl.held[0], q_rel)


def test_the_release_snap_re_arms_on_the_next_drive():
    backend, _s, _h = _backend(
        law_kind="wrench", hold_at_tick=False, hold_on_release=True)
    ctrl = FakeHoldController()
    assert backend._maybe_hold(ctrl, np.zeros(7)) is True
    ctrl.input_active = True
    assert backend._maybe_hold(ctrl, np.zeros(7)) is False
    ctrl.input_active = False
    assert backend._maybe_hold(ctrl, np.ones(7)) is True
    assert len(ctrl.held) == 2


def test_a_reset_re_arms_the_release_snap():
    """A pendant stop or a rail-osc float may leave the arm somewhere else.

    Resuming into the pre-stop target would lunge back to it -- the same rule as
    law.reset() and the joint hold.
    """
    backend, _s, _h = _backend(
        law_kind="wrench", hold_at_tick=False, hold_on_release=True)
    ctrl = FakeHoldController()
    assert backend._maybe_hold(ctrl, np.zeros(7)) is True
    assert backend._maybe_hold(ctrl, np.zeros(7)) is False
    backend._hold_release_reset()
    assert backend._maybe_hold(ctrl, np.ones(7)) is True
    assert np.allclose(ctrl.held[-1], np.ones(7))


def test_the_release_snap_is_wrench_law_only():
    backend, _s, _h = _backend(
        law_kind="velocity", hold_at_tick=False, hold_on_release=True)
    ctrl = FakeHoldController()
    assert backend._maybe_hold(ctrl, np.zeros(7)) is False


def test_both_policies_off_freezes_the_target_for_the_push_test():
    backend, _s, _h = _backend(
        law_kind="wrench", hold_at_tick=False, hold_on_release=False)
    ctrl = FakeHoldController()
    assert backend._maybe_hold(ctrl, np.zeros(7)) is False


def test_a_controller_without_the_hook_is_not_an_error():
    backend, _s, _h = _backend(law_kind="wrench")
    ctrl = FakeHoldController(has_hold=False)
    assert backend._maybe_hold(ctrl, np.zeros(7)) is False


def test_the_loop_reports_the_period_it_actually_held():
    """(D/M)*T decides whether the damper shakes the arm, and nothing reported T.

    The run that ended in ROBOT_EMERGENCY_STOP was at ~23 ms against an 8 ms
    nominal; the only hint was the stale-tau warnings.
    """
    backend, _s, _h = _backend(hz=125.0)
    assert backend.stats()["hz_meas"] == 0.0        # no second tick yet
    backend.tick(now=1.000)
    backend.tick(now=1.040)                          # a 40 ms tick
    st = backend.stats()
    assert st["hz"] == 125.0
    assert 24.0 < st["hz_meas"] < 26.0
    assert 39.0 < st["dt_max_ms"] < 41.0


# --- the publish-side staleness bound ---------------------------------------
#
# The streamer chops tau when the last tick is older than this. Two control
# periods (16 ms at 125 Hz) sat *below* the period this loop actually holds
# (dt_max 25 ms measured), so it chopped a live loop several times a second
# and the on-robot latch applied every edge. The bound has to clear the
# measured jitter, not the nominal one.


def test_the_stale_bound_clears_the_jitter_a_healthy_loop_shows():
    backend, _s, held = _backend(hz=125.0)
    backend._ensure_streamer()
    assert backend._stale_tau_s >= 0.032        # four periods, not two
    assert backend._stale_tau_s >= DEFAULT_STALE_TAU_MIN_S


def test_a_slow_loop_gets_a_proportionally_longer_bound():
    backend, _s, _h = _backend(hz=50.0)
    assert backend._stale_tau_s == pytest.approx(0.08)


def test_the_floor_holds_when_the_nominal_rate_is_optimistic():
    """4/hz alone would be 8 ms at 500 Hz -- below any real host tick."""
    backend, _s, _h = _backend(hz=500.0)
    assert backend._stale_tau_s == pytest.approx(DEFAULT_STALE_TAU_MIN_S)


def test_an_explicit_bound_wins():
    backend, _s, _h = _backend(hz=125.0, stale_tau_s=0.01)
    assert backend._stale_tau_s == pytest.approx(0.01)


# --- which joint railed ------------------------------------------------------


def test_the_clipped_joint_is_named_not_just_counted():
    """A railed wrist and a railed shoulder are different faults.

    The wrist is the light, lightly-damped blade that shook the arm into
    C283A2; the shoulder is a friction grab doing what tau_max is for.
    """
    j = np.zeros((6, 6))
    j[3, 4] = 1.0                       # linear X blade drives joint 4 only
    ctrl = FakeWrenchController([0, 0, 0, 500.0, 0, 0], jacobian=j)
    backend, _s, _h = _wrench_backend(ctrl, tau_max=[30.0] * 6)
    backend._compute_wrench()
    st = backend.stats()
    assert st["clip_joint_worst"] == 4
    assert st["n_clip_joint"] == [0, 0, 0, 0, 1, 0]


def test_nothing_railed_reads_as_no_joint():
    ctrl = FakeWrenchController([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    backend, _s, _h = _wrench_backend(ctrl)
    backend._compute_wrench()
    assert backend.stats()["clip_joint_worst"] == -1


# --- the live plot feed ------------------------------------------------------
#
# The speedJ trace hangs off the controller and fires at the end of
# `compute_control_command`, which the wrench law never calls -- so the plot
# window was empty for every wrench-law run, and the operator lost the one
# view that shows |W_lin| between the 1 Hz console lines. It is traced from the
# backend instead, where both the wrench and the *clipped* tau exist.


class FakeTracedState:
    def __init__(self, qdot):
        self._qdot = np.asarray(qdot, dtype=float)

    def get_velocity(self):
        return self._qdot


def _traced_backend(controller, qdot=None, **kwargs):
    backend, state, held = _backend(law_kind="wrench", **kwargs)
    rs = FakeTracedState(np.zeros(6) if qdot is None else qdot)
    backend._control_states = lambda: (controller, rs, None, controller)
    seen = []
    backend.set_command_trace(
        lambda tau, **kw: seen.append((np.asarray(tau, dtype=float), kw)))
    return backend, seen


def test_the_trace_gets_the_clipped_tau_and_the_wrench():
    ctrl = FakeWrenchController([0.0, 0.0, 0.0, 500.0, 0.0, 0.0])
    backend, seen = _traced_backend(ctrl, tau_max=[30.0] * 6)
    backend._compute_wrench()
    tau, kw = seen[0]
    # Clipped, not raw: the plot has to show what the arm was actually sent,
    # or a railed run looks like it was asking for 500 N.m and getting it.
    assert np.allclose(tau, [0, 0, 0, 30, 0, 0])
    assert np.allclose(kw["wrench"], [0, 0, 0, 500, 0, 0])


def test_the_trace_carries_the_measured_joint_velocity():
    """"Is the arm actually moving" is the question a stalling loop leaves open."""
    ctrl = FakeWrenchController([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    qdot = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    backend, seen = _traced_backend(ctrl, qdot=qdot)
    backend._compute_wrench()
    np.testing.assert_allclose(seen[0][1]["measured"], qdot)


def test_a_broken_trace_does_not_take_down_the_loop():
    """It holds the robot's 500 Hz clock; a plot must never be able to stop it."""
    ctrl = FakeWrenchController([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    backend, _s, _h = _wrench_backend(ctrl)
    def boom(tau, **kw):
        raise RuntimeError("plot process died")
    backend.set_command_trace(boom)
    assert np.allclose(backend._compute_wrench(), [0, 0, 0, 5, 0, 0])
    assert backend._command_trace is None
    assert np.allclose(backend._compute_wrench(), [0, 0, 0, 5, 0, 0])


def test_no_trace_is_the_default():
    ctrl = FakeWrenchController([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    backend, _s, _h = _wrench_backend(ctrl)
    assert backend._command_trace is None
    assert np.allclose(backend._compute_wrench(), [0, 0, 0, 5, 0, 0])


# --- the per-tick record -----------------------------------------------------
#
# The 1 Hz console line has now hidden the failure three times. A
# protective-stopped run has to leave the record that explains it.


def test_the_log_records_one_row_per_tick_with_the_phase(tmp_path):
    path = tmp_path / "dtc.csv"
    backend, _s, _h = _backend(qdot_cmd_fn=lambda: np.zeros(6),
                               log_path=str(path))
    backend._log_open()
    backend.tick(now=1.000)
    backend.tick(now=1.008)
    backend._log_close()
    rows = _csv_lines(path)
    header = rows[0].split(",")
    assert header[:4] == ["t", "dt_ms", "phase", "safety"]
    assert header[4:10] == [f"w{i}" for i in range(6)]
    assert header[10:16] == [f"sm{i}" for i in range(6)]
    assert len(rows) == 3
    assert rows[1].split(",")[2] == "live"
    # dt is blank on the first row (no previous tick) and measured after.
    assert rows[1].split(",")[1] == ""
    assert float(rows[2].split(",")[1]) == pytest.approx(8.0, abs=0.1)


def test_the_log_records_spacemouse_raw_twist(tmp_path):
    """sm0..sm5 are the latest SpaceMouse Twist [ang(3), lin(3)]."""
    path = tmp_path / "dtc.csv"
    ctrl = FakeWrenchController([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    ctrl.raw_twist = np.array([0.1, -0.2, 0.3, 0.001, -0.002, 0.003])
    backend, _s, _h = _wrench_backend(ctrl, log_path=str(path))
    backend.set_controller(ctrl)
    backend._log_open()
    backend.tick(now=1.0)
    backend._log_close()
    header = _csv_lines(path)[0].split(",")
    row = _csv_lines(path)[1].split(",")
    sm_idx = [header.index(f"sm{i}") for i in range(6)]
    sm = [float(row[i]) for i in sm_idx]
    assert sm == pytest.approx([0.1, -0.2, 0.3, 0.001, -0.002, 0.003], abs=1e-5)


def test_the_log_says_when_the_tick_was_gated_or_held_at_zero(tmp_path):
    path = tmp_path / "dtc.csv"
    backend, _s, _h = _backend(qdot_cmd_fn=lambda: np.zeros(6),
                               hold_zero=True, log_path=str(path))
    backend._log_open()
    backend.tick(now=1.000)
    backend._safety_mode_fn = lambda: 3          # not NORMAL/REDUCED
    backend.tick(now=1.008)
    backend._log_close()
    phases = [r.split(",")[2] for r in _csv_lines(path)[1:]]
    assert phases == ["hold_zero", "gated"]


def test_the_log_records_the_tracking_error_and_field_conformity(tmp_path):
    """err*/n_deg/act are what separate "following the field" from "shoved at it".

    The wrench columns alone cannot: a loop tracking a target it reaches and a
    loop railing against one it never reaches both read as a large |W|. n_deg
    is the angle between live tool +Z and the commanded field normal, so it is
    the conformity error itself, and act marks the spring-gate edges where the
    target jumps from the live pose to the full field pose in one tick.
    """
    path = tmp_path / "dtc.csv"
    ctrl = FakeWrenchController([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    ctrl.conditioning = lambda: {
        "err_rx": 0.01, "err_ry": -0.02, "err_rz": 0.03,
        "err_x": 0.004, "err_y": -0.005, "err_z": 0.006,
    }
    # 30 deg off body +Z.
    ctrl.contact_normal_body = np.array([0.5, 0.0, np.sqrt(3.0) / 2.0])
    ctrl.input_active = True
    backend, _s, _h = _wrench_backend(ctrl, log_path=str(path))
    backend.set_controller(ctrl)
    backend._log_open()
    backend.tick(now=1.0)
    backend._log_close()
    lines = _csv_lines(path)
    header = lines[0].split(",")
    row = lines[1].split(",")
    # The three denominator columns trail act: Lambda's pre-clamp smallest
    # eigenvalue per blade, and the effective moment arm along the commanded
    # linear direction. The deadband is tau_friction / (lev * Lambda * K), so a
    # stuck run and a well-conditioned one are indistinguishable without them.
    # The pose block trails those, and the commanded-vs-actual block trails
    # it: absolute orientation of both frames, then the resolved-rate estimate
    # of the joints the pose error is asking for (there is no joint setpoint
    # under a torque law). All appended at the end, because every analyzer here
    # reads by header name and trailing columns are additive.
    assert header[-33:] == ["erx", "ery", "erz", "ex", "ey", "ez", "n_deg",
                            "act", "lam_ang", "lam_dir", "lam_hi", "lam_lo",
                            "lev",
                            "ee_x", "ee_y", "ee_z", "tgt_x", "tgt_y", "tgt_z",
                            "ee_qw", "ee_qx", "ee_qy", "ee_qz",
                            "tgt_qw", "tgt_qx", "tgt_qy", "tgt_qz",
                            "qc0", "qc1", "qc2", "qc3", "qc4", "qc5"]
    got = dict(zip(header, row))
    assert [float(got[k]) for k in ("erx", "ery", "erz")] == pytest.approx(
        [0.01, -0.02, 0.03])
    assert [float(got[k]) for k in ("ex", "ey", "ez")] == pytest.approx(
        [0.004, -0.005, 0.006])
    assert float(got["n_deg"]) == pytest.approx(30.0, abs=0.01)
    assert got["act"] == "1"


def test_leverage_is_the_moment_arm_along_the_commanded_direction():
    """``lev`` is the only direction-dependent term in the deadband.

    Lambda is symmetric by construction, so it treats ``+u`` and ``-u`` alike
    and cannot produce a one-sided dead zone; ``J.T`` can and does. Measured on
    dtc_20260904_105747.csv: 0.195 m along body x against 0.82 m along body z,
    and the settled deadband tracked that ratio (13.85 mm at 40 N vs 3.25 mm at
    8.4 N).
    """
    from geodude_teleoperation.torque_backend import _leverage_m
    jac = np.zeros((6, 6))
    jac[3, 0] = 0.2      # one N along blade x buys 0.2 N.m on joint 0
    jac[4, 1] = 0.8      # one N along blade y buys 0.8 N.m on joint 1
    assert _leverage_m(jac, [0, 0, 0, 1.0, 0, 0]) == pytest.approx(0.2)
    assert _leverage_m(jac, [0, 0, 0, 0, 3.0, 0]) == pytest.approx(0.8)
    # Sign-blind, which is what makes it an axis property rather than a bias.
    assert _leverage_m(jac, [0, 0, 0, -5.0, 0, 0]) == pytest.approx(0.2)
    # The angular blade is excluded: it is not what the linear deadband pays.
    assert _leverage_m(jac, [9.0, 9.0, 9.0, 1.0, 0, 0]) == pytest.approx(0.2)
    # No linear command means no direction, and a ratio built out of noise is
    # worse than no number at all.
    assert math.isnan(_leverage_m(jac, [1.0, 0, 0, 0, 0, 0]))


def test_the_tracking_columns_survive_a_controller_that_has_none(tmp_path):
    """A missing accessor must not take down the loop holding the 500 Hz clock."""
    path = tmp_path / "dtc.csv"
    ctrl = FakeWrenchController([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    backend, _s, _h = _wrench_backend(ctrl, log_path=str(path))
    backend.set_controller(ctrl)
    backend._log_open()
    backend.tick(now=1.0)
    backend._log_close()
    lines = _csv_lines(path)
    got = dict(zip(lines[0].split(","), lines[1].split(",")))
    assert float(got["ex"]) == 0.0
    assert got["n_deg"] == ""        # no normal published yet, not 0 degrees
    assert got["act"] == "0"


def test_no_log_path_costs_nothing(tmp_path):
    backend, _s, _h = _backend(qdot_cmd_fn=lambda: np.zeros(6), log_path="none")
    backend._log_open()
    backend.tick(now=1.0)
    assert backend._log_fh is None
    assert not list(tmp_path.iterdir())


# --- the joints tau = J.T @ W structurally cannot reach ---------------------


class _Ctrl:
    """Just enough controller for _joint_hold_tau: the input_active gate."""

    def __init__(self, active=False):
        self.input_active = active


def _hold_backend(active=False, q=None, **kw):
    backend, state, _h = _backend(**kw)
    state.q = np.zeros(6) if q is None else np.asarray(q, dtype=float)
    backend.tick(1.0)
    backend._controller = _Ctrl(active)
    return backend, state


def test_joint_hold_defaults_to_the_last_driven_joint():
    backend, _s, _h = _backend()
    assert backend._joint_hold_idx.tolist() == [5]


def test_joint_hold_is_silent_while_the_stick_is_quiet():
    """Idle re-snaps the target onto live, so the spring has nothing to say.

    Same discipline as _maybe_hold: idle is when "hold here" is a true
    statement, so it can never wind up against a slow commanded turn.
    """
    backend, state = _hold_backend(active=False, q=np.full(6, 0.3))
    assert np.allclose(backend._joint_hold_tau(np.zeros(6)), 0.0)


def test_joint_hold_opposes_drift_once_the_stick_is_live():
    """The failure this exists for: wrist 3 back-driving with tau = 0."""
    backend, state = _hold_backend(active=False, q=np.zeros(6))
    backend._joint_hold_tau(np.zeros(6))          # idle: latch the target at 0
    backend._controller.input_active = True
    state.q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
    backend.tick(1.0)
    tau = backend._joint_hold_tau(np.zeros(6))
    assert tau[5] < 0.0, "must push back toward the held angle"
    assert np.allclose(tau[:5], 0.0), "no other joint may be touched"


def test_joint_hold_yields_where_the_wrench_owns_the_joint():
    """The orthogonality that makes wrist 3 unreachable is a property of the
    current pose and field, not a theorem. If the wrench does command that
    joint, the hold must get out of the way rather than fight it."""
    backend, state = _hold_backend(active=False, q=np.zeros(6))
    backend._joint_hold_tau(np.zeros(6))
    backend._controller.input_active = True
    state.q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
    backend.tick(1.0)
    owned = np.zeros(6)
    owned[5] = 5.0                                 # the task wrench wants it
    assert np.allclose(backend._joint_hold_tau(owned), 0.0)


def test_joint_hold_never_exceeds_its_own_cap():
    backend, state = _hold_backend(active=False, q=np.zeros(6))
    backend._joint_hold_tau(np.zeros(6))
    backend._controller.input_active = True
    state.q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 100.0])
    backend.tick(1.0)
    tau = backend._joint_hold_tau(np.zeros(6))
    assert abs(tau[5]) <= backend._joint_hold_tau_max + 1e-9


def test_joint_hold_damps_with_no_target_at_all():
    """kp=0 leaves only the damper, which needs no target and so can never
    drive the arm anywhere. That is the reduction to the previous behavior."""
    backend, state = _hold_backend(active=True, q=np.zeros(6),
                                   joint_hold_kp=0.0)
    state.qdot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    backend.tick(1.0)
    tau = backend._joint_hold_tau(np.zeros(6))
    assert tau[5] < 0.0
    assert np.allclose(tau[:5], 0.0)


def test_joint_hold_disabled_by_empty_joint_list():
    backend, state = _hold_backend(active=True, q=np.full(6, 0.4),
                                   joint_hold_joints=[])
    assert np.allclose(backend._joint_hold_tau(np.zeros(6)), 0.0)


def test_joint_hold_forgets_its_target_on_a_safety_reset():
    """A pendant stop must not queue up a lunge back to a pre-stop angle."""
    backend, state = _hold_backend(active=False, q=np.zeros(6))
    backend._joint_hold_tau(np.zeros(6))
    assert backend._joint_hold_q is not None
    backend._joint_hold_reset()
    assert backend._joint_hold_q is None


def test_joint_hold_rejects_a_slot_outside_the_control_group():
    with pytest.raises(ValueError, match="outside"):
        _backend(joint_hold_joints=[9])


# --- the gain header has to record what was *asked* for, not only what ran ----
# _gain_note reads the live K and D on purpose, and that is blind to exactly one
# thing: the D ceiling. dtc_20260830_135043 (D_lin 15.08) and
# dtc_20260830_135527 (D_lin 12.00) were a --dtc-damping-ratio 0.3 -> 0.4
# experiment that came out *less* damped than it started, and neither header
# said which cap was in force.
class _GainCtrl:
    def __init__(self, k_ang=355.3, k_lin=631.7, d_ang=3.77, d_lin=15.08):
        self._stiffness = np.diag([k_ang] * 3 + [k_lin] * 3)
        self._damping = np.diag([d_ang] * 3 + [d_lin] * 3)
        self._err_clip_lin = 0.05
        self._err_clip_rot = 0.2
        self._rearm_wrench_s = 0.15


def test_the_gain_header_derives_zeta_from_the_live_matrices():
    backend, _s, _h = _backend(law_kind="wrench", use_lambda=True)
    backend._controller = _GainCtrl()
    note = backend._gain_note()
    # zeta = D / (2*sqrt(K)): 15.08 / (2*sqrt(631.7)) = 0.300
    assert "zeta_lin=0.300" in note
    # 3.77 / (2*sqrt(355.3)) = 0.100
    assert "zeta_ang=0.100" in note


def test_the_gain_header_shows_a_capped_zeta_as_the_capped_value():
    """The run whose --dtc-damping-ratio 0.4 arrived as D = 12."""
    backend, _s, _h = _backend(law_kind="wrench", use_lambda=True)
    backend._controller = _GainCtrl(d_lin=12.0)
    assert "zeta_lin=0.239" in backend._gain_note()


def test_the_gain_header_carries_the_callers_request_and_ceiling():
    backend, _s, _h = _backend(
        law_kind="wrench", use_lambda=True,
        gain_note_extra="zeta_req_lin=0.4  dmax=19.2")
    backend._controller = _GainCtrl(d_lin=12.0)
    note = backend._gain_note()
    assert "zeta_req_lin=0.4" in note and "dmax=19.2" in note
    # ...next to what actually ran, so the gap is visible in one line.
    assert "zeta_lin=0.239" in note


def test_the_gain_header_survives_a_controller_with_no_matrices():
    backend, _s, _h = _backend(law_kind="wrench")
    backend._controller = object()
    assert "law=wrench" in backend._gain_note()


# --- the cut-out for the failure tau_max cannot bound ------------------------
# dtc_20260830_141034.csv: with the stick quiet and the pose error at 0.0 mm,
# tau alternated +50/-50 N.m on j1 AND j2 at 32.1 Hz and the arm swung into the
# workstation. err_clip bounds the spring half of the wrench; the damping half
# -Lambda*D*v has no clip at all, so tau_max was the only thing bounding the
# amplitude -- which means the cap was setting how hard the arm got thrown.
_TMAX = np.array([50.0, 50.0, 50.0, 6.0, 6.0, 30.0])


def _run_guard(guard, taus, hz=200.0):
    """Feed a sequence of tau vectors at a fixed rate. Returns (trips, floated)."""
    floated = 0
    for k, tau in enumerate(taus):
        f, _ = guard.update(k / hz, np.asarray(tau, float), _TMAX)
        floated += bool(f)
    return guard.n_trips, floated


def test_rail_to_rail_alternation_trips_the_guard():
    g = RailOscillationGuard(6)
    # j1 slamming +50/-50 every other tick, 32 Hz-ish at 200 Hz.
    taus = []
    for k in range(60):
        t = np.zeros(6)
        t[1] = 50.0 if (k // 3) % 2 == 0 else -50.0
        taus.append(t)
    trips, floated = _run_guard(g, taus)
    self_j = g.last_joint
    assert trips >= 1
    assert self_j == 1
    assert floated > 0


def test_a_stall_on_one_sign_is_never_cut():
    """A joint held against breakaway friction sits on its rail, correctly."""
    g = RailOscillationGuard(6)
    taus = [np.array([0.0, 0.0, 0.0, 6.0, 0.0, 0.0])] * 4000   # 20 s at the rail
    trips, floated = _run_guard(g, taus)
    assert trips == 0
    assert floated == 0


def test_slow_reversals_are_not_a_limit_cycle():
    """Ordinary stick-slip reverses at the rail, just not 12 times in 0.3 s."""
    g = RailOscillationGuard(6)
    taus = []
    for k in range(2000):           # a flip every 0.25 s
        t = np.zeros(6)
        t[3] = 6.0 if (k // 50) % 2 == 0 else -6.0
        taus.append(t)
    trips, _ = _run_guard(g, taus)
    assert trips == 0


def test_the_float_lasts_hold_s_and_then_re_arms():
    g = RailOscillationGuard(6, flips=4, hold_s=0.5)
    trip_t = None
    for k in range(20):
        tau = np.array([0.0, 50.0 if k % 2 else -50.0, 0.0, 0.0, 0.0, 0.0])
        _, edge = g.update(k / 200.0, tau, _TMAX)
        if edge and trip_t is None:
            trip_t = k / 200.0
    assert g.n_trips == 1 and trip_t is not None
    # Still floating a tick before the hold expires, armed again just after.
    assert g.update(trip_t + 0.49, np.zeros(6), _TMAX)[0] is True
    assert g.update(trip_t + 0.51, np.zeros(6), _TMAX)[0] is False


def test_zero_flips_disables_the_guard():
    g = RailOscillationGuard(6, flips=0)
    taus = [np.array([0.0, 50.0 if k % 2 else -50.0, 0.0, 0.0, 0.0, 0.0])
            for k in range(200)]
    trips, floated = _run_guard(g, taus)
    assert not g.enabled
    assert trips == 0 and floated == 0


def test_reset_forgets_the_count():
    g = RailOscillationGuard(6, flips=8)
    for k in range(6):
        g.update(k / 200.0,
                 np.array([0.0, 50.0 if k % 2 else -50.0, 0, 0, 0, 0]), _TMAX)
    g.reset()
    for k in range(6, 12):
        g.update(k / 200.0,
                 np.array([0.0, 50.0 if k % 2 else -50.0, 0, 0, 0, 0]), _TMAX)
    assert g.n_trips == 0


class RailGuardReplayTest(unittest.TestCase):
    """The separation is empirical, so replay the actual logs through it.

    Healthy runs on this rig top out at 9 rail sign flips in 0.3 s and only on
    the 6 N.m wrist joints. The two incidents are nothing like it:
    dtc_20260830_141034.csv reads 20/11/21/18 on j1-j4 at once (including the
    50 N.m shoulder and elbow, which flip zero times in every other run) and
    dtc_20260830_005006.csv, the ROBOT_EMERGENCY_STOP run, reads 16 on j4.
    """

    LOGS = pathlib.Path(__file__).resolve().parents[1] / "speedj_logs"
    QUIET = ["dtc_20260830_002254.csv", "dtc_20260830_115039.csv",
             "dtc_20260830_122304.csv", "dtc_20260830_130541.csv",
             "dtc_20260830_132204.csv", "dtc_20260830_134133.csv",
             "dtc_20260830_135043.csv"]
    INCIDENTS = ["dtc_20260830_141034.csv", "dtc_20260830_005006.csv"]

    def _replay(self, name):
        path = self.LOGS / name
        if not path.exists():
            self.skipTest(f"{name} not in the tree")
        first = path.open().readline()
        skip = 1 if first.startswith("#") else 0
        d = np.genfromtxt(path, delimiter=",", names=True, skip_header=skip)
        tmax = _TMAX
        if skip and "tau_max=" in first:
            tmax = np.array([float(x) for x in
                             first.split("tau_max=")[1].split()[0].split(",")])
        t = d["t"]
        tau = np.stack([d[f"tau{i}"] for i in range(6)], 1)
        g = RailOscillationGuard(6)
        for k in range(t.size):
            g.update(float(t[k]), tau[k], tmax)
        return g

    def test_no_healthy_run_is_cut_out(self):
        for name in self.QUIET:
            with self.subTest(log=name):
                self.assertEqual(self._replay(name).n_trips, 0)

    def test_both_incidents_are_cut_out(self):
        for name in self.INCIDENTS:
            with self.subTest(log=name):
                self.assertGreater(self._replay(name).n_trips, 0)

    def test_it_fires_before_the_worst_of_it(self):
        """On the workstation run the trip lands 0.56 s before the 167 N peak."""
        path = self.LOGS / "dtc_20260830_141034.csv"
        if not path.exists():
            self.skipTest("log not in the tree")
        first = path.open().readline()
        d = np.genfromtxt(path, delimiter=",", names=True, skip_header=1)
        tmax = np.array([float(x) for x in
                         first.split("tau_max=")[1].split()[0].split(",")])
        t = d["t"]
        tau = np.stack([d[f"tau{i}"] for i in range(6)], 1)
        w = np.stack([d[f"w{i}"] for i in range(6)], 1)
        wl = np.linalg.norm(w[:, 3:], axis=1)
        g = RailOscillationGuard(6)
        trip_t = None
        for k in range(t.size):
            _, edge = g.update(float(t[k]), tau[k], tmax)
            if edge and trip_t is None:
                trip_t = float(t[k])
        self.assertIsNotNone(trip_t)
        self.assertLess(trip_t, float(t[int(np.argmax(wl))]))


def test_the_loop_floats_tau_when_a_joint_oscillates_at_its_rail(tmp_path):
    """End to end: rail-to-rail alternation reaches the guard and zeroes tau.

    Driven through the velocity law because it is the one that takes a plain
    qdot command, but the guard sits below both laws on the clipped tau, which
    is the only place the amplitude of a limit cycle is visible at all.
    """
    path = tmp_path / "dtc.csv"
    flip = {"k": 0}

    def cmd():
        flip["k"] += 1
        # kv * this is far past tau_max either way, so tau sits on the rail and
        # reverses every tick.
        return np.full(6, 100.0 if flip["k"] % 2 else -100.0)

    backend, _s, _h = _backend(qdot_cmd_fn=cmd, log_path=str(path),
                               rail_osc_flips=6, rail_osc_hold_s=0.5)
    backend._log_open()
    for k in range(20):
        backend.tick(now=1.0 + k * 0.005)
    backend._log_close()

    rows = _csv_lines(path)[1:]
    phases = [r.split(",")[2] for r in rows]
    assert "railosc" in phases
    assert backend.stats()["n_rail_osc"] >= 1
    # Every railosc tick commands exactly zero on every joint.
    for r in rows:
        f = r.split(",")
        if f[2] == "railosc":
            assert all(float(x) == 0.0 for x in f[16:22])


def test_an_honest_stall_reaches_the_rail_and_is_left_alone(tmp_path):
    """The same drive without the sign reversal must never be cut."""
    path = tmp_path / "dtc.csv"
    backend, _s, _h = _backend(qdot_cmd_fn=lambda: np.full(6, 100.0),
                               log_path=str(path), rail_osc_flips=6)
    backend._log_open()
    for k in range(200):
        backend.tick(now=1.0 + k * 0.005)
    backend._log_close()
    phases = [r.split(",")[2] for r in _csv_lines(path)[1:]]
    assert "railosc" not in phases
    assert backend.stats()["n_rail_osc"] == 0


def test_hold_zero_never_cuts_out(tmp_path):
    """Under the bring-up gate tau is 0 anyway; the guard must not fire."""
    backend, _s, _h = _backend(qdot_cmd_fn=lambda: np.full(6, 100.0),
                               hold_zero=True, rail_osc_flips=2)
    for k in range(50):
        backend.tick(now=1.0 + k * 0.005)
    assert backend.stats()["n_rail_osc"] == 0


# --- stiction breakaway assist ----------------------------------------------
# Three runs on 2026-08-30 sat wrist 3 at the Lambda floor's full 1.42 N.m
# ceiling and moved it 0.02-0.49 deg end to end -- stalled on 99.7-100% of the
# spring-on ticks -- so the roll error stayed open and leaked onto the other
# angular blades through Lambda's cross terms. The assist is the feedforward
# answer to that, and what makes it a friction compensator rather than a second
# controller is entirely in these conditions.


def _assist(backend, tau_task, qd5=0.0, ticks=60, sleep_s=0.005):
    """Run the assist to steady state, since it ramps rather than stepping."""
    qd = np.zeros(backend.n_joints)
    qd[-1] = qd5
    out = []
    for _ in range(ticks):
        backend._qd6 = qd.copy()
        out.append(backend._stiction_tau(tau_task)[-1])
        time.sleep(sleep_s)
    return out


def _task_tau(backend, nm):
    tau = np.zeros(backend.n_joints)
    tau[-1] = nm
    return tau


def test_assist_ramps_in_on_a_stalled_commanded_joint():
    backend, _s, _h = _backend()
    out = _assist(backend, _task_tau(backend, 1.42))
    assert out[0] == 0.0, "must not step in on the first tick"
    assert all(b >= a - 1e-12 for a, b in zip(out, out[1:])), "must be monotone"
    # Capped by the room under stiction_max_nm (2.5 - 1.42 = 1.08), not by it.
    assert 0.9 < out[-1] <= 1.0 + 1e-9


def test_assist_releases_once_the_joint_breaks_free():
    """The whole safety story: it pushes a stalled joint, never a moving one."""
    backend, _s, _h = _backend()
    _assist(backend, _task_tau(backend, 1.42))
    out = _assist(backend, _task_tau(backend, 1.42), qd5=0.5)
    assert out[-1] == 0.0


def test_assist_takes_its_direction_from_the_task_torque():
    """It has no direction of its own, so it cannot fight the field."""
    backend, _s, _h = _backend()
    assert _assist(backend, _task_tau(backend, -1.42))[-1] < 0.0


def test_assist_caps_the_total_not_its_own_share():
    backend, _s, _h = _backend()
    out = _assist(backend, _task_tau(backend, 2.4))
    assert abs(2.4 + out[-1]) <= 2.5 + 1e-9


def test_assist_stays_out_when_the_task_is_not_asking():
    """Below the deadband there is nothing to break away to."""
    backend, _s, _h = _backend()
    assert _assist(backend, _task_tau(backend, 0.1))[-1] == 0.0


def test_assist_off_restores_the_previous_behavior_exactly():
    backend, _s, _h = _backend(stiction_assist_nm=0.0)
    assert _assist(backend, _task_tau(backend, 1.42))[-1] == 0.0


def test_assist_is_forgotten_on_a_reset():
    """A pendant stop must stop the push, not queue one up."""
    backend, _s, _h = _backend()
    assert _assist(backend, _task_tau(backend, 1.42))[-1] > 0.0
    backend._joint_hold_reset()
    assert backend._stiction_tau(_task_tau(backend, 1.42))[-1] == 0.0


def test_log_header_records_the_command_line_and_the_joint_space_gains(tmp_path):
    """The header is the lab notebook: what was asked for AND what ran.

    A --dtc-joint-hold-kp sweep was run against a hold that
    DEFAULT_JOINT_HOLD_YIELD_NM had already disconnected, and nothing in the
    CSV distinguished those runs from each other.
    """
    log = tmp_path / "dtc.csv"
    backend, _s, _h = _backend(log_path=log, joint_hold_kp=4.0)
    backend._log_open()
    backend._log_close()
    head = log.read_text().splitlines()
    assert head[0].startswith("# argv: ")
    assert "jhold_kp=4" in head[1]
    assert "stiction_nm=" in head[1]


def test_float_now_latches_zero_tau_without_tearing_down():
    """The safe state under direct torque is float; teardown must reach it first.

    Observed on hardware: the wrist swung after Ctrl+C, because the control
    loop keeps driving at full authority through the whole teardown -- the
    device, the follow helper, the contact thread and the force socket all
    close before interface.shutdown() zeroes tau -- toward a carrot the main
    loop stopped updating the instant it exited.
    """
    backend, _state, _h = _backend(law_kind="wrench")
    assert backend.hold_zero is False
    streamer = FakeStreamer(object())     # as if start() had opened one
    backend._streamer = streamer
    backend.float_now()
    assert backend.hold_zero is True
    assert streamer.taus and np.allclose(streamer.taus[-1], 0.0)


def test_float_now_forgets_the_latches_a_restart_would_reopen_with():
    backend, _s, _h = _backend(
        law_kind="wrench", hold_at_tick=False, hold_on_release=True)
    ctrl = FakeHoldController()
    assert backend._maybe_hold(ctrl, np.zeros(7)) is True
    assert backend._maybe_hold(ctrl, np.zeros(7)) is False
    backend.float_now()
    assert backend._maybe_hold(ctrl, np.ones(7)) is True
