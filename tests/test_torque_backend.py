"""The velocity->torque law and the DTC seam.

Pure numpy: no gafropy, no ur_rtde, no hardware. The backend's model, RTDE
clients and tau streamer are all injected, so the control-group splice, the
staleness bound and the safety gate are exercised here without a robot.
"""
from __future__ import annotations

import numpy as np
import pytest

from geodude_teleoperation.torque_backend import (
    DEFAULT_KV,
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
