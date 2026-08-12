"""Phase 2 DTC helpers + Phase 3 tactile impedance tests."""
from __future__ import annotations

import numpy as np

import time

from geodude_teleoperation.teleop.direct_torque import (
    DirectTorqueBridge,
    parse_polyscope_version,
    polyscope_supports_dtc,
    verify_control_hz,
    verify_direct_torque_rate,
)
from geodude_teleoperation.teleop.tactile_impedance import (
    TactileAgent,
    TactileConfig,
    desired_wrench_along_normal,
    joint_torques_from_wrench,
)


def test_verify_control_hz_50_warns_not_500():
    r = verify_control_hz(50.0)
    assert r.ok is False
    assert any("500" in w for w in r.warnings)


def test_verify_control_hz_500_ok():
    r = verify_control_hz(500.0)
    assert r.ok is True


def test_polyscope_dtc_versions():
    assert polyscope_supports_dtc("5.23.0")
    assert polyscope_supports_dtc("5.24.1")
    assert not polyscope_supports_dtc("5.22.0")
    assert polyscope_supports_dtc("10.10.0")
    assert not polyscope_supports_dtc("10.9.0")
    assert parse_polyscope_version("v5.23.0-beta") == (5, 23, 0)


def test_direct_torque_bridge_latches():
    b = DirectTorqueBridge()
    tau = b.publish_torques([0.1] * 6)
    assert np.allclose(tau, 0.1)
    assert "direct_torque" in b.urscript_snippet()


class _FakeControl:
    def __init__(self, *, fail_after: int | None = None):
        self.fail_after = fail_after
        self.n = 0
        self.calls: list[list[float]] = []
        self.stopped = False

    def initPeriod(self):
        return time.perf_counter()

    def waitPeriod(self, t_mark):
        _ = t_mark
        time.sleep(0.002)

    def directTorque(self, tau, friction_comp=True):
        self.n += 1
        self.calls.append(list(tau))
        if self.fail_after is not None and self.n >= self.fail_after:
            return False
        return True

    def speedStop(self):
        self.stopped = True


def test_verify_direct_torque_rate_mock_ok():
    ctrl = _FakeControl()
    r = verify_direct_torque_rate(
        ctrl, duration_s=0.05, target_hz=500.0, hz_tol=150.0
    )
    assert r.commanded is True
    assert r.n_calls >= 10
    assert ctrl.stopped is True
    assert np.allclose(ctrl.calls[0], 0.0)
    assert r.ok is True


def test_verify_direct_torque_rate_missing_api():
    r = verify_direct_torque_rate(object(), duration_s=0.1)
    assert r.commanded is False
    assert r.ok is False


def test_joint_torques_from_wrench():
    J = np.eye(6)
    W = np.array([1.0, 0, 0, 0, 0, 0])
    tau = joint_torques_from_wrench(J, W)
    assert np.allclose(tau, [-1.0, 0, 0, 0, 0, 0])


def test_desired_wrench_compresses_along_approach():
    R = np.eye(3)
    R[:, 2] = [0.0, 0.0, -1.0]  # into -z ⇒ outward +z
    W = desired_wrench_along_normal(R, 5.0, compress=True)
    # compress into surface = −n_out = −z
    assert W[2] < 0
    assert abs(abs(W[2]) - 5.0) < 1e-12


def test_tactile_agent_speedj_vn():
    cfg = TactileConfig(enabled=True, desired_normal_force_n=5.0, wrench_to_vn_gain=0.01)
    agent = TactileAgent(cfg)
    R = np.eye(3)
    R[:, 2] = [0.0, 0.0, -1.0]
    # Measured force too low along compress direction → move into surface
    W_meas = np.zeros(6)
    out = agent.update(
        wrench_meas=W_meas, R_field=R, jacobian=np.eye(6), dt=0.02, mode="speedj"
    )
    assert float(np.linalg.norm(out["vn_world"])) > 0
