"""DTC host path: 500 Hz warning, PolyScope gate, streamer zeros on stop."""
from __future__ import annotations

import time

from geodude_teleoperation.teleop.direct_torque import (
    parse_polyscope_version,
    polyscope_supports_dtc,
    verify_control_hz,
    verify_direct_torque_rate,
)


def test_control_hz_50_warns_not_500():
    r = verify_control_hz(50.0)
    assert r.ok is False
    assert any("500" in w for w in r.warnings)


def test_polyscope_dtc_versions():
    assert polyscope_supports_dtc("5.23.0")
    assert not polyscope_supports_dtc("5.22.0")
    assert polyscope_supports_dtc("10.10.0")
    assert not polyscope_supports_dtc("10.9.0")
    assert parse_polyscope_version("v5.23.0-beta") == (5, 23, 0)


class _FakeControl:
    def __init__(self):
        self.n = 0
        self.calls: list[list[float]] = []
        self.stopped = False

    def directTorque(self, tau, friction_comp=True):
        # Simulate robot ready-wait (~500 Hz).
        time.sleep(0.002)
        self.n += 1
        self.calls.append(list(tau))
        _ = friction_comp
        return True

    def speedStop(self):
        self.stopped = True


def test_host_dtc_streamer_latches_and_zeros_on_stop():
    from geodude_teleoperation.teleop.direct_torque import HostDirectTorqueStreamer

    # Legacy host-500 path (no on-robot latch): fake ready-wait ≈ 500 Hz.
    ctrl = _FakeControl()
    streamer = HostDirectTorqueStreamer(
        ctrl, target_hz=500.0, robot_latch=False
    )
    streamer.set_tau([0.1, 0, 0, 0, 0, 0])
    streamer.start()
    time.sleep(0.15)
    st = streamer.stats()
    streamer.stop(zero=True)
    assert st["n_ok"] >= 30  # ~500 Hz × 0.15 s with 2 ms fake RTT
    assert st["measured_hz"] == st["measured_hz"] and st["measured_hz"] > 0.0
    assert st["measured_hz"] > 200.0
    assert st["robot_latch"] is False
    assert ctrl.stopped is True


def test_host_dtc_streamer_robot_latch_publishes_at_target_hz():
    from geodude_teleoperation.teleop.direct_torque import HostDirectTorqueStreamer

    class FastCtrl:
        def __init__(self):
            self.n = 0
            self.stopped = False

        def directTorque(self, tau, friction_comp=True):
            self.n += 1
            _ = tau, friction_comp
            return True

        def speedStop(self):
            self.stopped = True

    ctrl = FastCtrl()
    streamer = HostDirectTorqueStreamer(
        ctrl, target_hz=50.0, robot_latch=True
    )
    streamer.set_tau([0.2, 0, 0, 0, 0, 0])
    streamer.start()
    time.sleep(0.25)
    st = streamer.stats()
    streamer.stop(zero=True)
    assert st["robot_latch"] is True
    assert 8 <= st["n_ok"] <= 25  # ~50 Hz × 0.25 s
    assert ctrl.stopped is True


def test_verify_direct_torque_rate_missing_api():
    r = verify_direct_torque_rate(object(), duration_s=0.1)
    assert r.commanded is False
    assert r.ok is False
