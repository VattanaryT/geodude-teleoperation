"""Tests for RTDE teleop session and host loop (fake ur_rtde interfaces)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

import geodude_teleoperation.rtde_control_loop as rcl


class FakeControl:
    instances: list["FakeControl"] = []

    def __init__(self, ip: str):
        self.ip = ip
        self.speedj_calls: list[tuple[list[float], float, float]] = []
        FakeControl.instances.append(self)

    def speedJ(self, qdot, accel, dt):
        self.speedj_calls.append((list(qdot), float(accel), float(dt)))

    def disconnect(self):
        pass


class FakeReceive:
    instances: list["FakeReceive"] = []

    def __init__(self, ip: str):
        self.ip = ip
        self._q = [0.0] * 6
        self._tcp = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
        self._fault = False
        FakeReceive.instances.append(self)

    def getActualQ(self):
        return list(self._q)

    def getActualTCPPose(self):
        return list(self._tcp)

    def getRobotMode(self):
        return 7 if not self._fault else 4

    def isProtectiveStopped(self):
        return False

    def isEmergencyStopped(self):
        return False

    def disconnect(self):
        pass


@pytest.fixture
def fake_ur_rtde(monkeypatch):
    FakeControl.instances.clear()
    FakeReceive.instances.clear()
    control_mod = types.ModuleType("rtde_control")
    control_mod.RTDEControlInterface = FakeControl
    receive_mod = types.ModuleType("rtde_receive")
    receive_mod.RTDEReceiveInterface = FakeReceive
    monkeypatch.setitem(sys.modules, "rtde_control", control_mod)
    monkeypatch.setitem(sys.modules, "rtde_receive", receive_mod)
    return FakeControl, FakeReceive


def test_dry_run_opens_receive_only(fake_ur_rtde):
    FakeControl, FakeReceive = fake_ur_rtde
    session = rcl.open("192.168.1.20", dry_run=True)
    try:
        assert len(FakeReceive.instances) == 1
        assert len(FakeControl.instances) == 0
        assert session.control_open is False
        state = session.read_state()
        assert "q" in state and "tcp_pose" in state and "fault" in state
        with pytest.raises(RuntimeError, match="dry-run"):
            session.speedj([0.1] * 6, 0.5)
    finally:
        session.close()


def test_live_opens_control_and_receive(fake_ur_rtde):
    FakeControl, FakeReceive = fake_ur_rtde
    session = rcl.open("192.168.1.20", dry_run=False)
    try:
        assert len(FakeReceive.instances) == 1
        assert len(FakeControl.instances) == 1
        assert session.control_open is True
        session.speedj(np.zeros(6), 0.5)
        assert FakeControl.instances[0].speedj_calls
    finally:
        session.close()


def test_close_dry_run_no_speedj(fake_ur_rtde):
    FakeControl, FakeReceive = fake_ur_rtde
    session = rcl.open("192.168.1.20", dry_run=True)
    session.close()
    assert len(FakeControl.instances) == 0


def test_close_live_zeros_then_disconnects(fake_ur_rtde):
    FakeControl, FakeReceive = fake_ur_rtde
    session = rcl.open("192.168.1.20", dry_run=False)
    session.close()
    calls = FakeControl.instances[0].speedj_calls
    assert calls
    assert np.allclose(calls[-1][0], [0.0] * 6)


def test_watchdog_zeros(fake_ur_rtde, monkeypatch):
    FakeControl, FakeReceive = fake_ur_rtde
    session = rcl.open("192.168.1.20", dry_run=False)

    times = iter([0.0, 0.0, 0.6])
    monkeypatch.setattr(rcl.time, "monotonic", lambda: next(times, 0.6))

    assert rcl.watchdog_is_stale(0.0, 0.6, watchdog_s=0.5) is True
    assert rcl.watchdog_is_stale(0.55, 0.6, watchdog_s=0.5) is False

    qdot = rcl.apply_watchdog(np.ones(6), last_command_time=0.0, now=0.6, watchdog_s=0.5)
    assert np.allclose(qdot, 0.0)

    session.close()


def test_resolve_T_fixed_identity():
    args = MagicMock()
    args.base_frame = None
    args.tag0_frame = None
    args.T_tag0_base = None
    T, tf_ok, tf_age_ms = rcl.resolve_T_tag0_base(args)
    assert np.allclose(T, np.eye(4))
    assert tf_ok is True
    assert tf_age_ms == 0.0


def test_resolve_T_live_tf_unavailable():
    rcl._live_tf_fallback_warned = False
    rcl._ros_tf_unavailable_warned = False
    args = MagicMock()
    args.base_frame = "base"
    args.tag0_frame = "tag_0"
    args.T_tag0_base = None
    T, tf_ok, tf_age_ms = rcl.resolve_T_tag0_base(args)
    assert np.allclose(T, np.eye(4))
    assert tf_ok is True
    assert np.isnan(tf_age_ms)


def test_ee_motor_from_tcp_pose():
    motor = rcl.ee_motor_from_tcp_pose([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    t = motor.get_translator()
    assert t.x() == pytest.approx(0.1)
    assert t.y() == pytest.approx(0.2)
    assert t.z() == pytest.approx(0.3)
