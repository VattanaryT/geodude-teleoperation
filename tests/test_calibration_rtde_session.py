"""Tests for RTDE session wrapper (fake ur_rtde interfaces)."""

from __future__ import annotations

import sys
import types

import pytest

from geodude_teleoperation.calibration import rtde_session as rs


class FakeControl:
    def __init__(self, ip: str):
        self.ip = ip
        self.zeroed = False

    def zeroFtSensor(self):
        self.zeroed = True

    def disconnect(self):
        pass


class FakeReceive:
    def __init__(self, ip: str):
        self.ip = ip
        self.calls = 0

    def getActualTCPForce(self):
        self.calls += 1
        return [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]

    def disconnect(self):
        pass


@pytest.fixture
def fake_ur_rtde(monkeypatch):
    control_mod = types.ModuleType("rtde_control")
    control_mod.RTDEControlInterface = FakeControl
    receive_mod = types.ModuleType("rtde_receive")
    receive_mod.RTDEReceiveInterface = FakeReceive
    monkeypatch.setitem(sys.modules, "rtde_control", control_mod)
    monkeypatch.setitem(sys.modules, "rtde_receive", receive_mod)
    return FakeControl, FakeReceive


def test_zero_ft(fake_ur_rtde):
    with rs.RtdeSession("192.168.1.20") as session:
        session.zero_ft_sensor()
        assert session._control.zeroed is True


def test_average_tcp_force(fake_ur_rtde, monkeypatch):
    monkeypatch.setattr(rs.time, "sleep", lambda _s: None)
    with rs.RtdeSession("192.168.1.20") as session:
        wrench = session.average_tcp_force(samples=5, settle_s=0.0)
        assert wrench == [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
        assert session._receive.calls == 5


def test_missing_ur_rtde(monkeypatch):
    monkeypatch.setattr(
        rs,
        "_import_rtde",
        lambda: (_ for _ in ()).throw(
            ImportError("ur_rtde is required; pip install 'geodude-teleoperation[ur]'")
        ),
    )
    with pytest.raises(ImportError, match="ur_rtde"):
        with rs.RtdeSession("127.0.0.1"):
            pass
