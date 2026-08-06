"""Tests for tcp/payload/ft section builders and apply helpers."""

from __future__ import annotations

from geodude_teleoperation.calibration import ft_sensor, payload, tcp


class RecordingSession:
    def __init__(self):
        self.tcp = None
        self.payload = None
        self.zeroed = False
        self.samples = None

    def set_tcp(self, pose):
        self.tcp = list(pose)

    def set_payload(self, mass, cog):
        self.payload = (mass, list(cog))

    def zero_ft_sensor(self):
        self.zeroed = True

    def average_tcp_force(self, samples, settle_s=0.01):
        self.samples = samples
        return [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]


def test_build_tcp_section():
    section = tcp.build_tcp_section([0, 0, 0.15, 0, 0, 0])
    assert section["pose"][2] == 0.15
    assert section["frame"] == "tool0"
    assert section["source"] == "manual"


def test_build_payload_section():
    section = payload.build_payload_section(0.98, [0.01, 0.0, 0.05])
    assert section["mass_kg"] == 0.98
    assert section["cog"] == [0.01, 0.0, 0.05]


def test_build_ft_section():
    section = ft_sensor.build_ft_section([1, 2, 3, 4, 5, 6], sample_count=10, note="still")
    assert section["bias_wrench"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert section["sample_count"] == 10
    assert section["note"] == "still"
    assert "zeroed_at" in section


def test_apply_helpers():
    session = RecordingSession()
    tcp.apply_tcp(session, [0, 0, 0.2, 0, 0, 0])
    payload.apply_payload(session, 1.5, [0.0, 0.0, 0.1])
    ft_sensor.apply_ft_zero(session)
    bias = ft_sensor.sample_ft_bias(session, samples=7)
    assert session.tcp[2] == 0.2
    assert session.payload == (1.5, [0.0, 0.0, 0.1])
    assert session.zeroed is True
    assert session.samples == 7
    assert bias[0] == 0.1
