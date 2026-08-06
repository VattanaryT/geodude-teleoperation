"""Tests for FT section builders and apply helpers."""

from __future__ import annotations

from geodude_teleoperation.calibration import ft_sensor


class RecordingSession:
    def __init__(self):
        self.zeroed = False
        self.samples = None

    def zero_ft_sensor(self):
        self.zeroed = True

    def average_tcp_force(self, samples, settle_s=0.01):
        self.samples = samples
        return [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]


def test_build_ft_section():
    section = ft_sensor.build_ft_section([1, 2, 3, 4, 5, 6], sample_count=10, note="still")
    assert section["bias_wrench"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert section["sample_count"] == 10
    assert section["note"] == "still"
    assert "zeroed_at" in section


def test_apply_helpers():
    session = RecordingSession()
    ft_sensor.apply_ft_zero(session)
    bias = ft_sensor.sample_ft_bias(session, samples=7)
    assert session.zeroed is True
    assert session.samples == 7
    assert bias[0] == 0.1
