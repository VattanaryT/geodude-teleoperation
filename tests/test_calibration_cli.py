"""CLI smoke tests (no real robot)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from geodude_teleoperation.calibration.cli import main
from geodude_teleoperation.calibration.io import load_arm_calib, save_arm_calib


def test_cli_show(tmp_path: Path, capsys):
    save_arm_calib(
        tmp_path / "right.yaml",
        {"arm": "right", "tcp": {"pose": [0.0] * 6}, "payload": {"mass_kg": 0.49}},
    )
    rc = main(["show", "--arm", "right", "--calib-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "tcp:" in out
    assert "payload:" in out


def test_cli_ft_bias_offline(tmp_path: Path):
    rc = main(
        [
            "ft",
            "--arm",
            "right",
            "--bias",
            "0.1,0.2,0.3,0,0,0",
            "--samples",
            "5",
            "--calib-dir",
            str(tmp_path),
            "--ip",
            "10.0.0.1",
        ]
    )
    assert rc == 0
    data = load_arm_calib(tmp_path / "right.yaml")
    assert data["ft_sensor"]["bias_wrench"][0] == 0.1
    assert data["robot_ip"] == "10.0.0.1"


def test_cli_ft_apply_with_fake_rtde(tmp_path: Path, monkeypatch):
    class FakeC:
        def __init__(self, ip):
            self.ip = ip
            self.zeroed = False

        def zeroFtSensor(self):
            self.zeroed = True

        def disconnect(self):
            pass

    class FakeR:
        def __init__(self, ip):
            self.ip = ip

        def getActualTCPForce(self):
            return [0.0] * 6

        def disconnect(self):
            pass

    control_mod = types.ModuleType("rtde_control")
    control_mod.RTDEControlInterface = FakeC
    receive_mod = types.ModuleType("rtde_receive")
    receive_mod.RTDEReceiveInterface = FakeR
    monkeypatch.setitem(sys.modules, "rtde_control", control_mod)
    monkeypatch.setitem(sys.modules, "rtde_receive", receive_mod)

    rc = main(
        [
            "ft",
            "--arm",
            "right",
            "--bias",
            "0,0,0,0,0,0",
            "--calib-dir",
            str(tmp_path),
            "--ip",
            "192.168.0.9",
            "--apply",
            "--yes",
        ]
    )
    assert rc == 0
    assert (tmp_path / "right.yaml").is_file()


def test_cli_tcp_subcommand_removed():
    with pytest.raises(SystemExit) as exc:
        main(["tcp", "--arm", "right", "--pose", "0,0,0,0,0,0"])
    assert exc.value.code == 2
