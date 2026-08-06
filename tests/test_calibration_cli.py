"""CLI smoke tests (no real robot)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from geodude_teleoperation.calibration.cli import main
from geodude_teleoperation.calibration.io import load_arm_calib, save_arm_calib


def test_cli_tcp_writes_yaml(tmp_path: Path):
    rc = main(
        [
            "tcp",
            "--arm",
            "right",
            "--pose",
            "0,0,0.15,0,0,0",
            "--calib-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    data = load_arm_calib(tmp_path / "right.yaml")
    assert data["tcp"]["pose"][2] == 0.15
    assert data["arm"] == "right"


def test_cli_payload_merges(tmp_path: Path):
    assert (
        main(
            [
                "tcp",
                "--arm",
                "left",
                "--pose",
                "0,0,0.1,0,0,0",
                "--calib-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "payload",
                "--arm",
                "left",
                "--mass",
                "0.98",
                "--cog",
                "0.01,0,0.05",
                "--calib-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    data = load_arm_calib(tmp_path / "left.yaml")
    assert data["tcp"]["pose"][2] == 0.1
    assert data["payload"]["mass_kg"] == 0.98


def test_cli_show(tmp_path: Path, capsys):
    save_arm_calib(tmp_path / "right.yaml", {"arm": "right", "tcp": {"pose": [0.0] * 6}})
    rc = main(["show", "--arm", "right", "--calib-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "tcp:" in out


def test_cli_apply_missing_section(tmp_path: Path):
    save_arm_calib(tmp_path / "left.yaml", {"arm": "left"})
    rc = main(
        [
            "apply",
            "--arm",
            "left",
            "--what",
            "tcp",
            "--calib-dir",
            str(tmp_path),
            "--yes",
            "--ip",
            "127.0.0.1",
        ]
    )
    assert rc == 2


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


def test_cli_tcp_apply_with_fake_rtde(tmp_path: Path, monkeypatch):
    class FakeC:
        def __init__(self, ip):
            self.ip = ip
            self.tcp = None

        def setTcp(self, pose):
            self.tcp = list(pose)

        def disconnect(self):
            pass

    class FakeR:
        def __init__(self, ip):
            self.ip = ip

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
            "tcp",
            "--arm",
            "right",
            "--pose",
            "0,0,0.2,0,0,0",
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


def test_cli_bad_pose_exit_2(tmp_path: Path):
    rc = main(
        [
            "tcp",
            "--arm",
            "right",
            "--pose",
            "1,2,3",
            "--calib-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2
    assert not (tmp_path / "right.yaml").exists()
