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


def test_cli_tag_extrinsics(tmp_path: Path):
    from geodude_teleoperation.calibration import tag_extrinsics as te

    def _T(t=(0.0, 0.0, 0.0)):
        M = __import__("numpy").eye(4)
        M[:3, 3] = t
        return M

    cam = "wrist_cam_0"
    T_cam = _T((0.05, 0.0, 0.08))
    T_tag_base = _T((0.4, 0.1, 0.05))
    session = tmp_path / "session"
    for i in range(2):
        T_EE = _T((0.3, 0.01 * i, 0.2))
        T_tag_cam = te.invert_SE3(T_EE @ T_cam) @ T_tag_base
        cam_dir = session / "marked" / f"{i:04d}" / cam
        cam_dir.mkdir(parents=True)
        import json

        meta = {
            "T_EE_base": T_EE.tolist(),
            "T_cam_EE": T_cam.tolist(),
            "T_cam_base": (T_EE @ T_cam).tolist(),
        }
        tags = {
            "tags": [
                {
                    "id": 0,
                    "T_tag_cam": T_tag_cam.tolist(),
                    "T_tag_base": te.compose_T_tag_base(T_EE, T_cam, T_tag_cam).tolist(),
                }
            ]
        }
        (cam_dir / "meta.json").write_text(json.dumps(meta))
        (cam_dir / "tags.json").write_text(json.dumps(tags))

    calib = tmp_path / "calib"
    rc = main(
        [
            "tag-extrinsics",
            "--arm",
            "right",
            "--session",
            str(session),
            "--cam",
            cam,
            "--tag-id",
            "0",
            "--mode",
            "multiview",
            "--calib-dir",
            str(calib),
            "--max-trans",
            "0.001",
            "--max-rot",
            "1.0",
        ]
    )
    assert rc == 0
    data = load_arm_calib(calib / "right.yaml")
    assert data["tag_extrinsics"]["passed"] is True
