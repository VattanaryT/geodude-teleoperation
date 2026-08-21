"""CLI / SSoT session gates that actually block a run."""

import copy
from pathlib import Path

import pytest
import yaml

from geodude_teleoperation.ssot_dof import load_ssot_dof
from geodude_teleoperation.teleop.ur_stringer_teleop import parse_args, validate_args


def _parse(tmp_path: Path, cli: list[str] | None = None, **section_overrides):
    cfg = copy.deepcopy(load_ssot_dof())
    for key, val in section_overrides.items():
        if isinstance(val, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **val}
        else:
            cfg[key] = val
    cfg.setdefault("paths", {})["ply"] = "/tmp/x.ply"
    out = tmp_path / "SSoT_dof.yaml"
    out.write_text(yaml.dump(cfg))
    return parse_args(["--config", str(out), *(cli or [])])


def test_no_dry_run_clears_ssot_dry_run(tmp_path):
    args = _parse(
        tmp_path,
        ["--backend", "rtde", "--no-dry-run"],
        session={"dry_run": True},
        robot={"ip": "192.168.1.10"},
    )
    assert args.dry_run is False


def test_rtde_requires_ip(tmp_path):
    args = _parse(tmp_path, ["--backend", "rtde", "--dry-run"], robot={"ip": None})
    with pytest.raises(SystemExit):
        validate_args(args)
    validate_args(
        _parse(
            tmp_path,
            ["--backend", "rtde", "--dry-run"],
            robot={"ip": "192.168.1.10"},
        )
    )


def test_in_contact_exclusive_with_tactile(tmp_path):
    args = _parse(
        tmp_path,
        ["--backend", "rtde", "--dry-run", "--in-contact"],
        robot={"ip": "192.168.1.10"},
        tactile={"desired_normal_force": 5.0},
    )
    validate_args(args)
    assert args.in_contact is True
    with pytest.raises(SystemExit):
        validate_args(
            _parse(
                tmp_path,
                ["--backend", "rtde", "--in-contact", "--tactile"],
                robot={"ip": "10.0.0.1"},
            )
        )


def test_spacemouse_ssot_and_cli_override(tmp_path):
    args = _parse(
        tmp_path,
        control={
            "spacemouse": True,
            "spacemouse_linear_scale": 0.01,
            "spacemouse_hz": 40.0,
            "spacemouse_axis_sign": [-1, 1, 1, 1, 1, 1],
        },
    )
    assert args.spacemouse is True
    assert args.spacemouse_linear_scale == 0.01
    assert args.spacemouse_hz == 40.0
    assert _parse(tmp_path, ["--no-spacemouse"], control={"spacemouse": True}).spacemouse is False
