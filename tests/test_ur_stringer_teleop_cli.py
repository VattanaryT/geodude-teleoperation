import copy
from pathlib import Path

import pytest
import yaml

from geodude_teleoperation.ssot_dof import load_ssot_dof
from geodude_teleoperation.teleop.ur_stringer_teleop import parse_args, validate_args


def _write_ssot(tmp_path: Path, **section_overrides) -> Path:
    """Deep-merge top-level section overrides into package SSoT and write."""
    cfg = copy.deepcopy(load_ssot_dof())
    for key, val in section_overrides.items():
        if isinstance(val, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **val}
        else:
            cfg[key] = val
    # Avoid resolving real ply for unit tests that only validate CLI.
    if cfg.get("paths", {}).get("ply") is None:
        cfg.setdefault("paths", {})["ply"] = "/tmp/x.ply"
    out = tmp_path / "SSoT_dof.yaml"
    out.write_text(yaml.dump(cfg))
    return out


def _parse(tmp_path: Path, cli: list[str] | None = None, **section_overrides):
    cfg = _write_ssot(tmp_path, **section_overrides)
    return parse_args(["--config", str(cfg), *(cli or [])])


def test_default_backend_mujoco(tmp_path):
    args = _parse(tmp_path, session={"backend": "mujoco", "dry_run": False})
    assert args.backend == "mujoco"
    assert args.dry_run is False


def test_no_dry_run_clears_ssot_dry_run(tmp_path):
    args = _parse(
        tmp_path,
        ["--backend", "rtde", "--no-dry-run"],
        session={"dry_run": True},
        robot={"ip": "192.168.1.10"},
    )
    assert args.dry_run is False


def test_ssot_field_tuning_loaded(tmp_path):
    args = _parse(
        tmp_path,
        field={"diffusion_scalar": 4242.0},
        wos={"batch_size": 64, "convergence_threshold_edge_mult": 3.0},
    )
    assert args.diffusion_scalar == 4242.0
    assert args.wos_batch_size == 64
    assert args.wos_convergence_threshold_edge_mult == 3.0


def test_rtde_requires_ip_validation(tmp_path):
    args = _parse(
        tmp_path,
        ["--backend", "rtde", "--dry-run"],
        robot={"ip": None},
    )
    with pytest.raises(SystemExit):
        validate_args(args)


def test_rtde_with_ip_passes_validation(tmp_path):
    args = _parse(
        tmp_path,
        ["--backend", "rtde", "--dry-run"],
        robot={"ip": "192.168.1.10"},
    )
    validate_args(args)


def test_enable_key_must_not_be_t_or_v(tmp_path):
    args = _parse(tmp_path, control={"enable_key": "t"})
    with pytest.raises(SystemExit):
        validate_args(args)


def test_enable_key_enter_alias(tmp_path):
    args = _parse(tmp_path, control={"enable_key": "enter"})
    assert args.enable_key == "\n"


def test_viz_flags(tmp_path):
    args = _parse(
        tmp_path,
        ["--viz", "--viz-teleop-keys"],
        viz={"frame": "tag0", "frame_stride": 5},
    )
    assert args.viz is True
    assert args.viz_frame == "tag0"
    assert args.viz_frame_stride == 5
    assert args.viz_teleop_keys is True
    validate_args(args)


def test_viz_teleop_keys_implies_hide_object(tmp_path):
    args = _parse(tmp_path, ["--viz", "--viz-teleop-keys"])
    validate_args(args)
    assert args.viz_hide_object is True


def test_viz_show_object_overrides_hide(tmp_path):
    args = _parse(tmp_path, ["--viz", "--viz-teleop-keys", "--viz-show-object"])
    validate_args(args)
    assert args.viz_hide_object is False


def test_viz_frame_default_vention(tmp_path):
    args = _parse(tmp_path)
    assert args.viz_frame == "vention"


def test_motion_log_and_diag_flags(tmp_path):
    args = _parse(
        tmp_path,
        ["--backend", "rtde", "--plot-motion", "--diag-fixed-orient"],
        robot={"ip": "10.0.0.1"},
        paths={"log_motion": "/tmp/m.csv"},
    )
    validate_args(args)
    assert args.log_motion.name == "m.csv"
    assert args.plot_motion is True
    assert args.diag_fixed_orient is True


def test_diag_modes_exclusive(tmp_path):
    args = _parse(tmp_path, ["--diag-hold-zero", "--diag-fixed-orient"])
    with pytest.raises(SystemExit):
        validate_args(args)


def test_plot_motion_requires_log(tmp_path):
    args = _parse(tmp_path, ["--plot-motion"])
    with pytest.raises(SystemExit):
        validate_args(args)


def test_plot_motion_ok_with_record_mcap(tmp_path):
    args = _parse(tmp_path, ["--record-mcap", "--plot-motion"])
    validate_args(args)


def test_record_mcap_cli_defaults(tmp_path):
    args = _parse(tmp_path, ["--record-mcap"])
    assert args.record_mcap is True
    assert args.record_save_images is True
    assert args.record_sample_hz == 0.0
    validate_args(args)


def test_record_mcap_no_images_flag(tmp_path):
    args = _parse(tmp_path, ["--record-mcap", "--no-record-save-images"])
    assert args.record_save_images is False


def test_line_tracking_and_tactile_flags(tmp_path):
    args = _parse(
        tmp_path,
        [
            "--backend",
            "rtde",
            "--dry-run",
            "--line-tracking",
            "--soft-standoff",
            "--verify-500hz",
            "--tactile",
        ],
        robot={"ip": "192.168.1.10"},
        line_tracking={"standoff_distance": 0.003},
        tactile={"desired_normal_force": 4.0},
    )
    validate_args(args)
    assert args.line_tracking is True
    assert args.soft_standoff is True
    assert args.tactile is True
    assert abs(args.standoff_distance - 0.003) < 1e-12
    assert abs(args.desired_normal_force - 4.0) < 1e-12


def test_tactile_requires_rtde(tmp_path):
    args = _parse(
        tmp_path,
        ["--tactile"],
        session={"backend": "mujoco"},
    )
    with pytest.raises(SystemExit):
        validate_args(args)


def test_in_contact_flag(tmp_path):
    args = _parse(
        tmp_path,
        ["--backend", "rtde", "--dry-run", "--in-contact"],
        robot={"ip": "192.168.1.10"},
        tactile={"desired_normal_force": 5.0},
    )
    validate_args(args)
    assert args.in_contact is True
    assert abs(args.desired_normal_force - 5.0) < 1e-12


def test_in_contact_exclusive_with_tactile(tmp_path):
    args = _parse(
        tmp_path,
        ["--backend", "rtde", "--in-contact", "--tactile"],
        robot={"ip": "10.0.0.1"},
    )
    with pytest.raises(SystemExit):
        validate_args(args)
