"""Tests for SSoT_dof load / flatten."""
from pathlib import Path

import pytest
import yaml

from geodude_teleoperation.paths import ssot_dof_config_path
from geodude_teleoperation.ssot_dof import flatten_ssot, load_flat_ssot, load_ssot_dof


def test_package_ssot_exists_and_flattens():
    path = ssot_dof_config_path()
    assert path.is_file()
    flat = load_flat_ssot(path)
    assert flat["diffusion_scalar"] == 5000.0
    assert flat["wos_batch_size"] == 256
    assert flat["wos_convergence_threshold_edge_mult"] == 2.0
    assert flat["query_radius"] == 0.025
    assert flat["fit_mode"] == "sphere"
    assert flat["voxel_size"] == 0.001
    assert flat["viz"] is False
    assert flat["enable_key"] == "\n"
    assert flat["tip_offset"] == (0.0, 0.0, 0.0)
    assert flat["pose_gain"] == 4.0
    assert flat["max_linear_speed"] == 0.05
    assert flat["max_angular_speed"] == 0.4
    assert flat["max_twist_scale"] == 2.0
    assert flat["rtde_accel"] == 0.3
    assert flat["max_joint_velocity"] == 0.5


def test_flatten_missing_key_raises():
    with pytest.raises(KeyError, match="missing SSoT key"):
        flatten_ssot({"session": {"backend": "mujoco", "dry_run": False}})


def test_custom_ssot_path(tmp_path: Path):
    src = load_ssot_dof()
    src["field"]["diffusion_scalar"] = 1234.0
    src["wos"]["convergence_threshold_edge_mult"] = 3.5
    src["robot"]["ip"] = "10.0.0.9"
    out = tmp_path / "SSoT_dof.yaml"
    out.write_text(yaml.dump(src))
    flat = load_flat_ssot(out)
    assert flat["diffusion_scalar"] == 1234.0
    assert flat["wos_convergence_threshold_edge_mult"] == 3.5
    assert flat["robot_ip"] == "10.0.0.9"
