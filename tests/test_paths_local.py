"""Local path / config resolution."""
from pathlib import Path

from geodude_teleoperation.paths import (
    default_T_tag0_base_path,
    default_ply_path,
    ssot_dof_config_path,
    repo_root,
)
from geodude_teleoperation.field.stringer_field import config_defaults
from geodude_teleoperation.teleop.ur_stringer_teleop import parse_args


def test_repo_root_has_pyproject():
    root = repo_root()
    assert (root / "pyproject.toml").is_file()


def test_local_ssot_pointcloud_defaults():
    path = ssot_dof_config_path()
    assert path.is_file()
    cfg = config_defaults("stringer_layup")
    assert cfg["voxel_size"] == 0.004
    assert cfg["normal_orientation"] == 1


def test_parse_args_defaults_to_local_data():
    args = parse_args(["--dry-run"])
    assert args.ply == default_ply_path()
    if default_T_tag0_base_path().is_file():
        assert args.T_tag0_base == default_T_tag0_base_path()
    assert isinstance(args.ply, Path)
    assert args.diffusion_scalar == 5000.0
    assert args.wos_convergence_threshold_edge_mult == 2.0
