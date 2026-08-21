"""Local data defaults from SSoT (no CLI path maze)."""

from pathlib import Path

from geodude_teleoperation.field.stringer_field import config_defaults
from geodude_teleoperation.paths import repo_root, resolve_repo_path, ssot_dof_config_path
from geodude_teleoperation.ssot_dof import load_flat_ssot


def test_local_ssot_defaults():
    assert (repo_root() / "pyproject.toml").is_file()
    assert ssot_dof_config_path().is_file()
    cfg = config_defaults("stringer_layup")
    assert cfg["voxel_size"] == 0.004


def test_resolve_repo_path_relative_falls_back_to_worktree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rel = Path("data/processed/laser_profilometer/merged.ply")
    assert resolve_repo_path(rel) == repo_root() / rel
    assert resolve_repo_path("/abs/x.ply") == Path("/abs/x.ply")


def test_resolve_repo_path_prefers_existing_cwd_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "local.ply"
    local.write_text("x")
    assert resolve_repo_path("local.ply") == local


def test_package_ssot_relative_ply_resolves_when_cwd_is_not_worktree(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    ply = Path(load_flat_ssot(ssot_dof_config_path())["ply"])
    expected = repo_root() / "data" / "processed" / "laser_profilometer" / "merged.ply"
    assert ply == expected
