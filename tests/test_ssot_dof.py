"""SSoT flatten: package YAML + in-contact / SpaceMouse snapshots."""

from pathlib import Path

import pytest
import yaml

from geodude_teleoperation.paths import ssot_dof_config_path
from geodude_teleoperation.ssot_dof import flatten_ssot, load_flat_ssot, load_ssot_dof


def test_package_ssot_flattens_control_and_contact():
    flat = load_flat_ssot(ssot_dof_config_path())
    assert flat["enable_key"] == "\n"
    assert flat["orient_lock_on_enable"] is True
    assert flat["startup_max_joint_velocity"] > 0.0
    assert flat["contact_desired_force"] == 5.0
    assert flat["contact_actuation"] == "speedj"
    assert flat["force_kp"] == 0.001
    assert flat["spacemouse"] is False


def test_spacemouse_in_contact_snapshot_exposes_dualloop():
    root = ssot_dof_config_path().parent
    contact = load_flat_ssot(root / "SSoT_dof_spacemouse_rtde_in_contact.yaml")
    assert contact["in_contact"] is True
    assert contact["spacemouse"] is True
    assert contact["contact_actuation"] == "dtc"
    assert contact["contact_desired_force"] == 20.0
    assert contact["approach_stiffness_n_per_m"] > 0.0
    assert contact["force_kp"] > 0.0
    assert contact["wrench_in_tool_frame"] is True
    live = load_flat_ssot(root / "SSoT_dof_spacemouse_rtde_free_space.yaml")
    assert live["startup_max_joint_velocity"] < live["max_joint_velocity"]
    assert live["startup_ramp_s"] > 0.0


@pytest.mark.parametrize(
    "name",
    [
        "SSoT_dof.yaml",
        "SSoT_dof_keyboard_dryrun.yaml",
        "SSoT_dof_spacemouse_dryrun.yaml",
        "SSoT_dof_spacemouse_rtde_free_space.yaml",
        "SSoT_dof_spacemouse_rtde_in_contact.yaml",
    ],
)
def test_all_package_ssot_snapshots_flatten(name: str):
    """Every shipped YAML must include required SSoT keys (even unused DTC knobs)."""
    flat = load_flat_ssot(ssot_dof_config_path().parent / name)
    assert flat["dtc_min_fs_frac_of_fd"] is not None
    assert flat["dtc_exit_fs_frac_of_enter"] is not None
    assert flat["dtc_min_hold_s"] is not None
    assert flat["dtc_exit_hold_s"] is not None


def test_flatten_missing_key_raises_and_custom_path_overrides(tmp_path: Path):
    with pytest.raises(KeyError, match="missing SSoT key"):
        flatten_ssot({"session": {"backend": "mujoco", "dry_run": False}})
    src = load_ssot_dof()
    src["field"]["diffusion_scalar"] = 1234.0
    src["robot"]["ip"] = "10.0.0.9"
    out = tmp_path / "SSoT_dof.yaml"
    out.write_text(yaml.dump(src))
    flat = load_flat_ssot(out)
    assert flat["diffusion_scalar"] == 1234.0
    assert flat["robot_ip"] == "10.0.0.9"
