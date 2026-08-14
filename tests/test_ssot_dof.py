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
    assert flat["voxel_size"] == 0.004
    assert flat["viz"] is False
    assert flat["enable_key"] == "\n"
    assert flat["tip_offset"] == (0.0, 0.0, 0.0)
    assert flat["pose_gain"] == 4.0
    assert flat["max_linear_speed"] == 0.05
    assert flat["max_angular_speed"] == 0.4
    assert flat["orient_lock_on_enable"] is True
    assert flat["startup_max_joint_velocity"] > 0.0
    assert flat["startup_rtde_accel"] > 0.0
    assert flat["startup_max_angular_speed"] > 0.0
    assert flat["startup_ramp_s"] >= 0.0
    assert flat["orient_lock_tol_deg"] > 0.0
    assert flat["max_twist_scale"] == 2.0
    assert flat["vf_dry_query_period_s"] == 1.0
    assert flat["spacemouse"] is False
    assert flat["spacemouse_linear_scale"] == 0.05
    assert flat["spacemouse_angular_scale"] == 0.5
    assert flat["spacemouse_deadzone"] == 0.05
    assert flat["spacemouse_hz"] is None
    assert flat["spacemouse_axis_sign"] == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert flat["spacemouse_path"] is None
    assert flat["viz_latency"] is False


def test_keyboard_and_spacemouse_dryrun_configs_flatten():
    root = ssot_dof_config_path().parent
    dry = load_flat_ssot(root / "SSoT_dof_keyboard_dryrun.yaml")
    sm = load_flat_ssot(root / "SSoT_dof_spacemouse_dryrun.yaml")
    assert dry["dry_run"] is True
    assert dry["wos_batch_size"] == 32
    assert dry["control_hz"] == 500.0
    assert dry["spacemouse"] is False
    assert sm["dry_run"] is True
    assert sm["spacemouse"] is True
    assert sm["spacemouse_linear_scale"] > 0.0
    assert sm["spacemouse_hz"] is None or float(sm["spacemouse_hz"]) > 0.0
    assert sm["viz_latency"] is True
    assert dry["viz_latency"] is False
    assert sm["robot_ip"] == dry["robot_ip"]


def test_spacemouse_rtde_snapshot_exposes_motion_limits():
    root = ssot_dof_config_path().parent
    live = load_flat_ssot(root / "SSoT_dof_spacemouse_rtde.yaml")
    assert live["dry_run"] is False
    assert live["spacemouse"] is True
    assert live["orient_lock_on_enable"] is True
    assert live["startup_max_joint_velocity"] < live["max_joint_velocity"]
    assert live["startup_rtde_accel"] < live["rtde_accel"]
    assert live["startup_max_angular_speed"] < live["max_angular_speed"]
    assert live["startup_ramp_s"] > 0.0


def test_spacemouse_in_contact_snapshot_exposes_dualloop_tuning():
    root = ssot_dof_config_path().parent
    contact = load_flat_ssot(root / "SSoT_dof_spacemouse_rtde_in_contact.yaml")
    assert contact["in_contact"] is True
    assert contact["spacemouse"] is True
    assert contact["contact_desired_force"] == 5.0
    assert contact["force_kp"] > 0.0
    assert contact["align_k_ori"] > 0.0
    assert contact["tangent_scale"] > 0.0
    assert contact["wrench_in_tool_frame"] is True


def test_package_ssot_exposes_contact_force_pid_defaults():
    flat = load_flat_ssot()
    assert flat["contact_desired_force"] == 5.0
    assert flat["force_kp"] == 0.001
    assert flat["align_k_perp"] == 100.0
    assert flat["contact_damping"] == 0.05


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
