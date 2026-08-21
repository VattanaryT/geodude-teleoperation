"""Load and flatten ``config/SSoT_dof.yaml`` into teleop Namespace-compatible attrs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from geodude_teleoperation.paths import resolve_repo_path, ssot_dof_config_path

# Nested YAML path → argparse / Namespace attribute name used by teleop + viz.
_FLAT_MAP: tuple[tuple[str, str], ...] = (
    ("session.backend", "backend"),
    ("session.dry_run", "dry_run"),
    ("robot.arm", "arm"),
    ("robot.ip", "robot_ip"),
    ("paths.ply", "ply"),
    ("paths.T_tag0_base", "T_tag0_base"),
    ("paths.T_rtde_base_vention", "T_rtde_base_vention"),
    ("paths.T_left_base_vention", "T_left_base_vention"),
    ("paths.calib_meta", "calib_meta"),
    ("paths.object_offset", "object_offset"),
    ("paths.laser_ply", "laser_ply"),
    ("paths.laser_offset", "laser_offset"),
    ("paths.log_motion", "log_motion"),
    ("paths.record_output_dir", "record_output_dir"),
    ("pointcloud.voxel_size", "voxel_size"),
    ("pointcloud.normal_orientation", "normal_orientation"),
    ("field.mode", "field_mode"),
    ("field.flow", "field_flow"),
    ("field.end_line_fraction", "end_line_fraction"),
    ("field.swap_end_lines", "swap_end_lines"),
    ("field.diffusion_scalar", "diffusion_scalar"),
    ("wos.batch_size", "wos_batch_size"),
    ("wos.convergence_threshold_edge_mult", "wos_convergence_threshold_edge_mult"),
    ("query.contact_threshold_m", "contact_threshold"),
    ("query.query_radius_m", "query_radius"),
    ("query.fit_mode", "fit_mode"),
    ("vf.r_field_smoothing", "r_field_smoothing"),
    ("vf.relock_dist_m", "vf_relock_dist"),
    ("vf.relock_period_s", "vf_relock_period"),
    ("vf.motion_track", "vf_motion_track"),
    ("vf.dry_query_period_s", "vf_dry_query_period_s"),
    ("control.hz", "control_hz"),
    ("control.status_hz", "status_hz"),
    ("control.max_joint_velocity", "max_joint_velocity"),
    ("control.orient_lock_on_enable", "orient_lock_on_enable"),
    ("control.startup_max_joint_velocity", "startup_max_joint_velocity"),
    ("control.startup_rtde_accel", "startup_rtde_accel"),
    ("control.startup_max_angular_speed", "startup_max_angular_speed"),
    ("control.startup_ramp_s", "startup_ramp_s"),
    ("control.orient_lock_tol_deg", "orient_lock_tol_deg"),
    ("control.keyboard_step", "keyboard_step"),
    ("control.spacemouse", "spacemouse"),
    ("control.spacemouse_linear_scale", "spacemouse_linear_scale"),
    ("control.spacemouse_angular_scale", "spacemouse_angular_scale"),
    ("control.spacemouse_deadzone", "spacemouse_deadzone"),
    ("control.spacemouse_hz", "spacemouse_hz"),
    ("control.spacemouse_axis_sign", "spacemouse_axis_sign"),
    ("control.spacemouse_path", "spacemouse_path"),
    ("control.rtde_accel", "rtde_accel"),
    ("control.pose_gain", "pose_gain"),
    ("control.max_linear_speed", "max_linear_speed"),
    ("control.max_angular_speed", "max_angular_speed"),
    ("control.max_twist_scale", "max_twist_scale"),
    ("control.watchdog_ms", "watchdog_ms"),
    ("control.enable_key", "enable_key"),
    ("control.tip_offset", "tip_offset"),
    ("control.base_frame", "base_frame"),
    ("control.tag0_frame", "tag0_frame"),
    ("control.no_object_offset", "no_object_offset"),
    ("control.no_rtde_base_vention", "no_rtde_base_vention"),
    ("control.no_laser_offset", "no_laser_offset"),
    ("control.object_offset_step_m", "object_offset_step_m"),
    ("viz.frame", "viz_frame"),
    ("viz.frame_stride", "viz_frame_stride"),
    ("viz.hide_object", "viz_hide_object"),
    ("viz.show_object", "viz_show_object"),
    ("viz.teleop_keys", "viz_teleop_keys"),
    ("viz.laser", "viz_laser"),
    ("viz.latency_table", "viz_latency"),
    ("laser.voxel", "laser_voxel"),
    ("tag_crosscheck.camera_rgb_topic", "camera_rgb_topic"),
    ("tag_crosscheck.camera_info_topic", "camera_info_topic"),
    ("tag_crosscheck.tag_id", "tag_id"),
    ("tag_crosscheck.tag_size_m", "tag_size_m"),
    ("tag_crosscheck.tag_family", "tag_family"),
    ("record.mcap", "record_mcap"),
    ("record.sample_hz", "record_sample_hz"),
    ("record.save_images", "record_save_images"),
    ("record.camera_topics", "record_camera_topics"),
    ("record.target_frame", "record_target_frame"),
    ("record.plot_motion", "plot_motion"),
    ("line_tracking.enabled", "line_tracking"),
    ("line_tracking.soft_standoff", "soft_standoff"),
    ("line_tracking.standoff_distance", "standoff_distance"),
    ("line_tracking.standoff_gain", "standoff_gain"),
    ("line_tracking.standoff_max_speed", "standoff_max_speed"),
    ("line_tracking.no_keyboard_normal_ff", "no_keyboard_normal_ff"),
    ("tactile.enabled", "tactile"),
    ("tactile.desired_normal_force", "desired_normal_force"),
    ("tactile.wrench_kp", "wrench_kp"),
    ("tactile.wrench_ki", "wrench_ki"),
    ("tactile.wrench_kd", "wrench_kd"),
    ("tactile.wrench_to_vn", "wrench_to_vn"),
    ("tactile.max_vn", "tactile_max_vn"),
    ("tactile.ft_bias", "ft_bias"),
    ("in_contact.enabled", "in_contact"),
    ("in_contact.actuation", "contact_actuation"),
    ("in_contact.desired_force", "contact_desired_force"),
    ("in_contact.force_sign", "force_sign"),
    ("in_contact.force_kp", "force_kp"),
    ("in_contact.force_ki", "force_ki"),
    ("in_contact.force_kd", "force_kd"),
    ("in_contact.force_deadband_N", "force_deadband_N"),
    ("in_contact.force_lpf_hz", "force_lpf_hz"),
    ("in_contact.force_contact_threshold_N", "force_contact_threshold_N"),
    ("in_contact.wrench_in_tool_frame", "wrench_in_tool_frame"),
    ("in_contact.wrench_kp", "contact_wrench_kp"),
    ("in_contact.wrench_ki", "contact_wrench_ki"),
    ("in_contact.wrench_kd", "contact_wrench_kd"),
    ("in_contact.tangent_force_gain", "tangent_force_gain"),
    ("in_contact.max_tau_nm", "max_tau_nm"),
    ("in_contact.max_wrench_n", "max_wrench_n"),
    ("in_contact.K_line", "contact_K_line"),
    ("in_contact.D_twist", "contact_D_twist"),
    ("in_contact.align_first", "align_first"),
    ("in_contact.align_enter_tol_deg", "align_enter_tol_deg"),
    ("in_contact.align_exit_tol_deg", "align_exit_tol_deg"),
    ("in_contact.align_k_perp", "align_k_perp"),
    ("in_contact.align_k_ori", "align_k_ori"),
    ("in_contact.align_m_lin", "align_m_lin"),
    ("in_contact.align_m_ang", "align_m_ang"),
    ("in_contact.align_b_lin", "align_b_lin"),
    ("in_contact.align_b_ang", "align_b_ang"),
    ("in_contact.approach_v_max", "approach_v_max"),
    ("in_contact.force_max_v", "force_max_v"),
    ("in_contact.dtc_contact_dist_m", "dtc_contact_dist_m"),
    ("in_contact.dtc_require_force", "dtc_require_force"),
    ("in_contact.dtc_min_fs_frac_of_fd", "dtc_min_fs_frac_of_fd"),
    ("in_contact.dtc_exit_fs_frac_of_enter", "dtc_exit_fs_frac_of_enter"),
    ("in_contact.dtc_min_hold_s", "dtc_min_hold_s"),
    ("in_contact.dtc_exit_hold_s", "dtc_exit_hold_s"),
    ("in_contact.dtc_startup_ramp_s", "dtc_startup_ramp_s"),
    ("in_contact.dtc_startup_tau_nm", "dtc_startup_tau_nm"),
    ("in_contact.dtc_max_dtau_nm_s", "dtc_max_dtau_nm_s"),
    ("in_contact.approach_stiffness_n_per_m", "approach_stiffness_n_per_m"),
    ("in_contact.tangent_scale", "tangent_scale"),
    ("in_contact.tangent_hold_s", "tangent_hold_s"),
    ("in_contact.max_linear_m_s", "contact_max_linear_m_s"),
    ("in_contact.max_angular_deg_s", "contact_max_angular_deg_s"),
    ("in_contact.damping", "contact_damping"),
    ("in_contact.freeze_n_on_arm", "freeze_n_on_arm"),
    ("in_contact.bias_skip_load_N", "bias_skip_load_N"),
    ("diag.hold_zero", "diag_hold_zero"),
    ("diag.fixed_orient", "diag_fixed_orient"),
    ("diag.verify_500hz", "verify_500hz"),
    ("diag.dtc_verify_seconds", "dtc_verify_seconds"),
    ("diag.verify_500hz_exit", "verify_500hz_exit"),
    ("diag.polyscope_version", "polyscope_version"),
)

_PATH_KEYS = frozenset(
    {
        "ply",
        "T_tag0_base",
        "T_rtde_base_vention",
        "T_left_base_vention",
        "calib_meta",
        "object_offset",
        "laser_ply",
        "laser_offset",
        "log_motion",
        "record_output_dir",
    }
)


def _dig(cfg: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def load_ssot_dof(path: Path | None = None) -> dict[str, Any]:
    """Load nested SSoT YAML (default: package ``config/SSoT_dof.yaml``)."""
    cfg_path = ssot_dof_config_path() if path is None else Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"SSoT_dof config not found: {cfg_path}")
    with cfg_path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"SSoT_dof root must be a mapping, got {type(data).__name__}")
    return data


def flatten_ssot(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Map nested SSoT sections onto flat teleop Namespace attribute names."""
    out: dict[str, Any] = {}
    optional_defaults: dict[str, Any] = {
        "in_contact.actuation": "speedj",
        "in_contact.wrench_kp": 1.0,
        "in_contact.wrench_ki": 0.0,
        "in_contact.wrench_kd": 0.0,
        "in_contact.tangent_force_gain": 50.0,
        "in_contact.max_tau_nm": 20.0,
        "in_contact.max_wrench_n": 12.0,
        "in_contact.dtc_contact_dist_m": 0.008,
        "in_contact.dtc_require_force": True,
        "in_contact.dtc_startup_ramp_s": 3.0,
        "in_contact.dtc_startup_tau_nm": 2.0,
        "in_contact.dtc_max_dtau_nm_s": 8.0,
        "in_contact.approach_stiffness_n_per_m": 400.0,
        "in_contact.K_line": None,
        "in_contact.D_twist": None,
    }
    for dotted, dest in _FLAT_MAP:
        try:
            out[dest] = _dig(cfg, dotted)
        except KeyError as exc:
            if dotted in optional_defaults:
                out[dest] = optional_defaults[dotted]
            else:
                raise KeyError(f"missing SSoT key {dotted!r}") from exc

    # enable_key: YAML "enter" → newline (matches former CLI alias)
    ek = out.get("enable_key")
    if isinstance(ek, str) and ek.lower() in ("enter", "return", "newline", "\\n"):
        out["enable_key"] = "\n"
    elif isinstance(ek, str) and len(ek) == 1:
        out["enable_key"] = ek
    elif ek is not None and not isinstance(ek, str):
        raise ValueError(f"control.enable_key must be a string, got {ek!r}")

    tip = out.get("tip_offset")
    if tip is not None:
        tip_list = list(tip)
        if len(tip_list) != 3:
            raise ValueError("control.tip_offset must have length 3")
        out["tip_offset"] = tuple(float(x) for x in tip_list)

    bias = out.get("ft_bias")
    if bias is not None:
        bias_list = list(bias)
        if len(bias_list) != 6:
            raise ValueError("tactile.ft_bias must have length 6 when set")
        out["ft_bias"] = tuple(float(x) for x in bias_list)

    sm_hz = out.get("spacemouse_hz")
    if sm_hz is None or sm_hz == "":
        out["spacemouse_hz"] = None
    else:
        out["spacemouse_hz"] = float(sm_hz)

    signs = out.get("spacemouse_axis_sign")
    if signs is None:
        out["spacemouse_axis_sign"] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    else:
        sign_list = list(signs)
        if len(sign_list) != 6:
            raise ValueError("control.spacemouse_axis_sign must have length 6")
        out["spacemouse_axis_sign"] = tuple(float(x) for x in sign_list)

    sm_path = out.get("spacemouse_path")
    if sm_path is None or sm_path == "":
        out["spacemouse_path"] = None
    else:
        out["spacemouse_path"] = str(sm_path)

    for key in _PATH_KEYS:
        val = out.get(key)
        if val is None or val == "":
            out[key] = None
        else:
            out[key] = resolve_repo_path(val)

    # YAML 1.1 may parse bare off/on as bool — normalize viz.laser.
    laser = out.get("viz_laser")
    if laser is False:
        out["viz_laser"] = "off"
    elif laser is True:
        out["viz_laser"] = "on"
    elif laser is not None:
        out["viz_laser"] = str(laser)

    # Session viz open is CLI-driven; YAML teleop_keys is a default preference only.
    # Always start with viz=False unless --viz is passed.
    out["viz"] = False
    return out


def load_flat_ssot(path: Path | None = None) -> dict[str, Any]:
    """Convenience: load + flatten."""
    return flatten_ssot(load_ssot_dof(path))
