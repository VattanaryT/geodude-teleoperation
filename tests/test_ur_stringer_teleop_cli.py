import pytest

from geodude_teleoperation.teleop.ur_stringer_teleop import parse_args, validate_args


def test_default_backend_mujoco():
    args = parse_args(["--ply", "/tmp/x.ply"])
    assert args.backend == "mujoco"
    assert args.dry_run is False


def test_rtde_requires_ip_validation():
    args = parse_args(["--ply", "/tmp/x.ply", "--backend", "rtde", "--dry-run"])
    with pytest.raises(SystemExit):
        validate_args(args)


def test_rtde_with_ip_passes_validation():
    args = parse_args(
        [
            "--ply",
            "/tmp/x.ply",
            "--backend",
            "rtde",
            "--robot-ip",
            "192.168.1.10",
            "--dry-run",
        ]
    )
    validate_args(args)


def test_enable_key_must_not_be_t_or_v():
    args = parse_args(["--ply", "/tmp/x.ply", "--enable-key", "t"])
    with pytest.raises(SystemExit):
        validate_args(args)


def test_enable_key_enter_alias():
    args = parse_args(["--ply", "/tmp/x.ply", "--enable-key", "enter"])
    assert args.enable_key == "\n"


def test_viz_flags():
    args = parse_args(
        [
            "--ply",
            "/tmp/x.ply",
            "--viz",
            "--viz-frame",
            "tag0",
            "--viz-frame-stride",
            "5",
            "--viz-teleop-keys",
        ]
    )
    assert args.viz is True
    assert args.viz_frame == "tag0"
    assert args.viz_frame_stride == 5
    assert args.viz_teleop_keys is True
    validate_args(args)


def test_viz_teleop_keys_implies_hide_object():
    args = parse_args(["--ply", "/tmp/x.ply", "--viz", "--viz-teleop-keys"])
    validate_args(args)
    assert args.viz_hide_object is True


def test_viz_show_object_overrides_hide():
    args = parse_args(
        ["--ply", "/tmp/x.ply", "--viz", "--viz-teleop-keys", "--viz-show-object"]
    )
    validate_args(args)
    assert args.viz_hide_object is False


def test_viz_frame_default_vention():
    args = parse_args(["--ply", "/tmp/x.ply"])
    assert args.viz_frame == "vention"


def test_motion_log_and_diag_flags():
    args = parse_args(
        [
            "--ply",
            "/tmp/x.ply",
            "--backend",
            "rtde",
            "--robot-ip",
            "10.0.0.1",
            "--log-motion",
            "/tmp/m.csv",
            "--plot-motion",
            "--diag-fixed-orient",
        ]
    )
    validate_args(args)
    assert args.log_motion.name == "m.csv"
    assert args.plot_motion is True
    assert args.diag_fixed_orient is True


def test_diag_modes_exclusive():
    args = parse_args(
        [
            "--ply",
            "/tmp/x.ply",
            "--diag-hold-zero",
            "--diag-fixed-orient",
        ]
    )
    with pytest.raises(SystemExit):
        validate_args(args)


def test_plot_motion_requires_log():
    args = parse_args(["--ply", "/tmp/x.ply", "--plot-motion"])
    with pytest.raises(SystemExit):
        validate_args(args)


def test_plot_motion_ok_with_record_mcap():
    args = parse_args(["--ply", "/tmp/x.ply", "--record-mcap", "--plot-motion"])
    validate_args(args)


def test_record_mcap_cli_defaults():
    args = parse_args(["--ply", "/tmp/x.ply", "--record-mcap"])
    assert args.record_mcap is True
    assert args.record_save_images is True
    assert args.record_sample_hz == 0.0
    validate_args(args)


def test_record_mcap_no_images_flag():
    args = parse_args(
        ["--ply", "/tmp/x.ply", "--record-mcap", "--no-record-save-images"]
    )
    assert args.record_save_images is False


def test_line_tracking_and_tactile_flags():
    args = parse_args(
        [
            "--ply",
            "/tmp/x.ply",
            "--backend",
            "rtde",
            "--robot-ip",
            "192.168.1.10",
            "--dry-run",
            "--line-tracking",
            "--soft-standoff",
            "--standoff-distance",
            "0.003",
            "--verify-500hz",
            "--tactile",
            "--desired-normal-force",
            "4.0",
        ]
    )
    validate_args(args)
    assert args.line_tracking is True
    assert args.soft_standoff is True
    assert args.tactile is True
    assert abs(args.standoff_distance - 0.003) < 1e-12


def test_tactile_requires_rtde():
    args = parse_args(["--ply", "/tmp/x.ply", "--tactile"])
    with pytest.raises(SystemExit):
        validate_args(args)


def test_in_contact_flag():
    args = parse_args(
        [
            "--ply",
            "/tmp/x.ply",
            "--backend",
            "rtde",
            "--robot-ip",
            "192.168.1.10",
            "--dry-run",
            "--in-contact",
            "--desired-normal-force",
            "5.0",
        ]
    )
    validate_args(args)
    assert args.in_contact is True
    assert abs(args.desired_normal_force - 5.0) < 1e-12


def test_in_contact_exclusive_with_tactile():
    args = parse_args(
        [
            "--ply",
            "/tmp/x.ply",
            "--backend",
            "rtde",
            "--robot-ip",
            "10.0.0.1",
            "--in-contact",
            "--tactile",
        ]
    )
    with pytest.raises(SystemExit):
        validate_args(args)
