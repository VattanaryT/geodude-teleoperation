import pytest

from geodude_teleoperation.ur_stringer_teleop import parse_args, validate_args


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
