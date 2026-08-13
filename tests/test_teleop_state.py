from geodude_teleoperation.teleop.teleop_state import (
    TeleopState,
    format_status_line,
    should_emit_status,
)


def test_enable_defaults_teleop_active():
    s = TeleopState()
    assert s.motion_enabled is False
    s.enable()
    assert s.motion_enabled and s.teleop_active and s.mode == "teleop"


def test_gating():
    s = TeleopState()
    s.enable()
    assert s.commands_allowed(dry_run=False, fault=False, tf_ok=True)
    assert not s.commands_allowed(dry_run=True, fault=False, tf_ok=True)
    assert s.control_math_allowed(fault=False, tf_ok=True)
    assert not s.control_math_allowed(fault=True, tf_ok=True)
    s.teleop_active = False
    assert not s.commands_allowed(dry_run=False, fault=False, tf_ok=True)
    assert not s.control_math_allowed(fault=False, tf_ok=True)


def test_status_line_has_flags():
    s = TeleopState()
    s.enable()
    line = format_status_line(
        s,
        {
            "signed_d": 0.01,
            "in_contact": False,
            "used_wos": True,
            "tip_nn": 0.02,
            "query_ms": 1.5,
        },
        qdot_norm=0.0,
        tf_age_ms=0.0,
    )
    assert "motion_enabled=1" in line
    assert "teleop_active=1" in line


def test_should_emit_status_throttles():
    assert should_emit_status(float("-inf"), 0.0, status_hz=5.0) is True
    assert should_emit_status(0.0, 0.1, status_hz=5.0) is False
    assert should_emit_status(0.0, 0.2, status_hz=5.0) is True
    assert should_emit_status(0.0, 0.01, status_hz=0.0) is True
