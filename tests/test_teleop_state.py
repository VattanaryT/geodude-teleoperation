from geodude_teleoperation.teleop.teleop_state import TeleopState, format_status_line


def test_enable_gating():
    s = TeleopState()
    assert s.motion_enabled is False
    s.enable()
    assert s.motion_enabled and s.teleop_active
    assert s.commands_allowed(dry_run=False, fault=False, tf_ok=True)
    assert not s.commands_allowed(dry_run=True, fault=False, tf_ok=True)
    assert not s.control_math_allowed(fault=True, tf_ok=True)
    s.teleop_active = False
    assert not s.commands_allowed(dry_run=False, fault=False, tf_ok=True)
    line = format_status_line(
        s, {"signed_d": 0.01, "in_contact": False, "used_wos": True, "tip_nn": 0.02, "query_ms": 1.5},
        qdot_norm=0.0, tf_age_ms=0.0,
    )
    assert "motion_enabled=1" in line
