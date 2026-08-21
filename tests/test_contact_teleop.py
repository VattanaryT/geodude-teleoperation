"""Unit tests for in-contact DualLoop teleop bridge."""

from __future__ import annotations

import numpy as np

from geodude_teleoperation.teleop.contact_teleop import (
    ContactTeleopBridge,
    ContactTeleopConfig,
    field_tangent_world,
    is_dtc_actuation,
    is_dtc_pure_actuation,
    normalize_contact_actuation,
    orthonormal_R_from_approach,
    print_pendant_tcp_dbg,
)
from geodude_teleoperation.in_contact_control.force_pid import ForcePIDController


def test_orthonormal_R_approach_is_column2():
    n = np.array([0.0, 0.0, -1.0])
    R = orthonormal_R_from_approach(n)
    assert np.allclose(R[:, 2], n / np.linalg.norm(n))
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)


def test_field_tangent_world_kills_normal():
    R = np.eye(3)
    v, used_cga = field_tangent_world([1.0, 0.0, 5.0], R, scale=1.0)
    assert abs(np.dot(v, R[:, 2])) < 1e-12
    assert np.allclose(v, [1.0, 0.0, 0.0])
    # pygafro may or may not be present; flag must be bool either way.
    assert isinstance(used_cga, bool)


def test_wasd_uses_local_frame_not_tool_frame():
    """w must follow R_local[:,0], not R_tcp[:,0], when they differ."""
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(
            desired_force_n=5.0,
            force_sign=1.0,
            tangent_scale=1.0,
            align_first=False,
            freeze_n_on_arm=True,
            max_joint_velocity=5.0,
        )
    )
    # Surface / approach: +z; local teleop x along +y (width).
    R_approach = np.eye(3)
    R_local = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float
    ).T  # columns: x=+y, y=+x, z=+z
    # Tool x along +x (differs from local x=+y).
    R_tcp = np.eye(3)
    bridge.arm([0, 0, 0], R_tcp, R_approach)
    bridge.push_tangent_field_linear(
        [0.02, 0.0, 0.0], R_local, n_approach=R_approach[:, 2]
    )
    assert np.allclose(bridge._pending_v_tan, [0.0, 0.02, 0.0], atol=1e-9)
    # Not along tool +x.
    assert abs(bridge._pending_v_tan[0]) < 1e-9


def test_tip_track_and_freeze_n():
    cfg = ContactTeleopConfig(desired_force_n=5.0, freeze_n_on_arm=False)
    bridge = ContactTeleopBridge(cfg)
    R0 = np.eye(3)
    p0 = np.array([0.1, 0.2, 0.3])
    bridge.arm(p0, R0, R0)

    R_tilted = orthonormal_R_from_approach(np.array([0.1, 0.0, -1.0]), R_hint=R0)
    p1 = p0 + np.array([0.01, 0.0, 0.0])
    J = np.eye(6)
    bridge.step(
        p_tcp=p1,
        R_tcp=R0,
        R_field=R_tilted,
        force_base=[0.0, 0.0, 0.0],
        jacobian=J,
        dt=0.02,
    )
    assert bridge.dual.target is not None
    assert np.allclose(bridge.dual.target.p_star, p1)
    assert abs(np.dot(bridge.dual.target.n, R_tilted[:, 2])) > 0.99

    bridge.toggle_freeze()
    R_other = orthonormal_R_from_approach(np.array([0.0, 0.5, -1.0]))
    n_before = bridge.dual.target.n.copy()
    bridge.step(
        p_tcp=p1,
        R_tcp=R0,
        R_field=R_other,
        force_base=[0.0, 0.0, 0.0],
        jacobian=J,
        dt=0.02,
    )
    assert np.allclose(bridge.dual.target.n, n_before, atol=1e-9)
    assert bridge.last_diag["n_freeze"] == 1.0


def test_freeze_n_on_arm_locks_field_not_tcp_z():
    """freeze_n_on_arm: ALIGN follows live field; SEEK then locks that n*.

    It does **not** hold the current tool orientation, and it no longer freezes
    the Enter-time snapshot (that parked the TCP on a neighboring end-line).
    """
    R_tcp = np.eye(3)
    R_field = orthonormal_R_from_approach(np.array([0.0, -1.0, 0.0]))
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(desired_force_n=5.0, freeze_n_on_arm=True)
    )
    bridge.arm([0.4, -0.2, 0.3], R_tcp, R_field)
    assert bridge.phase == ContactTeleopBridge.PHASE_ALIGN
    assert bridge.n_frozen is False
    assert abs(np.dot(bridge._R_star[:, 2], R_field[:, 2])) > 0.99
    assert abs(np.dot(bridge._R_star[:, 2], R_tcp[:, 2])) < 0.2
    R_later = orthonormal_R_from_approach(np.array([1.0, 0.0, 0.0]))
    bridge._refresh_R_star(R_later)
    assert abs(np.dot(bridge._R_star[:, 2], R_later[:, 2])) > 0.99

    bridge._update_phase(0.0)
    assert bridge.phase == ContactTeleopBridge.PHASE_SEEK
    assert bridge.n_frozen is True
    z0 = bridge._R_star[:, 2].copy()
    bridge._refresh_R_star(orthonormal_R_from_approach(np.array([0.0, 0.0, 1.0])))
    assert np.allclose(bridge._R_star[:, 2], z0)


def test_ur_force_sign_makes_compression_positive_fs():
    """Live UR logs: compression increases F·n*; force_sign=+1 ⇒ Fs>0."""
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(desired_force_n=5.0, force_sign=1.0)
    )
    R = np.eye(3)
    bridge.arm([0, 0, 0], R, R)
    J = np.eye(6)
    # Free space: F≈0 → Fe≈Fd → seek along +n (into surface).
    qdot_seek = bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0.0, 0.0, 0.0],
        jacobian=J,
        dt=0.02,
    )
    assert bridge.last_diag["Fe_N"] > 0.0
    assert float(np.linalg.norm(qdot_seek)) > 0.0

    # Loaded: Fz positive along n* → Fs positive with force_sign=+1.
    for _ in range(20):
        bridge.step(
            p_tcp=[0, 0, 0],
            R_tcp=R,
            R_field=R,
            force_base=[0.0, 0.0, 5.0],
            jacobian=J,
            dt=0.02,
        )
    assert bridge.last_diag["Fs_N"] > 3.0
    assert abs(bridge.last_diag["Fe_N"]) < 2.0


def test_wrong_sign_fs_stops_approach():
    """Old force_sign=-1 impact mode: Fs≪0 must not push harder."""
    from geodude_teleoperation.in_contact_control.force_pid import ForcePIDController

    ctl = ForcePIDController()
    ctl.config.Fd_N = 2.0
    ctl.config.force_sign = -1.0
    ctl.config.wrench_in_tool_frame = False
    ctl.config.kp = 0.01
    n = np.array([0.0, 0.0, 1.0])
    # Compression F along +n with force_sign=-1 → Fs negative (live failure mode).
    cmd, diag = ctl.step(
        n_base=n, force_base=[0.0, 0.0, 25.0], dt=0.02, R_ee=np.eye(3)
    )
    assert diag["Fs_N"] < -2.0
    assert diag["wrong_sign"] == 1.0
    assert cmd.linear_z <= 1e-9


def test_high_fs_fault_allows_retract():
    """|Fs| overload must unload (retract), not freeze q̇=0."""
    from geodude_teleoperation.in_contact_control.force_pid import ForcePIDController

    ctl = ForcePIDController()
    ctl.config.Fd_N = 5.0
    ctl.config.force_sign = 1.0
    ctl.config.wrench_in_tool_frame = False
    ctl.config.fault_abs_fs_N = 20.0
    ctl.config.fault_fd_mult = 3.0
    ctl.config.kp = 0.001
    ctl.config.force_lpf_hz = 0.0  # no LPF — trip on first sample
    n = np.array([0.0, 0.0, 1.0])
    # Overload along +n with force_sign=+1 → Fs high, Fe negative → retract −n.
    cmd, diag = ctl.step(
        n_base=n, force_base=[0.0, 0.0, 26.0], dt=0.02, R_ee=np.eye(3)
    )
    assert diag["fault_fs"] == 1.0
    assert cmd.linear_z < 0.0


def test_wasd_survives_wrong_sign_and_overload():
    """Under high |Fs| jam, kill tangent (UR faulted on lateral speedJ); mild keeps it."""
    R = np.eye(3)
    J = np.eye(6)

    # High |Fs| wrong-sign jam: retract-only, no lateral speedJ.
    bridge_ws = ContactTeleopBridge(
        ContactTeleopConfig(
            desired_force_n=5.0,
            force_sign=-1.0,
            align_first=False,
            max_joint_velocity=5.0,
            tangent_scale=8.0,
            dtc_min_fs_frac_of_fd=0.25,
        )
    )
    bridge_ws.arm([0, 0, 0], R, R)
    bridge_ws.dual.config.force.wrench_in_tool_frame = False
    bridge_ws.push_tangent_field_linear([0.02, 0.0, 0.0], R)
    qdot_ws = bridge_ws.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0.0, 0.0, 25.0],
        jacobian=J,
        dt=0.02,
    )
    assert bridge_ws.last_diag["force_wrong_sign"] == 1.0
    assert bridge_ws.last_diag["v_tan_norm"] == 0.0
    assert abs(float(qdot_ws[0])) < 1e-6

    # Hard |Fs| overload with force_sign=+1: retract −n*, no lateral while jammed.
    bridge_ov = ContactTeleopBridge(
        ContactTeleopConfig(
            desired_force_n=5.0,
            force_sign=1.0,
            align_first=False,
            max_joint_velocity=5.0,
            tangent_scale=8.0,
        )
    )
    bridge_ov.arm([0, 0, 0], R, R)
    bridge_ov.dual.config.force.wrench_in_tool_frame = False
    bridge_ov.dual.config.force.fault_abs_fs_N = 20.0
    bridge_ov.dual.config.force.fault_fd_mult = 3.0
    bridge_ov.dual.config.force.force_lpf_hz = 0.0
    bridge_ov.push_tangent_field_linear([0.02, 0.0, 0.0], R)
    qdot_ov = bridge_ov.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0.0, 0.0, 26.0],
        jacobian=J,
        dt=0.02,
    )
    assert bridge_ov.last_diag["force_fault"] == 1.0
    assert bridge_ov.last_diag["v_tan_norm"] == 0.0
    assert abs(float(qdot_ov[0])) < 1e-6
    # Into-surface (+z) must not be commanded while overloaded.
    assert float(qdot_ov[2]) <= 1e-9


def test_raw_force_spike_retracts_despite_lpf():
    """Impact dFs/dt on raw Fs must unload even when 5 Hz LPF is slow."""
    ctl = ForcePIDController()
    ctl.config.Fd_N = 5.0
    ctl.config.force_sign = 1.0
    ctl.config.wrench_in_tool_frame = False
    ctl.config.force_lpf_hz = 5.0
    ctl.config.spike_dF_dt_N_s = 40.0
    ctl.config.kp = 0.001
    n = np.array([0.0, 0.0, 1.0])
    # Settle LPF near zero (free space)
    for _ in range(30):
        ctl.step(n_base=n, force_base=[0.0, 0.0, 0.0], dt=0.02, R_ee=np.eye(3))
    # Step impact like contact_motion.csv t≈2.94 (0 → ~8 N in one tick)
    cmd, diag = ctl.step(
        n_base=n, force_base=[0.0, 0.0, 8.2], dt=0.02, R_ee=np.eye(3)
    )
    assert diag["spike"] == 1.0 or diag["Fs_raw_N"] > 6.0
    # Must not keep pushing into the surface after the spike/overshoot.
    assert cmd.linear_z <= 1e-9


def test_align_first_gates_force_until_tool_z_aligned():
    """Misaligned EE: Force off + Align ω; once Z‖n*, Force seeks."""
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(
            desired_force_n=5.0,
            force_sign=1.0,
            align_first=True,
            align_enter_tol_deg=5.0,
            freeze_n_on_arm=True,
            max_joint_velocity=5.0,
        )
    )
    R_field = np.eye(3)  # n* = +z
    # Tip Z tilted ~20° about x
    ang = np.deg2rad(20.0)
    c, s = np.cos(ang), np.sin(ang)
    R_tcp = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    bridge.arm([0, 0, 0], R_tcp, R_field)
    assert bridge.phase == ContactTeleopBridge.PHASE_ALIGN
    assert bridge.dual.config.force.enable is False

    qdot = bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R_tcp,
        R_field=R_field,
        force_base=[0.0, 0.0, 0.0],
        jacobian=np.eye(6),
        dt=0.02,
    )
    assert bridge.last_diag["phase_align"] == 1.0
    assert bridge.last_diag["force_enable"] == 0.0
    assert bridge.last_diag["ang_z_deg"] > 5.0
    assert float(np.linalg.norm(qdot)) > 0.0  # Align ω

    # Already aligned TCP
    bridge2 = ContactTeleopBridge(
        ContactTeleopConfig(desired_force_n=5.0, align_first=True, force_sign=1.0)
    )
    bridge2.arm([0, 0, 0], R_field, R_field)
    assert bridge2.phase == ContactTeleopBridge.PHASE_SEEK
    bridge2.step(
        p_tcp=[0, 0, 0],
        R_tcp=R_field,
        R_field=R_field,
        force_base=[0.0, 0.0, 0.0],
        jacobian=np.eye(6),
        dt=0.02,
    )
    assert bridge2.last_diag["force_enable"] == 1.0
    assert bridge2.last_diag["Fe_N"] > 0.0


def test_align_phase_omega_only_no_tip_slide():
    """During align: rotate tool-Z → n* in place (no linear / no tangent)."""
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(
            desired_force_n=5.0,
            force_sign=1.0,
            align_first=True,
            align_enter_tol_deg=5.0,
            freeze_n_on_arm=True,
            max_joint_velocity=5.0,
            tangent_scale=8.0,
        )
    )
    R_field = np.eye(3)
    ang = np.deg2rad(25.0)
    c, s = np.cos(ang), np.sin(ang)
    R_tcp = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    bridge.arm([0, 0, 0], R_tcp, R_field)
    # Keyboard/SpaceMouse should not slide tip while aligning.
    bridge.push_tangent_field_linear([0.05, 0.0, 0.0], R_field)
    qdot = bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R_tcp,
        R_field=R_field,
        force_base=[0.0, 0.0, 0.0],
        jacobian=np.eye(6),
        dt=0.02,
    )
    assert bridge.last_diag["phase_align"] == 1.0
    assert bridge.last_diag["v_cmd_norm"] < 1e-9
    assert bridge.last_diag["v_tan_norm"] < 1e-9
    assert bridge.last_diag["w_cmd_norm"] > 1e-6
    # J=I → tip linear joints ≈ 0; angular joints move.
    assert float(np.linalg.norm(qdot[:3])) < 1e-6
    assert float(np.linalg.norm(qdot[3:])) > 1e-6


def test_continuous_tangent_persists_without_hold_decay():
    """SpaceMouse-style: keep last tangent until cleared (no hold timeout)."""
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(
            desired_force_n=5.0,
            force_sign=1.0,
            align_first=False,
            tangent_scale=1.0,
            tangent_hold_s=0.05,
            max_joint_velocity=5.0,
        )
    )
    R = np.eye(3)
    bridge.arm([0, 0, 0], R, R)
    bridge.push_tangent_field_linear([0.02, 0.0, 0.0], R, continuous=True)
    # dt ≫ hold_s — hold mode would have expired.
    bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0.0, 0.0, 5.0],
        jacobian=np.eye(6),
        dt=1.0,
    )
    assert bridge.last_diag["v_tan_norm"] > 0.01
    bridge.clear_tangent()
    bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0.0, 0.0, 5.0],
        jacobian=np.eye(6),
        dt=0.02,
    )
    assert bridge.last_diag["v_tan_norm"] < 1e-12


def test_hold_tangent_still_expires():
    """Keyboard one-shot: tangent dies after tangent_hold_s."""
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(
            desired_force_n=5.0,
            force_sign=1.0,
            align_first=False,
            tangent_scale=1.0,
            tangent_hold_s=0.05,
            max_joint_velocity=5.0,
        )
    )
    R = np.eye(3)
    bridge.arm([0, 0, 0], R, R)
    bridge.push_tangent_field_linear([0.02, 0.0, 0.0], R)
    bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0.0, 0.0, 5.0],
        jacobian=np.eye(6),
        dt=0.02,
    )
    assert bridge.last_diag["v_tan_norm"] > 0.01
    # Hold still applies on the tick that drains the timer…
    bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0.0, 0.0, 5.0],
        jacobian=np.eye(6),
        dt=0.1,
    )
    # …then the next tick is idle.
    bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0.0, 0.0, 5.0],
        jacobian=np.eye(6),
        dt=0.02,
    )
    assert bridge.last_diag["v_tan_norm"] < 1e-12


def test_dualloop_outputs_qdot_not_joint_torque():
    """Default actuation=speedj: DualLoop → q̇ (not DTC τ)."""
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(
            desired_force_n=5.0,
            force_sign=1.0,
            align_first=False,
            max_joint_velocity=5.0,
            actuation="speedj",
        )
    )
    assert bridge.actuation == "speedj"
    R = np.eye(3)
    bridge.arm([0, 0, 0], R, R)
    bridge.push_tangent_field_linear([0.02, 0.0, 0.0], R, continuous=True)
    qdot = bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0.0, 0.0, 5.0],
        jacobian=np.eye(6),
        dt=0.02,
    )
    assert qdot.shape == (6,)
    assert float(np.linalg.norm(qdot)) > 0.0
    assert not hasattr(bridge, "publish_torques")
    assert float(np.linalg.norm(bridge.last_tau)) == 0.0


def test_normalize_contact_actuation_aliases():
    assert normalize_contact_actuation("dtc-pure") == "dtc_pure"
    assert normalize_contact_actuation("pure_dtc") == "dtc_pure"
    assert is_dtc_pure_actuation("dtc_pure")
    assert is_dtc_actuation("dtc_pure")
    assert is_dtc_actuation("dtc")
    assert normalize_contact_actuation("dtc_hybrid") == "dtc"
    assert not is_dtc_pure_actuation("dtc")
    assert not is_dtc_actuation("speedj")
    assert normalize_contact_actuation("nope") == "speedj"


def test_print_pendant_tcp_dbg_reports_flange_offset(capsys):
    print_pendant_tcp_dbg(
        [0.4, -0.2, 0.25],
        [0.4, -0.2, 0.10],
        [0.0, 0.0, 0.0],
    )
    out = capsys.readouterr().out
    assert "pendant TCP" in out
    assert "150.0mm" in out
    assert "getActualTCPPose" in out

