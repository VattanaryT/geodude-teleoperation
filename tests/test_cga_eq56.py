"""CGA line tracking + Eq. 56 torque assembly (pygafro)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pygafro")

from geodude_teleoperation.cga.line_tracker import (
    alignment_twist_contact_se3,
    alignment_twist_orientation_se3,
    alignment_twist_se3,
)
from geodude_teleoperation.cga.tactile_law import (
    Eq56Controller,
    Eq56Gains,
    assemble_eq56_tau,
    default_eq56_gains,
)
from geodude_teleoperation.teleop.contact_teleop import (
    ContactTeleopBridge,
    ContactTeleopConfig,
    print_actuation_banner,
)


def test_line_aligned_twist_near_zero():
    p = np.array([0.1, 0.2, 0.3])
    R = np.eye(3)
    V = alignment_twist_se3(p, R, p, R)
    assert float(np.linalg.norm(V)) < 1e-9


def test_line_lateral_offset_has_translation():
    p = np.array([0.0, 0.0, 0.0])
    R = np.eye(3)
    x_q = np.array([0.05, 0.0, 0.0])
    V = alignment_twist_se3(p, R, x_q, R)
    # Should restore laterally (v_x dominant); free along z (line dir).
    assert abs(V[0]) > 1e-6
    assert abs(V[2]) < abs(V[0])


def test_line_motor_same_point_still_has_spatial_translation():
    """pygafro getMotor is a screw — ||v|| ≠ 0 even when lines meet at a far TCP."""
    p = np.array([0.4, -0.2, 0.3])
    R_tcp = np.eye(3)
    from geodude_teleoperation.teleop.contact_teleop import orthonormal_R_from_approach

    R_field = orthonormal_R_from_approach(np.array([0.0, -1.0, 0.0]))
    V = alignment_twist_se3(p, R_tcp, p, R_field)
    assert float(np.linalg.norm(V[:3])) > 0.1


def test_orientation_align_twist_is_tip_fixed():
    R_tcp = np.eye(3)
    from geodude_teleoperation.teleop.contact_teleop import orthonormal_R_from_approach

    R_field = orthonormal_R_from_approach(np.array([0.0, -1.0, 0.0]))
    V = alignment_twist_orientation_se3(R_tcp, R_field)
    assert np.allclose(V[:3], 0.0)
    # DualLoop e_R = n_ee × n* is +ω_x (z → −y). Eq.56 W_L = −e_R.
    assert V[3] < -0.5
    assert abs(V[4]) < 1e-9 and abs(V[5]) < 1e-9


def test_contact_align_twist_orbits_behind_pendant_tcp():
    """ω same as tip-fixed; v = ω × (lever z) and does not grow with |p_tcp|."""
    from geodude_teleoperation.teleop.contact_teleop import orthonormal_R_from_approach

    R_tcp = np.eye(3)
    R_field = orthonormal_R_from_approach(np.array([0.0, -1.0, 0.0]))
    V0 = alignment_twist_orientation_se3(R_tcp, R_field)
    V = alignment_twist_contact_se3(R_tcp, R_field, lever_m=0.08, v_cap_m=0.04)
    assert np.allclose(V[3:], V0[3:])
    assert float(np.linalg.norm(V[:3])) > 1e-6
    assert float(np.linalg.norm(V[:3])) <= 0.04 + 1e-9
    # v = ω × (lever ẑ) — no dependence on TCP distance from origin.


def test_eq56_align_does_not_translate_far_tcp():
    """Far TCP must not inflate Align; linear W_L is the bounded orbit term."""
    ctl = Eq56Controller(
        desired_force_n=5.0,
        wrench_kp=1.0,
        max_tau=50.0,
        approach_force_n=0.0,
        force_lpf_hz=0.0,
    )
    from geodude_teleoperation.teleop.contact_teleop import orthonormal_R_from_approach

    R_field = orthonormal_R_from_approach(np.array([0.0, -1.0, 0.0]))
    R_tcp = np.eye(3)
    p = np.array([0.4, -0.2, 0.3])
    tau = ctl.step(
        p_tcp=p,
        R_tcp=R_tcp,
        x_q=p,
        R_field=R_field,
        wrench_meas6=np.zeros(6),
        jacobian_ee=np.eye(6),
        qdot=np.zeros(6),
        v_tan_world=None,
        dt=0.02,
        enable_force=False,
        command_scale=1.0,
    )
    tau0 = ctl.step(
        p_tcp=np.zeros(3),
        R_tcp=R_tcp,
        x_q=np.zeros(3),
        R_field=R_field,
        wrench_meas6=np.zeros(6),
        jacobian_ee=np.eye(6),
        qdot=np.zeros(6),
        v_tan_world=None,
        dt=0.02,
        enable_force=False,
        command_scale=1.0,
    )
    assert np.allclose(tau, tau0, atol=1e-6)
    # Proximal (linear) channels are now active, but capped.
    assert float(np.linalg.norm(tau[:3])) > 1e-6
    assert float(np.linalg.norm(tau[:3])) < 20.0
    assert float(np.linalg.norm(tau[3:])) > 1.0
    # τ = −Jᵀ K W_L with W_L = −e_R ⇒ τ_ang aligns with DualLoop +e_R (+ω_x).
    assert float(tau[3]) > 0.0


def test_eq56_zero_wrenches_zero_tau():
    J = np.eye(6)
    tau = assemble_eq56_tau(
        J,
        W_L=np.zeros(6),
        W_V=np.zeros(6),
        W_C=np.zeros(6),
        W_tan=np.zeros(6),
    )
    assert np.allclose(tau, 0.0)


def test_eq56_wc_only_matches_jt():
    J = np.eye(6)
    W_C = np.array([1.0, 0, 0, 0, 0, 0])
    g = default_eq56_gains()
    # Contact wrench is applied via +Jᵀ W_C ⇒ τ = +W_C for J=I.
    tau = assemble_eq56_tau(
        J, W_L=np.zeros(6), W_V=np.zeros(6), W_C=W_C, gains=g
    )
    assert np.allclose(tau, W_C)


def test_eq56_damping_opposes_twist():
    """D W_V must brake ẋ (τ opposite measured ω), not accelerate through n*."""
    J = np.eye(6)
    W_V = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    g = default_eq56_gains()
    tau = assemble_eq56_tau(
        J, W_L=np.zeros(6), W_V=W_V, W_C=np.zeros(6), gains=g
    )
    assert float(tau[3]) < 0.0
    assert abs(float(tau[3]) + float(g.D_twist[3])) < 1e-9


def test_contact_bridge_dtc_returns_tau():
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(
            actuation="dtc",
            desired_force_n=5.0,
            align_first=False,
            max_tau_nm=50.0,
            wrench_kp=1.0,
        )
    )
    assert bridge.actuation == "dtc"
    R = np.eye(3)
    bridge.arm([0, 0, 0], R, R)
    # Tip laterally offset from x_q tracking (tip-track uses p as x_q → small line err)
    # Force a lateral geometric error by using a different R_field approach... tip-track
    # zeros point error; orientation mismatch produces align wrench.
    R_tilt = R.copy()
    R_tilt[:, 2] = np.array([0.2, 0.0, 1.0])
    R_tilt[:, 2] /= np.linalg.norm(R_tilt[:, 2])
    R_tilt[:, 0] = np.cross(R_tilt[:, 1], R_tilt[:, 2])
    R_tilt[:, 0] /= np.linalg.norm(R_tilt[:, 0])
    R_tilt[:, 1] = np.cross(R_tilt[:, 2], R_tilt[:, 0])
    tau = bridge.step_tau(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R_tilt,
        wrench6=[0, 0, 0, 0, 0, 0],
        jacobian=np.eye(6),
        dt=0.002,
        qdot_meas=np.zeros(6),
    )
    assert tau.shape == (6,)
    assert float(np.linalg.norm(tau)) > 0.0
    assert bridge.last_diag.get("actuation") == "dtc"


def test_print_actuation_banner_yellow(capsys):
    print_actuation_banner("dtc", desired_force_n=5.0)
    out = capsys.readouterr().out
    assert "ACTUATION=DTC" in out
    assert "\033[93m" in out
    print_actuation_banner("speedj")
    out = capsys.readouterr().out
    assert "ACTUATION=speedJ" in out
    print_actuation_banner("dtc_pure", desired_force_n=5.0)
    out = capsys.readouterr().out
    assert "ACTUATION=DTC_PURE" in out
    assert "No DualLoop speedJ" in out


def test_eq56_overshoot_holds_fd_not_reverse_yank():
    """Fs ≫ Fd must hold near Fd wrench — not reverse-yank (path deviation)."""
    ctl = Eq56Controller(
        desired_force_n=5.0,
        force_sign=1.0,
        wrench_kp=0.25,
        max_tau=40.0,
        max_wrench_n=12.0,
        overshoot_margin_n=2.0,
        retract_force_n=8.0,
        contact_threshold_n=0.6,
        approach_force_n=8.0,
    )
    R = np.eye(3)
    # Tool-frame compression already 28 N (> 1.5·Fd → mild bleed toward Fd).
    tau = ctl.step(
        p_tcp=np.zeros(3),
        R_tcp=R,
        x_q=np.zeros(3),
        R_field=R,
        wrench_meas6=np.array([0.0, 0.0, 28.0, 0.0, 0.0, 0.0]),
        jacobian_ee=np.eye(6),
        qdot=np.zeros(6),
        v_tan_world=None,
        dt=0.02,
        enable_force=True,
    )
    assert ctl.last_diag["force_overshoot"] > 0.5
    # Soft hold: still into-surface (same sign as Fd), not reverse yank.
    assert float(ctl.last_diag["f_cmd_base_z"]) > 0.0
    assert float(tau[2]) > 0.0
    assert float(ctl.last_diag["W_C_norm"]) <= 12.0 + 1e-6
    # Bleed: ‖W_C‖ ≤ Fd (not a large reverse retract).
    assert float(ctl.last_diag["W_C_norm"]) <= 5.0 + 1e-6


def test_eq56_pid_ignores_lateral_force():
    ctl = Eq56Controller(
        desired_force_n=5.0,
        wrench_kp=1.0,
        max_tau=50.0,
        max_wrench_n=30.0,
        contact_threshold_n=0.6,
        approach_force_n=0.0,
        force_lpf_hz=0.0,
    )
    R = np.eye(3)
    tau = ctl.step(
        p_tcp=np.zeros(3),
        R_tcp=R,
        x_q=np.zeros(3),
        R_field=R,
        wrench_meas6=np.array([20.0, 15.0, 5.0, 0.0, 0.0, 0.0]),
        jacobian_ee=np.eye(6),
        qdot=np.zeros(6),
        v_tan_world=None,
        dt=0.02,
        enable_force=True,
        signed_d=-0.001,  # in contact → no air spring rewrite
    )
    # At Fd=Fs_z=5: feedforward W_C≈Fd along Z only; lateral ignored.
    assert abs(float(ctl.last_diag["f_cmd_base_x"])) < 1e-6
    assert abs(float(ctl.last_diag["f_cmd_base_y"])) < 1e-6
    assert abs(float(ctl.last_diag["f_cmd_base_z"]) - 5.0) < 0.5
    _ = tau


def test_eq56_force_feedforward_reaches_fd_when_underloaded():
    """W_C ≈ Fd + kp·(Fd−Fs), not kp·(Fd−Fs) alone (stalls at ≪Fd)."""
    ctl = Eq56Controller(
        desired_force_n=70.0,
        wrench_kp=0.25,
        max_tau=50.0,
        max_wrench_n=80.0,
        approach_force_n=0.0,
        force_lpf_hz=0.0,
        contact_threshold_n=0.6,
    )
    R = np.eye(3)
    tau = ctl.step(
        p_tcp=np.zeros(3),
        R_tcp=R,
        x_q=np.zeros(3),
        R_field=R,
        wrench_meas6=np.array([0.0, 0.0, 4.5, 0.0, 0.0, 0.0]),
        jacobian_ee=np.eye(6),
        qdot=np.zeros(6),
        v_tan_world=None,
        dt=0.02,
        enable_force=True,
        signed_d=-0.001,
    )
    # Old P-only: 0.25*(70-4.5)≈16; FF: 70+16≈86 → clamp 80.
    assert float(ctl.last_diag["W_C_norm"]) >= 65.0
    assert float(ctl.last_diag["f_cmd_base_z"]) > 0.0
    _ = tau


def test_eq56_soft_approach_when_unloaded():
    """Tip-tracked lines have no n* slide — soft approach must create τ."""
    ctl = Eq56Controller(
        desired_force_n=5.0,
        wrench_kp=1.0,
        max_tau=50.0,
        contact_threshold_n=0.6,
        approach_force_n=8.0,
    )
    R = np.eye(3)
    # x_q == p ⇒ W_L ≈ 0; unloaded Fs ⇒ approach FF.
    tau = ctl.step(
        p_tcp=np.zeros(3),
        R_tcp=R,
        x_q=np.zeros(3),
        R_field=R,
        wrench_meas6=np.zeros(6),
        jacobian_ee=np.eye(6),
        qdot=np.zeros(6),
        v_tan_world=None,
        dt=0.02,
        enable_force=True,
    )
    assert float(np.linalg.norm(tau)) > 1.0
    assert ctl.last_diag["W_approach_norm"] > 0.0
    assert ctl.last_diag["W_C_norm"] > 0.0
    # Into +tool Z (= +world Z here): τ = +Jᵀ W ⇒ τ_z > 0 for J=I.
    assert float(tau[2]) > 0.0
    assert float(ctl.last_diag["f_cmd_base_z"]) > 0.0


def test_eq56_wrench_rotated_into_base_before_jt():
    """Tool +Z = −world Y must command base force along −Y, not world +Z."""
    ctl = Eq56Controller(
        desired_force_n=5.0,
        wrench_kp=1.0,
        max_tau=50.0,
        contact_threshold_n=0.6,
        approach_force_n=8.0,
        force_sign=1.0,
    )
    # Columns: tool x=world x, tool y=world z, tool z=−world y
    R = np.column_stack([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    tau = ctl.step(
        p_tcp=np.zeros(3),
        R_tcp=R,
        x_q=np.zeros(3),
        R_field=R,
        wrench_meas6=np.zeros(6),
        jacobian_ee=np.eye(6),
        qdot=np.zeros(6),
        v_tan_world=None,
        dt=0.02,
        enable_force=True,
    )
    # f_cmd_base ≈ R @ [0,0,f] = [0, -f, 0]; τ = +f_cmd for J=I.
    assert float(ctl.last_diag["f_cmd_base_y"]) < -1.0
    assert abs(float(ctl.last_diag["f_cmd_base_z"])) < 1e-6
    assert float(tau[1]) < -1.0


def test_dtc_seek_free_space_uses_speedj():
    """At d≫gate, DTC actuation must return DualLoop q̇ (teleop/approach live)."""
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(
            actuation="dtc",
            desired_force_n=5.0,
            align_first=False,
            max_tau_nm=40.0,
            wrench_kp=1.0,
            dtc_contact_dist_m=0.008,
            force_kp=0.01,
            approach_v_max_m_s=0.006,
            max_joint_velocity=1.0,
        )
    )
    R = np.eye(3)
    bridge.arm([0, 0, 0], R, R)
    bridge.push_tangent_field_linear([0.02, 0.0, 0.0], R, continuous=True)
    qdot = bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0, 0, 0],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 0, 0, 0, 0],
        signed_d=0.05,  # 5 cm away → speedJ seek
    )
    assert float(np.linalg.norm(qdot)) > 1e-4
    assert bridge.last_diag.get("actuation") == "dtc_seek_speedj"
    assert float(bridge.last_diag.get("dtc_use_speedj", 0.0)) > 0.5


def test_dtc_seek_near_uses_torque():
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(
            actuation="dtc",
            desired_force_n=5.0,
            align_first=False,
            max_tau_nm=40.0,
            wrench_kp=1.0,
            dtc_contact_dist_m=0.008,
            dtc_require_force=True,
            force_contact_threshold_N=0.6,
            dtc_startup_ramp_s=0.0,
            dtc_max_dtau_nm_s=0.0,
        )
    )
    R = np.eye(3)
    bridge.arm([0, 0, 0], R, R)
    # Geometry alone is not enough.
    qdot_air = bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0, 0, 0],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 0, 0, 0, 0],
        signed_d=0.002,
    )
    assert bridge.last_diag.get("actuation") == "dtc_seek_speedj"
    assert float(np.linalg.norm(qdot_air)) >= 0.0
    # Real compression → DTC.
    qdot = bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0, 0, 2.0],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 2.0, 0, 0, 0],
        signed_d=0.002,
    )
    assert np.allclose(qdot, 0.0)
    assert float(np.linalg.norm(bridge.last_tau)) > 0.5
    assert bridge.last_diag.get("actuation") == "dtc"
    assert float(bridge.last_diag.get("dtc_use_speedj", 1.0)) < 0.5


def test_dtc_seek_loaded_ignores_stale_signed_d():
    """High |Fs| must hand off to DTC even when VF signed_d is still large."""
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(
            actuation="dtc",
            desired_force_n=70.0,
            align_first=False,
            max_tau_nm=40.0,
            wrench_kp=0.35,
            dtc_contact_dist_m=0.015,
            dtc_require_force=True,
            dtc_min_fs_frac_of_fd=0.25,
            force_contact_threshold_N=0.6,
            max_wrench_n=80.0,
            dtc_startup_ramp_s=0.0,
            dtc_max_dtau_nm_s=0.0,
        )
    )
    R = np.eye(3)
    bridge.arm([0, 0, 0], R, R)
    # Matches hardware log: Fs≈45 N, d=+0.0285 (> geom gate).
    qdot = bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0, 0, 45.0],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 45.0, 0, 0, 0],
        signed_d=0.0285,
    )
    assert np.allclose(qdot, 0.0)
    assert bridge.last_diag.get("actuation") == "dtc"
    assert float(bridge.last_diag.get("dtc_use_speedj", 1.0)) < 0.5
    assert float(np.linalg.norm(bridge.last_tau)) > 0.5
    assert bridge._dtc_contact_ready(signed_d=0.0285, fs_n=45.0) is True
    assert bridge._dtc_contact_ready(signed_d=0.0285, fs_n=5.0) is False
    # Wrong force_sign → large negative Fs must still hand off (abs gate).
    assert bridge._dtc_contact_ready(signed_d=0.0285, fs_n=-115.0) is True
    # Near geom + tiny Fs must NOT hand off (live fault: Fs=0.85, d=6mm → |τ|=40).
    assert bridge._dtc_contact_ready(signed_d=0.0062, fs_n=0.85) is False
    assert bridge._dtc_contact_ready(signed_d=0.0062, fs_n=0.0) is False


def test_dtc_first_tick_not_slew_clamped_from_zero():
    """First hybrid DTC τ must not be slew-capped from 0 (path-deviation root)."""
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(
            actuation="dtc",
            desired_force_n=20.0,
            force_sign=1.0,
            align_first=False,
            max_tau_nm=40.0,
            wrench_kp=0.35,
            dtc_require_force=True,
            dtc_min_fs_frac_of_fd=0.25,
            force_contact_threshold_N=0.6,
            dtc_startup_ramp_s=1.5,
            dtc_startup_tau_nm=8.0,
            dtc_max_dtau_nm_s=20.0,
            max_wrench_n=40.0,
        )
    )
    R = np.eye(3)
    bridge.arm([0, 0, 0], R, R)
    # Loaded handoff (Fs≈22 like live log).
    bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0, 0, 22.0],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 22.0, 0, 0, 0],
        signed_d=0.001,
    )
    assert bridge.last_diag.get("actuation") == "dtc"
    # Must be well above the old 0.4 Nm first-tick slew ceiling.
    assert float(np.linalg.norm(bridge.last_tau)) > 2.0
    assert float(bridge.last_diag.get("command_scale", 0.0)) >= 0.3
    assert float(bridge.last_diag.get("tau_slew", 1.0)) < 0.5


def test_dtc_hysteresis_holds_through_soft_ramp_dip():
    """Once latched, stay on DTC when Fs dips below enter but above exit."""
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(
            actuation="dtc",
            desired_force_n=20.0,
            align_first=False,
            max_tau_nm=40.0,
            wrench_kp=0.35,
            dtc_require_force=True,
            dtc_min_fs_frac_of_fd=0.25,
            dtc_exit_fs_frac_of_enter=0.35,
            force_contact_threshold_N=0.6,
            dtc_startup_ramp_s=0.0,
            dtc_max_dtau_nm_s=0.0,
            dtc_min_hold_s=0.0,
            dtc_exit_hold_s=0.0,
            max_wrench_n=40.0,
        )
    )
    R = np.eye(3)
    bridge.arm([0, 0, 0], R, R)
    enter = bridge._dtc_fs_need_enter()  # 5.0
    exit_n = bridge._dtc_fs_need_exit()  # ~1.75
    assert enter == 5.0
    assert abs(exit_n - 0.35 * enter) < 1e-9 or exit_n >= 0.6

    # Enter at Fs=9.
    bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0, 0, 9.0],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 9.0, 0, 0, 0],
        signed_d=0.02,
    )
    assert bridge._dtc_latched is True
    assert bridge.last_diag.get("actuation") == "dtc"

    # Dip to Fs=4 (below enter=5, above exit≈1.75) — must stay DTC (no bounce).
    bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0, 0, 4.0],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 4.0, 0, 0, 0],
        signed_d=0.02,
    )
    assert bridge._dtc_latched is True
    assert bridge.last_diag.get("actuation") == "dtc"
    assert bridge._dtc_contact_ready(signed_d=0.02, fs_n=4.0, latched=True) is True
    assert bridge._dtc_contact_ready(signed_d=0.02, fs_n=4.0, latched=False) is False

    # Truly unload below exit → DualLoop.
    bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0, 0, 0.5],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 0.5, 0, 0, 0],
        signed_d=0.02,
    )
    assert bridge._dtc_latched is False
    assert bridge.last_diag.get("actuation") == "dtc_seek_speedj"


def test_dtc_min_hold_ignores_brief_fs_dip():
    """After handoff, stay on DTC through a brief Fs collapse (mode-switch bounce)."""
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(
            actuation="dtc",
            desired_force_n=20.0,
            force_sign=1.0,
            align_first=False,
            max_tau_nm=40.0,
            wrench_kp=0.35,
            dtc_require_force=True,
            dtc_min_fs_frac_of_fd=0.25,
            dtc_exit_fs_frac_of_enter=0.35,
            dtc_min_hold_s=1.5,
            dtc_exit_hold_s=0.35,
            dtc_startup_ramp_s=0.0,
            dtc_max_dtau_nm_s=0.0,
            max_wrench_n=40.0,
        )
    )
    R = np.eye(3)
    bridge.arm([0, 0, 0], R, R)
    bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0, 0, 11.0],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 11.0, 0, 0, 0],
        signed_d=0.018,
        tip_nn=0.007,
    )
    assert bridge._dtc_latched is True
    # Immediate Fs collapse like live log — must stay DTC during min hold.
    bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0, 0, -1.4],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, -1.4, 0, 0, 0],
        signed_d=0.018,
        tip_nn=0.011,
    )
    assert bridge._dtc_latched is True
    assert bridge.last_diag.get("actuation") == "dtc"


def test_dtc_hybrid_seek_stays_speedj_until_frac_fd():
    """Geom near + Fs≪frac·Fd must remain DualLoop (no early Eq.56 slam)."""
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(
            actuation="dtc",
            desired_force_n=30.0,
            align_first=False,
            max_tau_nm=40.0,
            wrench_kp=0.35,
            dtc_contact_dist_m=0.015,
            dtc_require_force=True,
            dtc_min_fs_frac_of_fd=0.25,
            force_contact_threshold_N=0.6,
            max_wrench_n=80.0,
            dtc_startup_ramp_s=1.5,
        )
    )
    R = np.eye(3)
    bridge.arm([0, 0, 0], R, R)
    qdot = bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0, 0, 0.85],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 0.85, 0, 0, 0],
        signed_d=0.0062,
    )
    assert bridge.last_diag.get("actuation") == "dtc_seek_speedj"
    assert float(bridge.last_diag.get("dtc_use_speedj", 0.0)) > 0.5
    assert float(np.linalg.norm(bridge.last_tau)) < 1e-6
    assert float(np.linalg.norm(qdot)) >= 0.0


def test_dtc_pure_align_orientation_only_then_seek_force():
    """dtc_pure ALIGN: W_L only. SEEK air: speedJ along n*. Near: Eq.56."""
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(
            actuation="dtc_pure",
            desired_force_n=50.0,
            align_first=True,
            align_enter_tol_deg=5.0,
            max_tau_nm=40.0,
            wrench_kp=0.25,
            dtc_contact_dist_m=0.008,
            dtc_require_force=True,
            force_contact_threshold_N=0.6,
            dtc_startup_ramp_s=0.0,
            dtc_max_dtau_nm_s=0.0,
            max_wrench_n=75.0,
        )
    )
    assert bridge.actuation == "dtc_pure"
    R = np.eye(3)
    R_tilt = R.copy()
    R_tilt[:, 2] = np.array([0.3, 0.0, 1.0])
    R_tilt[:, 2] /= np.linalg.norm(R_tilt[:, 2])
    R_tilt[:, 0] = np.cross(R_tilt[:, 1], R_tilt[:, 2])
    R_tilt[:, 0] /= np.linalg.norm(R_tilt[:, 0])
    R_tilt[:, 1] = np.cross(R_tilt[:, 2], R_tilt[:, 0])

    bridge.arm([0, 0, 0], R_tilt, R)
    qdot_align = bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R_tilt,
        R_field=R,
        force_base=[0, 0, 0],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 0, 0, 0, 0],
        signed_d=0.05,
    )
    assert np.allclose(qdot_align, 0.0)
    assert float(np.linalg.norm(bridge.last_tau)) > 0.0
    assert float(bridge.last_diag.get("phase_align", 0.0)) > 0.5
    # ALIGN must not dump Fd approach wrench (that snapped the live UR).
    assert float(bridge.last_diag.get("force_enable", 1.0)) < 0.5
    assert float(bridge.last_diag.get("W_C_norm", 1.0)) < 1e-6
    assert float(bridge.last_diag.get("dtc_use_speedj", 1.0)) < 0.5
    assert bridge.last_diag.get("actuation") == "dtc_pure_align"

    bridge.arm([0, 0, 0], R, R)
    qdot_seek = bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0, 0, 0],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 0, 0, 0, 0],
        signed_d=0.05,
    )
    # Air SEEK: speedJ along n* (open-loop DTC cannot close the gap).
    assert float(np.linalg.norm(qdot_seek)) > 1e-4
    assert float(bridge.last_diag.get("phase_align", 1.0)) < 0.5
    assert float(bridge.last_diag.get("force_enable", 0.0)) > 0.5
    assert float(bridge.last_diag.get("in_air", 0.0)) > 0.5
    assert float(bridge.last_diag.get("dtc_use_speedj", 0.0)) > 0.5
    assert bridge.last_diag.get("actuation") == "dtc_pure_air"
    assert np.allclose(bridge.last_tau, 0.0)
    # Near surface → Eq.56 DTC.
    qdot_near = bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0, 0, 0],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 0, 0, 0, 0],
        signed_d=0.002,
    )
    assert np.allclose(qdot_near, 0.0)
    assert float(np.linalg.norm(bridge.last_tau)) > 0.0
    assert float(bridge.last_diag.get("dtc_use_speedj", 1.0)) < 0.5
    assert bridge.last_diag.get("actuation") == "dtc_pure"
    assert abs(float(bridge.last_tau[2])) > 1.0


def test_dtc_tau_slew_limits_snap():
    """dtc_max_dtau_nm_s caps ‖Δτ‖/dt after the first tick (first tick is free)."""
    kwargs = dict(
        actuation="dtc_pure",
        desired_force_n=5.0,
        align_first=True,
        max_tau_nm=40.0,
        wrench_kp=1.0,
        dtc_startup_ramp_s=0.0,
        dtc_max_dtau_nm_s=5.0,
    )
    R = np.eye(3)
    R_tilt = R.copy()
    R_tilt[:, 2] = np.array([0.3, 0.0, 1.0])
    R_tilt[:, 2] /= np.linalg.norm(R_tilt[:, 2])
    R_tilt[:, 0] = np.cross(R_tilt[:, 1], R_tilt[:, 2])
    R_tilt[:, 0] /= np.linalg.norm(R_tilt[:, 0])
    R_tilt[:, 1] = np.cross(R_tilt[:, 2], R_tilt[:, 0])

    limited = ContactTeleopBridge(ContactTeleopConfig(**kwargs))
    limited.arm([0, 0, 0], R_tilt, R)
    limited.step(
        p_tcp=[0, 0, 0],
        R_tcp=R_tilt,
        R_field=R,
        force_base=[0, 0, 0],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 0, 0, 0, 0],
    )
    # First tick: no slew-from-zero (hybrid handoff safety).
    assert float(np.linalg.norm(limited.last_tau)) > 1.0
    assert float(limited.last_diag.get("tau_slew", 1.0)) < 0.5

    # Artificial jump from τ=0 → large: slew must engage.
    limited._tau_cmd_prev = np.zeros(6)
    limited.step(
        p_tcp=[0, 0, 0],
        R_tcp=R_tilt,
        R_field=R,
        force_base=[0, 0, 0],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 0, 0, 0, 0],
    )
    tau_slew = float(np.linalg.norm(limited.last_tau))
    assert tau_slew <= 5.0 * 0.02 + 1e-6
    assert float(limited.last_diag.get("tau_slew", 0.0)) > 0.5

    unlimited = ContactTeleopBridge(
        ContactTeleopConfig(**{**kwargs, "dtc_max_dtau_nm_s": 0.0})
    )
    unlimited.arm([0, 0, 0], R_tilt, R)
    unlimited.step(
        p_tcp=[0, 0, 0],
        R_tcp=R_tilt,
        R_field=R,
        force_base=[0, 0, 0],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 0, 0, 0, 0],
    )
    assert float(np.linalg.norm(unlimited.last_tau)) > tau_slew + 0.5


def test_eq56_command_scale_reduces_align_wrench():
    ctl = Eq56Controller(
        desired_force_n=5.0,
        wrench_kp=1.0,
        max_tau=50.0,
        approach_force_n=0.0,
        force_lpf_hz=0.0,
    )
    R = np.eye(3)
    R_tilt = R.copy()
    R_tilt[:, 2] = np.array([0.3, 0.0, 1.0])
    R_tilt[:, 2] /= np.linalg.norm(R_tilt[:, 2])
    R_tilt[:, 0] = np.cross(R_tilt[:, 1], R_tilt[:, 2])
    R_tilt[:, 0] /= np.linalg.norm(R_tilt[:, 0])
    R_tilt[:, 1] = np.cross(R_tilt[:, 2], R_tilt[:, 0])
    kwargs = dict(
        p_tcp=np.zeros(3),
        R_tcp=R_tilt,
        x_q=np.zeros(3),
        R_field=R,
        wrench_meas6=np.zeros(6),
        jacobian_ee=np.eye(6),
        qdot=np.zeros(6),
        v_tan_world=None,
        dt=0.02,
        enable_force=False,
    )
    tau_full = ctl.step(**kwargs, command_scale=1.0)
    ctl.reset()
    tau_soft = ctl.step(**kwargs, command_scale=0.0, max_tau_limit=2.0)
    assert float(np.linalg.norm(tau_full)) > 1.0
    assert float(np.linalg.norm(tau_soft)) < 0.25 * float(np.linalg.norm(tau_full))
    assert float(ctl.last_diag["command_scale"]) == 0.0


def test_eq56_raw_force_spike_retracts():
    ctl = Eq56Controller(
        desired_force_n=5.0,
        wrench_kp=1.0,
        max_tau=40.0,
        max_wrench_n=12.0,
        contact_threshold_n=0.6,
        approach_force_n=8.0,
        force_lpf_hz=5.0,
        spike_dF_dt_N_s=40.0,
        overshoot_margin_n=20.0,
    )
    R = np.eye(3)
    kwargs = dict(
        p_tcp=np.zeros(3),
        R_tcp=R,
        x_q=np.zeros(3),
        R_field=R,
        jacobian_ee=np.eye(6),
        qdot=np.zeros(6),
        v_tan_world=None,
        dt=0.02,
        enable_force=True,
    )
    for _ in range(30):
        ctl.step(**kwargs, wrench_meas6=np.zeros(6))
    tau = ctl.step(**kwargs, wrench_meas6=np.array([0.0, 0.0, 8.2, 0.0, 0.0, 0.0]))
    assert ctl.last_diag["spike"] > 0.5
    # Spike: hold zero contact wrench (not reverse yank).
    assert abs(float(tau[2])) < 1e-6


def test_eq56_wrong_sign_stops_push():
    ctl = Eq56Controller(
        desired_force_n=5.0,
        force_sign=1.0,
        wrench_kp=1.0,
        max_tau=40.0,
        approach_force_n=8.0,
        force_lpf_hz=0.0,
    )
    R = np.eye(3)
    kwargs = dict(
        p_tcp=np.zeros(3),
        R_tcp=R,
        x_q=np.zeros(3),
        R_field=R,
        jacobian_ee=np.eye(6),
        qdot=np.zeros(6),
        v_tan_world=None,
        dt=0.02,
        enable_force=True,
        signed_d=0.001,
    )
    ctl.step(**kwargs, wrench_meas6=np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0]))
    tau = ctl.step(
        **kwargs, wrench_meas6=np.array([0.0, 0.0, -25.0, 0.0, 0.0, 0.0])
    )
    assert ctl.last_diag["wrong_sign"] > 0.5
    assert float(ctl.last_diag["f_cmd_base_z"]) <= 1e-9
    assert float(tau[2]) <= 1e-9


def test_eq56_seek_approaches_despite_ft_gravity_bias():
    """Air SEEK must push along n* even if F/T gravity looks like 8 N compression."""
    ctl = Eq56Controller(
        desired_force_n=10.0,
        force_sign=1.0,
        wrench_kp=0.25,
        max_tau=40.0,
        approach_force_n=10.0,
        force_lpf_hz=0.0,
        max_wrench_n=15.0,
    )
    R = np.eye(3)
    tau = ctl.step(
        p_tcp=np.zeros(3),
        R_tcp=R,
        x_q=np.zeros(3),
        R_field=R,
        wrench_meas6=np.array([0.0, 0.0, 8.0, 0.0, 0.0, 0.0]),
        jacobian_ee=np.eye(6),
        qdot=np.zeros(6),
        v_tan_world=None,
        dt=0.02,
        enable_force=True,
        signed_d=0.05,
    )
    assert ctl.last_diag["real_contact"] < 0.5
    assert float(ctl.last_diag["W_C_norm"]) > 5.0
    assert float(tau[2]) > 1.0


def test_eq56_air_seek_zeros_w_l_and_springs_on_signed_d():
    """Air SEEK: W_L=0 (τ for approach only); ‖W_C‖ ≥ k·signed_d along n*."""
    ctl = Eq56Controller(
        desired_force_n=10.0,
        force_sign=1.0,
        wrench_kp=0.25,
        max_tau=40.0,
        approach_force_n=10.0,
        approach_stiffness_n_per_m=400.0,
        force_lpf_hz=0.0,
        max_wrench_n=40.0,
        gains=Eq56Gains(
            K_line=np.array([200.0, 200.0, 200.0, 125.0, 125.0, 125.0]),
            D_twist=np.array([10.0, 10.0, 10.0, 150.0, 150.0, 50.0]),
        ),
    )
    R = np.eye(3)
    R_tilt = R.copy()
    R_tilt[:, 2] = np.array([0.05, 0.0, 1.0])
    R_tilt[:, 2] /= np.linalg.norm(R_tilt[:, 2])
    R_tilt[:, 0] = np.cross(R_tilt[:, 1], R_tilt[:, 2])
    R_tilt[:, 0] /= np.linalg.norm(R_tilt[:, 0])
    R_tilt[:, 1] = np.cross(R_tilt[:, 2], R_tilt[:, 0])
    d = 0.033
    tau = ctl.step(
        p_tcp=np.zeros(3),
        R_tcp=R_tilt,
        x_q=np.zeros(3),
        R_field=R,
        wrench_meas6=np.zeros(6),
        jacobian_ee=np.eye(6),
        qdot=np.zeros(6),
        v_tan_world=None,
        dt=0.02,
        enable_force=True,
        signed_d=d,
    )
    assert float(ctl.last_diag["W_L_norm"]) < 1e-9
    assert float(ctl.last_diag["in_air"]) > 0.5
    # Spring: max(Fd, k·d) = max(10, 400*0.033)=13.2
    assert float(ctl.last_diag["W_C_norm"]) >= 13.0
    assert float(ctl.last_diag["f_cmd_base_z"]) > 10.0
    assert float(tau[2]) > 10.0


def test_eq56_fault_unloads():
    ctl = Eq56Controller(
        desired_force_n=5.0,
        force_sign=1.0,
        wrench_kp=1.0,
        max_tau=40.0,
        force_lpf_hz=0.0,
        fault_abs_fs_N=20.0,
        fault_fd_mult=3.0,
    )
    R = np.eye(3)
    tau = ctl.step(
        p_tcp=np.zeros(3),
        R_tcp=R,
        x_q=np.zeros(3),
        R_field=R,
        wrench_meas6=np.array([0.0, 0.0, 26.0, 0.0, 0.0, 0.0]),
        jacobian_ee=np.eye(6),
        qdot=np.zeros(6),
        v_tan_world=None,
        dt=0.02,
        enable_force=True,
    )
    assert ctl.last_diag["fault_fs"] > 0.5
    # Soft hold toward Fd (into-surface), not reverse yank.
    assert float(tau[2]) > 0.0
    assert float(ctl.last_diag["f_cmd_base_z"]) > 0.0


def test_dtc_pure_enter_does_not_snap_to_field():
    """First tick after Enter must not dump full K_line τ toward R_field."""
    kwargs = dict(
        actuation="dtc_pure",
        desired_force_n=5.0,
        align_first=True,
        max_tau_nm=20.0,
        wrench_kp=1.0,
        dtc_startup_tau_nm=2.0,
        dtc_max_dtau_nm_s=0.0,  # isolate ramp vs slew
    )
    R = np.eye(3)
    R_tilt = R.copy()
    R_tilt[:, 2] = np.array([0.3, 0.0, 1.0])
    R_tilt[:, 2] /= np.linalg.norm(R_tilt[:, 2])
    R_tilt[:, 0] = np.cross(R_tilt[:, 1], R_tilt[:, 2])
    R_tilt[:, 0] /= np.linalg.norm(R_tilt[:, 0])
    R_tilt[:, 1] = np.cross(R_tilt[:, 2], R_tilt[:, 0])

    snap = ContactTeleopBridge(ContactTeleopConfig(**kwargs, dtc_startup_ramp_s=0.0))
    snap.arm([0, 0, 0], R_tilt, R)
    snap.step(
        p_tcp=[0, 0, 0],
        R_tcp=R_tilt,
        R_field=R,
        force_base=[0, 0, 0],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 0, 0, 0, 0],
    )
    tau_snap = float(np.linalg.norm(snap.last_tau))

    soft = ContactTeleopBridge(ContactTeleopConfig(**kwargs, dtc_startup_ramp_s=3.0))
    soft.arm([0, 0, 0], R_tilt, R)
    soft.step(
        p_tcp=[0, 0, 0],
        R_tcp=R_tilt,
        R_field=R,
        force_base=[0, 0, 0],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 0, 0, 0, 0],
    )
    tau_soft = float(np.linalg.norm(soft.last_tau))
    assert tau_snap > 1.0
    assert tau_soft < 2.5
    assert tau_soft < 0.5 * tau_snap
    assert float(soft.last_diag.get("command_scale", 1.0)) < 0.05
    assert float(soft.last_diag.get("phase_align", 0.0)) > 0.5


def test_dtc_startup_ramp_ignores_arm_to_first_tick_delay():
    """Ramp clock starts at first step_tau — not at Enter (WoS can stall seconds)."""
    import time

    kwargs = dict(
        actuation="dtc_pure",
        desired_force_n=50.0,
        align_first=True,
        max_tau_nm=100.0,
        wrench_kp=0.25,
        dtc_startup_ramp_s=3.0,
        dtc_startup_tau_nm=2.0,
        max_wrench_n=75.0,
        K_line=[30.0, 30.0, 30.0, 750.0, 750.0, 300.0],
    )
    R = np.eye(3)
    R_tilt = R.copy()
    R_tilt[:, 2] = np.array([0.25, 0.0, 1.0])
    R_tilt[:, 2] /= np.linalg.norm(R_tilt[:, 2])
    R_tilt[:, 0] = np.cross(R_tilt[:, 1], R_tilt[:, 2])
    R_tilt[:, 0] /= np.linalg.norm(R_tilt[:, 0])
    R_tilt[:, 1] = np.cross(R_tilt[:, 2], R_tilt[:, 0])

    bridge = ContactTeleopBridge(ContactTeleopConfig(**kwargs))
    bridge.arm([0, 0, 0], R_tilt, R)
    # Simulate VF/WoS stall after Enter (live logs showed query_ms≈2s).
    time.sleep(0.35)
    bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R_tilt,
        R_field=R,
        force_base=[0, 0, 0],
        jacobian=np.eye(6),
        dt=0.02,
        wrench6=[0, 0, 0, 0, 0, 0],
    )
    scale = float(bridge.last_diag.get("command_scale", 1.0))
    tau_n = float(np.linalg.norm(bridge.last_tau))
    max_lim = float(bridge.last_diag.get("max_tau_limit", 100.0))
    # Must still be at the soft end of the ramp, not ~0.35/3 into it.
    assert scale < 0.05, f"command_scale={scale} (ramp started at arm?)"
    assert max_lim <= 2.5 + 1e-6, f"max_tau_limit={max_lim}"
    assert tau_n <= 2.5 + 1e-6, f"‖τ‖={tau_n} exceeds dtc_startup_tau_nm"
