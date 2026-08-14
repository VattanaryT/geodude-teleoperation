"""CGA line tracking + Eq. 56 torque assembly (pygafro)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pygafro")

from geodude_teleoperation.cga.line_tracker import alignment_twist_se3
from geodude_teleoperation.cga.tactile_law import (
    Eq56Controller,
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
    # Unloaded + small d must stay on DualLoop (air-gap DTC freeze).
    qdot_air = bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R_tilt,
        force_base=[0, 0, 0],
        jacobian=np.eye(6),
        dt=0.002,
        wrench6=[0, 0, 0, 0, 0, 0],
        signed_d=0.002,
    )
    assert bridge.last_diag.get("actuation") == "dtc_seek_speedj"
    assert float(bridge.last_diag.get("dtc_use_speedj", 0.0)) > 0.5
    # Compression along tool+Z (= world Z here after re-arm with aligned R).
    bridge.arm([0, 0, 0], R, R)
    qdot = bridge.step(
        p_tcp=[0, 0, 0],
        R_tcp=R,
        R_field=R,
        force_base=[0, 0, 2.0],  # tool-frame Fz → Fs ≥ threshold
        jacobian=np.eye(6),
        dt=0.002,
        wrench6=[0, 0, 2.0, 0, 0, 0],
        signed_d=0.002,
    )
    assert np.allclose(qdot, 0.0)
    assert float(np.linalg.norm(bridge.last_tau)) > 0.0
    assert bridge.last_diag.get("actuation") == "dtc"


def test_print_actuation_banner_yellow(capsys):
    print_actuation_banner("dtc", desired_force_n=5.0)
    out = capsys.readouterr().out
    assert "ACTUATION=DTC" in out
    assert "\033[93m" in out
    print_actuation_banner("speedj")
    out = capsys.readouterr().out
    assert "ACTUATION=speedJ" in out


def test_eq56_controller_tick():
    ctl = Eq56Controller(desired_force_n=5.0, wrench_kp=1.0, max_tau=50.0)
    R = np.eye(3)
    tau = ctl.step(
        p_tcp=np.zeros(3),
        R_tcp=R,
        x_q=np.array([0.02, 0.0, 0.0]),
        R_field=R,
        wrench_meas6=np.zeros(6),
        jacobian_ee=np.eye(6),
        qdot=np.zeros(6),
        v_tan_world=np.array([0.01, 0.0, 0.0]),
        dt=0.002,
        enable_force=True,
    )
    assert tau.shape == (6,)
    assert ctl.last_diag["W_tan_norm"] > 0.0


def test_eq56_overshoot_unloads_not_pushes():
    """Fs ≫ Fd must command retract (negative tool-Z), not saturate push."""
    ctl = Eq56Controller(
        desired_force_n=5.0,
        wrench_kp=0.25,
        max_tau=40.0,
        max_wrench_n=12.0,
        overshoot_margin_n=2.0,
        retract_force_n=8.0,
        contact_threshold_n=0.6,
        approach_force_n=8.0,
    )
    R = np.eye(3)
    # Tool-frame compression already 28 N.
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
    assert ctl.last_diag["W_C_norm"] <= 12.0 + 1e-6
    # Retract: W_C_z < 0 ⇒ τ_z < 0 for J=I with +Jᵀ W_C convention... 
    # W_C_ee_z = -retract; W_base_z = -retract; τ = +Jᵀ W = negative z.
    assert float(tau[2]) < 0.0
    assert float(ctl.last_diag["f_cmd_base_z"]) < 0.0


def test_eq56_pid_ignores_lateral_force():
    ctl = Eq56Controller(
        desired_force_n=5.0,
        wrench_kp=1.0,
        max_tau=50.0,
        max_wrench_n=30.0,
        contact_threshold_n=0.6,
        approach_force_n=0.0,
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
    )
    # At Fd=Fs_z=5, normal error ~0; lateral must not create large W_C.
    assert ctl.last_diag["W_C_norm"] < 1.0
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
