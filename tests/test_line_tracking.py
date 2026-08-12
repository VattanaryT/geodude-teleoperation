"""Tests for Phase 1.5 line tracking + soft stand-off."""
from __future__ import annotations

import numpy as np

from geodude_teleoperation.teleop.line_tracking import (
    LineTrackingConfig,
    apply_line_tracking_pose_error,
    combine_normal_feedforward,
    config_from_args,
    kill_normal_linear_error,
    outward_normal_from_R_field,
    soft_standoff_velocity,
    split_field_linear_tangent_normal,
    snap_ref_position_to_line,
)


def test_kill_normal_zeros_outward_component():
    n = np.array([0.0, 0.0, 1.0])
    err = np.array([0.1, -0.2, 0.5, 0.0, 0.0, 0.1])
    out = kill_normal_linear_error(err, n)
    assert abs(out[2]) < 1e-9
    assert abs(out[0] - 0.1) < 1e-9
    assert abs(out[5] - 0.1) < 1e-9


def test_snap_ref_to_line():
    p_cur = np.zeros(3)
    p_ref = np.array([1.0, 2.0, 3.0])
    n = np.array([0.0, 0.0, 1.0])
    snapped = snap_ref_position_to_line(p_cur, p_ref, n)
    assert abs(snapped[2]) < 1e-9
    assert abs(snapped[0] - 1.0) < 1e-9


def test_soft_standoff_moves_toward_surface_when_too_far():
    n_out = np.array([0.0, 0.0, 1.0])
    # d=0.01, d*=0.002 → too far outside → v along -n_out
    v = soft_standoff_velocity(0.01, d_star=0.002, gain=1.0, n_outward=n_out)
    assert v[2] < 0


def test_outward_from_approach_R_field():
    # Approach into -z ⇒ outward +z
    R = np.eye(3)
    R[:, 2] = [0.0, 0.0, -1.0]
    n = outward_normal_from_R_field(R)
    assert np.allclose(n, [0, 0, 1])


def test_apply_line_tracking_pose_error():
    R = np.eye(3)
    R[:, 2] = [0.0, 0.0, -1.0]  # into -z
    err = np.array([0.0, 0.0, 0.4, 0, 0, 0])
    # outward = +z; kill err_z
    out = apply_line_tracking_pose_error(err, R, enabled=True)
    assert abs(out[2]) < 1e-12
    out_off = apply_line_tracking_pose_error(err, R, enabled=False)
    assert abs(out_off[2] - 0.4) < 1e-12


def test_split_field_linear():
    t, n = split_field_linear_tangent_normal([0.1, -0.2, 0.3])
    assert np.allclose(t, [0.1, -0.2, 0.0])
    assert abs(n - 0.3) < 1e-12


def test_combine_includes_standoff_and_keyboard():
    cfg = LineTrackingConfig(enabled=True, standoff_enabled=True, standoff_gain=1.0)
    R = np.eye(3)
    R[:, 2] = [0.0, 0.0, -1.0]
    v = combine_normal_feedforward(
        cfg=cfg,
        signed_d=0.01,
        R_field=R,
        keyboard_v_world=np.array([0.0, 0.0, -0.01]),
    )
    assert v[2] < 0  # both standoff and keyboard into -z / -n_out


def test_config_from_args_soft_standoff_implies_line():
    class A:
        line_tracking = False
        soft_standoff = True
        standoff_distance = 0.003
        standoff_gain = 2.0
        standoff_max_speed = 0.04
        no_keyboard_normal_ff = False

    cfg = config_from_args(A())
    assert cfg.enabled is True
    assert cfg.standoff_enabled is True
    assert abs(cfg.standoff_distance_m - 0.003) < 1e-12
