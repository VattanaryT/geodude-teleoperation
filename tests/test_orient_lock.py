"""Enter → R_field orientation lock + startup UR motion-limit ramp."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from geodude_teleoperation.rtde.orient_lock import (
    EnableOrientLock,
    MotionLimits,
    apply_orient_lock_reference,
    blend_startup_motion_limits,
    orientation_error_deg,
    recapture_pose_reference,
    twist_linear_only,
)
from geodude_teleoperation.se3 import NumpyTwist


def test_orientation_error_deg_identity_is_zero():
    R = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()
    assert orientation_error_deg(R, R) < 1e-9


def test_orientation_error_deg_known_angle():
    R0 = np.eye(3)
    R1 = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    assert abs(orientation_error_deg(R0, R1) - 30.0) < 1e-6


def test_blend_at_t0_uses_startup_except_linear():
    startup = MotionLimits(
        max_joint_velocity=0.15,
        rtde_accel=0.2,
        max_linear_speed=0.01,
        max_angular_speed=0.08,
    )
    general = MotionLimits(
        max_joint_velocity=1.0,
        rtde_accel=1.0,
        max_linear_speed=0.05,
        max_angular_speed=0.4,
    )
    out = blend_startup_motion_limits(
        t_since_enable=0.0,
        ramp_s=2.0,
        startup=startup,
        general=general,
    )
    assert out.max_joint_velocity == 0.15
    assert out.rtde_accel == 0.2
    assert out.max_angular_speed == 0.08
    assert out.max_linear_speed == 0.05


def test_blend_after_ramp_uses_general():
    startup = MotionLimits(0.15, 0.2, 0.01, 0.08)
    general = MotionLimits(1.0, 1.0, 0.05, 0.4)
    out = blend_startup_motion_limits(
        t_since_enable=2.0,
        ramp_s=2.0,
        startup=startup,
        general=general,
    )
    assert out.max_joint_velocity == 1.0
    assert out.rtde_accel == 1.0
    assert out.max_angular_speed == 0.4
    assert out.max_linear_speed == 0.05


def test_blend_mid_ramp_interpolates():
    startup = MotionLimits(0.0, 0.0, 0.01, 0.0)
    general = MotionLimits(1.0, 2.0, 0.05, 0.4)
    out = blend_startup_motion_limits(
        t_since_enable=1.0,
        ramp_s=2.0,
        startup=startup,
        general=general,
    )
    assert abs(out.max_joint_velocity - 0.5) < 1e-12
    assert abs(out.rtde_accel - 1.0) < 1e-12
    assert abs(out.max_angular_speed - 0.2) < 1e-12


def test_blend_zero_ramp_is_general():
    startup = MotionLimits(0.15, 0.2, 0.01, 0.08)
    general = MotionLimits(1.0, 1.0, 0.05, 0.4)
    out = blend_startup_motion_limits(
        t_since_enable=0.0,
        ramp_s=0.0,
        startup=startup,
        general=general,
    )
    assert out.max_joint_velocity == 1.0
    assert out.rtde_accel == 1.0


def test_enable_orient_lock_settles_below_tol():
    lock = EnableOrientLock()
    assert lock.locking(ang_err_deg=20.0, tol_deg=2.0) is False
    lock.on_enable(now=1.0)
    assert lock.t_since_enable(1.5) == 0.5
    assert lock.locking(ang_err_deg=20.0, tol_deg=2.0) is True
    assert lock.locking(ang_err_deg=1.0, tol_deg=2.0) is False
    assert lock.locking(ang_err_deg=20.0, tol_deg=2.0) is False  # latched


def test_enable_orient_lock_freezes_R_star():
    lock = EnableOrientLock()
    lock.on_enable(now=0.0)
    R0 = Rotation.from_euler("z", 20, degrees=True).as_matrix()
    R1 = Rotation.from_euler("z", 80, degrees=True).as_matrix()
    assert lock.capture_R_star(R0) is not None
    assert np.allclose(lock.R_star, R0)
    lock.capture_R_star(R1)
    assert np.allclose(lock.R_star, R0)


def test_enable_orient_lock_holds_until_second_enter():
    lock = EnableOrientLock()
    lock.on_enable(now=0.0)
    assert lock.motion_hold()
    assert not lock.awaiting_approve
    lock.locking(ang_err_deg=20.0, tol_deg=2.0)
    assert lock.motion_hold()
    lock.locking(ang_err_deg=0.5, tol_deg=2.0)
    assert lock.settled
    assert lock.awaiting_approve
    assert lock.motion_hold()
    assert not lock.approved
    lock.approve()
    assert lock.approved
    assert not lock.awaiting_approve
    assert not lock.motion_hold()
    assert lock.vf_frozen is True


def test_enable_orient_lock_toggle_pose_hold_keeps_vf_frozen():
    lock = EnableOrientLock()
    lock.on_enable(now=0.0)
    lock.approve()
    assert lock.vf_frozen is True
    assert lock.pose_hold is False
    assert lock.motion_hold() is False
    assert lock.toggle_pose_hold() is True
    assert lock.pose_hold is True
    assert lock.vf_frozen is True
    assert lock.toggle_pose_hold() is False
    assert lock.pose_hold is False
    assert lock.vf_frozen is True


def test_recapture_pose_reference_matches_live_tcp():
    class _Kin:
        _T_ref = np.eye(4)
        _T_ref[:3, 3] = [9.0, 9.0, 9.0]

    tcp = np.array([0.41, -0.19, 0.32, 0.2, -0.1, 0.05])
    kin = _Kin()
    assert recapture_pose_reference(kin, tcp)
    from geodude_teleoperation.se3 import tcp_pose_to_matrix

    T = tcp_pose_to_matrix(tcp)
    assert np.allclose(kin._T_ref, T)


def test_blend_while_locking_zeros_linear_speed():
    startup = MotionLimits(0.15, 0.2, 0.01, 0.08)
    general = MotionLimits(1.0, 1.0, 0.05, 0.4)
    out = blend_startup_motion_limits(
        t_since_enable=0.0,
        ramp_s=2.0,
        startup=startup,
        general=general,
        locking=True,
    )
    assert out.max_linear_speed == 0.0
    assert out.max_joint_velocity == 0.15
    assert out.max_angular_speed == 0.08


def test_apply_orient_lock_reference_follows_tcp_position():
    class _Kin:
        _T_ref = None

    R_star = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()
    tcp0 = np.array([0.4, -0.2, 0.3, 0.0, 0.0, 0.0])
    tcp1 = np.array([0.41, -0.19, 0.32, 0.1, 0.0, 0.0])
    kin = _Kin()
    assert apply_orient_lock_reference(kin, tcp0, R_star)
    assert np.allclose(kin._T_ref[:3, 3], tcp0[:3])
    assert np.allclose(kin._T_ref[:3, :3], R_star)
    apply_orient_lock_reference(kin, tcp1, R_star)
    assert np.allclose(kin._T_ref[:3, 3], tcp1[:3])
    assert np.allclose(kin._T_ref[:3, :3], R_star)


def test_twist_linear_only_drops_angular():
    tw = NumpyTwist.from_array([0.1, -0.2, 0.3, 0.04, -0.05, 0.06])
    out = twist_linear_only(tw)
    assert np.allclose(out.get_angular(), 0.0)
    assert np.allclose(out.get_linear(), [0.04, -0.05, 0.06])
