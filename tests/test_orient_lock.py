"""Enter → R* lock, startup ramp, second-Enter approve, `f` freeze."""

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
)


def test_startup_ramp_zeros_linear_while_locking():
    startup = MotionLimits(0.15, 0.2, 0.01, 0.08)
    general = MotionLimits(1.0, 1.0, 0.05, 0.4)
    t0 = blend_startup_motion_limits(
        t_since_enable=0.0, ramp_s=2.0, startup=startup, general=general, locking=True
    )
    assert t0.max_linear_speed == 0.0
    assert t0.max_joint_velocity == 0.15
    mid = blend_startup_motion_limits(
        t_since_enable=1.0, ramp_s=2.0, startup=startup, general=general
    )
    assert abs(mid.max_joint_velocity - 0.575) < 1e-9  # 0.15 → 1.0 at 50%
    done = blend_startup_motion_limits(
        t_since_enable=2.0, ramp_s=2.0, startup=startup, general=general
    )
    assert done.max_joint_velocity == 1.0
    assert abs(orientation_error_deg(np.eye(3), Rotation.from_euler("z", 30, degrees=True).as_matrix()) - 30.0) < 1e-6


def test_enable_holds_until_second_enter_then_f_unfreezes():
    lock = EnableOrientLock()
    lock.on_enable(now=0.0)
    R0 = Rotation.from_euler("z", 20, degrees=True).as_matrix()
    assert lock.capture_R_star(R0) is not None
    lock.capture_R_star(Rotation.from_euler("z", 80, degrees=True).as_matrix())
    assert np.allclose(lock.R_star, R0)
    assert lock.locking(ang_err_deg=20.0, tol_deg=2.0)
    assert lock.motion_hold()
    lock.locking(ang_err_deg=0.5, tol_deg=2.0)
    assert lock.awaiting_approve and lock.motion_hold()
    lock.approve()
    assert lock.approved and lock.vf_frozen and not lock.motion_hold()
    assert lock.toggle_pose_hold() is True
    assert lock.pose_hold and lock.vf_frozen
    assert lock.toggle_pose_hold() is False
    assert not lock.pose_hold and not lock.vf_frozen


def test_orient_lock_reference_follows_tcp_position_not_stale_pose():
    class _Kin:
        _T_ref = np.eye(4)
        _T_ref[:3, 3] = [9.0, 9.0, 9.0]

    R_star = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()
    tcp = np.array([0.41, -0.19, 0.32, 0.2, -0.1, 0.05])
    kin = _Kin()
    assert recapture_pose_reference(kin, tcp)
    from geodude_teleoperation.se3 import tcp_pose_to_matrix

    assert np.allclose(kin._T_ref, tcp_pose_to_matrix(tcp))
    apply_orient_lock_reference(kin, tcp, R_star)
    assert np.allclose(kin._T_ref[:3, 3], tcp[:3])
    assert np.allclose(kin._T_ref[:3, :3], R_star)
