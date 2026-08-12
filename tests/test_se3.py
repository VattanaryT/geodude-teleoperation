"""Numpy SE3 helpers for gafropy-free RTDE teleop."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from geodude_teleoperation.se3 import (
    NumpyTwist,
    body_twist_increment,
    map_jacobian_to_point,
    pose_error_current_minus_ref,
    tcp_pose_to_matrix,
)


def test_numpy_twist_from_array_blade_order():
    # [e12, e13, e23, e1i, e2i, e3i] = [angular, linear]
    tw = NumpyTwist.from_array([0.1, 0.0, 0.0, 0.0, 0.0, 0.05])
    assert np.allclose(tw.get_angular(), [0.1, 0.0, 0.0])
    assert np.allclose(tw.get_linear(), [0.0, 0.0, 0.05])


def test_body_twist_increment_translates_in_ee_frame():
    T = np.eye(4)
    T[:3, 3] = [1.0, 0.0, 0.0]
    tw = NumpyTwist.from_array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
    T2 = body_twist_increment(T, tw)
    assert np.allclose(T2[:3, 3], [1.1, 0.0, 0.0], atol=1e-9)


def test_map_jacobian_identity_when_same_point():
    J = np.random.default_rng(0).normal(size=(6, 6))
    p = np.array([0.2, 0.0, 0.4])
    assert np.allclose(map_jacobian_to_point(J, p, p), J)


def test_map_jacobian_shifts_linear_by_omega_cross_r():
    J = np.zeros((6, 6))
    J[3:, 0] = [0.0, 0.0, 1.0]  # pure ωz on joint 0
    p_f = np.zeros(3)
    p_t = np.array([0.1, 0.0, 0.0])
    Jt = map_jacobian_to_point(J, p_f, p_t)
    # v = ω × r = ẑ × 0.1 x̂ = 0.1 ŷ
    assert np.allclose(Jt[:3, 0], [0.0, 0.1, 0.0], atol=1e-12)
    assert np.allclose(Jt[3:, 0], [0.0, 0.0, 1.0])


def test_pose_error_and_qdot_drive_toward_ref():
    from geodude_teleoperation.teleop.controller import qdot_from_error

    T_cur = np.eye(4)
    T_ref = np.eye(4)
    T_ref[:3, 3] = [0.0, 0.0, 0.1]
    err = pose_error_current_minus_ref(T_cur, T_ref)
    assert err[2] < 0  # current below ref → negative z error
    J = np.eye(6)
    qdot = qdot_from_error(err, J, gain=1.0, damping=0.0, max_joint_velocity=10.0)
    assert qdot[2] > 0  # move +z toward ref


def test_tcp_pose_to_matrix():
    R = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix()
    rotvec = Rotation.from_matrix(R).as_rotvec()
    pose = [0.1, 0.2, 0.3, *rotvec]
    T = tcp_pose_to_matrix(pose)
    assert np.allclose(T[:3, 3], [0.1, 0.2, 0.3])
    assert np.allclose(T[:3, :3], R, atol=1e-9)
