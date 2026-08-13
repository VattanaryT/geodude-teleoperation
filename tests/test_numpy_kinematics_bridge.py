"""Numpy UR kinematics bridge for RTDE live backup."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from geodude_teleoperation.rtde.numpy_kinematics_bridge import NumpyUrKinematicsBridge
from geodude_teleoperation.se3 import NumpyTwist
from geodude_teleoperation.rtde.ur5e_kinematics import forward_kinematics


class _FakeVF:
    """Identity project; exposes a field basis for twist-frame tests."""

    def __init__(self, R_field=None, distance=0.05):
        self._d = float(distance)
        self._R = (
            np.eye(3)
            if R_field is None
            else np.asarray(R_field, dtype=float).reshape(3, 3)
        )

    @property
    def status(self):
        return {"R_field_base": self._R}

    def project(self, motor):
        from geodude_teleoperation.se3 import _DuckMotor

        return _DuckMotor(motor.get_translator(), self._R)

    def get_distance(self, position):
        return self._d


def test_try_create_returns_bridge():
    kin = NumpyUrKinematicsBridge.try_create()
    assert kin is not None
    assert kin.backend_name == "numpy"


def test_compute_qdot_moves_toward_reference():
    kin = NumpyUrKinematicsBridge.try_create()
    assert kin is not None
    q = np.array([0.0, -1.2, 1.4, -1.6, -1.57, 0.0])
    T_f, _ = forward_kinematics(q)
    tcp = np.concatenate(
        [T_f[:3, 3], Rotation.from_matrix(T_f[:3, :3]).as_rotvec()]
    )

    kin.sync_joints(q)
    kin.set_tcp_pose(tcp)
    kin.ensure_reference_from_tcp()

    tw = NumpyTwist.from_array([0.0, 0.0, 0.0, 0.0, 0.0, 0.02])
    kin.update_command(tw, vf=_FakeVF(), min_scale=1.0, max_scale=1.0)

    qdot = kin.compute_qdot(
        gain=16.0, damping=0.05, max_joint_velocity=1.0
    )
    assert qdot.shape == (6,)
    assert np.linalg.norm(qdot) > 1e-4


def test_vf_ad_moves_along_field_y_not_body_y():
    """A/D are field ±y; must not be interpreted on misaligned TCP body axes."""
    kin = NumpyUrKinematicsBridge.try_create()
    assert kin is not None

    # TCP at origin with body axes = world; field y = world x (90° about z).
    R_tcp = np.eye(3)
    R_field = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    tcp = np.zeros(6)
    tcp[3:] = Rotation.from_matrix(R_tcp).as_rotvec()

    kin.sync_joints(np.zeros(6))
    kin.set_tcp_pose(tcp)
    kin.ensure_reference_from_tcp()

    # d → linear +y in field frame (keyboard index 4).
    step = 0.04
    tw = NumpyTwist.from_array([0.0, 0.0, 0.0, 0.0, step, 0.0])
    vf = _FakeVF(R_field=R_field, distance=0.05)
    kin.update_command(tw, vf=vf, min_scale=1.0, max_scale=1.0)

    T_ref = kin._T_ref
    assert T_ref is not None
    # linear_scale = 0.25 * factor; at d=0.05 → factor=1 → 0.25*step along field +y
    expected_dp = R_field @ np.array([0.0, 0.25 * step, 0.0])
    assert np.allclose(T_ref[:3, 3], expected_dp, atol=1e-9)
    # Pure linear keeps TCP orientation (no R_field snap).
    assert np.allclose(T_ref[:3, :3], R_tcp, atol=1e-9)
    # Must NOT have moved along body +y (world y).
    assert abs(T_ref[1, 3]) < 1e-9


def test_vf_pure_linear_keeps_tcp_orientation():
    kin = NumpyUrKinematicsBridge.try_create()
    assert kin is not None
    R_field = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()
    # Misaligned TCP — old path snapped R_ref to R_field and saturated q̇.
    R_tcp = Rotation.from_euler("xyz", [40, 5, -15], degrees=True).as_matrix()
    tcp = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0])
    tcp[3:] = Rotation.from_matrix(R_tcp).as_rotvec()
    kin.sync_joints(np.zeros(6))
    kin.set_tcp_pose(tcp)
    kin.ensure_reference_from_tcp()

    tw = NumpyTwist.from_array([0.0, 0.0, 0.0, 0.0, -0.04, 0.0])  # a
    kin.update_command(tw, vf=_FakeVF(R_field=R_field), min_scale=1.0, max_scale=1.0)
    assert np.allclose(kin._T_ref[:3, :3], R_tcp, atol=1e-9)
    qdot = kin.compute_qdot(
        gain=4.0,
        damping=0.05,
        max_joint_velocity=0.5,
        vf=_FakeVF(R_field=R_field),
        max_linear_speed=0.05,
        max_angular_speed=0.4,
    )
    # Linear-only step must not demand large rotational joint rates.
    assert float(np.linalg.norm(qdot)) < 0.5


def test_max_linear_speed_zero_kills_translation_error():
    """Lock-in-place: max_linear_speed=0 must not chase a stale T_ref position."""
    kin = NumpyUrKinematicsBridge.try_create()
    assert kin is not None
    q = np.array([0.0, -1.2, 1.4, -1.6, -1.57, 0.0])
    T_f, _ = forward_kinematics(q)
    R_tcp = T_f[:3, :3]
    tcp = np.concatenate(
        [T_f[:3, 3], Rotation.from_matrix(R_tcp).as_rotvec()]
    )
    R_star = Rotation.from_euler("xyz", [25, -35, 40], degrees=True).as_matrix()
    kin.sync_joints(q)
    kin.set_tcp_pose(tcp)
    kin.ensure_reference_from_tcp()
    kin._T_ref = kin._T_ref.copy()
    kin._T_ref[:3, :3] = R_star
    kin._T_ref[:3, 3] = T_f[:3, 3] + np.array([0.08, -0.05, 0.02])

    qdot = kin.compute_qdot(
        gain=4.0,
        damping=0.05,
        max_joint_velocity=0.15,
        max_linear_speed=0.0,
        max_angular_speed=0.08,
    )
    J = None
    from geodude_teleoperation.rtde.ur5e_kinematics import compute_jacobian
    from geodude_teleoperation.se3 import map_jacobian_to_point, tcp_pose_to_matrix

    T_tcp = tcp_pose_to_matrix(tcp)
    J_f = compute_jacobian(q)
    J = map_jacobian_to_point(J_f, T_f[:3, 3], T_tcp[:3, 3])
    twist = J @ qdot
    # Must not command a large restorative TCP translation.
    assert float(np.linalg.norm(twist[:3])) < 0.01, twist[:3]
    assert float(np.linalg.norm(qdot)) > 1e-4


def test_idle_ensure_reference_does_not_chase_field_orientation():
    """Enter alone must not yank TCP toward live R_field (idle wobble root cause)."""
    kin = NumpyUrKinematicsBridge.try_create()
    assert kin is not None
    q = np.array([0.0, -1.2, 1.4, -1.6, -1.57, 0.0])
    T_f, _ = forward_kinematics(q)
    R_tcp = T_f[:3, :3]
    tcp = np.concatenate(
        [T_f[:3, 3], Rotation.from_matrix(R_tcp).as_rotvec()]
    )
    R_field = Rotation.from_euler("xyz", [25, -35, 40], degrees=True).as_matrix()
    assert np.linalg.norm(Rotation.from_matrix(R_tcp @ R_field.T).as_rotvec()) > 0.2

    kin.sync_joints(q)
    kin.set_tcp_pose(tcp)
    # Loop policy after enable: capture TCP only — no vf.project.
    kin.ensure_reference_from_tcp()
    assert np.allclose(kin._T_ref[:3, :3], R_tcp, atol=1e-9)

    qdot = kin.compute_qdot(
        gain=4.0,
        damping=0.05,
        max_joint_velocity=0.5,
        vf=_FakeVF(R_field=R_field),
    )
    assert float(np.linalg.norm(qdot)) < 1e-3
