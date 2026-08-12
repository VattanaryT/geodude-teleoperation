"""Slim calibrated UR5e FK / geometric Jacobian (UR controller base frame).

Vendored from ``diffusion_policy.real_world.ur5e_kinematics`` (calibrated URDF
chain to wrist_3 / flange). Default frame matches RTDE / pendant base — no
REP-103 ``R_180Z`` remap.
"""
from __future__ import annotations

import numpy as np

# Joint transforms from calibrated URDF (xyz, rpy) relative to parent.
# Same numbers as diffusion_policy.real_world.ur5e_kinematics.CALIBRATED_JOINTS.
CALIBRATED_JOINTS = [
    {
        "xyz": np.array([0.0, 0.0, 0.1625269213099744]),
        "rpy": np.array([0.0, 0.0, -2.9298064682709875e-08]),
    },
    {
        "xyz": np.array([0.0001655127603476787, 0.0, 0.0]),
        "rpy": np.array([1.569956524571092, 0.0, -6.769377576441778e-10]),
    },
    {
        "xyz": np.array([-0.42543123286729523, 0.0, 0.0]),
        "rpy": np.array([3.1401294729209073, 3.1401513871885234, 3.1415926282859967]),
    },
    {
        "xyz": np.array([-0.3926255442838572, -0.00018118202218157404, 0.13378567426143276]),
        "rpy": np.array([0.001354269897852176, -0.0010953222406931517, 2.9544587825348556e-06]),
    },
    {
        "xyz": np.array([7.719320843298656e-05, -0.0997722260056821, 6.2471135291889e-05]),
        "rpy": np.array([1.570170189345338, 0.0, -3.311599503007332e-08]),
    },
    {
        "xyz": np.array([9.607876543233622e-05, 0.09944214059801317, 0.00012685702594143576]),
        "rpy": np.array([1.5720720129010473, 3.141592653589793, 3.141592595496277]),
    },
]


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def forward_kinematics(joint_angles) -> tuple[np.ndarray, list[np.ndarray]]:
    """FK to wrist_3 (flange) in UR controller base frame.

    Returns:
        T: 4x4 base→flange
        transforms: identity + each joint frame (length 7)
    """
    joint_angles = np.asarray(joint_angles, dtype=float).reshape(6)
    T = np.eye(4)
    transforms = [T.copy()]

    for i in range(6):
        xyz = CALIBRATED_JOINTS[i]["xyz"]
        rpy = CALIBRATED_JOINTS[i]["rpy"]
        T_fixed = np.eye(4)
        T_fixed[:3, :3] = rpy_to_matrix(rpy)
        T_fixed[:3, 3] = xyz

        theta = joint_angles[i]
        ct, st = np.cos(theta), np.sin(theta)
        T_joint = np.eye(4)
        T_joint[:3, :3] = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])

        T = T @ T_fixed @ T_joint
        transforms.append(T.copy())

    return T, transforms


def compute_jacobian(joint_angles) -> np.ndarray:
    """Geometric Jacobian at flange: rows [linear; angular], shape (6, 6)."""
    joint_angles = np.asarray(joint_angles, dtype=float).reshape(6)
    T_ee, _ = forward_kinematics(joint_angles)
    p_ee = T_ee[:3, 3]

    J = np.zeros((6, 6))
    T = np.eye(4)

    for i in range(6):
        xyz = CALIBRATED_JOINTS[i]["xyz"]
        rpy = CALIBRATED_JOINTS[i]["rpy"]
        T_fixed = np.eye(4)
        T_fixed[:3, :3] = rpy_to_matrix(rpy)
        T_fixed[:3, 3] = xyz

        T_joint_frame = T @ T_fixed
        z_i = T_joint_frame[:3, 2]
        p_i = T_joint_frame[:3, 3]

        J[:3, i] = np.cross(z_i, p_ee - p_i)
        J[3:, i] = z_i

        theta = joint_angles[i]
        ct, st = np.cos(theta), np.sin(theta)
        T_joint_rot = np.eye(4)
        T_joint_rot[:3, :3] = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
        T = T_joint_frame @ T_joint_rot

    return J
