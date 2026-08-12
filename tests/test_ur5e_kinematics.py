"""Vendored UR5e FK / Jacobian (UR controller frame)."""
from __future__ import annotations

import numpy as np

from geodude_teleoperation.rtde.ur5e_kinematics import (
    compute_jacobian,
    forward_kinematics,
)


def test_forward_kinematics_shape_and_finite():
    q = np.zeros(6)
    T, transforms = forward_kinematics(q)
    assert T.shape == (4, 4)
    assert len(transforms) == 7
    assert np.allclose(T[3, :], [0, 0, 0, 1])
    assert np.isfinite(T).all()
    # Zero config reaches on the order of UR5e reach (not at origin).
    assert np.linalg.norm(T[:3, 3]) > 0.2


def test_jacobian_shape_zero_offset_matches_flange_point():
    q = np.array([0.1, -0.4, 0.5, -0.2, 0.3, 0.0])
    T, _ = forward_kinematics(q)
    J = compute_jacobian(q)
    assert J.shape == (6, 6)
    assert np.isfinite(J).all()
    # Finite-diff linear column 0 vs analytic.
    eps = 1e-6
    q2 = q.copy()
    q2[0] += eps
    T2, _ = forward_kinematics(q2)
    v_fd = (T2[:3, 3] - T[:3, 3]) / eps
    assert np.allclose(J[:3, 0], v_fd, atol=1e-4)
