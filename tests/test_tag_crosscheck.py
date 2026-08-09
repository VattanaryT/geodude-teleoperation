"""Unit tests for AprilTag vs TF cross-check helpers (no ROS required)."""

from __future__ import annotations

import numpy as np

from geodude_teleoperation.calibration import tag_extrinsics as te
from geodude_teleoperation.frames import lookup_T_tag0_base_tf
from geodude_teleoperation.tag_crosscheck import (
    apriltag_T_tag0_base,
    compute_tag_vs_tf_mm,
    tag_vs_tf_translation_mm,
)


def _T(t=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def test_disagree_mm():
    T1 = np.eye(4)
    T2 = np.eye(4)
    T2[0, 3] = 0.01
    assert abs(tag_vs_tf_translation_mm(T1, T2) - 10.0) < 1e-6


def test_apriltag_T_tag0_base_matches_compose():
    T_EE = _T((0.5, 0.0, 0.2))
    T_cam_EE = _T((0.05, 0.0, 0.08))
    T_tag_cam = _T((0.1, -0.02, 0.4))
    expected = te.compose_T_tag_base(T_EE, T_cam_EE, T_tag_cam)
    assert np.allclose(apriltag_T_tag0_base(T_EE, T_cam_EE, T_tag_cam), expected)


def test_compute_tag_vs_tf_mm():
    T_tf = np.eye(4)
    T_EE = _T((0.5, 0.0, 0.2))
    T_cam_EE = _T((0.05, 0.0, 0.08))
    T_tag_cam = _T((0.1, -0.02, 0.4))
    T_april = apriltag_T_tag0_base(T_EE, T_cam_EE, T_tag_cam)
    T_tf_shift = T_april.copy()
    T_tf_shift[0, 3] += 0.005
    mm = compute_tag_vs_tf_mm(T_tf_shift, T_EE, T_cam_EE, T_tag_cam)
    assert mm is not None
    assert abs(mm - 5.0) < 1e-6


def test_compute_tag_vs_tf_mm_missing_inputs():
    assert compute_tag_vs_tf_mm(None, np.eye(4), np.eye(4), np.eye(4)) is None


def test_lookup_without_ros_returns_none():
    T, age_ms = lookup_T_tag0_base_tf(None, "base", "tag_0", 0.5)
    assert T is None and age_ms is None


def test_lookup_without_buffer_returns_none():
    T, age_ms = lookup_T_tag0_base_tf(None, "base", "tag_0", 0.5)
    assert T is None and age_ms is None
