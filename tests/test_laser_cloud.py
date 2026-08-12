"""Laser profilometer → vention / tag0 transform helpers."""
from __future__ import annotations

import numpy as np

from geodude_teleoperation.frames import (
    load_T_left_base_vention,
    left_base_to_vention_points,
    vention_to_tag0_points,
)
from geodude_teleoperation.teleop.laser_cloud import (
    points_vention_in_viz_frame,
    voxel_downsample_points,
)


def test_default_T_left_base_vention_asset_loads():
    T = load_T_left_base_vention()
    assert T.shape == (4, 4)
    assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-6
    assert np.allclose(T[:3, 3], [-0.33, -0.828, 1.65])


def test_left_base_to_vention_translation():
    T = np.eye(4)
    T[:3, 3] = [-0.33, -0.828, 1.65]
    p = np.array([[0.1, 0.0, 0.0]])
    out = left_base_to_vention_points(p, T)
    assert np.allclose(out, [[-0.23, -0.828, 1.65]])


def test_vention_to_tag0_inverse():
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 0.0]
    p_v = np.array([[0.6, 0.1, 0.0]])
    p_t = vention_to_tag0_points(p_v, T)
    assert np.allclose(p_t, [[0.1, 0.1, 0.0]])


def test_points_vention_in_viz_frame_modes():
    V = np.array([[1.0, 2.0, 3.0]])
    T_tag = np.eye(4)
    T_tag[:3, 3] = [0.5, 0.0, 0.0]
    T_rv = np.eye(4)
    T_rv[:3, 3] = [0.2, 0.0, 0.0]
    assert np.allclose(
        points_vention_in_viz_frame(
            V, T_tag0_vention=T_tag, T_rtde_base_vention=T_rv, viz_frame="vention"
        ),
        V,
    )
    assert np.allclose(
        points_vention_in_viz_frame(
            V, T_tag0_vention=T_tag, T_rtde_base_vention=T_rv, viz_frame="tag0"
        ),
        [[0.5, 2.0, 3.0]],
    )
    assert np.allclose(
        points_vention_in_viz_frame(
            V, T_tag0_vention=T_tag, T_rtde_base_vention=T_rv, viz_frame="rtde"
        ),
        [[0.8, 2.0, 3.0]],
    )


def test_voxel_downsample_points_reduces():
    # 100 points in a 1cm cube → 2mm voxel collapses heavily
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 0.01, size=(100, 3))
    ds = voxel_downsample_points(pts, 0.002)
    assert 0 < len(ds) < len(pts)


def test_viz_laser_cli_default_off():
    from geodude_teleoperation.teleop.ur_stringer_teleop import parse_args

    args = parse_args(["--ply", "/tmp/x.ply", "--viz"])
    assert args.viz_laser == "off"
    assert args.laser_voxel == 0.002


def test_viz_laser_overlay_requires_viz():
    from geodude_teleoperation.teleop.ur_stringer_teleop import parse_args, validate_args

    args = parse_args(["--ply", "/tmp/x.ply", "--viz-laser", "overlay"])
    try:
        validate_args(args)
        raised = False
    except SystemExit:
        raised = True
    assert raised
