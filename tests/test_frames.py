import numpy as np
from geodude_teleoperation.frames import (
    base_to_tag0_point,
    compose_T_tag0_rtde,
    load_T_tag0_base,
    load_T_rtde_base_vention,
    tag0_rot_to_base,
    tip_from_tcp,
    warn_tip_offset_xy,
)


def test_identity_roundtrip():
    T = np.eye(4)
    p = np.array([0.1, 0.2, 0.3])
    assert np.allclose(base_to_tag0_point(p, T), p)


def test_crop_meta_convention():
    # T_tag0_base = pose of tag_0 in base: origin of tag_0 at t, axes R
    T = np.eye(4)
    T[:3, 3] = [1.0, 0.0, 0.0]
    p_base = np.array([1.1, 0.0, 0.0])
    assert np.allclose(base_to_tag0_point(p_base, T), [0.1, 0.0, 0.0])


def test_tag0_rot_to_base():
    T = np.eye(4)
    R = np.eye(3)
    assert np.allclose(tag0_rot_to_base(R, T), R)


def test_tip_from_tcp():
    p = np.zeros(3)
    R = np.eye(3)
    tip = tip_from_tcp(p, R, np.array([0.0, 0.0, 0.1]))
    assert np.allclose(tip, [0.0, 0.0, 0.1])


def test_warn_tip_offset_xy():
    assert warn_tip_offset_xy(np.array([0.01, 0.0, 0.1])) is not None
    assert warn_tip_offset_xy(np.array([0.0, 0.0, 0.1])) is None


def test_load_none_is_identity():
    assert np.allclose(load_T_tag0_base(None), np.eye(4))


def test_compose_T_tag0_rtde():
    T_tag = np.eye(4)
    T_tag[:3, 3] = [1.0, 0.0, 0.0]
    T_rv = np.eye(4)
    T_rv[:3, 3] = [0.25, 0.0, 0.0]
    T = compose_T_tag0_rtde(T_tag, T_rv)
    assert np.allclose(T[:3, 3], [0.75, 0.0, 0.0])


def test_default_T_rtde_base_vention_asset_loads():
    T = load_T_rtde_base_vention()
    assert T.shape == (4, 4)
    assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-6


def test_default_T_left_base_vention_matches_known_t():
    from geodude_teleoperation.frames import load_T_left_base_vention

    T = load_T_left_base_vention()
    assert np.allclose(T[:3, 3], [-0.33, -0.828, 1.65])
    Tr = load_T_rtde_base_vention()
    assert np.allclose(T[:3, :3], Tr[:3, :3], atol=1e-9)


def test_viz_flags_vention_default():
    from geodude_teleoperation.teleop.ur_stringer_teleop import parse_args

    args = parse_args(["--viz"])
    assert args.viz_frame == "vention"
