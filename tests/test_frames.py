import numpy as np
from geodude_teleoperation.frames import (
    base_to_tag0_point,
    load_T_tag0_base,
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
