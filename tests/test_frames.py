import numpy as np
from geodude_teleoperation.frames import (
    base_to_tag0_point,
    compose_T_tag0_rtde,
    load_T_rtde_base_vention,
    load_T_tag0_base,
    tip_from_tcp,
)


def test_tag0_and_tip_conventions():
    T = np.eye(4)
    T[:3, 3] = [1.0, 0.0, 0.0]
    assert np.allclose(base_to_tag0_point(np.array([1.1, 0.0, 0.0]), T), [0.1, 0.0, 0.0])
    assert np.allclose(tip_from_tcp(np.zeros(3), np.eye(3), np.array([0.0, 0.0, 0.1])), [0.0, 0.0, 0.1])
    assert np.allclose(load_T_tag0_base(None), np.eye(4))


def test_rtde_mount_compose_and_asset():
    T_tag = np.eye(4)
    T_tag[:3, 3] = [1.0, 0.0, 0.0]
    T_rv = np.eye(4)
    T_rv[:3, 3] = [0.25, 0.0, 0.0]
    assert np.allclose(compose_T_tag0_rtde(T_tag, T_rv)[:3, 3], [0.75, 0.0, 0.0])
    T = load_T_rtde_base_vention()
    assert T.shape == (4, 4)
    assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-6
    from geodude_teleoperation.frames import load_T_left_base_vention

    Tl = load_T_left_base_vention()
    assert np.allclose(Tl[:3, 3], [-0.33, -0.828, 1.65])
