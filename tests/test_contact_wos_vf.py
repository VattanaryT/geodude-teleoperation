import numpy as np
import open3d as o3d
import pytest
from diffused_fields.manifold import Pointcloud
from scipy.spatial.transform import Rotation

from geodude_teleoperation.contact_wos_vf import (
    ContactWosOrientationField,
    tip_tag0_from_tcp,
)
from geodude_teleoperation.field_query import query_field_basis
from geodude_teleoperation.frames import base_to_tag0_point, tip_from_tcp


@pytest.fixture(autouse=True)
def _contact_wos_test_infra():
    """Open3D / Pointcloud patches for coplanar synthetic clouds (this module only)."""
    orig_orient = o3d.geometry.PointCloud.orient_normals_consistent_tangent_plane

    def _orient_normals_safe(self, k=30):
        try:
            orig_orient(self, k)
        except RuntimeError:
            self.orient_normals_towards_camera_location(
                camera_location=np.array([0.0, 0.0, -1.0])
            )

    o3d.geometry.PointCloud.orient_normals_consistent_tangent_plane = _orient_normals_safe

    had_num_vertices = hasattr(Pointcloud, "num_vertices")
    if not had_num_vertices:
        Pointcloud.num_vertices = property(lambda self: len(self.vertices))

    yield

    o3d.geometry.PointCloud.orient_normals_consistent_tangent_plane = orig_orient
    if not had_num_vertices:
        del Pointcloud.num_vertices


def _planar_cloud():
    xs, ys = np.meshgrid(np.linspace(-0.1, 0.1, 21), np.linspace(-0.1, 0.1, 21))
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    pc = Pointcloud(vertices=verts, voxel_size=0.01, normal_orientation=1)
    pc.get_normals()
    n = pc.num_vertices
    pc.local_bases = np.tile(np.eye(3), (n, 1, 1))
    pc.get_kd_tree()
    return pc


class FakeWos:
    def get_batch_from_point(self, x):
        return x

    def diffuse_rotations(self, batch):
        return np.eye(3), None, np.array(["converged"])


class DuckMotor:
    """Minimal motor stand-in when gafropy is unavailable."""

    def __init__(self, p, R=None):
        self._p = np.asarray(p, dtype=float).reshape(3)
        self._R = np.eye(3) if R is None else np.asarray(R, dtype=float).reshape(3, 3)

    def get_translator(self):
        p = self._p

        class _T:
            def x(self):
                return float(p[0])

            def y(self):
                return float(p[1])

            def z(self):
                return float(p[2])

        return _T()

    def get_rotor(self):
        return Rotation.from_matrix(self._R)


def _make_motor(p, R=None):
    gafropy = pytest.importorskip("gafropy")
    translator = gafropy.Translator(p[0], p[1], p[2])
    rotor = gafropy.Rotor.from_matrix(np.eye(3) if R is None else R)
    return gafropy.Motor(translator, rotor)


def test_tip_tag0_query_path_pure_numpy():
    """Tip → tag_0 → field query without gafropy."""
    pc = _planar_cloud()
    T = np.eye(4)
    T[:3, 3] = [1.0, 0.0, 0.0]
    p_tcp = np.array([1.0, 0.0, 0.02])
    R_tcp = np.eye(3)
    tip_offset = np.zeros(3)

    p_tip_tag0 = tip_tag0_from_tcp(p_tcp, R_tcp, tip_offset, T)
    assert np.allclose(p_tip_tag0, [0.0, 0.0, 0.02])

    _, signed_d, contact, diag = query_field_basis(
        pc,
        FakeWos(),
        p_tip_tag0,
        contact_threshold=0.001,
        fit_mode="plane",
        query_radius=0.05,
    )
    assert signed_d > 0
    assert contact is False
    assert diag.get("used_wos") is True


def test_get_distance_is_abs_cached():
    gafropy = pytest.importorskip("gafropy")
    pc = _planar_cloud()
    vf = ContactWosOrientationField(
        pc,
        FakeWos(),
        tip_offset=np.zeros(3),
        contact_threshold=0.001,
        T_tag0_base=np.eye(4),
        fit_mode="plane",
        query_radius=0.05,
    )
    ee_above = _make_motor(np.array([0.0, 0.0, 0.02]))
    vf.update_from_state(ee_above)
    d = vf.get_distance(np.array([0, 0, 0.02]))
    assert d >= 0
    assert abs(d - abs(vf.status["signed_d"])) < 1e-9


def test_continuity_flip(monkeypatch):
    pytest.importorskip("gafropy")
    pc = _planar_cloud()
    vf = ContactWosOrientationField(
        pc,
        FakeWos(),
        tip_offset=np.zeros(3),
        contact_threshold=0.001,
        T_tag0_base=np.eye(4),
        fit_mode="plane",
        query_radius=0.05,
    )

    R_first = np.eye(3)
    R_second = Rotation.from_euler("xyz", [0.2, 0.1, 2.9], degrees=True).as_matrix()
    q1 = Rotation.from_matrix(R_first).as_quat()
    q2 = Rotation.from_matrix(R_second).as_quat()
    if np.dot(q1, q2) >= 0:
        R_second = Rotation.from_quat(-q2).as_matrix()

    calls = iter([R_first, R_second])

    def _fake_query(*args, **kwargs):
        R = next(calls)
        return R, 0.02, False, {"used_wos": True, "nearest_idx": 0}

    monkeypatch.setattr(
        "geodude_teleoperation.contact_wos_vf.query_field_basis",
        _fake_query,
    )

    ee = _make_motor(np.array([0.0, 0.0, 0.02]))
    vf.update_from_state(ee)
    R_after_first = vf.status["R_field_base"].copy()
    vf.update_from_state(ee)
    R_after_second = vf.status["R_field_base"]

    assert np.isclose(np.linalg.det(R_after_second), 1.0, atol=1e-5)
    q_prev = Rotation.from_matrix(R_after_first).as_quat()
    q_curr = Rotation.from_matrix(R_after_second).as_quat()
    assert np.dot(q_prev, q_curr) >= 0.0


def test_project_keeps_translation():
    gafropy = pytest.importorskip("gafropy")
    pc = _planar_cloud()
    vf = ContactWosOrientationField(
        pc,
        FakeWos(),
        tip_offset=np.zeros(3),
        contact_threshold=0.001,
        T_tag0_base=np.eye(4),
        fit_mode="plane",
        query_radius=0.05,
    )
    p = np.array([0.05, -0.03, 0.04])
    ee = _make_motor(p)
    vf.update_from_state(ee)
    projected = vf.project(ee)
    tr = projected.get_translator()
    assert np.allclose([tr.x(), tr.y(), tr.z()], p, atol=1e-9)


def test_set_T_tag0_base_updates_query_frame():
    pc = _planar_cloud()
    vf = ContactWosOrientationField(
        pc,
        FakeWos(),
        tip_offset=np.zeros(3),
        contact_threshold=0.001,
        T_tag0_base=np.eye(4),
        fit_mode="plane",
        query_radius=0.05,
    )
    T = np.eye(4)
    T[:3, 3] = [1.0, 0.0, 0.0]
    vf.set_T_tag0_base(T)

    p_tcp = np.array([1.0, 0.0, 0.02])
    expected_tip_tag0 = base_to_tag0_point(tip_from_tcp(p_tcp, np.eye(3), np.zeros(3)), T)
    assert np.allclose(tip_tag0_from_tcp(p_tcp, np.eye(3), np.zeros(3), T), expected_tip_tag0)

    duck = DuckMotor(p_tcp)
    vf.update_from_state(duck, T_tag0_base=T)
    assert vf.status["signed_d"] > 0
