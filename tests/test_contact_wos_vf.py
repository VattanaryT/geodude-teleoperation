import numpy as np
import open3d as o3d
import pytest
from diffused_fields.manifold import Pointcloud
from scipy.spatial.transform import Rotation

from geodude_teleoperation.teleop.contact_wos_vf import (
    ContactWosOrientationField,
    tip_tag0_from_tcp,
)
from geodude_teleoperation.field.field_query import query_field_basis
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
    ee_above = DuckMotor(np.array([0.0, 0.0, 0.02]))
    vf.update_from_state(ee_above)
    d = vf.get_distance(np.array([0, 0, 0.02]))
    assert d >= 0
    assert abs(d - abs(vf.status["signed_d"])) < 1e-9


def test_continuity_flip(monkeypatch):
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
        "geodude_teleoperation.teleop.contact_wos_vf.query_field_basis",
        _fake_query,
    )

    ee = DuckMotor(np.array([0.0, 0.0, 0.02]))
    vf.update_from_state(ee)
    R_after_first = vf.status["R_field_base"].copy()
    vf.update_from_state(ee)
    R_after_second = vf.status["R_field_base"]

    assert np.isclose(np.linalg.det(R_after_second), 1.0, atol=1e-5)
    q_prev = Rotation.from_matrix(R_after_first).as_quat()
    q_curr = Rotation.from_matrix(R_after_second).as_quat()
    assert np.dot(q_prev, q_curr) >= 0.0


def test_project_keeps_translation():
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
    ee = DuckMotor(p)
    vf.update_from_state(ee)
    projected = vf.project(ee)
    tr = projected.get_translator()
    assert np.allclose([tr.x(), tr.y(), tr.z()], p, atol=1e-9)
    R_proj = projected.get_rotor().as_matrix()
    assert np.allclose(R_proj, vf.status["R_field_base"], atol=1e-9)


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


def test_smooth_R_field_ema():
    from geodude_teleoperation.teleop.contact_wos_vf import smooth_R_field, slerp_R

    R0 = np.eye(3)
    R1 = Rotation.from_euler("z", 40, degrees=True).as_matrix()
    Rs = smooth_R_field(R0, R1, smoothing=0.75)
    expect = slerp_R(R0, R1, 0.25)
    assert np.allclose(Rs, expect, atol=1e-9)
    assert np.allclose(smooth_R_field(None, R1, smoothing=0.75), R1)


def _line_cloud():
    verts = np.zeros((20, 3))
    verts[:, 0] = np.linspace(-0.1, 0.1, 20)
    verts[:, 2] = 0.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.estimate_normals()
    pcloud = Pointcloud(vertices=np.asarray(pcd.points), voxel_size=0.01, normal_orientation=1)
    pcloud.local_bases = np.tile(np.eye(3), (len(pcloud.vertices), 1, 1))
    pcloud.normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(pcloud.vertices), 1))
    return pcloud


class _WosOk:
    def get_batch_from_point(self, x):
        return x

    def diffuse_rotations(self, batch):
        return np.eye(3), None, np.array(["converged"])


def test_motion_relock_throttles_field_query(monkeypatch):
    """Stationary tip: second update within relock period skips WoS/surface query."""
    from geodude_teleoperation.teleop import contact_wos_vf as cvf

    calls = {"n": 0}
    real = cvf.query_field_basis

    def counting_query(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(cvf, "query_field_basis", counting_query)

    vf = ContactWosOrientationField(
        _line_cloud(),
        _WosOk(),
        tip_offset=np.zeros(3),
        contact_threshold=0.05,
        relock_dist_m=0.05,
        relock_period_s=10.0,
        r_field_smoothing=0.0,
        motion_track=1.0,
        fit_mode="plane",
        query_radius=0.2,
    )
    duck = DuckMotor(np.array([0.0, 0.0, 0.02]))
    vf.update_from_state(duck)
    n1 = calls["n"]
    vf.update_from_state(duck)
    assert calls["n"] == n1  # throttled
    assert vf.status["relock_reason"] == "hold"
    assert "R_motion_base" in vf.status


def test_frozen_vf_skips_relock_query(monkeypatch):
    """f freeze: update_from_state must not relock R_motion or re-query WoS."""
    from geodude_teleoperation.teleop import contact_wos_vf as cvf

    calls = {"n": 0}
    real = cvf.query_field_basis

    def counting_query(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(cvf, "query_field_basis", counting_query)

    vf = ContactWosOrientationField(
        _line_cloud(),
        _WosOk(),
        tip_offset=np.zeros(3),
        contact_threshold=0.05,
        relock_dist_m=0.001,
        relock_period_s=0.0,
        r_field_smoothing=0.0,
        motion_track=1.0,
        fit_mode="plane",
        query_radius=0.2,
    )
    vf.update_from_state(DuckMotor(np.array([0.0, 0.0, 0.02])))
    R0 = vf.motion_basis().copy()
    n0 = calls["n"]
    vf.set_frozen(True)
    vf.update_from_state(DuckMotor(np.array([0.08, 0.0, 0.02])))
    assert calls["n"] == n0
    assert np.allclose(vf.motion_basis(), R0)
    assert vf.status["relock_reason"] == "frozen"


def test_relock_rate_limits_when_tip_jumps_every_tick(monkeypatch):
    """Fast tip travel > relock_dist every sample must not re-query every tick.

    Period is a hard minimum interval (metres/seconds units verified here).
    """
    from geodude_teleoperation.teleop import contact_wos_vf as cvf

    calls = {"n": 0}
    real = cvf.query_field_basis

    def counting_query(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(cvf, "query_field_basis", counting_query)
    t = {"now": 1000.0}
    monkeypatch.setattr(cvf.time, "monotonic", lambda: t["now"])

    vf = ContactWosOrientationField(
        _line_cloud(),
        _WosOk(),
        tip_offset=np.zeros(3),
        contact_threshold=0.05,
        relock_dist_m=0.012,  # 12 mm
        relock_period_s=0.10,  # 100 ms hard floor
        r_field_smoothing=0.0,
        motion_track=1.0,
        fit_mode="plane",
        query_radius=0.2,
    )
    # Tip jumps 20 mm each 20 ms tick (>12 mm) — old OR-logic re-queried every sample.
    for i in range(5):  # t=0..80ms; still inside 100ms floor
        t["now"] = 1000.0 + i * 0.02
        vf.update_from_state(DuckMotor(np.array([i * 0.020, 0.0, 0.02])))
    assert calls["n"] == 1  # cold only; rest rate-limited
    assert vf.status["relock_reason"] == "rate_limit"
    assert vf.status["tip_delta_m"] >= 0.012

    t["now"] = 1000.0 + 0.11
    vf.update_from_state(DuckMotor(np.array([0.15, 0.0, 0.02])))
    assert calls["n"] == 2
    assert vf.status["relocked"] is True


def test_twist_field_alignment_rad_a_along_t2():
    from geodude_teleoperation.teleop.contact_wos_vf import twist_field_alignment_rad

    R = np.eye(3)
    # a/d → field ±y = t̂2
    ang = twist_field_alignment_rad(np.array([0.0, 1.0, 0.0]), R)
    assert ang < 1e-6
    # 45° between t1 and t2 plane axes
    ang45 = twist_field_alignment_rad(np.array([1.0, 1.0, 0.0]) / np.sqrt(2), R)
    assert abs(ang45 - np.pi / 4) < 1e-6
