import numpy as np
from diffused_fields.manifold import Pointcloud
from geodude_teleoperation.field_query import query_basis_surface, query_field_basis


def _planar_cloud():
    xs, ys = np.meshgrid(np.linspace(-0.1, 0.1, 21), np.linspace(-0.1, 0.1, 21))
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    pc = Pointcloud(vertices=verts, voxel_size=0.01, normal_orientation=1)
    pc.get_normals()
    # identity-ish local bases: z = normal
    n = pc.num_vertices
    bases = np.tile(np.eye(3), (n, 1, 1))
    pc.local_bases = bases
    pc.get_kd_tree()
    return pc


def test_surface_signed_distance_above_plane():
    pc = _planar_cloud()
    x = np.array([0.0, 0.0, 0.02])
    R, idx, d, diag = query_basis_surface(pc, x, fit_mode="plane", query_radius=0.05)
    assert d > 0
    assert abs(abs(d) - 0.02) < 0.005
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_in_contact_band():
    pc = _planar_cloud()
    class FakeWos:
        def get_batch_from_point(self, x): return x
        def diffuse_rotations(self, batch):
            return np.eye(3), None, np.array(["converged"])
    R, d, contact, diag = query_field_basis(
        pc, FakeWos(), np.array([0.0, 0.0, 0.0005]),
        contact_threshold=0.001, fit_mode="plane", query_radius=0.05,
    )
    assert contact is True
    assert diag.get("used_wos") is False
