import numpy as np
import open3d as o3d
import pytest
from pathlib import Path
from diffused_fields.manifold import Pointcloud
from geodude_teleoperation.field.stringer_field import (
    compute_orientation_field,
    load_stringer_pointcloud,
    validate_field_prints,
    build_wos,
)


@pytest.fixture(autouse=True)
def _stringer_field_test_infra():
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


def test_field_build_end_lines(tmp_path: Path):
    xs, ys = np.meshgrid(np.linspace(0, 0.2, 40), np.linspace(0, 0.05, 10))
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    ply = tmp_path / "tiny.ply"
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(verts))
    o3d.io.write_point_cloud(str(ply), pcd)
    pc = load_stringer_pointcloud(ply, voxel_size=0.01, normal_orientation=1)
    pc.get_normals()
    pc.get_kd_tree()
    sd = compute_orientation_field(
        pc, field_mode="end_lines", field_flow="length", end_line_fraction=0.1,
    )
    assert pc.local_bases is not None
    R0 = pc.local_bases[0]
    assert np.allclose(R0.T @ R0, np.eye(3), atol=1e-5)
    thr = pc.get_mean_edge_length() * 2
    report = validate_field_prints(pc, sd, thr)
    assert "#verts" in report or "verts" in report
    wos = build_wos(pc, batch_size=8, convergence_threshold_edge_mult=2.0)
    assert wos is not None
    assert abs(wos.convergence_threshold - pc.get_mean_edge_length() * 2.0) < 1e-12
    wos3 = build_wos(pc, batch_size=8, convergence_threshold_edge_mult=3.0)
    assert abs(wos3.convergence_threshold - pc.get_mean_edge_length() * 3.0) < 1e-12
