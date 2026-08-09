"""Pytest hooks. Coplanar synthetic clouds can trigger Qhull errors in open3d normal orientation."""
import numpy as np
import open3d as o3d
from diffused_fields.manifold import Pointcloud

_orig_orient = o3d.geometry.PointCloud.orient_normals_consistent_tangent_plane


def _orient_normals_safe(self, k=30):
    try:
        _orig_orient(self, k)
    except RuntimeError:
        self.orient_normals_towards_camera_location(
            camera_location=np.array([0.0, 0.0, -1.0])
        )


o3d.geometry.PointCloud.orient_normals_consistent_tangent_plane = _orient_normals_safe

if not hasattr(Pointcloud, "num_vertices"):
    Pointcloud.num_vertices = property(lambda self: len(self.vertices))
