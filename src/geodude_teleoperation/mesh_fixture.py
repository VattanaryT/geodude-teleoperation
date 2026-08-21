import numpy as np
import trimesh

from geodude_teleoperation.df_runtime import prefer_env_diffused_fields

prefer_env_diffused_fields()

from diffused_fields.manifold.pointcloud import Pointcloud  # noqa: E402
from scipy.spatial.transform import Rotation


def compute_resting_orientation(points):
    centered = points - points.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
    thin_axis = eigvecs[:, 0]  # smallest eigenvalue = least variance

    rotation, _ = Rotation.align_vectors([[0.0, 0.0, 1.0]], [thin_axis])
    return rotation.as_matrix()


def read_ascii_pcd(path):
    points = []
    with open(path) as f:
        in_data = False
        for line in f:
            if not in_data:
                if line.startswith("DATA"):
                    if "ascii" not in line:
                        raise ValueError(f"read_ascii_pcd: {
                                         path} is not ASCII-encoded (binary .pcd not supported)")
                    in_data = True
                continue
            values = line.split()
            points.append(
                (float(values[0]), float(values[1]), float(values[2])))
    return np.asarray(points, dtype=np.float64)


def load_points(path):
    if str(path).lower().endswith(".pcd"):
        return read_ascii_pcd(path)
    return np.asarray(trimesh.load(path).vertices)


def load_object_pointcloud(path, rotation=None, translation=None, normal_orientation=1):
    """Load path (mesh or point cloud file) as a positioned, normal-estimated
    Pointcloud. Pointcloud's own filename= loader (via trimesh) handles
    everything except .pcd, which trimesh can't parse -- read that case's raw
    points explicitly and feed them through Pointcloud's vertices= path
    instead, so both cases still get the same rotation/translation/normal
    handling.
    """
    if str(path).lower().endswith(".pcd"):
        pointcloud = Pointcloud(
            vertices=read_ascii_pcd(path), rotation=rotation, translation=translation,
            is_cluster_points=False, normal_orientation=normal_orientation)
    else:
        pointcloud = Pointcloud(
            filename=str(path), rotation=rotation, translation=translation,
            is_cluster_points=False, normal_orientation=normal_orientation)
    pointcloud.get_normals()
    return pointcloud
