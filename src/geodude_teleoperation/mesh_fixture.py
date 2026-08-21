# Imported before pymeshlab, deliberately: pymeshlab bundles its own (older) libcrypto.so.3,
# which -- once loaded into the process -- shadows the system OpenSSL for any *later* import of
# the standard library's ssl module (e.g. via viser, transitively, when gafropy.viz.Visualizer is
# constructed), breaking it with "version `OPENSSL_3.3.0' not found". Importing ssl first forces
# the correct system libcrypto to be resolved and loaded before pymeshlab ever enters the
# process, so pymeshlab's own bundled copy never gets the chance to shadow it. Reproduced and
# confirmed directly: `python -c "import pymeshlab; import ssl"` fails with exactly this error;
# `python -c "import ssl; import pymeshlab"` does not.
#
# pymeshlab itself is imported lazily in reconstruct_mesh_from_points: free-space teleop (and
# any other caller that never reconstructs a point cloud) must not require it at package import.
import ssl  # noqa: F401

import numpy as np
import trimesh
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
                        raise ValueError(
                            f"read_ascii_pcd: {path} is not ASCII-encoded (binary .pcd not supported)")
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


def reconstruct_mesh_from_points(points: np.ndarray) -> trimesh.Trimesh:
    """Reconstructs a triangle mesh from an unstructured point cloud via pymeshlab's screened
    Poisson surface reconstruction. Needs per-point normals first (pymeshlab's own
    compute_normal_for_point_clouds, PCA over each point's local neighborhood).

    This is what replaces diffused_fields' point-cloud-only Laplacian path for OrientationField:
    that path needed no explicit mesh at all (diffused_fields.solver.LaplacianSolver handles
    face-less point clouds directly via a "tufted cover" construction), but gafropy's geometry
    stack (SimplicialComplex/ConnectionLaplacian) requires genuine triangle faces -- so raw
    point-cloud input (e.g. a .pcd scan) is triangulated once here instead of diffused directly.
    """
    try:
        import pymeshlab
    except ImportError as exc:
        raise ImportError(
            "pymeshlab is required to reconstruct a mesh from a point cloud. "
            "Install it with: pip install pymeshlab"
        ) from exc
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=points))
    ms.compute_normal_for_point_clouds()
    ms.generate_surface_reconstruction_screened_poisson()
    reconstructed = ms.current_mesh()
    return trimesh.Trimesh(vertices=reconstructed.vertex_matrix(), faces=reconstructed.face_matrix(), process=True)


def load_object_mesh(path, rotation=None, translation=None, normal_orientation=1) -> trimesh.Trimesh:
    """Loads path (mesh or point-cloud file) as a positioned trimesh.Trimesh with outward-facing
    normals. Mesh formats (.ply/.obj/.stl/...) load directly via trimesh; .pcd (no faces) is
    triangulated first via reconstruct_mesh_from_points. Replaces the old
    load_object_pointcloud (diffused_fields.manifold.pointcloud.Pointcloud) -- same
    rotation/translation/normal_orientation handling, now returning a trimesh.Trimesh (the
    representation gafropy's SimplicialComplex/ConnectionLaplacian need) instead.
    """
    if str(path).lower().endswith(".pcd"):
        mesh = reconstruct_mesh_from_points(read_ascii_pcd(path))
    else:
        loaded = trimesh.load(str(path), force="mesh")
        if len(loaded.faces) == 0:
            # trimesh found no real face data for this file (e.g. a point-cloud-only .ply --
            # PLY can store either; force="mesh" silently returns an empty Trimesh rather than
            # raising). Reconstruct a real mesh via pymeshlab from the raw points instead.
            points = np.asarray(trimesh.load(str(path)).vertices)
            mesh = reconstruct_mesh_from_points(points)
        else:
            mesh = loaded

    if normal_orientation < 0:
        mesh.invert()

    if rotation is not None:
        mesh.vertices = mesh.vertices @ np.asarray(rotation, dtype=np.float64).T
    if translation is not None:
        mesh.vertices = mesh.vertices + np.asarray(translation, dtype=np.float64)

    return mesh
