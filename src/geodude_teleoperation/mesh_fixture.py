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
    """Rotation that sits the object on z=0 with its broader end down.

    PCA's thin-axis eigenvector has an arbitrary sign, so aligning it to +Z
    inverts a hemisphere that was already a dome (~50% of the time). After
    that align, put the wider cap on the table so a dome stays a dome (flat
    rim down, pole up) instead of a bowl hanging through the table.
    """
    centered = points - points.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
    thin_axis = eigvecs[:, 0]  # smallest eigenvalue = least variance

    rotation, _ = Rotation.align_vectors([[0.0, 0.0, 1.0]], [thin_axis])
    R = np.asarray(rotation.as_matrix(), dtype=np.float64)
    rotated = centered @ R.T
    z = rotated[:, 2]
    zmin = float(z.min())
    zmax = float(z.max())
    span = zmax - zmin
    band = 0.1 * span if span > 1e-12 else 1.0
    xy_center = rotated.mean(axis=0)[:2]
    high = rotated[z > zmax - band]
    low = rotated[z < zmin + band]
    radius_high = float(np.linalg.norm(high[:, :2] - xy_center, axis=1).mean())
    radius_low = float(np.linalg.norm(low[:, :2] - xy_center, axis=1).mean())
    # Dome: narrow cap up, wide rim on the table. Bowl is the opposite.
    if radius_high > radius_low:
        R = np.diag([1.0, -1.0, -1.0]) @ R
    return R


def table_placement_translation(points, rotation, table_xyz, xy_offset=(0.15, 0.0)):
    """Translation matching load_object_mesh (vertices @ R.T + t, no centering).

    Lowest rotated point lands on table_xyz[2]; xy centroid lands on
    table_xyz[:2] + xy_offset. Do not mix a centroid-subtracted min-z with the
    uncentered mesh transform: that hung a PCA-flipped unit hemisphere ~1 m
    below the table.
    """
    rotated = np.asarray(points, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64).T
    table_xyz = np.asarray(table_xyz, dtype=np.float64)
    offset = np.asarray(xy_offset, dtype=np.float64)
    translation = np.empty(3, dtype=np.float64)
    translation[:2] = table_xyz[:2] + offset - rotated.mean(axis=0)[:2]
    translation[2] = table_xyz[2] - rotated[:, 2].min()
    return translation


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
