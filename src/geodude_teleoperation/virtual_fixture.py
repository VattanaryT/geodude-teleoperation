import gafropy
import numpy as np
from diffused_fields.manifold._rotor_interop import as_scipy_rotation
from diffused_fields.manifold.pointcloud import Pointcloud
from diffused_fields.problem import DiffusionProblem
from diffused_fields.solver import LaplacianSolver, WalkOnSpheresSolver, farthest_from_boundary_vertex
from scipy.spatial.transform import Rotation


def _orient_approach_into_surface(local_basis):

    approach_basis = local_basis.copy()
    approach_basis[:, 1] *= -1
    approach_basis[:, 2] *= -1
    return approach_basis


def _motor_position(motor):
    translator = motor.get_translator()
    return translator, np.array([translator.x(), translator.y(), translator.z()])


def _nlerp_matrices(old, new, t):
    q_old = as_scipy_rotation(old).as_quat()
    q_new = as_scipy_rotation(new).as_quat()
    if np.dot(q_old, q_new) < 0:
        q_new = -q_new
    q = (1 - t) * q_old + t * q_new
    q /= np.linalg.norm(q)
    return Rotation.from_quat(q).as_matrix()


def _subsample_boundary(pointcloud, num_points, seed):
    n = min(num_points, len(pointcloud.vertices))
    if n >= len(pointcloud.vertices):
        return pointcloud
    indices = np.random.default_rng(seed).choice(
        len(pointcloud.vertices), size=n, replace=False)
    boundary = Pointcloud(
        vertices=pointcloud.vertices[indices].copy(), is_cluster_points=False)
    boundary.local_bases = pointcloud.local_bases[indices].copy()
    return boundary


class OrientationField:

    def __init__(self, pointcloud, key_direction, source_vertex=None,
                 diffusion_scalar=100, smoothing=0.75, num_boundary_points=3000,
                 boundary_seed=0, **wos_kwargs):
        vertex = source_vertex if source_vertex is not None else farthest_from_boundary_vertex(pointcloud)
        if not hasattr(pointcloud, "normals"):
            pointcloud.get_normals()

        direction = np.asarray(key_direction, dtype=float)
        normal = pointcloud.normals[vertex]
        tangent = direction - np.dot(direction, normal) * normal
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-9:
            raise ValueError(
                "OrientationField: key_direction is parallel to the surface "
                "normal at the source vertex -- no tangential component left to diffuse"
            )
        tangent = tangent / tangent_norm

        solver = LaplacianSolver(diffusion_scalar=diffusion_scalar)
        tangent_problem = DiffusionProblem(
            pointcloud, source_vertices=[vertex], source_values=tangent[np.newaxis, :])
        tangent_field = solver.solve_vector(tangent_problem)
        pointcloud.get_bases_from_tangent_vector_and_normal(tangent_field.values)

        wos_boundary = _subsample_boundary(
            pointcloud, num_boundary_points, boundary_seed)
        wos_kwargs.setdefault(
            "convergence_threshold", pointcloud.get_mean_edge_length() * 2)
        wos_solver = WalkOnSpheresSolver(**wos_kwargs)

        self._pointcloud = pointcloud
        self._solver = solver
        self._field = wos_solver.solve_quaternion(DiffusionProblem(wos_boundary))
        self._smoothing = smoothing
        self._last_basis = np.eye(3)
        self._has_last = False

    def get_distance(self, position):
        distances, _ = self._field.manifold.get_closest_points(
            np.asarray(position)[np.newaxis, :])
        return float(np.asarray(distances).reshape(-1)[0])

    def project(self, proposed_reference):
        translator, position = _motor_position(proposed_reference)

        try:
            local_basis = as_scipy_rotation(self._field(position)).as_matrix()
        except ValueError:
            local_basis = self._last_basis if self._has_last else np.eye(3)

        if self._has_last and self._smoothing > 0:
            local_basis = _nlerp_matrices(
                self._last_basis, local_basis, 1 - self._smoothing)
        self._last_basis = local_basis
        self._has_last = True

        approach_basis = _orient_approach_into_surface(local_basis)
        rotor = gafropy.Rotor.from_matrix(approach_basis)
        return gafropy.Motor(translator, rotor)
