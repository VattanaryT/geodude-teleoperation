import gafropy
import numpy as np
from diffused_fields.diffusion.pointcloud_quaternion_diffusion import (
    PointcloudQuaternionDiffusion,
)
from diffused_fields.diffusion.walk_on_spheres import WalkOnSpheresDiffusion
from diffused_fields.manifold._rotor_interop import as_scipy_rotation
from diffused_fields.manifold.pointcloud import Pointcloud
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


class WalkOnSpheresVirtualFixture:

    def __init__(self, wos, smoothing=0.75):
        self._wos = wos
        self._last_basis = np.eye(3)
        self._has_last = False
        self._smoothing = smoothing

    def get_distance(self, position):
        distances, _ = self._wos.boundaries[0].get_closest_points(
            np.asarray(position)[np.newaxis, :])
        return float(np.asarray(distances).reshape(-1)[0])

    def project(self, proposed_reference):
        translator, position = _motor_position(proposed_reference)

        batch_points = self._wos.get_batch_from_point(position)
        try:
            local_basis, _, _ = self._wos.diffuse_rotations(batch_points)
        except ValueError:
            local_basis = self._last_basis if self._has_last else np.eye(3)

        if self._has_last and self._smoothing > 0:
            local_basis = _nlerp_matrices(
                self._last_basis, local_basis, 1 - self._smoothing)
        self._last_basis = local_basis
        self._has_last = True

        approach_basis = _orient_approach_into_surface(local_basis)
        rotor = gafropy.Rotor.from_matrix(approach_basis)
        corrected = gafropy.Motor(translator, rotor)
        return corrected


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
        diffusion = PointcloudQuaternionDiffusion(
            pointcloud, diffusion_scalar=diffusion_scalar)

        tangent_field = diffusion.diffuse_key_direction_as_tangent_field(
            key_direction, vertex=source_vertex)
        pointcloud.get_bases_from_tangent_vector_and_normal(tangent_field)

        wos_boundary = _subsample_boundary(
            pointcloud, num_boundary_points, boundary_seed)
        wos_kwargs.setdefault(
            "convergence_threshold", pointcloud.get_mean_edge_length() * 2)
        wos = WalkOnSpheresDiffusion(boundaries=[wos_boundary], **wos_kwargs)

        self._pointcloud = pointcloud
        self._diffusion = diffusion
        self._wos_fixture = WalkOnSpheresVirtualFixture(
            wos, smoothing=smoothing)

    def get_distance(self, position):
        return self._wos_fixture.get_distance(position)

    def project(self, proposed_reference):
        return self._wos_fixture.project(proposed_reference)
