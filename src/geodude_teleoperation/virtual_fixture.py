import gafropy
import numpy as np
from diffused_fields.manifold._rotor_interop import as_scipy_rotation
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
    # Normalized-quaternion blend toward `new` by fraction t (t=1 -> new, t=0 -> old),
    # flipping the antipodal sign first so the two don't fight each other (q and -q are
    # the same rotation, but interpolating between q_old and -q_new instead of q_new would
    # spin the "wrong way around").
    q_old = as_scipy_rotation(old).as_quat()
    q_new = as_scipy_rotation(new).as_quat()
    if np.dot(q_old, q_new) < 0:
        q_new = -q_new
    q = (1 - t) * q_old + t * q_new
    q /= np.linalg.norm(q)
    return Rotation.from_quat(q).as_matrix()


class ManifoldVirtualFixture:

    def __init__(self, manifold):
        self._manifold = manifold

    def project(self, proposed_reference):
        translator = proposed_reference.get_translator()
        position = np.array([translator.x(), translator.y(), translator.z()])

        distances, directions = self._manifold.get_closest_points(
            position[np.newaxis, :])
        closest_point = position + directions[0]

        corrected = gafropy.Motor(translator, proposed_reference.get_rotor())
        return corrected, float(distances[0])


class DiffusedSurfaceVirtualFixture:

    def __init__(self, pointcloud):
        if not hasattr(pointcloud, "local_bases"):
            raise AttributeError(
                "pointcloud.local_bases is not populated yet; run a diffusion pass "
                "(e.g. PointcloudScalarDiffusion(pointcloud).get_local_bases()) first."
            )
        self._pointcloud = pointcloud

    def project(self, proposed_reference):
        translator, position = _motor_position(proposed_reference)

        distances, indices = self._pointcloud.get_closest_points(
            position[np.newaxis, :])
        index = int(np.asarray(indices).reshape(-1)[0])

        approach_basis = _orient_approach_into_surface(
            self._pointcloud.local_bases[index])
        rotor = gafropy.Rotor.from_matrix(approach_basis)

        corrected = gafropy.Motor(translator, rotor)
        return corrected


class WalkOnSpheresVirtualFixture:
    """See mesh_fixture.load_object_surface / build_wos_boundary for how the boundary this
    wraps gets built.

    Each project() call is an independent Monte Carlo estimate (WalkOnSpheresDiffusion
    averages a batch of random walkers that land on different, if nearby, boundary points
    each time) -- even at a fixed query position, consecutive calls can land several
    degrees apart purely from resampling. `smoothing` exponentially blends each new solve
    with the previous one (nlerp on the quaternion, see _nlerp_matrices) to damp that
    frame-to-frame jitter in continuous interactive use, at the cost of a small lag behind
    the true instantaneous field value -- separate from and on top of whatever WoS
    batch_size/max_iterations/convergence_threshold tuning already reduces the raw
    per-solve noise by (see the WalkOnSpheresDiffusion construction this fixture wraps).
    0 disables smoothing (use the raw solve every call); closer to 1 is smoother but laggier
    -- measured on spot.ply at update_command's ~50Hz call rate: steady-state spread (at a
    fixed query position) drops from ~2.6 deg mean unsmoothed to ~1.3 deg at 0.7, ~0.55 deg
    at 0.85, roughly an EMA filter's usual sqrt(smoothing/(2-smoothing)) noise reduction. The
    tradeoff is settling time if the query position itself is moving (the EE tracking a
    twist command) -- ballpark 170ms to settle within 5% of a step change at 0.7, ~370ms at
    0.85 -- so there's a point past which "smoother" starts reading as "lags behind where
    the arm actually is" rather than "steadier."
    """

    def __init__(self, wos, smoothing=0.75):
        self._wos = wos
        self._last_basis = np.eye(3)
        self._has_last = False
        self._smoothing = smoothing

    def get_distance(self, position):
        """Distance from `position` to the nearest boundary point -- a plain nearest-point
        lookup, independent of (and much cheaper than) the WoS solve in project(). Kept as
        a separate call rather than folded into project()'s return value (unlike
        ManifoldVirtualFixture/DiffusedSurfaceVirtualFixture, which do bundle it in) so
        callers that don't need it don't pay for it and project()'s signature stays plain.
        For ufactory_demos-style distance-based logic (e.g. TeleoperationController/
        AdmittanceTeleoperationController's adaptive twist-scaling factor).
        """
        distances, _ = self._wos.boundaries[0].get_closest_points(
            np.asarray(position)[np.newaxis, :])
        return float(np.asarray(distances).reshape(-1)[0])

    def project(self, proposed_reference):
        translator, position = _motor_position(proposed_reference)

        batch_points = self._wos.get_batch_from_point(position)
        try:
            local_basis, _, _ = self._wos.diffuse_rotations(batch_points)
        except ValueError:
            # No walker converged for this query (typical only far from the boundary) --
            # diffuse_rotations has nothing to average and raises rather than returning a
            # value. Reuse the last basis instead of propagating the exception into the
            # device thread's callback.
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
