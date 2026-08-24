import gafropy
import numpy as np
from scipy.spatial import cKDTree


def _orient_approach_into_surface(local_basis):

    approach_basis = local_basis.copy()
    approach_basis[:, 1] *= -1
    approach_basis[:, 2] *= -1
    return approach_basis


def _motor_position(motor):
    translator = motor.get_translator()
    return translator, np.array([translator.x(), translator.y(), translator.z()])


def _nlerp_matrices(old, new, t):
    q_old = np.asarray(gafropy.Rotor.from_matrix(old).to_quaternion())
    q_new = np.asarray(gafropy.Rotor.from_matrix(new).to_quaternion())
    if np.dot(q_old, q_new) < 0:
        q_new = -q_new
    q = (1 - t) * q_old + t * q_new
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.asarray(gafropy.Rotor.from_quaternion(float(w), float(x), float(y), float(z)).to_rotation_matrix())


def _tangent_direction(mesh, vertex_index, reference):
    """A unit tangent vector at a single vertex: `reference` projected onto that vertex's own
    tangent plane, falling back to a different reference if nearly parallel to the normal."""
    normal = np.asarray(mesh.vertex_normals[vertex_index])
    ref = reference
    if abs(np.dot(ref, normal)) > 0.99:
        ref = np.array([1.0, 0.0, 0.0])
    tangent = ref - np.dot(ref, normal) * normal
    return tangent / np.linalg.norm(tangent)


def _boundary_frame(tangent, normal):
    """Exact orthonormal frame at a surface point: columns [tangent, y, normal]."""
    t = tangent / np.linalg.norm(tangent)
    n = normal - np.dot(normal, t) * t
    n = n / np.linalg.norm(n)
    y = np.cross(n, t)
    y = y / np.linalg.norm(y)
    return np.column_stack([t, y, n])


def _quaternion_to_fiber_rotor(w, x, y, z):
    """gafro's raw Cl(3,0,0) rotor representation (blade 0 = scalar, blade 3 = e12, blade 5 =
    e13, blade 6 = e23) -- the representation RotorFieldDiffusion consumes directly -- from a
    standard Hamilton quaternion. See gafropy's own rotor sign convention: blade0=w, blade3=-z,
    blade5=y, blade6=-x."""
    r = gafropy.FiberMultivector()
    r.set(0, float(w))
    r.set(3, -float(z))
    r.set(5, float(y))
    r.set(6, -float(x))
    return r


def _fiber_rotor_to_rotation_matrix(rotor):
    w = rotor.get(0)
    z = -rotor.get(3)
    y = rotor.get(5)
    x = -rotor.get(6)
    return np.asarray(gafropy.Rotor.from_quaternion(float(w), float(x), float(y), float(z)).to_rotation_matrix())


def _continuous_boundary_rotors(mesh, tangent, normal):
    """Per-vertex boundary rotors (as gafropy.FiberMultivector), made sign-continuous across the
    mesh via face-adjacency BFS -- needed because RotorFieldDiffusion's Riemannian mean is
    sign-sensitive (R and -R are the same physical rotation but sit at S^3-geodesic distance pi
    apart)."""
    n = len(tangent)
    quats_wxyz = np.zeros((n, 4))
    for i in range(n):
        cga_rotor = gafropy.Rotor.from_matrix(_boundary_frame(tangent[i], normal[i]))
        x, y, z, w = cga_rotor.to_quaternion()
        quats_wxyz[i] = [w, x, y, z]

    fixed = np.zeros(n, dtype=bool)
    faces = mesh.faces
    num_faces = len(faces)
    face_neighbors = [[] for _ in range(num_faces)]
    for f1, f2 in mesh.face_adjacency:
        face_neighbors[f1].append(f2)
        face_neighbors[f2].append(f1)

    face_visited = np.zeros(num_faces, dtype=bool)
    for seed in range(num_faces):
        if face_visited[seed]:
            continue
        face_visited[seed] = True
        for v in faces[seed]:
            fixed[v] = True
        queue = [seed]
        while queue:
            f = queue.pop()
            for nb in face_neighbors[f]:
                if face_visited[nb]:
                    continue
                face_visited[nb] = True
                queue.append(nb)
                verts = faces[nb]
                reference = next((v for v in verts if fixed[v]), None)
                for v in verts:
                    if fixed[v]:
                        continue
                    if reference is not None and np.dot(quats_wxyz[v], quats_wxyz[reference]) < 0.0:
                        quats_wxyz[v] = -quats_wxyz[v]
                    fixed[v] = True

    return [_quaternion_to_fiber_rotor(*quats_wxyz[i]) for i in range(n)]


DEFAULT_MAX_BOUNDARY_SAMPLES = 2048
# Heat-kernel length in mean-edge units. mean_edge^2 is too local on laser
# reconstructions (mm edges): orientation then tracks scan noise as the probe moves.
DEFAULT_T_SPATIAL_EDGE_FACTOR = 5.0


def boundary_sample_indices(n_vertices, num_boundary_samples=None,
                            max_default=DEFAULT_MAX_BOUNDARY_SAMPLES):
    """Vertex subset used as RotorFieldDiffusion samples.

    Small meshes keep every vertex. Large reconstructed clouds default to a cap so
    each project() Frechet mean stays cheap enough for interactive motion.
    """
    n = int(n_vertices)
    if n <= 0:
        return np.arange(0, dtype=int)
    if num_boundary_samples is not None:
        k = min(max(int(num_boundary_samples), 1), n)
    elif n <= max_default:
        return np.arange(n)
    else:
        k = max_default
    if k >= n:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, k, dtype=int))


class OrientationField:
    """Ambient orientation field around a mesh's surface, via gafropy.RotorFieldDiffusion.

    A tangent direction (`key_direction`, projected onto a source vertex's tangent plane) is
    diffused across the mesh surface via gafropy.ConnectionLaplacian (the Vector Heat Method)
    into a per-vertex boundary rotor, combining tangent + surface normal. That boundary rotor
    field is then extended into ambient R^3 via gafropy.RotorFieldDiffusion: a spatial-heat-
    kernel-weighted Riemannian (Frechet-mean) average over the boundary rotors, which by
    construction never leaves the unit-rotor manifold (see
    gafro/geometry/RotorFieldDiffusion.hpp's own docstring for the exact math).

    Query cost is capped: reconstructed clouds default to
    `DEFAULT_MAX_BOUNDARY_SAMPLES` Frechet-mean samples. Spatial kernel default is
    `(5 * mean_edge)^2` so mm-scale laser edges do not make the field twitch along a path.

    Deterministic -- unlike the previous diffused_fields Walk-on-Spheres implementation,
    rerunning gives bit-for-bit identical results, and there is no Monte Carlo
    batch_size/convergence_threshold to tune.
    """

    def __init__(self, mesh, key_direction, source_vertex=None, t_surface=None, t_spatial=None,
                 num_boundary_samples=None, max_iterations=50, smoothing=0.75):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        mean_edge = float(mesh.edges_unique_length.mean())

        vertex = source_vertex if source_vertex is not None else int(np.argmax(vertices[:, 2]))

        direction = np.asarray(key_direction, dtype=np.float64)
        direction = direction / np.linalg.norm(direction)
        v0 = _tangent_direction(mesh, vertex, direction)

        complex_ = gafropy.SimplicialComplex.build_from_faces(len(vertices), mesh.faces.tolist())
        connection = gafropy.ConnectionLaplacian.build(complex_, list(vertices))
        t_surface = t_surface if t_surface is not None else 5.0 * connection.get_mean_edge_length_squared()

        diffused = np.asarray(connection.diffuse_vector(v0, vertex, t_surface))
        magnitude = np.linalg.norm(diffused, axis=1)
        degenerate = magnitude < 1e-9
        tangent = np.where(degenerate[:, None], v0[None, :], diffused / np.maximum(magnitude, 1e-30)[:, None])

        rotors = _continuous_boundary_rotors(mesh, tangent, normals)

        indices = boundary_sample_indices(len(vertices), num_boundary_samples)
        positions = [vertices[i] for i in indices]
        self._field = gafropy.RotorFieldDiffusion(positions, [rotors[i] for i in indices])
        self._t_spatial = (
            t_spatial if t_spatial is not None
            else (DEFAULT_T_SPATIAL_EDGE_FACTOR * mean_edge) ** 2)
        self._max_iterations = max_iterations
        self._kdtree = cKDTree(np.asarray(positions))
        self._smoothing = smoothing
        self._last_basis = np.eye(3)
        self._has_last = False
        self._last_approach_basis = None
        self._has_last_approach = False

    def get_distance(self, position):
        distance, _ = self._kdtree.query(np.asarray(position))
        return float(distance)

    def project(self, proposed_reference):
        translator, position = _motor_position(proposed_reference)

        try:
            rotor = self._field.diffuse(position, self._t_spatial, max_iterations=self._max_iterations)
            local_basis = _fiber_rotor_to_rotation_matrix(rotor)
        except RuntimeError:
            # Every boundary sample's spatial weight underflowed to zero -- position too far
            # from the object at this t_spatial. Fall back to the last known-good orientation
            # rather than raising through this control-critical call path.
            local_basis = self._last_basis if self._has_last else np.eye(3)

        if self._has_last and self._smoothing > 0:
            local_basis = _nlerp_matrices(
                self._last_basis, local_basis, 1 - self._smoothing)
        self._last_basis = local_basis
        self._has_last = True

        approach_basis = _orient_approach_into_surface(local_basis)
        rotor = gafropy.Rotor.from_matrix(approach_basis)
        return gafropy.Motor(translator, rotor)

    def approach_basis_at(self, position):
        """Approach frame at ``position``. Does not update project() smoothing state.

        Spatial-weight underflow keeps the last successful approach frame
        (flipped identity if none yet) instead of snapping to world -Z.
        """
        position = np.asarray(position, dtype=float).reshape(3)
        try:
            rotor = self._field.diffuse(
                position, self._t_spatial, max_iterations=self._max_iterations)
            local_basis = _fiber_rotor_to_rotation_matrix(rotor)
        except RuntimeError:
            if self._has_last_approach:
                return np.array(self._last_approach_basis, copy=True)
            return _orient_approach_into_surface(np.eye(3))
        basis = _orient_approach_into_surface(local_basis)
        self._last_approach_basis = basis
        self._has_last_approach = True
        return np.array(basis, copy=True)

    def display_basis_at(self, position):
        """Object +Z frame whose +Z matches /field_z arrows (out of the working face)."""
        B = np.array(self.approach_basis_at(position), copy=True)
        B[:, 1] *= -1
        B[:, 2] *= -1
        return B

    def field_z_arrows(self, points, *, max_count=256, length=0.03, outward=True):
        """Start/end segments for viser. Default ``outward=True`` draws +normal
        (out of the working face). EE snap still uses into-surface approach Z."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        idx = boundary_sample_indices(len(points), max_count)
        pts = points[idx]
        z_dirs = np.array([self.approach_basis_at(p)[:, 2] for p in pts], dtype=float)
        return field_z_segments(pts, field_z_display_dirs(z_dirs, outward=outward), length)


def field_z_display_dirs(approach_z, *, outward=True):
    z = np.asarray(approach_z, dtype=float)
    return -z if outward else z


def field_z_segments(origins, z_dirs, length):
    origins = np.asarray(origins, dtype=float).reshape(-1, 3)
    z = np.asarray(z_dirs, dtype=float).reshape(-1, 3)
    n = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.maximum(n, 1e-30)
    return np.stack([origins, origins + float(length) * z], axis=1)


class FieldAlignPreview:
    """Paint-only field readout at a capped rate. Does not enable the pose spring."""

    def __init__(self, vf, *, min_dt=1.0 / 15.0):
        self._vf = vf
        self._min_dt = float(min_dt)
        self._last_t = float("-inf")
        self.motor = None
        self.basis = None

    def update(self, live_ee, now):
        if self.motor is not None and (now - self._last_t) < self._min_dt:
            return self.motor
        self.motor = self._vf.project(live_ee)
        self._last_t = now
        return self.motor

    def update_display_basis(self, xyz, now):
        """Outward object basis (+Z = /field_z arrows) at ``xyz``."""
        if self.basis is not None and (now - self._last_t) < self._min_dt:
            return self.basis
        self.basis = self._vf.display_basis_at(xyz)
        self._last_t = now
        return self.basis

    def update_approach_basis(self, xyz, now):
        """EE/flange basis (+Z into the surface, anti-parallel to /field_z)."""
        if self.basis is not None and (now - self._last_t) < self._min_dt:
            return self.basis
        self.basis = self._vf.approach_basis_at(xyz)
        self._last_t = now
        return self.basis

