import gafropy
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from geodude_teleoperation.stringer_field import length_tangents


def _orient_approach_into_surface(local_basis):

    approach_basis = local_basis.copy()
    approach_basis[:, 1] *= -1
    approach_basis[:, 2] *= -1
    return approach_basis


def lock_roll_to_axis(approach, axis, min_sin=0.15):
    """Keep tool +Z; yaw so +X is ``axis`` projected into the tool XY plane.

    Rotor interpolation and far-field nlerp can leave +X diagonal even when
    surface heat is along the stringer. Reclocking to the PCA length axis
    removes that extra roll. Degenerate when ``axis`` is parallel to tool Z
    (end-caps): keep the field's own +X.
    """
    R = np.asarray(approach, dtype=float).reshape(3, 3)
    z = R[:, 2]
    a = np.asarray(axis, dtype=float).reshape(3)
    nrm = np.linalg.norm(a)
    if nrm < 1e-12:
        return np.array(R, copy=True)
    a = a / nrm
    x = a - np.dot(a, z) * z
    mag = np.linalg.norm(x)
    if mag < float(min_sin):
        return np.array(R, copy=True)
    x = x / mag
    if np.dot(x, R[:, 0]) < 0.0:
        x = -x
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    return np.column_stack([x, y, z])


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
# Past the object, blend toward a table-flat frame (vention +Z) so the EE
# does not keep the last into-surface tilt and drive through the table.
# ``NEAR`` is negative: the outer hull already overlaps the table field.
FAR_FIELD_OVERLAP = 0.06
FAR_FIELD_NEAR = -FAR_FIELD_OVERLAP
FAR_FIELD_FAR = 0.45


def far_field_weight(distance, near=FAR_FIELD_NEAR, far=FAR_FIELD_FAR):
    """0 in the object core, 1 once ``distance`` clears ``far``.

    Ease-out cubic (not smoothstep): the rim has immediate slope so leaving the
    part does not keep a dead zone of object tilt. ``near`` may be negative to
    overlap the table field into the hull's outer band.
    """
    d = float(distance)
    near = float(near)
    far = float(far)
    if far <= near:
        return 0.0 if d <= near else 1.0
    if d <= near:
        return 0.0
    if d >= far:
        return 1.0
    t = (d - near) / (far - near)
    return 1.0 - (1.0 - t) ** 3


def planar_hull_signed_distance(points_xy, query_xy):
    """Negative inside the 2D convex hull, positive outside. Unsigned fallback."""
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    q = np.asarray(query_xy, dtype=float).reshape(2)
    if len(pts) < 3:
        distance, _ = cKDTree(pts).query(q)
        return float(distance)
    try:
        equations = ConvexHull(pts).equations
    except Exception:
        distance, _ = cKDTree(pts).query(q)
        return float(distance)
    vals = equations[:, :2] @ q + equations[:, 2]
    norms = np.linalg.norm(equations[:, :2], axis=1)
    return float(np.max(vals / np.maximum(norms, 1e-30)))


def table_flat_basis(up=None, tangent=None):
    """Outward table frame: +Z along ``up`` (default world/vention +Z)."""
    n = (np.array([0.0, 0.0, 1.0], dtype=float) if up is None
         else np.asarray(up, dtype=float).reshape(3))
    t = (np.array([1.0, 0.0, 0.0], dtype=float) if tangent is None
         else np.asarray(tangent, dtype=float).reshape(3))
    return _boundary_frame(t, n)


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

    Surface frames follow the old stringer path: PCA end-cap heat,
    ``x = ∇u``, ``z`` = surface normal, ``y = z × x``. Default ``flow='length'``
    puts source/sink on the short end-caps (heat along the long axis).
    ``flow='width'`` uses the long sides instead. That boundary rotor
    field is extended into ambient R^3 via RotorFieldDiffusion
    (startup only; queries are deterministic Frechet means, not Walk-on-Spheres).

    Query cost is capped: reconstructed clouds default to
    `DEFAULT_MAX_BOUNDARY_SAMPLES` Frechet-mean samples. Spatial kernel default is
    `(5 * mean_edge)^2` so mm-scale laser edges do not make the field twitch along a path.
    Queries use a planar hull signed distance: the outer ~6 cm of the object
    already overlaps the table-flat field, then eases out to full table by 45 cm.
    """

    def __init__(self, mesh, key_direction=None, source_vertex=None, t_surface=None,
                 t_spatial=None, num_boundary_samples=None, max_iterations=50,
                 smoothing=0.75, end_line_fraction=0.06, swap_end_lines=False,
                 diffusion_scalar=5000.0, flow="length", roll_flip=False):
        del key_direction, source_vertex, t_surface
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        mean_edge = float(mesh.edges_unique_length.mean())

        tangent, sink, source, axis = length_tangents(
            mesh, fraction=end_line_fraction, swap=swap_end_lines,
            flow=flow, diffusion_scalar=diffusion_scalar)
        mean_t = np.mean(tangent, axis=0)
        if np.dot(mean_t, axis) < 0.0:
            axis = -np.asarray(axis, dtype=float)
        self._length_axis = axis / np.linalg.norm(axis)
        # Tool orientation A (False) or B (True). Roll about tool Z is locked
        # to the PCA length axis, so the opposite axis IS the 180 deg flip --
        # exactly, and with no separate rotation to compose. Tool +Z, the
        # surface normal and therefore every distance the press latch measures
        # are untouched by it.
        self._roll_flip = bool(roll_flip)
        self.sink_points = vertices[sink]
        self.source_points = vertices[source]

        rotors = _continuous_boundary_rotors(mesh, tangent, normals)

        indices = boundary_sample_indices(len(vertices), num_boundary_samples)
        positions = [vertices[i] for i in indices]
        self._field = gafropy.RotorFieldDiffusion(positions, [rotors[i] for i in indices])
        self._t_spatial = (
            t_spatial if t_spatial is not None
            else (DEFAULT_T_SPATIAL_EDGE_FACTOR * mean_edge) ** 2)
        self._max_iterations = max_iterations
        self._kdtree = cKDTree(np.asarray(positions))
        xy = vertices[:, :2]
        self._xy_tree = cKDTree(xy)
        try:
            self._hull_eq = ConvexHull(xy).equations if len(xy) >= 3 else None
        except Exception:
            self._hull_eq = None
        self._smoothing = smoothing
        self._last_basis = np.eye(3)
        self._has_last = False
        self._table_flat = table_flat_basis(tangent=self._length_axis)
        self._far_near = FAR_FIELD_NEAR
        self._far_far = FAR_FIELD_FAR

    def get_distance(self, position):
        distance, _ = self._kdtree.query(np.asarray(position))
        return float(distance)

    def get_planar_signed_distance(self, position):
        p = np.asarray(position, dtype=float).reshape(3)[:2]
        if self._hull_eq is None:
            distance, _ = self._xy_tree.query(p)
            return float(distance)
        vals = self._hull_eq[:, :2] @ p + self._hull_eq[:, 2]
        norms = np.linalg.norm(self._hull_eq[:, :2], axis=1)
        return float(np.max(vals / np.maximum(norms, 1e-30)))

    def _object_local_basis(self, position):
        try:
            rotor = self._field.diffuse(
                position, self._t_spatial, max_iterations=self._max_iterations)
            return _fiber_rotor_to_rotation_matrix(rotor)
        except RuntimeError:
            return None

    def _blended_local_basis(self, position, *, fallback):
        local = self._object_local_basis(position)
        if local is None:
            local = np.array(fallback, copy=True)
        w = far_field_weight(
            self.get_planar_signed_distance(position), self._far_near, self._far_far)
        if w >= 1.0:
            return np.array(self._table_flat, copy=True)
        if w > 0.0:
            local = _nlerp_matrices(local, self._table_flat, w)
        return local

    @property
    def roll_flip(self) -> bool:
        return bool(getattr(self, "_roll_flip", False))

    def set_roll_flip(self, flip) -> bool:
        """Select tool orientation A (0 deg) or B (rolled 180 deg about tool Z).

        The stringer looks the same from either side, but the tool does not:
        one roll puts the chamfer's working face down the part and the other
        puts it up. Both are legitimate approaches to the same surface, so this
        is a choice, not a correction.

        Applied as ``R @ Rz(pi)`` *after* the roll lock, not by negating the
        roll-lock axis: ``lock_roll_to_axis`` re-flips a candidate +X that
        opposes the field's own, so negating the axis is a no-op (measured --
        it returns the identical matrix).
        """
        self._roll_flip = bool(flip)
        return self._roll_flip

    def _approach_from_local(self, local_basis):
        approach = _orient_approach_into_surface(local_basis)
        axis = getattr(self, "_length_axis", None)
        if axis is None:
            return approach
        approach = lock_roll_to_axis(approach, axis)
        if getattr(self, "_roll_flip", False):
            # Rz(pi) in the TOOL frame: negate +X and +Y, keep +Z. Negating the
            # roll-lock axis instead does nothing, because lock_roll_to_axis
            # re-flips a candidate +X that opposes the field's own -- measured,
            # it returns the identical matrix.
            approach = approach @ np.diag([-1.0, -1.0, 1.0])
        return approach

    def project(self, proposed_reference):
        translator, position = _motor_position(proposed_reference)
        fallback = self._last_basis if self._has_last else self._table_flat
        local_basis = self._blended_local_basis(position, fallback=fallback)

        if self._has_last and self._smoothing > 0:
            local_basis = _nlerp_matrices(
                self._last_basis, local_basis, 1 - self._smoothing)
        self._last_basis = local_basis
        self._has_last = True

        approach_basis = self._approach_from_local(local_basis)
        rotor = gafropy.Rotor.from_matrix(approach_basis)
        return gafropy.Motor(translator, rotor)

    def approach_basis_at(self, position):
        """Approach frame at ``position``. Does not update project() smoothing state.

        Near the object this is into-surface field Z. Far away it blends to a
        table-flat approach (EE +Z = world −Z) so travel across the workstation
        stays level instead of keeping the last stringer tilt through the table.
        Roll about tool Z is the PCA length axis in the tool XY plane.
        """
        position = np.asarray(position, dtype=float).reshape(3)
        local_basis = self._blended_local_basis(position, fallback=self._table_flat)
        return self._approach_from_local(local_basis)

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


def workstation_field_z_arrows(points, *, pad=0.4, spacing=0.08, length=0.03,
                               exclude_radius=0.02, up=None):
    """Table-plane +Z arrows around the object (the expanded far field).

    Default exclude is small so light-blue arrows overlap the object's outer
    edge instead of leaving a gap.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    z = float(points[:, 2].min())
    lo = points[:, :2].min(axis=0) - float(pad)
    hi = points[:, :2].max(axis=0) + float(pad)
    xs = np.arange(lo[0], hi[0] + 0.5 * float(spacing), float(spacing))
    ys = np.arange(lo[1], hi[1] + 0.5 * float(spacing), float(spacing))
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])
    dist, _ = cKDTree(points).query(grid)
    origins = grid[dist > float(exclude_radius)]
    if len(origins) == 0:
        return np.zeros((0, 2, 3))
    z_dir = table_flat_basis(up=up)[:, 2]
    z_dirs = np.broadcast_to(z_dir, origins.shape)
    return field_z_segments(origins, z_dirs, length)


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

