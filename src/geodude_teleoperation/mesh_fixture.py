"""Builds a diffused-orientation Pointcloud from an arbitrary object placed on the
table -- either a mesh (simulating a top-view/ceiling-camera scan of it) or an
already-scanned point cloud (e.g. a real RGBD capture) -- then diffused_fields
propagates a full orientation field across that partial surface
(PointcloudScalarDiffusion.get_local_bases(), sourced from two keypoints at the
extremes of the cloud's longest axis -- see _diffuse_local_bases for why not the
whole silhouette boundary).

Pairs with virtual_fixture.DiffusedSurfaceVirtualFixture (nearest-point lookup) and
WalkOnSpheresVirtualFixture (live query, meaningful anywhere in the surrounding
workspace), which project a proposed EE reference using the resulting pointcloud's
diffused local basis.
"""

import numpy as np
import trimesh
from diffused_fields.diffusion.pointcloud_scalar_diffusion import PointcloudScalarDiffusion
from diffused_fields.manifold.pointcloud import Pointcloud
from scipy.spatial.transform import Rotation


def compute_resting_orientation(points):
    """PCA-based heuristic for laying an arbitrary (already-loaded, as-modeled/as-scanned)
    object down naturally on the table instead of using whatever orientation it happened to
    be authored/scanned in: rotates so the object's axis of *least* spatial variance (a flat
    object's cut-face normal, or a curved/elongated one's cross-sectional thickness
    direction) aligns with world Z -- i.e. the object's "thinnest" direction ends up
    vertical, its "widest spread" ends up lying across the table.

    Not something every object needs (assets/objects/*.stl and bunny_real.pcd were already
    modeled/scanned resting in a sensible orientation) -- meant for a newly added object
    that isn't, e.g. assets/objects/banana_half.ply, which is long and only lightly curved
    in one direction: with no rotation at all it would stand up on an edge instead of lying
    on its side. This only resolves the rotation about two axes (which way the thin axis
    points); the remaining rotational freedom about that axis (yaw) is whatever
    Rotation.align_vectors happens to return, not something this tries to control.

    Args:
        points (np.ndarray): (N, 3) points, in the object's own local/model frame (i.e.
            before any table_pose is applied).

    Returns:
        np.ndarray: (3, 3) rotation matrix -- use as the rotation part of a table_pose.
    """
    centered = points - points.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
    thin_axis = eigvecs[:, 0]  # smallest eigenvalue = least variance

    rotation, _ = Rotation.align_vectors([[0.0, 0.0, 1.0]], [thin_axis])
    return rotation.as_matrix()


def _select_keypoints(pointcloud):
    """Two-pass farthest-point selection, via *diffusion* distance rather than straight-line
    Euclidean distance -- respects the point cloud's actual connectivity, unlike a
    bounding-box heuristic (which can pick two points that are geodesically close despite
    being Euclidean-far apart, e.g. across a concavity or a wraparound shape).

    Standard two-pass farthest-point approximation of a point set's diameter: an arbitrary
    seed -> the point farthest from it (A) -> the point farthest from A alone (B). Built on
    PointcloudScalarDiffusion.sample_points, which already implements exactly this
    "diffuse, take the argmin as the next source" step (already in diffused_fields, unused
    anywhere else in the codebase) -- except its own random starting seed
    (`random.randint(0, len(vertices))`, inclusive on both ends in Python's `random` module)
    can select an out-of-range index, and a repeatable demo wants deterministic keypoints
    across runs anyway, so the seed is chosen explicitly here (nearest point to the cloud's
    centroid -- itself an arbitrary choice, but a stable, reproducible one) instead of relying
    on sample_points' own default.

    A short diffusion_scalar (10, distinct from the much larger one used for the tangent
    field's actual diffusion below) is deliberately used for this selection step: the heat
    method's short-time behavior approximates true geodesic distance (Varadhan's formula);
    a longer diffusion time saturates toward a near-uniform value, washing out the spatial
    signal sample_points ranks "farthest" by.
    """
    n = len(pointcloud.vertices)
    diffusion = PointcloudScalarDiffusion(pointcloud, diffusion_scalar=10)

    seed = int(np.argmin(np.linalg.norm(
        pointcloud.vertices - pointcloud.vertices.mean(axis=0), axis=1)))
    u0 = np.zeros(n)
    u0[seed] = 1
    point_a = int(diffusion.sample_points(1, u0=u0)[0])

    u0 = np.zeros(n)
    u0[point_a] = 1
    point_b = int(diffusion.sample_points(1, u0=u0)[0])

    return [point_a, point_b]


def _diffuse_local_bases(pointcloud):
    # Sourced from two keypoints (see _select_keypoints), not the whole silhouette boundary
    # (source_vertices=[], the "no explicit sources" default) -- and with diffusion_scalar=1e6,
    # not the class default of 1.
    #
    # Root cause this addresses: local_bases' *tangent* axis (unlike its normal, which is
    # estimated independently and is fine either way) comes from the gradient of a diffused
    # scalar potential. Diffusing from the whole boundary at once (a closed curve, not a
    # point) tends to create multiple internal local extrema of that potential on any
    # complex, non-convex shape (a real object has protrusions -- e.g. the bunny's ears,
    # legs -- each pulling toward its own local peak) -- and *between* extrema, the gradient
    # necessarily passes through a watershed/saddle region where its direction reverses
    # near-discontinuously (confirmed: max angular disagreement pinned at ~90 deg between
    # nearby points, independent of diffusion_scalar over 9 orders of magnitude -- a
    # topological fact about gradient fields on complex surfaces (see the "hairy ball
    # theorem" family of results), not a numerical conditioning issue a bigger diffusion
    # timescale can fix). Confirmed present identically in dof_old's own original
    # implementation on this exact point cloud -- not a regression from porting.
    #
    # Diffusing from two well-separated keypoints instead establishes one single global flow
    # direction end-to-end, sidestepping the multi-peak problem for any query region that
    # isn't itself very close to a source (a source point's own immediate neighborhood has
    # the opposite problem -- gradient direction sweeps through all angles radially around a
    # point source, confirmed empirically worse there) or unlucky enough to sit in whatever
    # single watershed the two-source configuration still has somewhere.
    # Measured on the bunny cloud: end-to-end live WalkOnSpheresVirtualFixture query noise
    # (independent solves at the same position) dropped from ~32-51 deg mean / up to 172 deg
    # max (whole-boundary source) to ~6-15 deg mean / 12-28 deg max (this).
    #
    # diffusion_scalar needs to be *much* larger for a 2-point source than for a whole-
    # boundary one (1e6 here vs. 1000 there): heat diffusing from an entire boundary curve
    # starts everywhere already close to some source point, while two point sources have to
    # spread the full distance across the object -- verified the resulting gradient's
    # *direction* pattern (not just magnitude) is stable across diffusion_scalar spanning
    # 1e4 to 1e9, i.e. 1e6 is comfortably in the converged, not still-transient, regime.
    #
    # (This still doesn't fully fix an exactly symmetric test object like debug_box.stl's
    # flat rectangular top under a whole-boundary source, whose center-point gradient is
    # forced to exactly zero *by symmetry* for any diffusion_scalar -- not the situation this
    # fix targets; real/scanned objects generally aren't exactly symmetric like that. In
    # practice the keypoint-based source above sidesteps it too, since the keypoints won't
    # generally land exactly on that object's axis of symmetry.)
    diffusion = PointcloudScalarDiffusion(
        pointcloud, source_vertices=_select_keypoints(pointcloud), diffusion_scalar=1e6)
    diffusion.get_local_bases()
    return pointcloud


def scan_top_view(mesh, camera_height=1.0, resolution=0.002):
    """Ray-casts a grid of downward rays through `mesh`'s XY footprint, from
    `camera_height` above its highest point, keeping only the first surface each
    ray hits. This is occlusion-aware (unlike filtering vertices by upward-facing
    normal, which would also keep e.g. the underside of a self-occluded bump) --
    it is what a ceiling-mounted depth camera actually sees.

    Args:
        mesh (trimesh.Trimesh): mesh in world coordinates (already transformed to
            its placement on the table).
        camera_height (float): ray origin height above the mesh's highest point.
        resolution (float): grid spacing (m) of the ray origins over the mesh's
            XY footprint.

    Returns:
        tuple[np.ndarray, np.ndarray]: (N, 3) world-frame hit points, and their
        (N, 3) exact face normals from the mesh (not re-estimated) -- every hit is
        by construction a front face as seen from the ray origin, so these are
        already consistently oriented toward the camera, unlike a PCA-based
        re-estimate over the resulting sparse point cloud (which has no such
        guarantee and was observed to flip sign on some points of a thin scan).
    """
    bounds = mesh.bounds
    x = np.arange(bounds[0, 0], bounds[1, 0], resolution)
    y = np.arange(bounds[0, 1], bounds[1, 1], resolution)
    xx, yy = np.meshgrid(x, y)

    origins = np.column_stack(
        [xx.ravel(), yy.ravel(), np.full(xx.size, bounds[1, 2] + camera_height)]
    )
    directions = np.tile(np.array([0.0, 0.0, -1.0]), (origins.shape[0], 1))

    locations, _, triangle_indices = mesh.ray.intersects_location(
        origins, directions, multiple_hits=False)

    if locations.shape[0] == 0:
        raise RuntimeError(
            "scan_top_view: no rays hit the mesh -- check camera_height/resolution "
            "against the mesh's actual placement/bounds"
        )

    return locations, mesh.face_normals[triangle_indices]


def build_diffused_surface(mesh_path, table_pose=None, camera_height=1.0, resolution=0.002):
    """Loads `mesh_path`, places it at `table_pose`, scans its top-view-visible
    surface, and returns a diffused_fields Pointcloud with `local_bases` populated.

    Args:
        mesh_path (str): path to a mesh file trimesh can load (.stl, .obj, ...).
        table_pose (np.ndarray, optional): 4x4 homogeneous transform placing the
            mesh in world coordinates (e.g. on geodude's table). Identity if omitted.
        camera_height (float): see scan_top_view.
        resolution (float): see scan_top_view. No downsampling option here (unlike
            most other diffused_fields pipelines) -- it would desync the normals
            array from the vertices it was computed for; use a coarser resolution
            directly if fewer points are wanted.

    Returns:
        diffused_fields.manifold.pointcloud.Pointcloud: with `.vertices`,
        `.normals`, and `.local_bases` populated, ready for
        DiffusedSurfaceVirtualFixture.
    """
    mesh = trimesh.load(mesh_path, force="mesh")

    if table_pose is not None:
        mesh.apply_transform(table_pose)

    points, normals = scan_top_view(
        mesh, camera_height=camera_height, resolution=resolution)

    # is_cluster_points=False, voxel_size left unset: both would drop/reorder points relative
    # to `points`, which would desync them from `normals` (computed once, per-point, from the
    # ray hits -- not re-derivable after reindexing). Not needed anyway for an already-clean
    # raycasted scan.
    pointcloud = Pointcloud(vertices=points, is_cluster_points=False)
    # Assigned directly rather than via get_normals(): that re-estimates via PCA over the
    # sparse point cloud, which has no guaranteed sign consistency (observed flipping on part
    # of a thin scan) -- these normals are exact, straight from the mesh triangles that were
    # actually hit, and are already camera-facing by construction of a visible ray hit.
    pointcloud.normals = normals

    return _diffuse_local_bases(pointcloud)


def read_ascii_pcd(path):
    """Minimal reader for ASCII-encoded .pcd (PCL point cloud data) files --
    trimesh has no .pcd loader. Only the x/y/z fields are used; any others
    (rgb, normals, ...) present in the file are ignored.

    Returns:
        np.ndarray: (N, 3) points, in the order they appear in the file.
    """
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


def _build_diffused_surface_from_points(points, table_pose=None, normal_orientation=1):
    """Shared implementation behind build_diffused_surface_from_pointcloud (loads .pcd) and
    load_object_surface (loads anything else trimesh can read as a point cloud) -- once
    points are in hand as a plain (N, 3) array, loading format is irrelevant to everything
    that follows.
    """
    if table_pose is not None:
        points = points @ table_pose[:3, :3].T + table_pose[:3, 3]

    # There's no camera-direction ground truth here to derive exact normals from (unlike
    # build_diffused_surface's raycast path) -- this is already the input data, not something
    # scanned from a known viewpoint -- so this falls back to Pointcloud's own PCA-based normal
    # estimation (get_normals(), the ordinary path every other diffused_fields pointcloud
    # example uses).
    #
    # normal_orientation is a genuine per-object property, not a universal estimator constant
    # -- an earlier version of this docstring claimed +1 always matches dof_old's original
    # (open3d-based) implementation, verified on the bunny point cloud specifically. That's
    # true for the bunny, but turned out NOT to generalize: get_normals()'s underlying
    # orient_normals_consistent_tangent_plane only guarantees *local* agreement between
    # neighboring points' normals, not which overall polarity ("inward" vs "outward") the
    # propagation converges to -- that depends on which point it happens to start from, which
    # is a property of each specific point cloud, not of the sklearn/open3d estimator itself.
    # Confirmed empirically: neither "normals point away from the centroid on average" nor
    # the same check restricted to convex-hull-only points reliably predicts the right value
    # -- both said flip-it for the bunny, which is verified wrong. So this has to be a
    # caller-supplied parameter, checked per new object (e.g. query the resulting field
    # directly above the object and confirm the approach z-axis points down into the surface,
    # not up away from it -- see WalkOnSpheresVirtualFixture) rather than assumed.
    pointcloud = Pointcloud(
        vertices=points, is_cluster_points=False, normal_orientation=normal_orientation)
    pointcloud.get_normals()

    return _diffuse_local_bases(pointcloud)


def build_diffused_surface_from_pointcloud(pcd_path, table_pose=None, normal_orientation=1):
    """Loads an already-scanned point cloud (e.g. a real RGBD capture, as
    opposed to a synthetic mesh -- see build_diffused_surface for that case),
    places it at `table_pose`, and returns a diffused_fields Pointcloud with
    `local_bases` populated.

    Args:
        pcd_path (str): path to an ASCII .pcd file (see read_ascii_pcd).
        table_pose (np.ndarray, optional): 4x4 homogeneous transform placing the
            points in world coordinates (e.g. on geodude's table). Identity if omitted.
        normal_orientation (1 or -1): sign correction for the PCA-estimated normals'
            overall polarity -- a property of this specific point cloud, not a universal
            constant; see _build_diffused_surface_from_points for why. 1 is correct for
            assets/objects/bunny_real.pcd (verified against ground truth); other point
            clouds may need -1 instead.

    Returns:
        diffused_fields.manifold.pointcloud.Pointcloud: with `.vertices`,
        `.normals`, and `.local_bases` populated, ready for
        DiffusedSurfaceVirtualFixture.
    """
    return _build_diffused_surface_from_points(
        read_ascii_pcd(pcd_path), table_pose=table_pose, normal_orientation=normal_orientation)


def load_points(path):
    """Just the raw (N, 3) vertices of a mesh or point cloud file, with no diffusion --
    e.g. for computing a centroid to place an arbitrary "--object" argument on the table
    before running the (much more expensive) full load_object_surface pipeline on it. Same
    format dispatch as load_object_surface: .pcd via read_ascii_pcd, anything else via
    trimesh (mesh or point-cloud-only, either way trimesh's own .vertices covers it).
    """
    if str(path).lower().endswith(".pcd"):
        return read_ascii_pcd(path)
    return np.asarray(trimesh.load(path).vertices)


def load_object_surface(path, table_pose=None, camera_height=1.0, resolution=0.002,
                         normal_orientation=1):
    """Single entry point for either build_diffused_surface or
    build_diffused_surface_from_pointcloud, dispatching on what `path` actually contains
    rather than requiring the caller to know in advance -- lets a demo take an arbitrary
    "--object" file argument without needing a second flag for its kind.

    .pcd is always treated as a point cloud (trimesh has no native .pcd loader -- see
    read_ascii_pcd); anything else is loaded once via trimesh and dispatched on whether that
    came back with faces (a real mesh, e.g. .stl/.obj -- the top-view-raycast path) or not
    (a point-cloud-only file, e.g. many .ply exports -- the direct-point-cloud path).

    Args:
        path (str): path to a mesh (.stl, .obj, ...) or point cloud (.pcd, or any
            point-cloud .ply/similar trimesh can load) file.
        table_pose (np.ndarray, optional): 4x4 homogeneous transform placing the object
            in world coordinates. Identity if omitted.
        camera_height, resolution: forwarded to build_diffused_surface if `path` turns out
            to be a mesh; ignored for a point cloud.
        normal_orientation (1 or -1): only used for the point-cloud path (a mesh's normals
            come from exact raycasts, not PCA, so this doesn't apply) -- see
            _build_diffused_surface_from_points for why this is a per-object property, not
            a constant. Wrong-by-default here shows up as the virtual fixture's approach
            axis pointing away from the object instead of into it.

    Returns:
        diffused_fields.manifold.pointcloud.Pointcloud: with `.vertices`, `.normals`, and
        `.local_bases` populated, ready for DiffusedSurfaceVirtualFixture/
        WalkOnSpheresVirtualFixture.
    """
    if str(path).lower().endswith(".pcd"):
        return build_diffused_surface_from_pointcloud(
            path, table_pose=table_pose, normal_orientation=normal_orientation)

    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) > 0:
        return build_diffused_surface(
            path, table_pose=table_pose, camera_height=camera_height, resolution=resolution)

    points = np.asarray(loaded.vertices)
    return _build_diffused_surface_from_points(
        points, table_pose=table_pose, normal_orientation=normal_orientation)


def build_wos_boundary(pointcloud, num_points=3000, seed=0):
    """Builds a downsampled copy of `pointcloud` for use as
    WalkOnSpheresDiffusion's boundary in a *live*, per-query solve (see
    WalkOnSpheresVirtualFixture) -- `pointcloud` itself must already have
    `local_bases` populated by a surface diffusion pass
    (build_diffused_surface / build_diffused_surface_from_pointcloud).

    Only downsamples; the diffused local_bases values themselves are copied
    from the corresponding full-resolution points, not recomputed -- no
    second diffusion pass needed, and the returned boundary's orientation
    field is exactly as accurate as the source cloud's, just sparser.

    Downsampling matters because WalkOnSpheresDiffusion's per-iteration cost
    is dominated by a scipy cKDTree nearest-point query against the boundary,
    and that cost is highly sensitive to the boundary's *point count*, not
    just its geometric extent (measured on a 55.8k-point bunny scan: ~3.6ms
    for a single k=1 query of 128 points against the full cloud, vs ~0.16ms
    against a 2000-point subsample of the same surface -- roughly tracking
    the ~28x point-count reduction, not some fixed per-call floor). A full
    WoS solve needs ~max_iterations sequential queries (each iteration's
    walkers depend on the last, so they can't be parallelized away) -- at
    num_points=3000, batch_size=128, max_iterations=24, that's ~6ms/solve end
    to end, well under the ~20ms budget of a 50Hz device-callback loop (see
    TeleoperationController.update_command, which is where this boundary is
    meant to be queried from -- ufactory_demos' own GetLocalBasis-backed
    virtual fixture was likewise called from the 3D-mouse device thread's own
    ~50Hz loop, decoupled from the robot's separate 1kHz control loop, not
    from the control loop itself).

    Args:
        pointcloud: diffused_fields Pointcloud with local_bases already
            populated.
        num_points (int): boundary size after downsampling. Trades query
            speed for how finely the orientation field varies near the
            surface -- since this only feeds a virtual fixture guiding
            approach orientation from a distance, not a position constraint,
            a coarser boundary than the full scan is normally fine.
        seed: RNG seed for the (uniform, without replacement) point subset.

    Returns:
        diffused_fields.manifold.pointcloud.Pointcloud: with `.vertices`,
        `.normals`, `.local_bases` populated (subsampled, index-aligned).
    """
    n = min(num_points, len(pointcloud.vertices))
    indices = np.random.default_rng(seed).choice(
        len(pointcloud.vertices), size=n, replace=False)

    boundary = Pointcloud(
        vertices=pointcloud.vertices[indices].copy(), is_cluster_points=False)
    boundary.normals = pointcloud.normals[indices].copy()
    boundary.local_bases = pointcloud.local_bases[indices].copy()
    return boundary
