"""Contact / free-space field query helpers (ported from offline demo)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from diffused_fields.diffusion import WalkOnSpheresDiffusion
from diffused_fields.manifold import Pointcloud


def _mean_quaternion(quats: np.ndarray) -> np.ndarray:
    """Markley-style quaternion mean (xyzw). Hemispherizes to quats[0]."""
    q = np.asarray(quats, dtype=float).reshape(-1, 4).copy()
    if len(q) == 1:
        return q[0] / np.linalg.norm(q[0])
    q0 = q[0]
    for i in range(1, len(q)):
        if np.dot(q[i], q0) < 0:
            q[i] = -q[i]
    A = q.T @ q
    _, vecs = np.linalg.eigh(A)
    q_mean = vecs[:, -1]
    if q_mean[3] < 0:
        q_mean = -q_mean
    return q_mean / np.linalg.norm(q_mean)


def _orient_normal(n: np.ndarray, ref: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=float).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    if np.dot(n, ref) < 0:
        n = -n
    return n


def fit_local_tangent_plane(
    points: np.ndarray, ref_normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Least-squares tangent plane via PCA.

    Returns (centroid, unit normal, residual_rms).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    c = pts.mean(axis=0)
    centered = pts - c
    if len(pts) < 3:
        n = _orient_normal(ref_normal, ref_normal)
        return c, n, 0.0
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    n = _orient_normal(vt[-1], ref_normal)
    resid = centered @ n
    rms = float(np.sqrt(np.mean(resid**2)))
    return c, n, rms


def fit_local_tangent_sphere(
    points: np.ndarray, ref_normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Algebraic least-squares sphere fit.

    Returns (center, outward unit normal at neighborhood centroid,
             signed geometric radius, residual_rms).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 4:
        c, n, rms = fit_local_tangent_plane(pts, ref_normal)
        return c, n, np.inf, rms

    A = np.concatenate([2.0 * pts, np.ones((len(pts), 1))], axis=1)
    b = np.sum(pts**2, axis=1)
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        c, n, rms = fit_local_tangent_plane(pts, ref_normal)
        return c, n, np.inf, rms

    center = sol[:3]
    rad2 = float(sol[3] + np.dot(center, center))
    if not np.isfinite(rad2) or rad2 <= 1e-10:
        c, n, rms = fit_local_tangent_plane(pts, ref_normal)
        return c, n, np.inf, rms

    radius = float(np.sqrt(rad2))
    centroid = pts.mean(axis=0)
    n_raw = centroid - center
    if np.linalg.norm(n_raw) < 1e-12:
        c, n, rms = fit_local_tangent_plane(pts, ref_normal)
        return c, n, np.inf, rms
    n = _orient_normal(n_raw, ref_normal)
    resid = np.abs(np.linalg.norm(pts - center, axis=1) - radius)
    rms = float(np.sqrt(np.mean(resid**2)))
    return center, n, radius, rms


def sphere_signed_distance(
    x: np.ndarray, center: np.ndarray, radius: float, outward_normal_hint: np.ndarray
) -> float:
    """Signed distance to sphere surface. Positive outside (along outward hint)."""
    v = np.asarray(x, dtype=float).reshape(3) - np.asarray(center, dtype=float).reshape(3)
    dist = float(np.linalg.norm(v) - radius)
    radial = v / (np.linalg.norm(v) + 1e-12)
    if np.dot(radial, outward_normal_hint) < 0:
        dist = -dist
    return dist


def query_neighborhood(
    pcloud: Pointcloud,
    x: np.ndarray,
    *,
    query_radius: float,
    min_neighbors: int,
    k_fallback: int,
    max_neighbors: int,
) -> tuple[np.ndarray, int]:
    """
    Radius ball around tip; fall back to k-NN if too sparse.
    Keeps at most the closest `max_neighbors` points (local low-pass).

    Returns (indices, nearest_idx).
    """
    if not hasattr(pcloud, "kd_tree"):
        pcloud.get_kd_tree()
    x = np.asarray(x, dtype=float).reshape(3)
    idxs = pcloud.kd_tree.query_ball_point(x, r=float(query_radius))
    idxs = np.asarray(idxs, dtype=int)
    if len(idxs) < int(min_neighbors):
        k = min(int(k_fallback), len(pcloud.vertices))
        _, knn = pcloud.kd_tree.query(x.reshape(1, 3), k=k)
        idxs = np.atleast_1d(knn[0]).astype(int)
    d = np.linalg.norm(pcloud.vertices[idxs] - x.reshape(1, 3), axis=1)
    order = np.argsort(d)
    idxs = idxs[order]
    if len(idxs) > int(max_neighbors):
        idxs = idxs[: int(max_neighbors)]
    nearest = int(idxs[0])
    return idxs, nearest


def orthonormalize_basis_with_normal(t1_hint: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Build R = [t1, t2, n] with n fixed and t1 projected onto the tangent plane."""
    n = n / (np.linalg.norm(n) + 1e-12)
    t1 = np.asarray(t1_hint, dtype=float).reshape(3)
    t1 = t1 - np.dot(t1, n) * n
    if np.linalg.norm(t1) < 1e-9:
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = a - np.dot(a, n) * n
    t1 = t1 / (np.linalg.norm(t1) + 1e-12)
    t2 = np.cross(n, t1)
    t2 = t2 / (np.linalg.norm(t2) + 1e-12)
    return np.stack([t1, t2, n], axis=1)


def query_basis_surface(
    pcloud: Pointcloud,
    x: np.ndarray,
    *,
    query_radius: float = 0.025,
    fit_mode: str = "sphere",
    min_neighbors: int = 12,
    k_fallback: int = 24,
    max_neighbors: int = 48,
) -> tuple[np.ndarray, int, float, dict]:
    """
    Robust surface query: radius-neighborhood plane/sphere fit for n̂ and
    signed distance; quaternion-mean of neighborhood local_bases for tangent
    direction, then re-orthonormalized onto the fitted normal.

    Returns (R, nearest_idx, signed_dist, diagnostics).
    """
    idxs, nearest = query_neighborhood(
        pcloud,
        x,
        query_radius=query_radius,
        min_neighbors=min_neighbors,
        k_fallback=k_fallback,
        max_neighbors=max_neighbors,
    )
    pts = pcloud.vertices[idxs]
    pn = np.asarray(pcloud.normals[idxs], dtype=float).copy()
    for i in range(1, len(pn)):
        if np.dot(pn[i], pn[0]) < 0:
            pn[i] = -pn[i]
    ref_n = pn.mean(axis=0)
    if np.linalg.norm(ref_n) < 1e-12:
        ref_n = pcloud.normals[nearest]
    ref_n = ref_n / (np.linalg.norm(ref_n) + 1e-12)

    used_mode = "plane"
    radius = np.inf
    rms = 0.0
    x = np.asarray(x, dtype=float).reshape(3)

    def _plane_fit():
        centroid, n_fit, plane_rms = fit_local_tangent_plane(pts, ref_n)
        signed = float(np.dot(x - centroid, n_fit))
        return n_fit, signed, plane_rms

    if fit_mode == "sphere":
        center, n_fit, radius, rms = fit_local_tangent_sphere(pts, ref_n)
        sphere_ok = (
            np.isfinite(radius)
            and (0.02 <= radius <= 5.0)
            and (rms < 0.5 * float(query_radius))
        )
        if sphere_ok:
            used_mode = "sphere"
            signed = sphere_signed_distance(x, center, radius, n_fit)
            foot_dir = x - center
            if np.linalg.norm(foot_dir) > 1e-12:
                n_fit = _orient_normal(foot_dir, ref_n)
        else:
            n_fit, signed, rms = _plane_fit()
            radius = np.inf
    else:
        n_fit, signed, rms = _plane_fit()

    quats = R.from_matrix(pcloud.local_bases[idxs]).as_quat()
    R_mean = R.from_quat(_mean_quaternion(quats)).as_matrix()
    basis = orthonormalize_basis_with_normal(R_mean[:, 0], n_fit)

    diag = {
        "n_neighbors": int(len(idxs)),
        "fit_mode": used_mode,
        "fit_rms": float(rms),
        "sphere_radius": float(radius) if np.isfinite(radius) else None,
    }
    return basis, nearest, float(signed), diag


def query_basis_wos(wos: WalkOnSpheresDiffusion, x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Mandatory free-space orientation via Walk-on-Spheres (ref for double-cover)."""
    batch = wos.get_batch_from_point(np.asarray(x, dtype=float))
    basis, _, status = wos.diffuse_rotations(batch)
    if int(np.sum(status == "converged")) == 0 or not np.isfinite(basis).all():
        return ref
    q = R.from_matrix(basis).as_quat()
    q_ref = R.from_matrix(ref).as_quat()
    if np.dot(q, q_ref) < 0:
        q = -q
    return R.from_quat(q).as_matrix()


def query_field_basis(pcloud, wos, x, *, contact_threshold, **kwargs):
    basis, nearest, sd, diag = query_basis_surface(pcloud, x, **kwargs)
    in_contact = abs(sd) <= contact_threshold
    used_wos = False
    if not in_contact:
        basis = query_basis_wos(wos, x, basis)
        used_wos = True
    diag = {**diag, "nearest_idx": nearest, "used_wos": used_wos}
    return basis, float(sd), bool(in_contact), diag
