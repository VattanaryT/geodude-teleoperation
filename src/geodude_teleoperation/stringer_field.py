"""Stringer length-field BCs: PCA end-caps, scalar heat, ∇u as along-part tangent.

Copied from the previous diffused_fields stringer path (end_lines + flow=length).
Ambient queries stay on RotorFieldDiffusion; this module only builds surface frames.
"""
from __future__ import annotations

import numpy as np
import robust_laplacian
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


def end_cap_source_sink(vertices, fraction=0.06, swap=False, flow="length"):
    """PCA end-cap (length) or long-side (width) source/sink bands.

    Axis 0 ≈ length, 1 ≈ width, 2 ≈ thickness.
    ``flow='length'`` (default) bands the short end-caps so ∇u follows the long axis.
    ``flow='width'`` bands the long sides.
    """
    if not (0.0 < float(fraction) < 0.5):
        raise ValueError(f"end-line-fraction must be in (0, 0.5), got {fraction}")
    if flow not in ("length", "width"):
        raise ValueError(f"flow must be 'length' or 'width', got {flow!r}")
    verts = np.asarray(vertices, dtype=float).reshape(-1, 3)
    centered = verts - verts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0 if flow == "length" else 1]
    proj = centered @ axis
    lo = np.percentile(proj, float(fraction) * 100.0)
    hi = np.percentile(proj, 100.0 - float(fraction) * 100.0)
    sink = np.where(proj <= lo)[0]
    source = np.where(proj >= hi)[0]
    if len(sink) == 0 or len(source) == 0:
        raise RuntimeError(
            f"end-cap bands empty (sink={len(sink)}, source={len(source)}); "
            "try a larger end_line_fraction.")
    if swap:
        sink, source = source, sink
    return sink, source, axis


def bases_from_tangent_and_normal(tangent, normal, eps=1e-9):
    """Orthonormal frames: columns [x=tangent, y=z×x, z=normal]."""
    t = np.asarray(tangent, dtype=float).reshape(-1, 3)
    n = np.asarray(normal, dtype=float).reshape(-1, 3)
    t = t / np.clip(np.linalg.norm(t, axis=1, keepdims=True), eps, None)
    n = n - np.sum(n * t, axis=1, keepdims=True) * t
    n = n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), eps, None)
    y = np.cross(n, t)
    y = y / np.clip(np.linalg.norm(y, axis=1, keepdims=True), eps, None)
    return np.stack([t, y, n], axis=2)


def _in_tangent_plane(vec, normal, eps=1e-9):
    n = np.asarray(normal, dtype=float).reshape(-1, 3)
    n = n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), eps, None)
    t = np.asarray(vec, dtype=float).reshape(-1, 3)
    t = t - np.sum(t * n, axis=1, keepdims=True) * n
    mag = np.linalg.norm(t, axis=1, keepdims=True)
    ok = mag[:, 0] >= eps
    out = np.zeros_like(t)
    out[ok] = t[ok] / mag[ok]
    for fb in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        need = ~ok
        if not np.any(need):
            break
        t2 = np.array(fb, dtype=float) - np.sum(n[need] * fb, axis=1, keepdims=True) * n[need]
        m2 = np.linalg.norm(t2, axis=1, keepdims=True)
        good = m2[:, 0] >= eps
        idx = np.where(need)[0][good]
        out[idx] = t2[good] / m2[good]
        ok[idx] = True
    return out


def _face_gradients(vertices, faces, u):
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    u0 = u[faces[:, 0]]
    u1 = u[faces[:, 1]]
    u2 = u[faces[:, 2]]
    normal = np.cross(p1 - p0, p2 - p0)
    rhs = (
        u0[:, None] * (p2 - p1)
        + u1[:, None] * (p0 - p2)
        + u2[:, None] * (p1 - p0))
    denom = np.einsum("ij,ij->i", normal, normal)[:, None]
    grad = np.cross(normal, rhs) / np.maximum(denom, 1e-30)
    area = 0.5 * np.linalg.norm(normal, axis=1)
    return grad, area


def length_tangents(mesh, fraction=0.06, swap=False, flow="length",
                    diffusion_scalar=5000.0):
    """Heat from the source band; ∇u is the in-plane tangent.

    Default ``flow='length'`` puts BCs on the short end-caps so +X follows
    the long axis of the part.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    sink, source, axis = end_cap_source_sink(
        vertices, fraction=fraction, swap=swap, flow=flow)

    lap, mass = robust_laplacian.mesh_laplacian(
        np.ascontiguousarray(vertices), np.ascontiguousarray(faces))
    mean_edge = float(mesh.edges_unique_length.mean())
    dt = float(diffusion_scalar) * mean_edge ** 2
    u0 = np.zeros(len(vertices), dtype=np.float64)
    u0[source] = 1.0
    heat = spsolve(csc_matrix(mass + dt * lap), mass @ u0)

    face_grad, area = _face_gradients(vertices, faces, np.asarray(heat))
    grad = np.zeros_like(vertices)
    weight = np.zeros(len(vertices))
    for k in range(3):
        np.add.at(grad, faces[:, k], face_grad * area[:, None])
        np.add.at(weight, faces[:, k], area)
    grad = grad / np.maximum(weight[:, None], 1e-30)

    n = normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9, None)
    mag = np.linalg.norm(grad, axis=1, keepdims=True)
    candidate = np.where(mag < 1e-9, np.broadcast_to(axis, grad.shape), grad)
    tangent = _in_tangent_plane(candidate, n)
    return tangent, sink, source, axis
