"""Viser framing for right_base stringer replays.

The half-stringer scan is thin along robot +Y (press / flange Z), not +Z.
Default viser +Z-up then makes it look like a vertical wall, and the camera
sits at the origin ~3 m away. Sit the part with its thin PCA axis as up.

Press / wrist-cam look along +Y into the surface, so sitting up is −Y:
the camera looks down onto the stringer the same way the tool does.
"""
from __future__ import annotations

import numpy as np


def sitting_up_axis(points: np.ndarray) -> np.ndarray:
    """Unit axis of least variance. Sign: −Y (anti-press), else +Z."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    centered = pts - pts.mean(axis=0)
    _, eigvecs = np.linalg.eigh(np.cov(centered.T))
    up = np.asarray(eigvecs[:, 0], dtype=float)
    up = up / (np.linalg.norm(up) + 1e-30)
    if abs(up[1]) >= abs(up[2]) and abs(up[1]) >= abs(up[0]):
        if up[1] > 0.0:
            up = -up
    elif up[2] < 0.0:
        up = -up
    return up


def framed_camera(
    points: np.ndarray, *, distance_scale: float = 1.35
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (look_at, up, eye) for a close 3/4 view of ``points``."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    center = pts.mean(axis=0)
    up = sitting_up_axis(pts)
    span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    dist = max(0.28, float(distance_scale) * span)
    _, eigvecs = np.linalg.eigh(np.cov((pts - center).T))
    long_ax = np.asarray(eigvecs[:, 2], dtype=float)
    horiz = long_ax - up * float(long_ax @ up)
    if np.linalg.norm(horiz) < 1e-8:
        horiz = np.array([1.0, 0.0, 0.0]) - up * float(up[0])
    horiz = horiz / (np.linalg.norm(horiz) + 1e-30)
    # PCA sign is arbitrary; a minus-X horiz yaws the stringer 180° vs right_base.
    if horiz[0] < 0.0:
        horiz = -horiz
    view = np.cross(up, horiz)
    view = view / (np.linalg.norm(view) + 1e-30)
    eye = center + dist * (0.50 * view + 0.72 * up + 0.22 * horiz)
    return center, up, eye


def apply_stringer_view(server, points: np.ndarray) -> None:
    """Z-up scene → sit stringer horizontally; frame each new client on it."""
    center, up, eye = framed_camera(points)
    server.scene.set_up_direction(up)

    @server.on_client_connect
    def _frame(client) -> None:
        client.camera.up_direction = up
        client.camera.position = eye
        client.camera.look_at = center
