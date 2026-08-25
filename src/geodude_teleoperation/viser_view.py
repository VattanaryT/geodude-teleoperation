"""Viser framing for right_base stringer replays.

The half-stringer scan is thin along robot +Y (press / flange Z), not +Z.
Default viser +Z-up then makes it look like a vertical wall, and the camera
sits at the origin ~3 m away. Sit the part with its thin PCA axis as up.

Press / wrist-cam look along +Y into the surface, so sitting up is −Y:
the camera looks down onto the stringer the same way the tool does.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

_AXIS_RGB = np.array([[255, 40, 40], [40, 200, 40], [40, 80, 255]], dtype=np.uint8)
AXIS_RGB = _AXIS_RGB


def polyline_segments(xyz: np.ndarray) -> np.ndarray:
    return np.stack([xyz[:-1], xyz[1:]], axis=1)


def wxyz_from_R(R: np.ndarray) -> np.ndarray:
    q_xyzw = Rotation.from_matrix(np.asarray(R, dtype=float).reshape(3, 3)).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def contact_segment_colors(in_contact: np.ndarray, n_seg: int) -> np.ndarray:
    flags = np.asarray(in_contact[:n_seg], dtype=bool)
    colors = np.zeros((n_seg, 2, 3), dtype=np.uint8)
    colors[:, :, :] = (80, 180, 90)
    colors[flags] = (220, 60, 50)
    return colors


def axis_arrows(origin: np.ndarray, R: np.ndarray, length: float) -> np.ndarray:
    o = np.asarray(origin, dtype=float).reshape(3)
    R = np.asarray(R, dtype=float).reshape(3, 3)
    return np.stack([np.stack([o, o + length * R[:, ax]]) for ax in range(3)])


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


def draw_arm_skeletons(
    server,
    arms: dict[str, np.ndarray],
    *,
    prefix: str = "/arm",
    handles: dict | None = None,
) -> dict:
    """Draw left/right UR5e FK chains. Reuse ``handles`` so the sticks move."""
    if handles is None:
        handles = {}
    colors = {"right": (80, 180, 255), "left": (170, 170, 175)}
    for side, xyz in arms.items():
        pts = np.asarray(xyz, dtype=float).reshape(-1, 3)
        if len(pts) < 2:
            continue
        rgb = colors.get(side, (160, 160, 160))
        segs = np.asarray(polyline_segments(pts), dtype=np.float32)
        key = f"{prefix}/{side}"
        if key in handles:
            handles[key].points = segs
            for i, p in enumerate(pts):
                handles[f"{key}/j{i}"].position = tuple(float(x) for x in p)
            continue
        handles[key] = server.scene.add_line_segments(
            key, points=segs, colors=rgb, thickness=0.028,
        )
        for i, p in enumerate(pts):
            handles[f"{key}/j{i}"] = server.scene.add_icosphere(
                f"{key}/j{i}", radius=0.022, color=rgb, position=p)
    return handles
