"""Local Polyscope helpers (no diffused_fields.visualization dependency)."""
from __future__ import annotations

from typing import Any

import numpy as np


def set_camera_and_plane(
    *, up: str = "z_up", front: str = "neg_x_front"
) -> None:
    """Upright stringer/vention view: layup is roughly XY-thin in Z."""
    import polyscope as ps

    ps.set_up_dir(up)
    ps.set_front_dir(front)
    ps.set_ground_plane_mode("shadow_only")


def frame_camera_on_points(
    points: Any, *, distance_scale: float = 1.8
) -> None:
    """Aim the camera at the bbox center of ``points`` (ignore distant refs)."""
    import polyscope as ps

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return
    center = pts.mean(axis=0)
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    dist = max(extent * float(distance_scale), 0.25)
    # Match neg_x_front: sit on -X looking toward +X, slight elevation.
    cam = center + np.array([-dist, 0.15 * dist, 0.35 * dist], dtype=float)
    ps.look_at(cam.tolist(), center.tolist())


def plot_orientation_field(
    vertices: Any,
    bases: Any = None,
    name: str = "",
    vector_length: float | None = None,
    vector_radius: float | None = None,
    point_radius: float = 0.01,
    enable: bool = True,
    enable_vector: bool = False,
    enable_x: bool = True,
    enable_z: bool = False,
    color: Any = None,
) -> Any:
    """Register a point cloud (+ optional frame axes) in Polyscope."""
    import polyscope as ps

    verts = np.asarray(vertices, dtype=float)
    if (len(verts.shape) != 2) or (verts.shape[1] not in (2, 3)):
        verts = np.asarray([vertices], dtype=float)

    kwargs: dict[str, Any] = {
        "radius": point_radius,
        "transparency": 1,
        "color": [0.0, 0.0, 0.0] if color is None else color,
        "enabled": enable,
    }
    ps_field = ps.register_point_cloud(name, verts, **kwargs)
    ps.set_ground_plane_mode("none")
    if bases is None:
        return ps_field
    bases_arr = np.asarray(bases, dtype=float)
    ps_field.add_vector_quantity(
        f"x_{name}",
        bases_arr[:, :, 0],
        color=[1, 0, 0],
        length=vector_length,
        radius=vector_radius,
        enabled=enable_x,
    )
    ps_field.add_vector_quantity(
        f"y_{name}",
        bases_arr[:, :, 1],
        color=[0, 1, 0],
        length=vector_length,
        radius=vector_radius,
        enabled=enable_vector,
    )
    ps_field.add_vector_quantity(
        f"z_{name}",
        bases_arr[:, :, 2],
        color=[0, 0, 1],
        length=vector_length,
        radius=vector_radius,
        enabled=enable_z,
    )
    return ps_field
