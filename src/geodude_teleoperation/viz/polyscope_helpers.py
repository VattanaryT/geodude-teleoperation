"""Local Polyscope helpers (no diffused_fields.visualization dependency)."""
from __future__ import annotations

from typing import Any

import numpy as np


def set_camera_and_plane(*, up: str = "y_up") -> None:
    """Match historical diffused_fields defaults; callers may override up-dir after."""
    import polyscope as ps

    ps.set_up_dir(up)
    ps.set_front_dir("neg_x_front")
    ps.set_ground_plane_mode("shadow_only")


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
