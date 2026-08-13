"""Stringer layup pointcloud load, field BC, and startup validation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from diffused_fields.diffusion import PointcloudScalarDiffusion, WalkOnSpheresDiffusion
from diffused_fields.manifold import Pointcloud

from geodude_teleoperation.paths import ssot_dof_config_path

_DEFAULTS = {"voxel_size": 0.004, "normal_orientation": 1}


def config_defaults(object_name: str = "stringer_layup") -> dict:
    """Load voxel_size / normal_orientation from ``config/SSoT_dof.yaml``."""
    del object_name  # kept for call-site compatibility
    defaults = dict(_DEFAULTS)
    try:
        from geodude_teleoperation.ssot_dof import load_ssot_dof

        cfg_path = ssot_dof_config_path()
        if not cfg_path.exists():
            return defaults
        cfg = load_ssot_dof(cfg_path)
        entry = cfg.get("pointcloud", {}) or {}
        defaults["voxel_size"] = entry.get("voxel_size", defaults["voxel_size"])
        defaults["normal_orientation"] = entry.get(
            "normal_orientation", defaults["normal_orientation"]
        )
    except Exception:
        pass
    return defaults


def load_stringer_pointcloud(
    ply: Path, voxel_size: float, normal_orientation: int
) -> Pointcloud:
    """Load a PLY, voxel-downsample, and wrap as a Pointcloud."""
    ply = ply.resolve()
    if not ply.exists():
        raise FileNotFoundError(ply)
    raw = o3d.io.read_point_cloud(str(ply))
    if len(raw.points) == 0:
        raise RuntimeError(f"Empty point cloud: {ply}")
    return pointcloud_from_points(
        np.asarray(raw.points),
        voxel_size=voxel_size,
        normal_orientation=normal_orientation,
    )


def pointcloud_from_points(
    points: np.ndarray,
    voxel_size: float,
    normal_orientation: int,
    *,
    object_name: str = "stringer_layup",
) -> Pointcloud:
    """Wrap Nx3 points as a Pointcloud (Pointcloud applies ``voxel_size``)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        raise RuntimeError("Empty point cloud")
    pcloud = Pointcloud(
        vertices=pts,
        voxel_size=voxel_size,
        normal_orientation=normal_orientation,
    )
    pcloud.object_name = object_name
    return pcloud


def side_source_sink_lines(
    pcloud: Pointcloud,
    fraction: float = 0.06,
    swap: bool = False,
    side: str = "sides",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build source/sink index sets as thin bands on a pair of opposite edges.

    PCA axes (descending variance):
      0 ≈ length, 1 ≈ width, 2 ≈ thickness.
    side='sides' → bands on the long edges (axis 1).
    side='ends'  → bands on the short end-caps (axis 0).

    Returns:
        sink_idx, source_idx, selection_axis (unit vector in world coords)
    """
    if not (0.0 < fraction < 0.5):
        raise ValueError(f"end-line-fraction must be in (0, 0.5), got {fraction}")
    if side not in ("sides", "ends"):
        raise ValueError(f"side must be 'sides' or 'ends', got {side!r}")

    verts = np.asarray(pcloud.vertices, dtype=float)
    center = verts.mean(axis=0)
    centered = verts - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis_idx = 1 if side == "sides" else 0
    axis = vt[axis_idx]
    proj = centered @ axis
    lo = np.percentile(proj, fraction * 100.0)
    hi = np.percentile(proj, 100.0 - fraction * 100.0)
    band_a = np.where(proj <= lo)[0]
    band_b = np.where(proj >= hi)[0]
    if len(band_a) == 0 or len(band_b) == 0:
        raise RuntimeError(
            f"Side lines empty (a={len(band_a)}, b={len(band_b)}); "
            "try a larger end_line_fraction."
        )
    sink_idx, source_idx = band_a, band_b
    if swap:
        sink_idx, source_idx = source_idx, sink_idx
    return sink_idx, source_idx, axis


def _field_flow_to_side(field_flow: str) -> str:
    """Map field_flow length→ends, width→sides."""
    if field_flow == "length":
        return "ends"
    if field_flow == "width":
        return "sides"
    raise ValueError(f"field_flow must be 'length' or 'width', got {field_flow!r}")


def compute_orientation_field(
    pcloud: Pointcloud,
    *,
    field_mode: str,
    field_flow: str = "length",
    end_line_fraction: float = 0.06,
    swap_end_lines: bool = False,
    diffusion_scalar: float = 5000.0,
) -> PointcloudScalarDiffusion:
    """Build scalar diffusion BC and compute local orientation bases.

    Defaults match ``diffused_fields_robotics/scripts/stringer_layup_ee_teleop.py``
    (``diffusion_scalar=5000``, end_lines + length→ends bands).

    ``diffusion_scalar`` is dimensionless diffusion time τ in units of ``h²``
    (squared mean edge length): ``dt = diffusion_scalar * mean_edge_length**2``.
    That keeps field smoothness independent of point-cloud density.

    Startup builds the Laplacian / heat operator and surface bases once; WoS
    queries remain in the real-time control loop via ``build_wos``.
    """
    sd = PointcloudScalarDiffusion(pcloud, diffusion_scalar=diffusion_scalar)
    if field_mode == "boundary":
        sd.source_vertices = []
        sd.sink_vertices = None
        sd.set_sources()
    else:
        side = _field_flow_to_side(field_flow)
        sink_idx, source_idx, _ = side_source_sink_lines(
            pcloud,
            fraction=end_line_fraction,
            swap=swap_end_lines,
            side=side,
        )
        sd.sink_vertices = sink_idx.tolist()
        sd.source_vertices = source_idx.tolist()
        sd.set_sources()
    sd.get_local_bases()
    return sd


def validate_field_prints(
    pcloud: Pointcloud, sd: PointcloudScalarDiffusion, wos_threshold: float
) -> str:
    """Multi-line startup report: verts, BC counts, basis sanity, WoS threshold."""
    n_verts = len(pcloud.vertices)
    sink = getattr(sd, "sink_vertices", None)
    source = getattr(sd, "source_vertices", None) or []
    n_sink = len(sink) if sink is not None else int(getattr(pcloud, "is_boundary_arr", np.array([])).sum())
    n_source = len(source)

    R0 = pcloud.local_bases[0]
    ortho_err = float(np.max(np.abs(R0.T @ R0 - np.eye(3))))
    det = float(np.linalg.det(R0))
    mean_edge = float(pcloud.get_mean_edge_length())

    lines = [
        f"#verts={n_verts}",
        f"#source={n_source}  #sink={n_sink}",
        f"sample basis: ortho_err={ortho_err:.2e}  det={det:.6f}",
        f"mean edge length={mean_edge:.6f} m",
        f"WoS convergence_threshold={wos_threshold:.6f} m",
    ]
    return "\n".join(lines)


def build_wos(
    pcloud: Pointcloud,
    batch_size: int,
    *,
    convergence_threshold_edge_mult: float = 2.0,
) -> WalkOnSpheresDiffusion:
    """Walk-on-Spheres solver for off-surface orientation queries.

    ``convergence_threshold`` is ``mean_edge_length * convergence_threshold_edge_mult``
    (see ``wos.convergence_threshold_edge_mult`` in ``SSoT_dof.yaml``).
    """
    return WalkOnSpheresDiffusion(
        boundaries=[pcloud],
        batch_size=batch_size,
        convergence_threshold=pcloud.get_mean_edge_length()
        * float(convergence_threshold_edge_mult),
    )
