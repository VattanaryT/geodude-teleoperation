"""Polyscope debug view for stringer field + tip/TCP alignment.

Invoked via ``ur_stringer_teleop --viz`` (not as a standalone module).
See the worktree ``README.md`` for operator commands. Local data defaults live
under ``data/processed/``; Polyscope helpers are in ``viz.polyscope_helpers``
(no sibling ``diffused_fields`` checkout required for viz).
"""
from __future__ import annotations

import sys
import time
from typing import Any, Sequence

import numpy as np

from geodude_teleoperation.frames import transform_points
from geodude_teleoperation.tag_crosscheck import load_T_cam_EE
from geodude_teleoperation.viz.live_state import LiveVizState, latency_table_rows


def draw_latency_imgui(imgui, live_viz: LiveVizState) -> None:
    """Second Polyscope ImGui window: loop / WoS / relock / SpaceMouse / viz age."""
    if imgui is None or not getattr(live_viz, "show_latency_table", False):
        return
    begin = getattr(imgui, "Begin", None)
    end = getattr(imgui, "End", None)
    text = getattr(imgui, "TextUnformatted", None)
    if begin is None or end is None or text is None:
        return
    opened = bool(begin("Latency", True))
    try:
        if not opened:
            return
        snap = live_viz.latency_snapshot()
        if not snap:
            text("waiting for RTDE ticks…")
            return
        rows = latency_table_rows(snap, now_mono=time.monotonic())
        begin_table = getattr(imgui, "BeginTable", None)
        end_table = getattr(imgui, "EndTable", None)
        if begin_table is not None and end_table is not None and begin_table(
            "latency_rows", 2
        ):
            next_row = getattr(imgui, "TableNextRow", lambda: None)
            next_col = getattr(imgui, "TableNextColumn", lambda: None)
            for name, val, warn in rows:
                next_row()
                next_col()
                text(str(name))
                next_col()
                text(f"{val}{'  !' if warn else ''}")
            end_table()
        else:
            for name, val, warn in rows:
                text(f"{name}: {val}{'  !' if warn else ''}")
    finally:
        end()

# Lazily used; tests monkeypatch these names on this module.
ps = None  # type: ignore
plot_orientation_field = None  # type: ignore
set_camera_and_plane = None  # type: ignore
frame_camera_on_points = None  # type: ignore


def _ensure_polyscope() -> None:
    global ps, plot_orientation_field, set_camera_and_plane, frame_camera_on_points
    if ps is not None and plot_orientation_field is not None:
        return
    try:
        import polyscope as _ps
        from geodude_teleoperation.viz.polyscope_helpers import (
            frame_camera_on_points as _frame_camera_on_points,
            plot_orientation_field as _plot_orientation_field,
            set_camera_and_plane as _set_camera_and_plane,
        )
    except ImportError as exc:
        raise ImportError(
            "Polyscope required for --viz; install polyscope in this environment"
        ) from exc
    ps = _ps
    plot_orientation_field = _plot_orientation_field
    set_camera_and_plane = _set_camera_and_plane
    frame_camera_on_points = _frame_camera_on_points


def points_in_viz_frame(
    vertices: np.ndarray,
    *,
    T_tag0_vention: np.ndarray,
    T_rtde_base_vention: np.ndarray | None,
    viz_frame: str,
) -> np.ndarray:
    """Map tag_0 vertices into tag0 / vention / rtde viz frame."""
    V = np.asarray(vertices, dtype=float).reshape(-1, 3)
    if viz_frame == "tag0":
        return V.copy()
    if viz_frame == "vention":
        return transform_points(T_tag0_vention, V)
    if viz_frame == "rtde":
        if T_rtde_base_vention is None:
            raise ValueError("viz_frame=rtde requires T_rtde_base_vention")
        T_tag0_rtde = np.linalg.inv(T_rtde_base_vention) @ np.asarray(
            T_tag0_vention, dtype=float
        ).reshape(4, 4)
        return transform_points(T_tag0_rtde, V)
    if viz_frame == "base":
        # Back-compat alias: historically meant crop_meta "base" (= vention).
        return transform_points(T_tag0_vention, V)
    raise ValueError(
        f"viz_frame must be tag0|vention|rtde|base, got {viz_frame!r}"
    )


def rotate_bases_to_viz_frame(
    bases: np.ndarray,
    *,
    T_tag0_vention: np.ndarray,
    T_rtde_base_vention: np.ndarray | None,
    viz_frame: str,
) -> np.ndarray:
    bases = np.asarray(bases, dtype=float)
    if viz_frame in ("tag0",):
        return bases.copy()
    if viz_frame in ("vention", "base"):
        R = np.asarray(T_tag0_vention, dtype=float).reshape(4, 4)[:3, :3]
        return np.einsum("ij,njk->nik", R, bases)
    if viz_frame == "rtde":
        assert T_rtde_base_vention is not None
        T_tag0_rtde = np.linalg.inv(T_rtde_base_vention) @ np.asarray(
            T_tag0_vention, dtype=float
        ).reshape(4, 4)
        R = T_tag0_rtde[:3, :3]
        return np.einsum("ij,njk->nik", R, bases)
    raise ValueError(viz_frame)


def bc_clouds_tag0(pcloud, sd) -> dict[str, np.ndarray]:
    """Source/sink (or boundary) point sets in tag_0."""
    V = np.asarray(pcloud.vertices, dtype=float).reshape(-1, 3)
    out: dict[str, np.ndarray] = {}
    sink_idx = getattr(sd, "sink_vertices", None)
    source_idx = getattr(sd, "source_vertices", None) or []
    if sink_idx is not None and len(sink_idx) > 0:
        out["sink_line"] = V[np.asarray(sink_idx, dtype=int)]
    if len(source_idx) > 0:
        out["source_line"] = V[np.asarray(source_idx, dtype=int)]
    if not out and hasattr(pcloud, "is_boundary_arr"):
        bidx = np.where(np.asarray(pcloud.is_boundary_arr))[0]
        if len(bidx):
            show = bidx[:: max(1, len(bidx) // 400)]
            out["boundary_sinks"] = V[show]
    return out


def _T_ee_rtde(tcp_pose_rtde: Sequence[float]) -> np.ndarray:
    from geodude_teleoperation.tag_crosscheck import tcp_pose_to_T_EE_base

    return tcp_pose_to_T_EE_base(tcp_pose_rtde)


def se3_rtde_to_viz(
    T_rtde: np.ndarray,
    *,
    T_tag0_vention: np.ndarray,
    T_rtde_base_vention: np.ndarray | None,
    viz_frame: str,
) -> np.ndarray:
    """Map an SE3 expressed in RTDE base into the viz frame."""
    T_rtde = np.asarray(T_rtde, dtype=float).reshape(4, 4)
    if viz_frame == "rtde":
        return T_rtde.copy()
    if T_rtde_base_vention is None:
        T_v = T_rtde
    else:
        T_v = np.asarray(T_rtde_base_vention, dtype=float).reshape(4, 4) @ T_rtde
    if viz_frame in ("vention", "base"):
        return T_v
    if viz_frame == "tag0":
        return np.linalg.inv(np.asarray(T_tag0_vention, dtype=float).reshape(4, 4)) @ T_v
    raise ValueError(viz_frame)


def tip_tcp_markers(
    tcp_pose_rtde: Sequence[float] | None,
    tip_offset: np.ndarray,
    *,
    T_tag0_vention: np.ndarray,
    T_rtde_base_vention: np.ndarray | None,
    viz_frame: str,
) -> dict[str, np.ndarray]:
    """TCP / tip from RTDE pose, expressed in the viz frame.

    Also returns ``T_tcp`` (4x4) in the viz frame when TCP is available.
    """
    if tcp_pose_rtde is None:
        return {}
    pose = np.asarray(tcp_pose_rtde, dtype=float).reshape(6)
    T_tcp_rtde = _T_ee_rtde(pose)
    T_tcp_viz = se3_rtde_to_viz(
        T_tcp_rtde,
        T_tag0_vention=T_tag0_vention,
        T_rtde_base_vention=T_rtde_base_vention,
        viz_frame=viz_frame,
    )
    tip_off = np.asarray(tip_offset, dtype=float).reshape(3)
    tip_viz = (T_tcp_viz @ np.array([tip_off[0], tip_off[1], tip_off[2], 1.0]))[:3]
    return {
        "tcp": T_tcp_viz[:3, 3].copy(),
        "tip": np.asarray(tip_viz, dtype=float),
        "T_tcp": T_tcp_viz,
    }


def camera_pose_in_viz(
    tcp_pose_rtde: Sequence[float] | None,
    T_cam_EE: np.ndarray | None,
    *,
    T_tag0_vention: np.ndarray,
    T_rtde_base_vention: np.ndarray | None,
    viz_frame: str,
) -> np.ndarray | None:
    """``T_cam`` in viz frame: ``T_EE_rtde @ T_cam_EE``, then mapped to viz."""
    if tcp_pose_rtde is None or T_cam_EE is None:
        return None
    T_ee = _T_ee_rtde(tcp_pose_rtde)
    T_cam_rtde = T_ee @ np.asarray(T_cam_EE, dtype=float).reshape(4, 4)
    return se3_rtde_to_viz(
        T_cam_rtde,
        T_tag0_vention=T_tag0_vention,
        T_rtde_base_vention=T_rtde_base_vention,
        viz_frame=viz_frame,
    )


def _register_axes(
    name: str, T: np.ndarray, *, axis_len: float = 0.08, enabled: bool = True
) -> None:
    """Draw RGB axes at SE3 pose ``T`` (columns of R)."""
    assert ps is not None
    T = np.asarray(T, dtype=float).reshape(4, 4)
    o = T[:3, 3]
    for i, (c, label) in enumerate(
        zip([(1.0, 0.15, 0.15), (0.15, 0.85, 0.15), (0.2, 0.4, 1.0)], ("x", "y", "z"))
    ):
        seg = np.vstack([o, o + axis_len * T[:3, i]])
        ps.register_curve_network(
            f"{name}_{label}",
            seg,
            np.array([[0, 1]], dtype=int),
            radius=0.0022,
            color=c,
            enabled=enabled,
        )
    ps.register_point_cloud(
        name,
        o.reshape(1, 3),
        radius=0.006,
        color=(0.95, 0.95, 0.95),
        enabled=enabled,
    )


def contact_frame_overlays(
    tip: np.ndarray,
    R_field_viz: np.ndarray,
    *,
    tangent_len: float = 0.10,
    approach_len: float = 0.06,
) -> dict[str, np.ndarray]:
    """Segments for in-contact preview: tangent (±field-x) and approach (field-z).

    Returns dict of Nx3 polylines (2 points each) in the viz frame.
    """
    tip = np.asarray(tip, dtype=float).reshape(3)
    R = np.asarray(R_field_viz, dtype=float).reshape(3, 3)
    tx = R[:, 0]
    tz = R[:, 2]
    return {
        "tangent_line": np.vstack(
            [tip - float(tangent_len) * tx, tip + float(tangent_len) * tx]
        ),
        "approach_n": np.vstack([tip, tip + float(approach_len) * tz]),
        "field_y": np.vstack(
            [tip, tip + 0.5 * float(tangent_len) * R[:, 1]]
        ),
    }


def _register_contact_overlays(
    tip: np.ndarray,
    R_field_viz: np.ndarray,
    *,
    tangent_len: float = 0.10,
    approach_len: float = 0.06,
) -> None:
    """Draw expected tangent line + approach normal at the tip for --in-contact viz."""
    assert ps is not None
    segs = contact_frame_overlays(
        tip, R_field_viz, tangent_len=tangent_len, approach_len=approach_len
    )
    ps.register_curve_network(
        "contact_tangent",
        segs["tangent_line"],
        np.array([[0, 1]], dtype=int),
        radius=0.0030,
        color=(1.0, 0.35, 0.85),
    )
    ps.register_curve_network(
        "contact_approach",
        segs["approach_n"],
        np.array([[0, 1]], dtype=int),
        radius=0.0026,
        color=(0.2, 0.85, 1.0),
    )
    ps.register_curve_network(
        "contact_field_y",
        segs["field_y"],
        np.array([[0, 1]], dtype=int),
        radius=0.0020,
        color=(0.3, 0.9, 0.35),
    )
    T = np.eye(4)
    T[:3, :3] = np.asarray(R_field_viz, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(tip, dtype=float).reshape(3)
    _register_axes("contact_R_field", T, axis_len=0.05)


def _register_camera_frustum(
    name: str, T_cam: np.ndarray, *, scale: float = 0.05
) -> Any:
    """Simple pinhole frustum: optical +Z forward (OpenCV), pyramid in viz frame."""
    assert ps is not None
    T = np.asarray(T_cam, dtype=float).reshape(4, 4)
    # Corners in camera frame (z forward, x right, y down)
    z, w, h = scale, scale * 0.6, scale * 0.45
    corners_c = np.array(
        [
            [0.0, 0.0, 0.0],
            [w, h, z],
            [-w, h, z],
            [-w, -h, z],
            [w, -h, z],
        ]
    )
    corners = transform_points(T, corners_c)
    edges = np.array(
        [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]], dtype=int
    )
    net = ps.register_curve_network(
        name, corners, edges, radius=0.0018, color=(1.0, 0.55, 0.1)
    )
    origin = ps.register_point_cloud(
        f"{name}_origin", corners[0:1], radius=0.007, color=(1.0, 0.45, 0.05)
    )
    return {"net": net, "origin": origin, "corners_c": corners_c, "scale": float(scale)}


def _update_camera_frustum(handles: dict, name: str, T_cam: np.ndarray) -> None:
    assert ps is not None
    T = np.asarray(T_cam, dtype=float).reshape(4, 4)
    if name not in handles:
        handles[name] = _register_camera_frustum(name, T, scale=0.06)
        return
    pack = handles[name]
    corners = transform_points(T, pack["corners_c"])
    pack["net"].update_node_positions(corners)
    pack["origin"].update_point_positions(corners[0:1])


# Keyboard linear keys → (axis index 0..2, sign) matching keyboard._KEY_TWIST.
_TELEOP_LINEAR_KEYS = (
    ("w", 0, 1.0),
    ("s", 0, -1.0),
    ("d", 1, 1.0),
    ("a", 1, -1.0),
    ("x", 2, 1.0),
    ("y", 2, -1.0),
)


def R_tcp_base_to_viz(
    R_tcp_base: np.ndarray,
    *,
    T_tag0_vention: np.ndarray,
    T_rtde_base_vention: np.ndarray | None,
    viz_frame: str,
) -> np.ndarray:
    """Map a rotation expressed in RTDE/TCP base into the viz frame."""
    R = np.asarray(R_tcp_base, dtype=float).reshape(3, 3)
    if viz_frame == "rtde":
        return R.copy()
    if T_rtde_base_vention is None:
        R_v = R
    else:
        R_v = np.asarray(T_rtde_base_vention, dtype=float).reshape(4, 4)[:3, :3] @ R
    if viz_frame in ("vention", "base"):
        return R_v
    if viz_frame == "tag0":
        R_tv = np.asarray(T_tag0_vention, dtype=float).reshape(4, 4)[:3, :3]
        return R_tv.T @ R_v
    raise ValueError(viz_frame)


def teleop_key_ghosts(
    tip: np.ndarray,
    R_field_viz: np.ndarray,
    *,
    keyboard_step: float,
    linear_scale: float,
) -> dict[str, np.ndarray]:
    """Ghost tip positions after one scaled linear keypress in field frame."""
    tip = np.asarray(tip, dtype=float).reshape(3)
    R = np.asarray(R_field_viz, dtype=float).reshape(3, 3)
    step = float(linear_scale) * float(keyboard_step)
    out: dict[str, np.ndarray] = {}
    for key, axis, sign in _TELEOP_LINEAR_KEYS:
        out[key] = tip + sign * step * R[:, axis]
    return out


def query_R_field_at_tcp(
    pcloud,
    wos,
    tcp_pose_rtde: Sequence[float],
    tip_offset: np.ndarray,
    *,
    T_tag0_tcp_base: np.ndarray,
    contact_threshold: float,
    query_kwargs: dict | None = None,
) -> tuple[np.ndarray, float, bool, bool]:
    """Return ``(R_field_tcp_base, abs_d, in_contact, used_wos)`` via live VF path."""
    from geodude_teleoperation.teleop.contact_wos_vf import ContactWosOrientationField
    from geodude_teleoperation.rtde.control_loop import ee_motor_from_tcp_pose

    vf = ContactWosOrientationField(
        pcloud,
        wos,
        tip_offset=tip_offset,
        contact_threshold=contact_threshold,
        T_tag0_base=T_tag0_tcp_base,
        **(query_kwargs or {}),
    )
    ee = ee_motor_from_tcp_pose(tcp_pose_rtde)
    vf.update_from_state(ee)
    st = vf.status
    return (
        np.asarray(st["R_field_base"], dtype=float).reshape(3, 3),
        abs(float(st["signed_d"])),
        bool(st["in_contact"]),
        bool(st["used_wos"]),
    )


def point_viz_to_tag0(
    point: np.ndarray,
    *,
    T_tag0_vention: np.ndarray,
    T_rtde_base_vention: np.ndarray | None,
    viz_frame: str,
) -> np.ndarray:
    """Inverse of ``points_in_viz_frame`` for a single point."""
    p = np.asarray(point, dtype=float).reshape(1, 3)
    if viz_frame == "tag0":
        return p[0].copy()
    if viz_frame in ("vention", "base"):
        T_inv = np.linalg.inv(np.asarray(T_tag0_vention, dtype=float).reshape(4, 4))
        return transform_points(T_inv, p)[0]
    if viz_frame == "rtde":
        if T_rtde_base_vention is None:
            raise ValueError("viz_frame=rtde requires T_rtde_base_vention")
        T_tag0_rtde = np.linalg.inv(
            np.asarray(T_rtde_base_vention, dtype=float).reshape(4, 4)
        ) @ np.asarray(T_tag0_vention, dtype=float).reshape(4, 4)
        return transform_points(np.linalg.inv(T_tag0_rtde), p)[0]
    raise ValueError(viz_frame)


def query_R_field_at_tip_tag0(
    pcloud,
    wos,
    tip_tag0: np.ndarray,
    *,
    T_tag0_tcp_base: np.ndarray,
    contact_threshold: float,
    query_kwargs: dict | None = None,
    prev_R_tag0: np.ndarray | None = None,
    prev_R_field: np.ndarray | None = None,
    r_field_smoothing: float = 0.75,
) -> tuple[np.ndarray, float, bool, bool, np.ndarray]:
    """Live VF orientation at a tip in tag_0 (with EMA ``R_field`` smoothing).

    Returns ``(R_field_tcp_base, abs_d, in_contact, used_wos, R_tag0)``.
    """
    from geodude_teleoperation.teleop.contact_wos_vf import (
        _continuity_flip,
        _orient_approach_into_surface,
        smooth_R_field,
    )
    from geodude_teleoperation.field.field_query import query_field_basis
    from geodude_teleoperation.frames import tag0_rot_to_base

    R_tag0, signed_d, in_contact, diag = query_field_basis(
        pcloud,
        wos,
        np.asarray(tip_tag0, dtype=float).reshape(3),
        contact_threshold=contact_threshold,
        **(query_kwargs or {}),
    )
    R_tag0 = _continuity_flip(R_tag0, prev_R_tag0)
    R_base = tag0_rot_to_base(R_tag0, T_tag0_tcp_base)
    R_field_raw = _orient_approach_into_surface(R_base)
    R_field = smooth_R_field(prev_R_field, R_field_raw, smoothing=r_field_smoothing)
    return (
        R_field,
        abs(float(signed_d)),
        bool(in_contact),
        bool(diag.get("used_wos", False)),
        R_tag0,
    )


def _update_axis_curves(handles: dict, name: str, T: np.ndarray, axis_len: float) -> None:
    """Update or create RGB axis curve networks for SE3 ``T``."""
    assert ps is not None
    T = np.asarray(T, dtype=float).reshape(4, 4)
    o = T[:3, 3]
    for i, (c, label) in enumerate(
        zip([(1.0, 0.15, 0.15), (0.15, 0.85, 0.15), (0.2, 0.4, 1.0)], ("x", "y", "z"))
    ):
        seg = np.vstack([o, o + axis_len * T[:3, i]])
        key = f"{name}_{label}"
        if key in handles:
            handles[key].update_node_positions(seg)
        else:
            handles[key] = ps.register_curve_network(
                key, seg, np.array([[0, 1]], dtype=int), radius=0.0022, color=c
            )
    origin_key = name
    if origin_key in handles:
        handles[origin_key].update_point_positions(o.reshape(1, 3))
    else:
        handles[origin_key] = ps.register_point_cloud(
            origin_key, o.reshape(1, 3), radius=0.006, color=(0.95, 0.95, 0.95)
        )


def _run_interactive_teleop_keys(
    *,
    tip0: np.ndarray,
    R0_viz: np.ndarray,
    pcloud,
    wos,
    T_tag0_vention: np.ndarray,
    T_rtde_base_vention: np.ndarray | None,
    T_tag0_tcp_base: np.ndarray,
    viz_frame: str,
    contact_threshold: float,
    query_kwargs: dict,
    keyboard_step: float,
    linear_scale0: float,
    abs_d0: float,
    in_contact0: bool,
    used_wos0: bool,
) -> None:
    """Polyscope user-callback: hold WASD/XYZ to move tip along R_field (live-like)."""
    assert ps is not None
    from polyscope import imgui

    from geodude_teleoperation.teleop.contact_wos_vf import (
        DEFAULT_MOTION_TRACK,
        DEFAULT_RELOCK_DIST_M,
        DEFAULT_RELOCK_PERIOD_S,
        DEFAULT_R_FIELD_SMOOTHING,
        slerp_R,
        smooth_R_field,
    )
    from geodude_teleoperation.teleop.controller import _adaptive_linear_scale

    tip = np.asarray(tip0, dtype=float).reshape(3).copy()
    R_viz = np.asarray(R0_viz, dtype=float).reshape(3, 3).copy()
    R_move = R_viz.copy()
    R_field_filt = R_viz.copy()
    linear_scale = float(linear_scale0)
    abs_d = float(abs_d0)
    in_contact = bool(in_contact0)
    used_wos = bool(used_wos0)
    prev_R_tag0: np.ndarray | None = None
    last_t = time.time()
    last_relock_t = 0.0
    dist_since_relock = 0.0
    was_holding = False
    speed = max(0.03, 8.0 * float(keyboard_step))
    relock_dist_m = DEFAULT_RELOCK_DIST_M
    relock_period_s = DEFAULT_RELOCK_PERIOD_S
    move_track = DEFAULT_MOTION_TRACK
    r_field_smoothing = DEFAULT_R_FIELD_SMOOTHING

    handles: dict = {}
    tip_cloud = ps.register_point_cloud(
        "ee_tip", tip.reshape(1, 3), radius=0.007, color=(1.0, 0.55, 0.1)
    )
    T0 = np.eye(4)
    T0[:3, :3] = R_viz
    T0[:3, 3] = tip
    _update_axis_curves(handles, "ee_frame", T0, axis_len=0.06)

    key_colors = {
        "w": (0.9, 0.2, 0.2),
        "s": (0.6, 0.1, 0.1),
        "d": (0.2, 0.85, 0.2),
        "a": (0.1, 0.55, 0.1),
        "x": (0.25, 0.45, 1.0),
        "y": (0.15, 0.25, 0.7),
    }
    ghosts = teleop_key_ghosts(
        tip, R_viz, keyboard_step=keyboard_step, linear_scale=linear_scale
    )
    ghost_clouds: dict = {}
    ghost_curves: dict = {}
    for key, ghost in ghosts.items():
        ghost_clouds[key] = ps.register_point_cloud(
            f"teleop_ghost_{key}",
            ghost.reshape(1, 3),
            radius=0.0045,
            color=key_colors[key],
        )
        ghost_curves[key] = ps.register_curve_network(
            f"teleop_key_{key}",
            np.vstack([tip, ghost]),
            np.array([[0, 1]], dtype=int),
            radius=0.0016,
            color=key_colors[key],
        )

    key_map = [
        (imgui.ImGuiKey_W, 0, 1.0),
        (imgui.ImGuiKey_S, 0, -1.0),
        (imgui.ImGuiKey_D, 1, 1.0),
        (imgui.ImGuiKey_A, 1, -1.0),
        (imgui.ImGuiKey_X, 2, 1.0),
        (imgui.ImGuiKey_Y, 2, -1.0),
    ]

    def _refresh_geometry() -> None:
        nonlocal ghosts
        tip_cloud.update_point_positions(tip.reshape(1, 3))
        T = np.eye(4)
        T[:3, :3] = R_viz
        T[:3, 3] = tip
        _update_axis_curves(handles, "ee_frame", T, axis_len=0.06)
        ghosts = teleop_key_ghosts(
            tip, R_move, keyboard_step=keyboard_step, linear_scale=linear_scale
        )
        for key, ghost in ghosts.items():
            ghost_clouds[key].update_point_positions(ghost.reshape(1, 3))
            ghost_curves[key].update_node_positions(np.vstack([tip, ghost]))

    def _requery_field() -> None:
        nonlocal R_viz, R_field_filt, linear_scale, abs_d, in_contact, used_wos
        nonlocal prev_R_tag0
        tip_tag0 = point_viz_to_tag0(
            tip,
            T_tag0_vention=T_tag0_vention,
            T_rtde_base_vention=T_rtde_base_vention,
            viz_frame=viz_frame,
        )
        R_tcp, abs_d, in_contact, used_wos, prev_R_tag0 = query_R_field_at_tip_tag0(
            pcloud,
            wos,
            tip_tag0,
            T_tag0_tcp_base=T_tag0_tcp_base,
            contact_threshold=contact_threshold,
            query_kwargs=query_kwargs,
            prev_R_tag0=prev_R_tag0,
            prev_R_field=None,
            r_field_smoothing=0.0,
        )
        R_raw_viz = R_tcp_base_to_viz(
            R_tcp,
            T_tag0_vention=T_tag0_vention,
            T_rtde_base_vention=T_rtde_base_vention,
            viz_frame=viz_frame,
        )
        R_field_filt = smooth_R_field(
            R_field_filt, R_raw_viz, smoothing=r_field_smoothing
        )
        R_viz = R_field_filt
        linear_scale = 0.25 * _adaptive_linear_scale(abs_d, 1.0, 10.0)

    def callback() -> None:
        nonlocal tip, last_t, last_relock_t, dist_since_relock, was_holding, R_move
        now = time.time()
        dt = min(max(now - last_t, 0.0), 0.05)
        last_t = now

        imgui.Begin("Teleop key preview", True)
        imgui.TextWrapped(
            "Hold w/s/a/d/x/y (focus Polyscope). Motion tracks smoothed R_field; "
            f"re-locks every {relock_dist_m*100:.0f} mm / {relock_period_s*1000:.0f} ms."
        )
        imgui.Text(
            f"|d|={abs_d:.4f}  contact={int(in_contact)}  wos={int(used_wos)}  "
            f"scale={linear_scale:.2f}  speed={speed:.3f} m/s  "
            f"since_relock={dist_since_relock*1000:.0f} mm"
        )
        imgui.End()

        move = np.zeros(3, dtype=float)
        for imgui_key, axis, sign in key_map:
            if imgui.IsKeyDown(imgui_key):
                move += sign * R_move[:, axis]
        holding = float(np.linalg.norm(move)) > 1e-12
        if holding:
            if not was_holding:
                R_move = R_viz.copy()
                dist_since_relock = 0.0
                last_relock_t = now
                move = np.zeros(3, dtype=float)
                for imgui_key, axis, sign in key_map:
                    if imgui.IsKeyDown(imgui_key):
                        move += sign * R_move[:, axis]
            n = np.linalg.norm(move)
            if n > 1e-12:
                step = (move / n) * (speed * dt)
                tip = tip + step
                dist_since_relock += float(np.linalg.norm(step))
            need_relock = (
                dist_since_relock >= relock_dist_m
                or (now - last_relock_t) >= relock_period_s
            )
            if need_relock:
                _requery_field()
                R_move = slerp_R(R_move, R_viz, move_track)
                dist_since_relock = 0.0
                last_relock_t = now
            _refresh_geometry()
        elif was_holding:
            _requery_field()
            R_move = R_viz.copy()
            dist_since_relock = 0.0
            last_relock_t = now
            _refresh_geometry()
        was_holding = holding

    print(
        "[viz] interactive teleop keys: focus Polyscope, hold w/s/a/d/x/y — "
        "R_field EMA smoothed; motion re-locks on distance/time — close window to continue",
        flush=True,
    )
    ps.set_user_callback(callback)
    ps.show()
    ps.clear_user_callback()


def _run_live_rtde_markers(
    *,
    live_viz: LiveVizState,
    V: np.ndarray,
    tip_offset: np.ndarray,
    T_tag0_vention_raw: np.ndarray,
    T_rtde_base_vention: np.ndarray | None,
    viz_frame: str,
    T_cam_EE: np.ndarray | None,
    in_contact_mode: bool,
    seed_tcp: Sequence[float] | None,
    pcloud,
    bc_tag0: dict[str, np.ndarray],
    field_idx: np.ndarray,
    object_cloud: Any | None,
    field_cloud: Any | None,
    bc_clouds: dict[str, Any],
    object_offset_t: np.ndarray,
    object_offset_step_m: float,
    object_offset_path: Any | None,
    laser_vention: np.ndarray | None = None,
    laser_cloud: Any | None = None,
    laser_offset_t: np.ndarray | None = None,
    laser_offset_path: Any | None = None,
) -> None:
    """Keep tip/TCP/camera synced to RTDE; j/l i/k u/o nudge object offset.

    Hold Shift with the same keys to nudge the laser overlay (vention XYZ).
    """
    assert ps is not None
    try:
        from polyscope import imgui as _imgui
    except Exception:  # pragma: no cover - headless / tests
        _imgui = None

    from geodude_teleoperation.teleop.laser_cloud import points_vention_in_viz_frame
    from geodude_teleoperation.teleop.laser_offset import (
        apply_laser_t_offset,
        save_laser_offset,
    )
    from geodude_teleoperation.teleop.object_offset import (
        apply_object_t_offset,
        delta_viz_to_vention,
        save_object_offset,
    )

    # Real imgui.IsKeyDown segfaults outside an active Polyscope window (unit tests).
    imgui = None if getattr(ps, "_is_fake_for_tests", False) else _imgui

    handles: dict[str, Any] = {}
    last_seq = -1
    last_print_t = 0.0
    tip_offset = np.asarray(tip_offset, dtype=float).reshape(3)
    T_raw = np.asarray(T_tag0_vention_raw, dtype=float).reshape(4, 4)
    t_off = np.asarray(object_offset_t, dtype=float).reshape(3)
    live_viz.object_offset_t = t_off
    step = float(object_offset_step_m)
    save_path = object_offset_path
    V_work = np.asarray(V, dtype=float).reshape(-1, 3).copy()
    laser_raw = (
        None
        if laser_vention is None
        else np.asarray(laser_vention, dtype=float).reshape(-1, 3).copy()
    )
    t_laser = (
        np.zeros(3, dtype=float)
        if laser_offset_t is None
        else np.asarray(laser_offset_t, dtype=float).reshape(3)
    )
    laser_save_path = laser_offset_path
    live_viz.laser_offset_t = t_laser

    def _T_eff() -> np.ndarray:
        return apply_object_t_offset(T_raw, t_off)

    def _laser_vention_eff() -> np.ndarray | None:
        if laser_raw is None:
            return None
        return apply_laser_t_offset(laser_raw, t_laser)

    def _refresh_object_geometry() -> None:
        nonlocal V_work
        T_eff = _T_eff()
        V_work = points_in_viz_frame(
            pcloud.vertices,
            T_tag0_vention=T_eff,
            T_rtde_base_vention=T_rtde_base_vention,
            viz_frame=viz_frame,
        )
        if object_cloud is not None:
            object_cloud.update_point_positions(V_work)
        if field_cloud is not None:
            field_cloud.update_point_positions(V_work[field_idx])
        for name, cloud in bc_clouds.items():
            pts = bc_tag0.get(name)
            if pts is None or len(pts) == 0:
                continue
            cloud.update_point_positions(
                points_in_viz_frame(
                    pts,
                    T_tag0_vention=T_eff,
                    T_rtde_base_vention=T_rtde_base_vention,
                    viz_frame=viz_frame,
                )
            )
        if laser_cloud is not None and laser_raw is not None:
            laser_cloud.update_point_positions(
                points_vention_in_viz_frame(
                    _laser_vention_eff(),
                    T_tag0_vention=T_eff,
                    T_rtde_base_vention=T_rtde_base_vention,
                    viz_frame=viz_frame,
                )
            )
        live_viz.bump_offset_seq()

    def _apply_tcp(tcp_pose: np.ndarray, R_field_tcp: np.ndarray | None) -> float | None:
        T_eff = _T_eff()
        markers = tip_tcp_markers(
            tcp_pose,
            tip_offset,
            T_tag0_vention=T_eff,
            T_rtde_base_vention=T_rtde_base_vention,
            viz_frame=viz_frame,
        )
        if "T_tcp" not in markers:
            return None
        _update_axis_curves(handles, "tcp", markers["T_tcp"], axis_len=0.08)
        tip = markers["tip"].reshape(3)
        if "tip" in handles:
            handles["tip"].update_point_positions(tip.reshape(1, 3))
        else:
            handles["tip"] = ps.register_point_cloud(
                "tip", tip.reshape(1, 3), radius=0.005, color=(1.0, 0.85, 0.1)
            )
        nn_i = int(np.linalg.norm(V_work - tip.reshape(1, 3), axis=1).argmin())
        seg = np.vstack([tip, V_work[nn_i : nn_i + 1]])
        if "tip_to_nn" in handles:
            handles["tip_to_nn"].update_node_positions(seg)
        else:
            handles["tip_to_nn"] = ps.register_curve_network(
                "tip_to_nn",
                seg,
                np.array([[0, 1]], dtype=int),
                radius=0.0015,
                color=(1.0, 0.85, 0.1),
            )
        nn = float(np.linalg.norm(tip - V_work[nn_i]))

        T_cam = camera_pose_in_viz(
            tcp_pose,
            T_cam_EE,
            T_tag0_vention=T_eff,
            T_rtde_base_vention=T_rtde_base_vention,
            viz_frame=viz_frame,
        )
        if T_cam is not None:
            _update_axis_curves(handles, "camera", T_cam, axis_len=0.05)
            _update_camera_frustum(handles, "camera_frustum", T_cam)

        if in_contact_mode and R_field_tcp is not None:
            R_viz = R_tcp_base_to_viz(
                R_field_tcp,
                T_tag0_vention=T_eff,
                T_rtde_base_vention=T_rtde_base_vention,
                viz_frame=viz_frame,
            )
            segs = contact_frame_overlays(tip, R_viz)
            for key, color, radius in (
                ("contact_tangent", (1.0, 0.35, 0.85), 0.0030),
                ("contact_approach", (0.2, 0.85, 1.0), 0.0026),
                ("contact_field_y", (0.3, 0.9, 0.35), 0.0020),
            ):
                pts = segs[
                    {
                        "contact_tangent": "tangent_line",
                        "contact_approach": "approach_n",
                        "contact_field_y": "field_y",
                    }[key]
                ]
                if key in handles:
                    handles[key].update_node_positions(pts)
                else:
                    handles[key] = ps.register_curve_network(
                        key,
                        pts,
                        np.array([[0, 1]], dtype=int),
                        radius=radius,
                        color=color,
                    )
            T_c = np.eye(4)
            T_c[:3, :3] = R_viz
            T_c[:3, 3] = tip
            _update_axis_curves(handles, "contact_R_field", T_c, axis_len=0.05)
        elif R_field_tcp is not None:
            # Free-space: show R_field at tip so operators can see VF orientation.
            R_viz = R_tcp_base_to_viz(
                R_field_tcp,
                T_tag0_vention=T_eff,
                T_rtde_base_vention=T_rtde_base_vention,
                viz_frame=viz_frame,
            )
            T_c = np.eye(4)
            T_c[:3, :3] = R_viz
            T_c[:3, 3] = tip
            _update_axis_curves(handles, "R_field", T_c, axis_len=0.06)
        return nn

    if seed_tcp is not None:
        _apply_tcp(np.asarray(seed_tcp, dtype=float).reshape(6), None)

    nudge_map: list = []
    prev_down: dict = {}
    teleop_key_map: list = []
    if imgui is not None:
        nudge_map = [
            (imgui.ImGuiKey_J, np.array([-1.0, 0.0, 0.0])),
            (imgui.ImGuiKey_L, np.array([1.0, 0.0, 0.0])),
            (imgui.ImGuiKey_I, np.array([0.0, 1.0, 0.0])),
            (imgui.ImGuiKey_K, np.array([0.0, -1.0, 0.0])),
            (imgui.ImGuiKey_U, np.array([0.0, 0.0, 1.0])),
            (imgui.ImGuiKey_O, np.array([0.0, 0.0, -1.0])),
        ]
        prev_down = {int(k): False for k, _ in nudge_map}
        for extra in ("LeftBracket", "RightBracket", "R", "P"):
            prev_down[extra] = False
        # Linear + gates only (j/k/u/i reserved for object offset in this window).
        teleop_key_map = [
            (imgui.ImGuiKey_W, "w"),
            (imgui.ImGuiKey_S, "s"),
            (imgui.ImGuiKey_A, "a"),
            (imgui.ImGuiKey_D, "d"),
            (imgui.ImGuiKey_X, "x"),
            (imgui.ImGuiKey_Y, "y"),
            (imgui.ImGuiKey_T, "t"),
            (imgui.ImGuiKey_V, "v"),
            (imgui.ImGuiKey_F, "f"),
            (imgui.ImGuiKey_Q, "q"),
        ]
        for key_enum, _ch in teleop_key_map:
            prev_down[f"teleop_{int(key_enum)}"] = False
        for name in ("Enter", "KeypadEnter"):
            prev_down[f"teleop_{name}"] = False

    def _edge(key_id, key_enum) -> bool:
        if imgui is None:
            return False
        down = bool(imgui.IsKeyDown(key_enum))
        was = bool(prev_down.get(key_id, False))
        prev_down[key_id] = down
        return down and not was

    def callback() -> None:
        nonlocal last_seq, last_print_t, step, save_path, laser_save_path
        if imgui is not None:
            shift = bool(
                imgui.IsKeyDown(getattr(imgui, "ImGuiKey_LeftShift", 0))
                or imgui.IsKeyDown(getattr(imgui, "ImGuiKey_RightShift", 0))
            )
            # Teleop keys → RTDE worker (stdin is inactive while Polyscope has focus).
            # Movement: report held keys every frame (control loop applies at control_hz).
            # Gates (t/v/f/q/Enter): one-shot only.
            _teleop_move = {"w", "s", "a", "d", "x", "y"}

            def _pressed(key_enum, *, repeat: bool = False) -> bool:
                if key_enum is None:
                    return False
                pressed = getattr(imgui, "IsKeyPressed", None)
                if pressed is not None:
                    try:
                        return bool(pressed(key_enum, repeat))
                    except TypeError:
                        return bool(pressed(key_enum))
                return _edge(f"fallback_{int(key_enum)}", key_enum)

            enter = getattr(imgui, "ImGuiKey_Enter", None)
            if _pressed(enter, repeat=False):
                live_viz.push_teleop_key("\n")
            kp_enter = getattr(imgui, "ImGuiKey_KeypadEnter", None)
            if _pressed(kp_enter, repeat=False):
                live_viz.push_teleop_key("\n")
            held: set[str] = set()
            for key_enum, ch in teleop_key_map:
                if ch in _teleop_move:
                    if imgui.IsKeyDown(key_enum):
                        held.add(ch)
                elif _pressed(key_enum, repeat=False):
                    live_viz.push_teleop_key(ch)
            live_viz.set_held_teleop_keys(held)

            for key_enum, delta_viz in nudge_map:
                if _edge(int(key_enum), key_enum):
                    d_v = delta_viz_to_vention(
                        step * delta_viz,
                        viz_frame=viz_frame,
                        T_tag0_vention=T_raw,
                        T_rtde_base_vention=T_rtde_base_vention,
                    )
                    if shift and laser_raw is not None:
                        t_laser[:] = t_laser + d_v
                        _refresh_object_geometry()
                        print(
                            f"[viz-laser-offset] t_vention_m={t_laser.tolist()}  "
                            f"step={step:.4f} m",
                            flush=True,
                        )
                    else:
                        t_off[:] = t_off + d_v
                        _refresh_object_geometry()
                        print(
                            f"[viz-offset] t_vention_m={t_off.tolist()}  "
                            f"step={step:.4f} m",
                            flush=True,
                        )

            lb = getattr(imgui, "ImGuiKey_LeftBracket", None)
            rb = getattr(imgui, "ImGuiKey_RightBracket", None)
            if lb is not None and _edge("LeftBracket", lb):
                step = max(1e-4, step * 0.5)
                print(f"[viz-offset] step={step:.4f} m", flush=True)
            if rb is not None and _edge("RightBracket", rb):
                step = min(0.05, step * 2.0)
                print(f"[viz-offset] step={step:.4f} m", flush=True)
            if _edge("R", imgui.ImGuiKey_R):
                if shift and laser_raw is not None:
                    t_laser[:] = 0.0
                    _refresh_object_geometry()
                    print(
                        "[viz-laser-offset] reset t_vention_m=[0,0,0]",
                        flush=True,
                    )
                else:
                    t_off[:] = 0.0
                    _refresh_object_geometry()
                    print("[viz-offset] reset t_vention_m=[0,0,0]", flush=True)
            if _edge("P", imgui.ImGuiKey_P):
                if shift and laser_raw is not None:
                    path = laser_save_path
                    if path is None:
                        from pathlib import Path as _Path

                        path = _Path("laser_offset.json")
                    laser_save_path = save_laser_offset(path, t_laser, step_m=step)
                    print(f"[viz-laser-offset] saved {laser_save_path}", flush=True)
                else:
                    path = save_path
                    if path is None:
                        from pathlib import Path as _Path

                        path = _Path("object_offset.json")
                    save_path = save_object_offset(path, t_off, step_m=step)
                    print(f"[viz-offset] saved {save_path}", flush=True)

            imgui.TextUnformatted(
                f"object offset t_vention_m={np.array2string(t_off, precision=4)}  "
                f"step={step:.4f} m  (j/l i/k u/o nudge, [/] step, r reset, p save)"
            )
            if laser_raw is not None:
                imgui.TextUnformatted(
                    f"laser offset t_vention_m={np.array2string(t_laser, precision=4)}  "
                    f"(Shift+j/l i/k u/o, Shift+r reset, Shift+p save)"
                )
            imgui.TextUnformatted(
                "teleop (this window): Enter enable · Enter again approve · "
                "hold w/s a/d x/y move · t/v pause/VF · f hold / unlock VF · q quit"
            )
            draw_latency_imgui(imgui, live_viz)

        if live_viz.loop_done.is_set():
            return
        seq, tcp, R_field, signed_d, in_contact = live_viz.snapshot()
        if tcp is None or seq == last_seq:
            return
        last_seq = seq
        nn = _apply_tcp(tcp, R_field)
        now = time.time()
        if nn is not None and (now - last_print_t) >= 0.5:
            last_print_t = now
            d_s = "n/a" if signed_d is None else f"{signed_d:+.4f}"
            c_s = "n/a" if in_contact is None else str(int(in_contact))
            print(
                f"[viz-live] tcp={tcp[:3].tolist()}  tip_nn={nn:.4f}  "
                f"d={d_s}  in_contact={c_s}  "
                f"t_obj={t_off.tolist()}  t_laser={t_laser.tolist()}",
                flush=True,
            )

    print(
        "[viz] live RTDE + offsets: j/l i/k u/o object; Shift+same laser; "
        "Enter/w/s/a/d/x/y/t/v/f/q teleop in this window; "
        "p / Shift+p save; close window to drop viz (q quits teleop)"
        + (
            "; Latency table ON"
            if getattr(live_viz, "show_latency_table", False)
            else ""
        ),
        flush=True,
    )
    ps.set_user_callback(callback)
    ps.show()
    ps.clear_user_callback()
    live_viz.set_held_teleop_keys(())
    live_viz.mark_viz_closed()
    print(
        "[viz] window closed — RTDE teleop continues in background until terminal q",
        flush=True,
    )


def show_field_debug(
    pcloud,
    sd,
    *,
    T_tag0_vention: np.ndarray,
    T_rtde_base_vention: np.ndarray | None = None,
    viz_frame: str = "vention",
    frame_stride: int = 20,
    tcp_pose_rtde: Sequence[float] | None = None,
    tip_offset: np.ndarray | None = None,
    T_cam_EE: np.ndarray | None = None,
    show_teleop_keys: bool = False,
    hide_object: bool = False,
    wos=None,
    T_tag0_tcp_base: np.ndarray | None = None,
    keyboard_step: float = 0.01,
    contact_threshold: float = 0.001,
    query_kwargs: dict | None = None,
    in_contact_mode: bool = False,
    live_viz: LiveVizState | None = None,
    T_tag0_vention_raw: np.ndarray | None = None,
    object_offset_t: np.ndarray | None = None,
    object_offset_step_m: float = 0.001,
    object_offset_path: Any | None = None,
    laser_vention: np.ndarray | None = None,
    laser_offset_t: np.ndarray | None = None,
    laser_offset_path: Any | None = None,
) -> None:
    """Blocking Polyscope window: cloud, BCs, orientation frames, tip/TCP/camera."""
    _ensure_polyscope()
    assert ps is not None and plot_orientation_field is not None
    assert set_camera_and_plane is not None
    assert frame_camera_on_points is not None

    tip_offset = (
        np.zeros(3, dtype=float)
        if tip_offset is None
        else np.asarray(tip_offset, dtype=float).reshape(3)
    )
    T_tag0_vention = np.asarray(T_tag0_vention, dtype=float).reshape(4, 4)

    V = points_in_viz_frame(
        pcloud.vertices,
        T_tag0_vention=T_tag0_vention,
        T_rtde_base_vention=T_rtde_base_vention,
        viz_frame=viz_frame,
    )
    bases = rotate_bases_to_viz_frame(
        pcloud.local_bases,
        T_tag0_vention=T_tag0_vention,
        T_rtde_base_vention=T_rtde_base_vention,
        viz_frame=viz_frame,
    )

    bc_tag0 = bc_clouds_tag0(pcloud, sd)
    bc_viz = {
        name: points_in_viz_frame(
            pts,
            T_tag0_vention=T_tag0_vention,
            T_rtde_base_vention=T_rtde_base_vention,
            viz_frame=viz_frame,
        )
        for name, pts in bc_tag0.items()
    }
    markers = tip_tcp_markers(
        tcp_pose_rtde,
        tip_offset,
        T_tag0_vention=T_tag0_vention,
        T_rtde_base_vention=T_rtde_base_vention,
        viz_frame=viz_frame,
    )
    T_cam_viz = camera_pose_in_viz(
        tcp_pose_rtde,
        T_cam_EE,
        T_tag0_vention=T_tag0_vention,
        T_rtde_base_vention=T_rtde_base_vention,
        viz_frame=viz_frame,
    )

    teleop_preview = None
    contact_preview = None
    if show_teleop_keys or in_contact_mode:
        if tcp_pose_rtde is None or wos is None or "tip" not in markers:
            print(
                "[viz] contact/teleop preview skipped (need TCP pose + wos + tip)",
                flush=True,
            )
        else:
            T_query = (
                T_tag0_vention
                if T_tag0_tcp_base is None
                else np.asarray(T_tag0_tcp_base, dtype=float).reshape(4, 4)
            )
            from geodude_teleoperation.teleop.controller import _adaptive_linear_scale

            R_tcp, abs_d, in_contact, used_wos = query_R_field_at_tcp(
                pcloud,
                wos,
                tcp_pose_rtde,
                tip_offset,
                T_tag0_tcp_base=T_query,
                contact_threshold=contact_threshold,
                query_kwargs=query_kwargs,
            )
            R_viz = R_tcp_base_to_viz(
                R_tcp,
                T_tag0_vention=T_tag0_vention,
                T_rtde_base_vention=T_rtde_base_vention,
                viz_frame=viz_frame,
            )
            if in_contact_mode:
                contact_preview = {
                    "R": R_viz,
                    "abs_d": abs_d,
                    "in_contact": in_contact,
                    "used_wos": used_wos,
                }
            if show_teleop_keys:
                factor = _adaptive_linear_scale(abs_d, 1.0, 10.0)
                linear_scale = 0.25 * factor
                ghosts = teleop_key_ghosts(
                    markers["tip"],
                    R_viz,
                    keyboard_step=keyboard_step,
                    linear_scale=linear_scale,
                )
                teleop_preview = {
                    "R": R_viz,
                    "ghosts": ghosts,
                    "linear_scale": linear_scale,
                    "abs_d": abs_d,
                    "in_contact": in_contact,
                    "used_wos": used_wos,
                    "step_m": linear_scale * float(keyboard_step),
                }

    vmin, vmax = V.min(axis=0), V.max(axis=0)
    print(
        f"[viz] frame={viz_frame}  #verts={len(V)}  "
        f"bbox_min={vmin.tolist()}  bbox_max={vmax.tolist()}",
        flush=True,
    )
    if T_rtde_base_vention is not None:
        print(
            f"[viz] using T_rtde_base_vention t={np.asarray(T_rtde_base_vention)[:3,3].tolist()}",
            flush=True,
        )
    else:
        print(
            "[viz] WARNING: no T_rtde_base_vention — RTDE TCP treated as vention "
            "(likely wrong on Geodude)",
            flush=True,
        )
    for name, pts in bc_viz.items():
        print(f"[viz] {name}: {len(pts)} points", flush=True)
    if markers:
        tip = markers["tip"]
        tcp = markers["tcp"]
        nn = float(np.linalg.norm(V - tip.reshape(1, 3), axis=1).min())
        print(
            f"[viz] tcp={tcp.tolist()}  tip={tip.tolist()}  tip_nn_in_viz={nn:.4f} m",
            flush=True,
        )
    else:
        print("[viz] no TCP pose yet (TCP/tip markers omitted)", flush=True)
    if T_cam_viz is not None:
        print(
            f"[viz] camera origin={T_cam_viz[:3, 3].tolist()}  (from T_cam_EE)",
            flush=True,
        )
    elif tcp_pose_rtde is not None and T_cam_EE is None:
        print(
            "[viz] camera omitted (pass --calib-meta with T_cam_EE)",
            flush=True,
        )
    if teleop_preview is not None:
        print(
            f"[viz] teleop keys: step={teleop_preview['step_m']:.5f} m  "
            f"scale={teleop_preview['linear_scale']:.3f}  "
            f"|d|={teleop_preview['abs_d']:.4f}  "
            f"contact={int(teleop_preview['in_contact'])}  "
            f"wos={int(teleop_preview['used_wos'])}  "
            f"(w/s=±x d/a=±y x/y=±z in R_field)",
            flush=True,
        )
    if contact_preview is not None:
        print(
            f"[viz] in-contact preview: |d|={contact_preview['abs_d']:.4f}  "
            f"band_contact={int(contact_preview['in_contact'])}  "
            f"wos={int(contact_preview['used_wos'])}  "
            f"(magenta=tangent ±x, cyan=approach n*, green=field y)",
            flush=True,
        )

    ps.init()
    # Layup is roughly XY (thin in Z); z_up keeps it horizontal at startup.
    set_camera_and_plane(up="z_up")
    ps.set_ground_plane_mode("none")

    if hide_object:
        print(
            "[viz] object cloud hidden (field frames only); pass --viz-show-object to restore",
            flush=True,
        )
        object_cloud = None
    else:
        object_cloud = ps.register_point_cloud(
            "object", V, radius=0.0009, color=(0.6, 0.6, 0.6)
        )
        ut = getattr(sd, "ut", None)
        if ut is not None and len(np.asarray(ut)) == len(V):
            object_cloud.add_scalar_quantity(
                "ut", np.asarray(ut, dtype=float), enabled=True
            )

    laser_ps = None
    t_laser0 = (
        np.zeros(3, dtype=float)
        if laser_offset_t is None
        else np.asarray(laser_offset_t, dtype=float).reshape(3)
    )
    if laser_vention is not None and len(np.asarray(laser_vention).reshape(-1, 3)) > 0:
        from geodude_teleoperation.teleop.laser_cloud import points_vention_in_viz_frame
        from geodude_teleoperation.teleop.laser_offset import apply_laser_t_offset

        L = points_vention_in_viz_frame(
            apply_laser_t_offset(laser_vention, t_laser0),
            T_tag0_vention=T_tag0_vention,
            T_rtde_base_vention=T_rtde_base_vention,
            viz_frame=viz_frame,
        )
        laser_ps = ps.register_point_cloud(
            "laser", L, radius=0.0007, color=(0.2, 0.75, 0.95)
        )
        print(
            f"[viz] laser overlay: {len(L)} points  "
            f"t_vention_m={t_laser0.tolist()}",
            flush=True,
        )

    colors = {
        "sink_line": (0.15, 0.35, 1.0),
        "source_line": (1.0, 0.25, 0.15),
        "boundary_sinks": (0.15, 0.35, 1.0),
    }
    bc_clouds: dict[str, Any] = {}
    for name, pts in bc_viz.items():
        if len(pts) == 0:
            continue
        bc_clouds[name] = ps.register_point_cloud(
            name, pts, radius=0.0028, color=colors.get(name, (1.0, 1.0, 0.0))
        )

    # Denser field samples when the noisy object cloud is hidden.
    stride = max(1, int(frame_stride))
    if hide_object:
        stride = max(1, stride // 2)
    idx = np.arange(0, len(V), stride)
    field_cloud = plot_orientation_field(
        V[idx],
        bases[idx],
        name="field",
        vector_length=0.025,
        vector_radius=0.0015,
        point_radius=0.0012,
        enable_x=True,
        enable_z=False,
        enable_vector=False,
    )

    # Keep RTDE-base as a toggleable reference; hide so home view frames the object.
    if viz_frame == "vention" and T_rtde_base_vention is not None:
        T_rv = np.asarray(T_rtde_base_vention, dtype=float).reshape(4, 4)
        _register_axes("rtde_base", T_rv, axis_len=0.12, enabled=False)

    frame_camera_on_points(V)

    if live_viz is not None and not show_teleop_keys:
        t0 = (
            np.zeros(3, dtype=float)
            if object_offset_t is None
            else np.asarray(object_offset_t, dtype=float).reshape(3)
        )
        if T_tag0_vention_raw is not None:
            raw = np.asarray(T_tag0_vention_raw, dtype=float).reshape(4, 4)
        else:
            raw = np.asarray(T_tag0_vention, dtype=float).reshape(4, 4).copy()
            raw[:3, 3] = raw[:3, 3] - t0
        _run_live_rtde_markers(
            live_viz=live_viz,
            V=V,
            tip_offset=tip_offset,
            T_tag0_vention_raw=raw,
            T_rtde_base_vention=T_rtde_base_vention,
            viz_frame=viz_frame,
            T_cam_EE=T_cam_EE,
            in_contact_mode=in_contact_mode,
            seed_tcp=tcp_pose_rtde,
            pcloud=pcloud,
            bc_tag0=bc_tag0,
            field_idx=idx,
            object_cloud=object_cloud,
            field_cloud=field_cloud,
            bc_clouds=bc_clouds,
            object_offset_t=t0,
            object_offset_step_m=float(object_offset_step_m),
            object_offset_path=object_offset_path,
            laser_vention=laser_vention,
            laser_cloud=laser_ps,
            laser_offset_t=t_laser0,
            laser_offset_path=laser_offset_path,
        )
        return

    if "T_tcp" in markers:
        _register_axes("tcp", markers["T_tcp"], axis_len=0.08)
    elif "tcp" in markers:
        ps.register_point_cloud(
            "tcp", markers["tcp"].reshape(1, 3), radius=0.006, color=(0.1, 0.45, 1.0)
        )
    if "tip" in markers:
        tip = markers["tip"].reshape(1, 3)
        ps.register_point_cloud("tip", tip, radius=0.005, color=(1.0, 0.85, 0.1))
        nn_i = int(np.linalg.norm(V - tip, axis=1).argmin())
        seg = np.vstack([tip, V[nn_i : nn_i + 1]])
        ps.register_curve_network(
            "tip_to_nn",
            seg,
            np.array([[0, 1]], dtype=int),
            radius=0.0015,
            color=(1.0, 0.85, 0.1),
        )
        if contact_preview is not None:
            _register_contact_overlays(markers["tip"], contact_preview["R"])
    if T_cam_viz is not None:
        _register_axes("camera", T_cam_viz, axis_len=0.05)
        _register_camera_frustum("camera_frustum", T_cam_viz, scale=0.06)

    if teleop_preview is not None:
        T_query = (
            T_tag0_vention
            if T_tag0_tcp_base is None
            else np.asarray(T_tag0_tcp_base, dtype=float).reshape(4, 4)
        )
        _run_interactive_teleop_keys(
            tip0=markers["tip"],
            R0_viz=teleop_preview["R"],
            pcloud=pcloud,
            wos=wos,
            T_tag0_vention=T_tag0_vention,
            T_rtde_base_vention=T_rtde_base_vention,
            T_tag0_tcp_base=T_query,
            viz_frame=viz_frame,
            contact_threshold=contact_threshold,
            query_kwargs=query_kwargs or {},
            keyboard_step=keyboard_step,
            linear_scale0=teleop_preview["linear_scale"],
            abs_d0=teleop_preview["abs_d"],
            in_contact0=teleop_preview["in_contact"],
            used_wos0=teleop_preview["used_wos"],
        )
        return

    print("[viz] close the Polyscope window to continue", flush=True)
    ps.show()


def maybe_show_startup_viz(
    args: Any,
    pcloud,
    sd,
    *,
    T_tag0_vention: np.ndarray,
    T_rtde_base_vention: np.ndarray | None,
    tip_offset: np.ndarray,
    tcp_pose_rtde: Sequence[float] | None,
    T_cam_EE: np.ndarray | None = None,
    wos=None,
    T_tag0_tcp_base: np.ndarray | None = None,
    live_viz: LiveVizState | None = None,
    T_tag0_vention_raw: np.ndarray | None = None,
) -> None:
    if not getattr(args, "viz", False):
        return
    if T_cam_EE is None:
        T_cam_EE = load_T_cam_EE(getattr(args, "calib_meta", None))
    query_kwargs = {
        "query_radius": float(getattr(args, "query_radius", 0.025)),
        "fit_mode": str(getattr(args, "fit_mode", "sphere")),
    }
    hide_object = bool(getattr(args, "viz_hide_object", False))
    show_teleop_keys = bool(getattr(args, "viz_teleop_keys", False))

    laser_vention = None
    if getattr(args, "viz_laser", "off") == "overlay":
        try:
            from geodude_teleoperation.frames import load_T_left_base_vention
            from geodude_teleoperation.teleop.laser_cloud import (
                load_laser_points_vention,
                resolve_laser_ply,
            )

            T_left = load_T_left_base_vention(
                getattr(args, "T_left_base_vention", None)
            )
            ply = resolve_laser_ply(getattr(args, "laser_ply", None))
            print(
                f"[viz] loading laser overlay {ply} "
                f"(voxel={float(getattr(args, 'laser_voxel', 0.002))} m)",
                flush=True,
            )
            laser_vention = load_laser_points_vention(
                ply,
                voxel_m=float(getattr(args, "laser_voxel", 0.002)),
                T_left_base_vention=T_left,
                # Keep raw; offset applied in viz / live Shift-nudge.
                t_vention_m=None,
            )
        except Exception as exc:
            print(f"warning: --viz-laser overlay failed: {exc}", file=sys.stderr)
            laser_vention = None

    try:
        show_field_debug(
            pcloud,
            sd,
            T_tag0_vention=T_tag0_vention,
            T_rtde_base_vention=T_rtde_base_vention,
            viz_frame=getattr(args, "viz_frame", "vention"),
            frame_stride=int(getattr(args, "viz_frame_stride", 20)),
            tcp_pose_rtde=tcp_pose_rtde,
            tip_offset=tip_offset,
            T_cam_EE=T_cam_EE,
            show_teleop_keys=show_teleop_keys,
            hide_object=hide_object,
            wos=wos,
            T_tag0_tcp_base=T_tag0_tcp_base,
            keyboard_step=float(getattr(args, "keyboard_step", 0.01)),
            contact_threshold=float(getattr(args, "contact_threshold", 0.001)),
            query_kwargs=query_kwargs,
            in_contact_mode=bool(getattr(args, "in_contact", False)),
            live_viz=None if show_teleop_keys else live_viz,
            T_tag0_vention_raw=T_tag0_vention_raw,
            object_offset_t=getattr(args, "object_offset_t", None),
            object_offset_step_m=float(getattr(args, "object_offset_step_m", 0.001)),
            object_offset_path=getattr(args, "object_offset_path", None),
            laser_vention=laser_vention,
            laser_offset_t=getattr(args, "laser_offset_t", None),
            laser_offset_path=getattr(args, "laser_offset_path", None),
        )
    except ImportError as exc:
        print(f"error: --viz failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
