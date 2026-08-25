"""Clock a local-frame demo through the speedJ + DualLoop plants.

The pose-spring ``update_command`` treats stick linear as a body-frame pose
step (meters per callback). DualLoop contact treats the same linear as a
body-frame velocity (m/s). This module converts flange-body sandwich
velocity into the plant EE body and applies that distinction.

Default replay uses DualLoop in contact: score **XY** (and force), not 3D TCP.
``--follow-path`` keeps ``a_n`` and never enters DualLoop: score 3D ``err_mm``.
With bag start joints in the model, tracking error is absolute chamfer vs bag.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class TrackingRow:
    i: int
    t: float
    mode: str
    err_xy_mm: float
    err_z_mm: float
    err_mm: float
    abs_fn: float | None
    plant_vz: float


def body_twist_for_plant(
    v_flange: np.ndarray,
    *,
    dt_s: float,
    contact: bool,
    R_flange: np.ndarray,
    R_plant: np.ndarray,
) -> np.ndarray:
    """Flange-body linear (m/s) → gafro twist ``[w, v]`` in the plant EE.

    Approach: ``v * dt`` (meters this tick). Contact: ``v`` (m/s). Angular is 0.
    """
    v = np.asarray(v_flange, dtype=float).reshape(3)
    R_f = np.asarray(R_flange, dtype=float).reshape(3, 3)
    R_p = np.asarray(R_plant, dtype=float).reshape(3, 3)
    body = R_p.T @ (R_f @ v)
    if not contact:
        body = body * float(dt_s)
    twist = np.zeros(6, dtype=float)
    twist[3:] = body
    return twist


def apply_mode_edge(
    mode: dict,
    in_contact: bool,
    *,
    on_enter: Callable[[], None],
    on_exit: Callable[[], None],
) -> str:
    """Switch on raw contact. Returns the mode after the edge."""
    want = "contact" if in_contact else "approach"
    if mode.get("active") != want:
        if want == "contact":
            on_enter()
        else:
            on_exit()
        mode["active"] = want
    return str(mode["active"])


def relative_error_mm(
    *,
    live_xyz: np.ndarray,
    bag_xyz: np.ndarray,
    live0: np.ndarray,
    bag0: np.ndarray,
) -> tuple[float, float, float]:
    """(xy_mm, z_mm, norm_mm) of (live−live0) vs (bag−bag0)."""
    e = (np.asarray(live_xyz, dtype=float).reshape(3) - np.asarray(live0, dtype=float).reshape(3)) - (
        np.asarray(bag_xyz, dtype=float).reshape(3) - np.asarray(bag0, dtype=float).reshape(3)
    )
    xy = float(np.hypot(e[0], e[1]) * 1e3)
    z = float(e[2] * 1e3)
    n = float(np.linalg.norm(e) * 1e3)
    return xy, z, n


def absolute_error_mm(
    *,
    live_xyz: np.ndarray,
    bag_xyz: np.ndarray,
) -> tuple[float, float, float]:
    """(xy_mm, z_mm, norm_mm) of live chamfer vs bag TCP."""
    e = np.asarray(live_xyz, dtype=float).reshape(3) - np.asarray(
        bag_xyz, dtype=float
    ).reshape(3)
    xy = float(np.hypot(e[0], e[1]) * 1e3)
    z = float(e[2] * 1e3)
    n = float(np.linalg.norm(e) * 1e3)
    return xy, z, n


def path_follow_contact(raw: np.ndarray, *, follow_path: bool) -> np.ndarray:
    """Contact flags for sandwich: all-false keeps ``a_n`` (path follow)."""
    flags = np.asarray(raw, dtype=bool).reshape(-1)
    if follow_path:
        return np.zeros_like(flags)
    return flags.copy()


def replay_mode_label(in_contact: bool, *, follow_path: bool) -> str:
    if follow_path:
        return "path"
    return "contact" if in_contact else "approach"


def score_legend(*, follow_path: bool) -> str:
    if follow_path:
        return "score: 3D err_mm (path follow; DualLoop off)"
    return (
        "score: contact XY (+ |f_n|); mean_z is DualLoop-owned, not a tracking fail"
    )


def enter_speedj_contact(*, mode, contact_controller, approach_controller, interface=None):
    """Keep the pose-spring speedJ plant. Overlay DualLoop Z. Do not swap to line admittance."""
    del interface
    mode["active"] = "contact"
    approach_controller.hold_live_pose()
    approach_controller.set_input_active(False)
    approach_controller.set_tool_z_velocity(0.0)
    approach_controller.set_z_compliant(True)
    contact_controller.set_on_vz(approach_controller.set_tool_z_velocity)
    contact_controller.start()


def exit_speedj_contact(*, mode, contact_controller, approach_controller, interface=None):
    """Drop Z overlay and freeze on the same pose-spring plant."""
    del interface
    contact_controller.stop()
    contact_controller.set_on_vz(None)
    approach_controller.set_tool_z_velocity(0.0)
    approach_controller.set_z_compliant(False)
    approach_controller.hold_live_pose()
    approach_controller.set_input_active(False)
    mode["active"] = "approach"


def ur_reaction_fz(
    f_n: float, *, in_contact: bool, force_sign: float = 1.0
) -> float:
    """Bag ``f_n`` as UR ``ActualTCPForce`` tool-Z (press along +Z is negative)."""
    if not in_contact:
        return 0.0
    return -float(force_sign) * abs(float(f_n))


def format_tracking_table(rows: list[TrackingRow]) -> str:
    header = (
        f"{'#':>4}  {'mode':<9} {'t':>7}  "
        f"{'xy_mm':>8} {'z_mm':>8} {'err_mm':>8} {'|f_n|N':>8} {'vz_mm/s':>8}"
    )
    lines = [header]
    for r in rows:
        fn = f"{r.abs_fn:8.2f}" if r.abs_fn is not None else f"{'n/a':>8}"
        lines.append(
            f"{r.i:4d}  {r.mode:<9} {r.t:7.2f}  "
            f"{r.err_xy_mm:8.2f} {r.err_z_mm:8.2f} {r.err_mm:8.2f} "
            f"{fn} {r.plant_vz * 1e3:8.2f}"
        )
    return "\n".join(lines)


def summarize_tracking(rows: list[TrackingRow]) -> str:
    """One line per contiguous mode block: mean/max path error."""
    if not rows:
        return "(no samples)"
    header = (
        f"{'#':>2}  {'mode':<9} {'t0':>7} {'t1':>7} {'n':>4}  "
        f"{'mean_mm':>8} {'max_mm':>8} {'mean_xy':>8} {'mean_z':>8} {'|f_n|N':>8}"
    )
    lines = [header]
    i0 = 0
    n_seg = 0
    while i0 < len(rows):
        i1 = i0 + 1
        while i1 < len(rows) and rows[i1].mode == rows[i0].mode:
            i1 += 1
        block = rows[i0:i1]
        errs = np.array([r.err_mm for r in block], dtype=float)
        xys = np.array([r.err_xy_mm for r in block], dtype=float)
        zs = np.array([r.err_z_mm for r in block], dtype=float)
        fns = [r.abs_fn for r in block if r.abs_fn is not None]
        fn = f"{float(np.mean(fns)):8.2f}" if fns else f"{'n/a':>8}"
        n_seg += 1
        lines.append(
            f"{n_seg:2d}  {block[0].mode:<9} {block[0].t:7.2f} {block[-1].t:7.2f} "
            f"{len(block):4d}  {float(errs.mean()):8.2f} {float(errs.max()):8.2f} "
            f"{float(xys.mean()):8.2f} {float(zs.mean()):8.2f} {fn}"
        )
        i0 = i1
    return "\n".join(lines)
