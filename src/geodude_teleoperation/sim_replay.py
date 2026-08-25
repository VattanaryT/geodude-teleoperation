"""Clock a demo through the speedJ + DualLoop plants.

The pose-spring ``update_command`` treats stick linear as a body-frame pose
step (meters per callback). DualLoop contact treats the same linear as a
body-frame velocity (m/s). This module converts flange-body velocity into
the plant EE body and applies that distinction.

Test 1 (``--bag-tcp``): stick is recorded TCP Δp/Δt in the flange.
Test 2 (check 6): stick is CGA ``a_local`` sandwiched into the flange.

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


def bag_tcp_flange_velocity(
    bag_xyz: np.ndarray, R_flange: np.ndarray, dt_s: float
) -> np.ndarray:
    """Recorded TCP Δp/Δt in flange body (m/s). Not local-frame ``a_t*``."""
    p = np.asarray(bag_xyz, dtype=float).reshape(-1, 3)
    R = np.asarray(R_flange, dtype=float).reshape(-1, 3, 3)
    if p.shape[0] < 2:
        return np.zeros((0, 3), dtype=float)
    n = min(p.shape[0] - 1, R.shape[0])
    dp = np.diff(p[: n + 1], axis=0) / float(dt_s)
    return np.einsum("nji,nj->ni", R[:n], dp)


DEFAULT_REPLAY_SPEED = 0.2
MAX_START_GAP_MM = 25.0
DEFAULT_CMD_HZ = 10.0


def start_gap_mm(live_xyz: np.ndarray, bag_xyz: np.ndarray) -> float:
    """3D distance (mm) from live TCP to bag start."""
    return float(
        np.linalg.norm(
            np.asarray(live_xyz, dtype=float).reshape(3)
            - np.asarray(bag_xyz, dtype=float).reshape(3)
        )
        * 1e3
    )


def refuse_start_gap(gap_mm: float, limit_mm: float = MAX_START_GAP_MM) -> None:
    """Raise if the live TCP is too far from bag start to begin replay."""
    gap = float(gap_mm)
    limit = float(limit_mm)
    if gap > limit:
        raise RuntimeError(
            f"live TCP is {gap:.1f} mm from bag start (limit {limit:.1f} mm). "
            "Jog closer, then rerun."
        )


def ramp_duration_s(
    gap_m: float, *, speed_mps: float = 0.02, min_s: float = 2.0
) -> float:
    """Wall time to crawl ``gap_m`` at ``speed_mps``. Never faster than ``min_s``."""
    s = float(speed_mps)
    if s <= 0.0:
        raise ValueError("speed_mps must be > 0")
    return max(float(min_s), abs(float(gap_m)) / s)


def blend_pose_matrix(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    """Lerp XYZ + slerp rotation. ``alpha=0`` is ``T0``."""
    from scipy.spatial.transform import Rotation, Slerp

    A = np.asarray(T0, dtype=float).reshape(4, 4)
    B = np.asarray(T1, dtype=float).reshape(4, 4)
    a = float(np.clip(alpha, 0.0, 1.0))
    out = np.eye(4)
    key = Rotation.concatenate([
        Rotation.from_matrix(A[:3, :3]),
        Rotation.from_matrix(B[:3, :3]),
    ])
    out[:3, :3] = Slerp([0.0, 1.0], key)(a).as_matrix()
    out[:3, 3] = (1.0 - a) * A[:3, 3] + a * B[:3, 3]
    return out


def slow_replay_clock(
    t: np.ndarray, speed: float, cmd_hz: float = DEFAULT_CMD_HZ
) -> tuple[np.ndarray, np.ndarray]:
    """Stretch demo time by ``1/speed``. Returns ``(t_wall, t_src)``.

    ``t_src`` is the original-timeline sample for each wall-clock command.
    ``speed=0.2`` makes a 1 s demo take 5 s. ``cmd_hz`` keeps a dense
    Cartesian stream so the spring does not jump a 10 Hz waypoint at once.
    """
    s = float(speed)
    if s <= 0.0:
        raise ValueError("speed must be > 0")
    hz = float(cmd_hz)
    if hz <= 0.0:
        raise ValueError("cmd_hz must be > 0")
    t0 = np.asarray(t, dtype=float).reshape(-1)
    t0 = t0 - t0[0]
    duration = float(t0[-1]) / s if t0.size else 0.0
    n = max(2, int(round(duration * hz)) + 1)
    t_wall = np.linspace(0.0, duration, n)
    return t_wall, t_wall * s


def interpolate_tcp_xyzw(
    poses: np.ndarray, t_src: np.ndarray, t_query: np.ndarray
) -> np.ndarray:
    """Lerp XYZ + slerp quat. ``poses`` is ``(N, 7)`` xyz + xyzw."""
    from scipy.spatial.transform import Rotation, Slerp

    poses = np.asarray(poses, dtype=float).reshape(-1, 7)
    t_src = np.asarray(t_src, dtype=float).reshape(-1)
    t_src = t_src - t_src[0]
    t_query = np.clip(np.asarray(t_query, dtype=float).reshape(-1), t_src[0], t_src[-1])
    out = np.empty((t_query.size, 7), dtype=float)
    for k in range(3):
        out[:, k] = np.interp(t_query, t_src, poses[:, k])
    slerp = Slerp(t_src, Rotation.from_quat(poses[:, 3:7]))
    out[:, 3:7] = slerp(t_query).as_quat()
    return out


def task_twist_from_error(
    error: np.ndarray,
    kp: float,
    *,
    z_velocity: float | None = None,
    feedforward: np.ndarray | None = None,
) -> np.ndarray:
    """Map gafro pose error to a body twist. ``z_velocity`` replaces body-Z (DualLoop)."""
    twist = -float(kp) * np.asarray(error, dtype=float).reshape(6)
    if feedforward is not None:
        ff = np.asarray(feedforward, dtype=float).reshape(6)
        if z_velocity is None:
            return twist + ff
        out = twist.copy()
        out[:5] = out[:5] + ff[:5]
        out[5] = float(z_velocity)
        return out
    if z_velocity is not None:
        twist = twist.copy()
        twist[5] = float(z_velocity)
    return twist


def damped_least_squares_qdot(
    jacobian: np.ndarray, twist: np.ndarray, *, damping: float = 1e-6
) -> np.ndarray:
    """``qdot = (JᵀJ + λI)⁻¹ Jᵀ twist``. Locks no joints; caller zeros the rail."""
    J = np.asarray(jacobian, dtype=float)
    t = np.asarray(twist, dtype=float).reshape(J.shape[0])
    n = int(J.shape[1])
    j_inv = np.linalg.inv(J.T @ J + float(damping) * np.eye(n)) @ J.T
    return j_inv @ t


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
