"""Split a local-frame demo into approach vs contact (DualLoop-style).

Contact enters on the first in-contact sample and stays until ``lost_s`` of
free samples (same idea as DualLoop unload). In contact, field-normal motion
is dropped and body-Z is zeroed so the force loop owns press.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sticky_contact(
    raw: np.ndarray, *, dt_s: float, lost_s: float = 1.0
) -> np.ndarray:
    """Enter immediately; leave only after ``lost_s`` consecutive free samples.

    DualLoop unloads when ``(now - unload_since) >= lost_s``. Discretized: keep
    contact for ``round(lost_s / dt)`` free samples, then drop.
    """
    raw = np.asarray(raw, dtype=bool).reshape(-1)
    if float(lost_s) <= 0.0:
        return raw.copy()
    n_hold = max(1, int(round(float(lost_s) / float(dt_s))))
    out = np.zeros(len(raw), dtype=bool)
    in_contact = False
    free_run = 0
    for i, flag in enumerate(raw):
        if flag:
            in_contact = True
            free_run = 0
        elif in_contact:
            free_run += 1
            if free_run > n_hold:
                in_contact = False
                free_run = 0
        out[i] = in_contact
    return out


def body_linear_from_local(
    R_field: np.ndarray,
    R_ee: np.ndarray,
    a_local: np.ndarray,
    *,
    dt_s: float,
    contact: bool,
) -> np.ndarray:
    """EE-body linear velocity for one sample (m/s).

    ``R_ee`` columns are the flange axes in the same world as ``R_field``.
    """
    a = np.asarray(a_local, dtype=float).reshape(3).copy()
    if contact:
        a[2] = 0.0
    R_f = np.asarray(R_field, dtype=float).reshape(3, 3)
    R_b = np.asarray(R_ee, dtype=float).reshape(3, 3)
    v_world = (R_f @ a) / float(dt_s)
    v_body = R_b.T @ v_world
    if contact:
        v_body[2] = 0.0
    return v_body


def body_linear_series(
    R_field: np.ndarray,
    R_ee: np.ndarray,
    a_local: np.ndarray,
    contact: np.ndarray,
    *,
    dt_s: float,
) -> np.ndarray:
    R_f = np.asarray(R_field, dtype=float)
    R_b = np.asarray(R_ee, dtype=float)
    a = np.asarray(a_local, dtype=float).reshape(-1, 3)
    flags = np.asarray(contact, dtype=bool).reshape(-1)
    n = len(a)
    if R_f.shape[0] == n + 1:
        R_f = R_f[:-1]
    if R_b.shape[0] == n + 1:
        R_b = R_b[:-1]
    out = np.empty((n, 3), dtype=float)
    for i in range(n):
        out[i] = body_linear_from_local(
            R_f[i], R_b[i], a[i], dt_s=dt_s, contact=bool(flags[i]))
    return out


@dataclass(frozen=True)
class ModeSegment:
    mode: str
    i0: int
    i1: int
    t0: float
    t1: float


def segments_from_contact(
    contact: np.ndarray, timestamps: np.ndarray
) -> list[ModeSegment]:
    """Half-open index ranges. ``timestamps`` may be pose length (N = n_flags+1)."""
    flags = np.asarray(contact, dtype=bool).reshape(-1)
    t = np.asarray(timestamps, dtype=float).reshape(-1)
    segs: list[ModeSegment] = []
    if len(flags) == 0:
        return segs
    i0 = 0
    for i in range(1, len(flags) + 1):
        if i == len(flags) or flags[i] != flags[i0]:
            mode = "contact" if flags[i0] else "approach"
            t0 = float(t[min(i0, len(t) - 1)])
            t1 = float(t[min(i, len(t) - 1)])
            segs.append(ModeSegment(mode, i0, i, t0, t1))
            i0 = i
    return segs


@dataclass(frozen=True)
class SegmentStats:
    mode: str
    i0: int
    i1: int
    t0: float
    t1: float
    n: int
    mean_abs_at1: float
    mean_abs_at2: float
    mean_abs_an: float
    mean_abs_fn: float | None
    mean_abs_v_body: tuple[float, float, float]


def summarize_segments(
    segs: list[ModeSegment],
    a_local: np.ndarray,
    v_body: np.ndarray,
    f_n: np.ndarray | None = None,
) -> list[SegmentStats]:
    a = np.asarray(a_local, dtype=float).reshape(-1, 3)
    v = np.asarray(v_body, dtype=float).reshape(-1, 3)
    fn = None if f_n is None else np.asarray(f_n, dtype=float).reshape(-1)
    out: list[SegmentStats] = []
    for seg in segs:
        sl = slice(seg.i0, seg.i1)
        aa = np.abs(a[sl])
        vv = np.abs(v[sl])
        n = int(seg.i1 - seg.i0)
        if fn is None:
            mean_fn = None
        elif n == 0:
            mean_fn = 0.0
        else:
            mean_fn = float(np.abs(fn[sl]).mean())
        out.append(
            SegmentStats(
                mode=seg.mode,
                i0=seg.i0,
                i1=seg.i1,
                t0=seg.t0,
                t1=seg.t1,
                n=n,
                mean_abs_at1=float(aa[:, 0].mean()) if n else 0.0,
                mean_abs_at2=float(aa[:, 1].mean()) if n else 0.0,
                mean_abs_an=float(aa[:, 2].mean()) if n else 0.0,
                mean_abs_fn=mean_fn,
                mean_abs_v_body=tuple(float(x) for x in vv.mean(axis=0)) if n else (0.0, 0.0, 0.0),
            )
        )
    return out


def sandwich_local_actions(
    *,
    R_field: np.ndarray,
    R_ee: np.ndarray,
    a_local: np.ndarray,
    in_contact: np.ndarray,
    timestamps: np.ndarray,
    f_n: np.ndarray | None = None,
    lost_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[ModeSegment], list[SegmentStats]]:
    """Sticky-split a demo and map field actions to EE-body twists.

    Returns ``(contact, v_body, twist, segments, stats)``. ``twist`` is gafro
    order ``[wx, wy, wz, vx, vy, vz]``; angular is 0 (CGA ``a_local`` is 3-D).
    In contact, field-normal and body-Z are zeroed; ``f_n`` is a force label.
    """
    a = np.asarray(a_local, dtype=float).reshape(-1, 3)
    t = np.asarray(timestamps, dtype=float).reshape(-1)
    dt_s = float(np.diff(t).mean()) if len(t) > 1 else 0.1
    raw = np.asarray(in_contact, dtype=bool).reshape(-1)[: len(a)]
    contact = sticky_contact(raw, dt_s=dt_s, lost_s=lost_s)
    v_body = body_linear_series(
        R_field, R_ee, a, contact, dt_s=dt_s)
    twist = np.zeros((len(a), 6), dtype=float)
    twist[:, 3:] = v_body
    segs = segments_from_contact(contact, t)
    fn = None if f_n is None else np.asarray(f_n, dtype=float).reshape(-1)[: len(a)]
    stats = summarize_segments(segs, a, v_body, f_n=fn)
    return contact, v_body, twist, segs, stats


def format_segment_table(stats: list[SegmentStats]) -> str:
    """Numeric table: actions in mm, force in N, body velocity in mm/s."""
    header = (
        f"{'#':>2}  {'mode':<9} {'t0':>7} {'t1':>7} {'dur':>6} {'n':>4}  "
        f"{'|a_t1|mm':>9} {'|a_t2|mm':>9} {'|a_n|mm':>9} {'|f_n|N':>7}  "
        f"{'|vx|mm/s':>9} {'|vy|mm/s':>9} {'|vz|mm/s':>9}"
    )
    lines = [header]
    for i, s in enumerate(stats, start=1):
        fn = f"{s.mean_abs_fn:7.2f}" if s.mean_abs_fn is not None else f"{'n/a':>7}"
        vx, vy, vz = s.mean_abs_v_body
        lines.append(
            f"{i:2d}  {s.mode:<9} {s.t0:7.2f} {s.t1:7.2f} {s.t1 - s.t0:6.2f} {s.n:4d}  "
            f"{s.mean_abs_at1 * 1e3:9.3f} {s.mean_abs_at2 * 1e3:9.3f} "
            f"{s.mean_abs_an * 1e3:9.3f} {fn}  "
            f"{vx * 1e3:9.2f} {vy * 1e3:9.2f} {vz * 1e3:9.2f}"
        )
    return "\n".join(lines)
