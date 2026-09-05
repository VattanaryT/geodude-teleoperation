#!/usr/bin/env python3
"""Paired-push summary for a dtc log: is the dead band CENTRED or OFFSET?

A P-D law with no integrator settles exactly where spring force equals
friction, so *every* such controller has a dead band of width
``F_friction / K``.  Pushing from one side cannot tell that band's **width**
from its **centre**.  Push ``+u``, release, settle; push ``-u``, release,
settle.  The **midpoint** of the two settles is the **bias** -- a gravity or
model error, the same every time -- and the **half-separation** is the
**friction**.  A centred band returns from both sides.  A band offset by a bias
returns from one side and stops short by twice the bias from the other, which
is a one-sided dead zone.

Prints the same table as ``log_push_test.py`` in the diffusion_policy repo, so
the two controllers can be diffed line for line rather than by feel.

    python examples/analyze_paired_push.py                  # newest dtc log
    python examples/analyze_paired_push.py speedj_logs/dtc_*.csv

Settles are found from the log, not from labels: a window of at least
``--settle`` seconds where every joint is under ``--quiet`` rad/s.  Consecutive
settles are paired when the joint that moved most between them moved in
opposite directions -- i.e. a push and its opposite.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def _newest() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    hits = sorted(glob.glob(os.path.join(here, "..", "speedj_logs", "dtc_*.csv")))
    if not hits:
        sys.exit("no speedj_logs/dtc_*.csv found; pass a path")
    return hits[-1]


def load(path):
    """``(header_lines, {column: array})``.  ``#`` lines are the gain header."""
    head, rows, cols = [], [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                head.append(line.rstrip())
                continue
            parts = line.rstrip("\n").split(",")
            if cols is None:
                cols = parts
                continue
            rows.append(parts)
    if not rows:
        sys.exit(f"{path}: no data rows")
    out = {}
    for i, name in enumerate(cols):
        vals = np.array([r[i] if i < len(r) else "" for r in rows], dtype=object)
        try:
            out[name] = np.array(
                [np.nan if v in ("", None) else float(v) for v in vals],
                dtype=float)
        except (TypeError, ValueError):
            out[name] = vals
    return head, out


def _stack(d, prefix, n=6):
    keys = [f"{prefix}{i}" for i in range(n)]
    if not all(k in d for k in keys):
        return None
    return np.column_stack([d[k] for k in keys])


def find_settles(t, qd, settle_s, quiet):
    """Maximal runs where ``max|qd|`` stays under ``quiet`` for >= settle_s."""
    still = np.max(np.abs(qd), axis=1) < quiet
    out, i, n = [], 0, len(still)
    while i < n:
        if not still[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and still[j + 1]:
            j += 1
        if t[j] - t[i] >= settle_s:
            out.append((i, j))
        i = j + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=None)
    ap.add_argument("--settle", type=float, default=2.0,
                    help="seconds a window must stay quiet to count (2.0)")
    ap.add_argument("--quiet", type=float, default=0.01,
                    help="rad/s below which every joint counts as still (0.01)")
    ap.add_argument("--tail", type=float, default=1.0,
                    help="seconds at the END of each settle to average (1.0)")
    args = ap.parse_args()

    path = args.path or _newest()
    head, d = load(path)
    print(f"=== {os.path.basename(path)} ===")
    for h in head:
        print(h)

    t = d["t"]
    q = _stack(d, "q")
    qd = _stack(d, "qd")
    tau = _stack(d, "tau")
    if q is None or qd is None:
        sys.exit("log has no q*/qd* columns")
    ex = np.column_stack([d[k] for k in ("ex", "ey", "ez")]) if "ex" in d else None
    er = np.column_stack([d[k] for k in ("erx", "ery", "erz")]) if "erx" in d else None
    tgt = (np.column_stack([d[k] for k in ("tgt_x", "tgt_y", "tgt_z")])
           if "tgt_x" in d else None)
    ee = (np.column_stack([d[k] for k in ("ee_x", "ee_y", "ee_z")])
          if "ee_x" in d else None)

    segs = find_settles(t, qd, args.settle, args.quiet)
    if not segs:
        sys.exit(f"\nno settle >= {args.settle:g} s with every joint under "
                 f"{args.quiet:g} rad/s -- lower --quiet or hold longer")

    print(f"\n=== {len(segs)} settles (median over the last {args.tail:g} s of "
          f"each) ===")
    hdr = (f"{'t_end':>7} {'lin_mm':>7} {'ang_deg':>8}   "
           + " ".join(f"{j[:7]:>8}" for j in JOINTS))
    print(hdr)
    print(" " * 26 + "q, deg")
    S = []
    for i, j in segs:
        m = slice(np.searchsorted(t, t[j] - args.tail, "left"), j + 1)
        rec = dict(
            t=float(t[j]),
            q=np.degrees(np.median(q[m], axis=0)),
            tau=None if tau is None else np.median(tau[m], axis=0),
            lin=np.nan if ex is None else float(
                np.linalg.norm(np.median(ex[m], axis=0))) * 1e3,
            ang=np.nan if er is None else float(np.degrees(
                np.linalg.norm(np.median(er[m], axis=0)))),
            tgt=None if tgt is None else np.median(tgt[m], axis=0),
            ee=None if ee is None else np.median(ee[m], axis=0),
        )
        S.append(rec)
        print(f"{rec['t']:7.1f} {rec['lin']:7.2f} {rec['ang']:8.2f}   "
              + " ".join(f"{v:8.3f}" for v in rec["q"]))
    if tau is not None:
        print(" " * 26 + "tau, N.m")
        for rec in S:
            print(f"{rec['t']:7.1f} {'':>7} {'':>8}   "
                  + " ".join(f"{v:8.3f}" for v in rec["tau"]))

    # Did the target move?  This is the whole question a push test asks, and a
    # relative error cannot answer it: "the target held and the arm stopped
    # short" and "the target followed the arm" look identical in err*.
    if any(r["tgt"] is not None for r in S):
        base = S[0]["tgt"]
        print("\n=== target position, base frame (mm from the first settle) ===")
        print(f"{'t_end':>7} {'dtgt_x':>8} {'dtgt_y':>8} {'dtgt_z':>8} "
              f"{'|dtgt|':>8}   {'dee_x':>8} {'dee_y':>8} {'dee_z':>8}")
        for rec in S:
            dt_ = (rec["tgt"] - base) * 1e3
            de = ((rec["ee"] - S[0]["ee"]) * 1e3 if rec["ee"] is not None
                  else np.full(3, np.nan))
            print(f"{rec['t']:7.1f} " + " ".join(f"{v:8.2f}" for v in dt_)
                  + f" {np.linalg.norm(dt_):8.2f}   "
                  + " ".join(f"{v:8.2f}" for v in de))
        drift = max(float(np.linalg.norm(r["tgt"] - base)) for r in S) * 1e3
        if drift < 1.0:
            print(f"\n  target held to {drift:.2f} mm across every settle -- so "
                  f"a settle short of it is the ARM not arriving, not the "
                  f"target moving.")
        else:
            print(f"\n  target MOVED {drift:.2f} mm across the run. Any 'dead "
                  f"band' below that is the target chasing the arm, not "
                  f"friction. Check hold_at_tick= in the header.")

    # Every settle against the REST pose, which is the settle the target was
    # snapped onto and therefore the only one with err ~ 0. Pairing consecutive
    # settles (below) answers "did these two straddle", but the population of
    # offsets from rest is what says centred-vs-one-sided, and it uses every
    # push rather than throwing away all but one neighbour.
    rest = min(range(len(S)), key=lambda i: S[i]["lin"] if np.isfinite(S[i]["lin"])
               else np.inf)
    q0 = S[rest]["q"]
    print(f"\n=== every settle vs the REST pose (settle at t={S[rest]['t']:.1f}s, "
          f"err {S[rest]['lin']:.2f} mm), deg ===")
    print(f"{'t_end':>7}   " + " ".join(f"{j[:7]:>8}" for j in JOINTS))
    D = []
    for rec in S:
        if rec is S[rest]:
            continue
        d = rec["q"] - q0
        D.append(d)
        print(f"{rec['t']:7.1f}   " + " ".join(f"{v:+8.3f}" for v in d))
    if D:
        D = np.array(D)
        print(f"\n{'joint':>13} {'mean':>8} {'min':>8} {'max':>8} {'|max|':>8}"
              f"   verdict")
        for i, j in enumerate(JOINTS):
            col = D[:, i]
            lo, hi, mean = col.min(), col.max(), col.mean()
            span = max(abs(lo), abs(hi))
            if span < 0.15:
                v = "returns"
            elif len(D) < 3:
                # One or two pushes are always "all on one side" -- there was
                # no chance to be otherwise. Calling that one-sided is how a
                # single sample becomes a hardware campaign.
                v = ("only %d push%s: offset %.2f deg, direction unproven"
                     % (len(D), "" if len(D) == 1 else "es", span))
            elif lo > 0 or hi < 0:
                # never crossed the rest pose: the band is offset, not centred
                v = "ONE-SIDED (never returned past rest)"
            elif abs(mean) > 0.5 * span:
                v = "skewed (bias ~ %.2f deg)" % mean
            else:
                v = "centred band, half-width %.2f deg" % span
            print(f"{j:>13} {mean:+8.3f} {lo:+8.3f} {hi:+8.3f} {span:8.3f}   {v}")
        print("\n  ONE-SIDED means every push left that joint on the same side")
        print("  of rest -- a bias the spring cannot cross. A centred band is")
        print("  what every P-D law has: width = tau_friction / (lev*Lambda*K).")

    print("\n=== consecutive pairs: midpoint is BIAS, half-separation is FRICTION ===")
    if len(S) < 2:
        print("  need at least two settles")
        return
    printed = 0
    for a, b in zip(S, S[1:]):
        dq = b["q"] - a["q"]
        k = int(np.argmax(np.abs(dq)))
        if abs(dq[k]) < 0.2:
            continue        # the arm never really moved between these two
        printed += 1
        print(f"\n  settle {a['t']:.1f}s -> {b['t']:.1f}s   "
              f"(moved most about {JOINTS[k]}, {dq[k]:+.2f} deg)")
        print(f"    {'joint':>13} {'settle_a':>9} {'settle_b':>9} "
              f"{'bias':>9} {'friction':>9}")
        for i, j in enumerate(JOINTS):
            mid = (a["q"][i] + b["q"][i]) / 2.0
            half = abs(a["q"][i] - b["q"][i]) / 2.0
            print(f"    {j:>13} {a['q'][i]:9.3f} {b['q'][i]:9.3f} "
                  f"{mid:9.3f} {half:9.3f}")
    if not printed:
        print("  no pair moved more than 0.2 deg -- push harder, or the "
              "settles are all the same rest")
        return
    print("\n  Read it per joint, against the OTHER settle of the pair:")
    print("  * both settles on the same side of the rest pose = one-sided.")
    print("  * settles straddling it, similar magnitude = a centred band,")
    print("    which is what every P-D law has and is not a fault.")
    print("  Compare tau at the two settles: a joint that is 'not trying hard")
    print("  enough' shows a SMALL |tau| while its error is large. A joint at")
    print("  its own breakaway (UR5e wrist ~1.5-2.5 N.m, shoulder several) is")
    print("  trying as hard as the law allows and is friction-limited.")


if __name__ == "__main__":
    main()
