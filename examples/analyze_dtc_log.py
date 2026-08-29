"""Summarize a --dtc-log CSV: the wrench split, the rail, and which joint.

The 1 Hz console line samples; this reads every tick. Answers the two questions
a wrench-law run leaves open: is |W| actually asking for lateral newtons, and is
the clip counter still *rising* (cap too low) or flat (not the cap any more).

    python examples/analyze_dtc_log.py            # newest speedj_logs/dtc_*.csv
    python examples/analyze_dtc_log.py path.csv
"""
import csv, glob, pathlib, sys
import numpy as np


def newest():
    hits = sorted(glob.glob("speedj_logs/dtc_*.csv"), key=lambda p: pathlib.Path(p).stat().st_mtime)
    if not hits:
        sys.exit("no speedj_logs/dtc_*.csv found")
    return hits[-1]


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else newest()
    rows = list(csv.DictReader(open(path)))
    if not rows:
        sys.exit(f"{path}: empty")
    n = sum(1 for k in rows[0] if k.startswith("tau"))
    print(f"{path}: {len(rows)} ticks, {n} joints")

    phase = np.array([r["phase"] for r in rows])
    for p in sorted(set(phase)):
        print(f"  phase {p:<10} {int((phase == p).sum()):6d}  {(phase == p).mean():6.1%}")
    dt = np.array([float(r["dt_ms"]) for r in rows if r["dt_ms"]])
    print(f"  dt_ms  median {q(dt,50):6.2f}  p95 {q(dt,95):6.2f}  max {dt.max():6.2f}"
          f"   -> {1e3/max(q(dt,50),1e-9):.0f} Hz median")

    live = phase == "live"
    if not live.any():
        print("  no live ticks (hold_zero or gated the whole run)")
        live = np.ones(len(rows), bool)
    W = np.array([[float(r[f"w{i}"]) for i in range(6)] for r in rows])
    wa = np.linalg.norm(W[live, 0:3], axis=1)
    wl = np.linalg.norm(W[live, 3:6], axis=1)
    print(f"\n  |W_ang| N.m  median {q(wa,50):7.2f}  p95 {q(wa,95):7.2f}  max {wa.max():7.2f}")
    print(f"  |W_lin| N    median {q(wl,50):7.2f}  p95 {q(wl,95):7.2f}  max {wl.max():7.2f}")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(wl > 1e-6, wa / wl, np.nan)
    print(f"  ang/lin ratio median {np.nanmedian(ratio):7.2f}"
          "   (>>1 = the wrench is a moment; the wrists get it, the TCP does not slide)")

    tau = np.array([[float(r[f"tau{i}"]) for i in range(n)] for r in rows])
    clip = np.array([[r[f"clip{i}"] == "1" for i in range(n)] for r in rows])
    print("\n  joint   |tau| p95   |tau| max   clip%    clip by quartile of the run")
    quarters = np.array_split(np.arange(len(rows)), 4)
    for j in range(n):
        per_q = "  ".join(f"{clip[idx, j].mean():5.1%}" for idx in quarters)
        print(f"   j{j}    {q(np.abs(tau[:,j]),95):8.2f}   {np.abs(tau[:,j]).max():8.2f}"
              f"   {clip[:,j].mean():6.1%}   {per_q}")
    print("\n  rising left-to-right = still capped, raise --dtc-tau-max on that joint.")
    print("  flat/zero while the TCP will not slide = not the cap; look at |W_lin|.")


if __name__ == "__main__":
    main()
