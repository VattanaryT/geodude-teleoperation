"""Summarize a --dtc-log CSV: the wrench split, the rail, and which joint.

The 1 Hz console line samples; this reads every tick. Answers the three
questions a wrench-law run leaves open: is |W| actually asking for lateral
newtons, is the clip counter still *rising* (cap too low) or flat (not the cap
any more), and -- from the tracking columns -- is the tool actually conforming
to the object field or just being shoved at it.

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

    stall(rows, live, W, n)
    tracking(rows, live, W)


def stall(rows, live, W, n):
    """Stick-slip report: is the arm being pushed while it is not moving?

    The wrench and clip columns cannot see this. It is gated on the spring
    being *on*, which matters more than it looks: over all ticks
    dtc_20260829_232839.csv reads 50.1% stalled, but 98.6% of its spring-off
    ticks are stalled and correctly so -- an idle arm should not move. Against
    spring-on ticks only, the two runs read 11.9% and 6.7%, with |W_lin| at
    9-12 N on the stalled ones and |qdot| p95 of 0.17-0.26 rad/s: real
    loading-up against friction, but a second-order effect next to the
    conformity error below.

    Read it against that conformity number. Falling stall% with falling n_deg
    is the fix working; falling stall% with n_deg unchanged just means the push
    got weaker.
    """
    if f"qd{n - 1}" not in rows[0] or "act" not in rows[0]:
        print("\n  (no qd*/act columns -- log predates them; no stall report)")
        return
    qd = np.array([[float(r[f"qd{j}"]) for j in range(n)] for r in rows])
    act = np.array([r["act"] == "1" for r in rows])
    on = live & act
    if not on.any():
        print("\n  (spring never armed; no stall report)")
        return
    speed = np.linalg.norm(qd, axis=1)[on]
    wl = np.linalg.norm(W[on, 3:6], axis=1)
    slow = speed < 0.01
    print(f"\n  |qdot| rad/s median {q(speed,50):6.3f}  p95 {q(speed,95):6.3f}"
          f"  max {speed.max():6.3f}   (spring on)")
    print(f"  stalled (|qdot| < 0.01) {slow.mean():6.1%} of spring-on ticks")
    if slow.any() and (~slow).any():
        print(f"  |W_lin| N while stalled median {q(wl[slow],50):6.2f}"
              f"   while moving median {q(wl[~slow],50):6.2f}")
    print("     a high stall% carrying real newtons is stick-slip -- the loop "
          "is loading up against friction, not oscillating. Check the "
          "conformity number below before touching a gain.")


def tracking(rows, live, W):
    """Field-conformity report off the err*/n_deg/act columns.

    Written because the wrench columns alone cannot separate "the arm is
    tracking the field" from "the arm is being shoved at a target it never
    reaches": both read as a large |W|. n_deg is the angle between live tool +Z
    and the commanded field normal, so it IS the conformity error; err* is what
    the spring was correcting when that angle was held.
    """
    if "n_deg" not in rows[0]:
        print("\n  (no tracking columns -- log predates err*/n_deg/act; "
              "re-run to get the conformity report)")
        return
    def col(name):
        return np.array([float(r[name]) if r[name] else np.nan for r in rows])

    t = col("t")
    act = np.array([r["act"] == "1" for r in rows])
    n_deg = col("n_deg")
    e_lin = np.array([[float(r[k]) for k in ("ex", "ey", "ez")] for r in rows])
    e_ang = np.array([[float(r[k]) for k in ("erx", "ery", "erz")] for r in rows])
    el = np.linalg.norm(e_lin, axis=1) * 1000.0
    ea = np.degrees(np.linalg.norm(e_ang, axis=1))
    on = live & act
    print(f"\n  spring on {act.mean():6.1%} of ticks "
          f"({int((act & live).sum())} live)")
    if on.any():
        print(f"  |err_lin| mm  median {q(el[on],50):7.2f}  p95 {q(el[on],95):7.2f}"
              f"  max {el[on].max():7.2f}   (spring on)")
        print(f"  |err_ang| deg median {q(ea[on],50):7.2f}  p95 {q(ea[on],95):7.2f}"
              f"  max {ea[on].max():7.2f}   (spring on)")
    ok = np.isfinite(n_deg)
    if ok.any():
        print(f"  tool Z vs field normal deg  median {q(n_deg[ok],50):6.2f}"
              f"  p95 {q(n_deg[ok],95):6.2f}  max {n_deg[ok].max():6.2f}")
        print("     this is the conformity number. Sustained >5 deg while the "
              "spring is on = the arm is not reaching the field pose.")

    # Re-arm steps: the spring gate going 0->1 hands the loop a fresh target in
    # one tick. A big wrench on that edge is a step command, not tracking.
    wl = np.linalg.norm(W[:, 3:6], axis=1)
    wa = np.linalg.norm(W[:, 0:3], axis=1)
    edge = np.flatnonzero(act[1:] & ~act[:-1]) + 1
    if edge.size:
        step_l = wl[edge]
        step_a = wa[edge]
        print(f"\n  spring re-arm edges: {edge.size}"
              f"   |W_lin| on the edge median {q(step_l,50):6.2f} max {step_l.max():7.2f} N"
              f"   |W_ang| max {step_a.max():6.2f} N.m")
        worst = edge[np.argsort(step_l)[-5:][::-1]]
        for i in sorted(worst):
            print(f"    t={t[i]:8.3f}  |W_lin| {wl[i-1]:6.2f} -> {wl[i]:7.2f} N"
                  f"   |W_ang| {wa[i-1]:5.2f} -> {wa[i]:5.2f} N.m"
                  f"   err {el[i]:6.2f} mm / {ea[i]:5.2f} deg")
        print("    a large step here is a re-arm discontinuity: the target was "
              "the live pose while quiet and the full field pose one tick later.")


if __name__ == "__main__":
    main()
