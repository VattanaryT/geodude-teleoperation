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


# A SpaceMouse cannot be driven above a couple of Hz, so everything below this
# is the operator and everything above it is the loop talking to itself. Every
# frequency-domain verdict here is taken *above* this floor -- the plain argmax
# of the qdot spectrum is always the stick, which is what made a run with 75%
# of its qdot power at 8.45 Hz print "no limit cycle".
STICK_FLOOR_HZ = 2.0
# Ticks of coast-down to drop after the stick is released. The arm is still
# decelerating there, and that transient is not a limit cycle: on
# dtc_20260830_123245.csv the "quiet" |qdot| p95 read 1.111 rad/s, which looked
# like a hold failure and was entirely the first 0.1 s after each release
# (median 1.03 rad/s under 0.1 s, 0.0000 past 0.5 s -- the same shape as a run
# with a clean hold, which differed only in how many short stick bursts it had).
QUIET_SETTLE_S = 0.25
# Quiet-stick |qdot| p95 above this is worth explaining at all.
QUIET_HOLD_RADPS = 0.05
# ...and above either of these, a commanded wrench explains it. The idle hold
# keeps the field *orientation* commanded on purpose, so the angular bound is
# the one that fires; both sit well above the wrench a truly idle arm carries
# (|W| medians on the quiet ticks of a clean run are ~0).
QUIET_WRENCH_ANG_NM = 0.5
QUIET_WRENCH_LIN_N = 2.0


def _settled(quiet, act, dt):
    """``quiet`` with QUIET_SETTLE_S after each falling edge dropped.

    The arm is still decelerating there and that transient is not the hold
    failing: dtc_20260830_123245.csv read |qdot| p95 1.111 rad/s over raw quiet
    ticks and 0.003 over settled ones.
    """
    settle = max(1, int(round(QUIET_SETTLE_S / dt)))
    since = np.full(quiet.size, settle + 1, int)
    c = settle + 1
    for i in range(quiet.size):
        c = 0 if (quiet[i] and i > 0 and act[i - 1]) else (
            c + 1 if quiet[i] else settle + 1)
        since[i] = c
    return quiet & (since >= settle)


def _read(path):
    """Rows plus the leading '# ...' gain header the backend writes, if any."""
    meta = []
    with open(path) as fh:
        lines = fh.readlines()
    while lines and lines[0].lstrip().startswith("#"):
        meta.append(lines.pop(0).lstrip().lstrip("#").strip())
    return list(csv.DictReader(lines)), meta


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else newest()
    rows, meta = _read(path)
    if not rows:
        sys.exit(f"{path}: empty")
    n = sum(1 for k in rows[0] if k.startswith("tau"))
    print(f"{path}: {len(rows)} ticks, {n} joints")
    for line in meta:
        print(f"  {line}")

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

    oscillation(rows, live, W, n)
    stall(rows, live, W, n)
    tracking(rows, live, W)


def oscillation(rows, live, W, n):
    """Limit-cycle report: is the arm shaking at a frequency nobody commanded?

    This exists because dtc_20260830_005006.csv passed every other section of
    this script -- conformity 0.59 deg median, 2 re-arm edges under 2 N, err_lin
    6 mm -- while the operator watched the arm oscillate violently with the
    stick at rest. The only hint anywhere in the old output was j4's clip%, and
    a clip% reads as "cap too low", which is the opposite of the truth.

    A limit cycle is obvious the moment you look in the frequency domain and
    invisible in every percentile: tau and qdot on all six joints peaked at
    **24.5 Hz**, common to every joint, which is a structural mode rather than
    anything the control law has a pole at. What sustains it is the damper:
    -D*v applied one loop delay late is *negative* damping at any frequency
    past 90 deg of lag, and at ~12 ms of delay that is everything above ~20 Hz.

    So the two numbers that matter are the peak frequency and how much of the
    motion sits there -- and both must be read against a *quiet stick*, because
    a peak while the operator is driving may just be the operator.

    **Both of those numbers were wrong, and 2026-08-30 proved it twice.**

    - The dominant frequency was the plain argmax of the summed qdot spectrum,
      which is *always* the stick: a SpaceMouse lives under 2 Hz and carries
      most of the power. `--dtc-fn-lin 4.0` (dtc_20260830_123040.csv) put
      **75.6% of the qdot power into 2-40 Hz, peaked at 8.45 Hz on j0/j3/j4**,
      and this section printed "no limit cycle" because the argmax was still
      0.2 Hz. Everything frequency-domain is now measured above STICK_FLOOR_HZ.
    - The "quiet stick" population included coast-down. On
      dtc_20260830_123245.csv it read |qdot| p95 = 1.111 rad/s, which looks
      like the arm not holding still and is not: median 1.03 rad/s within
      0.1 s of a release and 0.0000 past 0.5 s, the same profile as a run that
      read 0.012 -- the two differed only in how many short stick bursts they
      had, so the metric was reporting the operator's cadence. QUIET_SETTLE_S
      of each release is dropped.

    The two failures are different and the report must not blur them: an
    above-floor band share says the *closed loop* is ringing (fix it with the
    gains -- lower fn, or lower D if it is up near the delay crossover), while
    motion long after a release says the *hold* is failing (a different bug,
    and coast-down is not evidence of it).
    """
    if f"qd{n - 1}" not in rows[0]:
        print("\n  (no qd* columns -- log predates them; no oscillation report)")
        return
    t = np.array([float(r["t"]) for r in rows])[live]
    if t.size < 64:
        print("\n  (too few live ticks for an oscillation report)")
        return
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return
    tau = np.array([[float(r[f"tau{j}"]) for j in range(n)] for r in rows])[live]
    qd = np.array([[float(r[f"qd{j}"]) for j in range(n)] for r in rows])[live]
    act = (np.array([r["act"] == "1" for r in rows])[live]
           if "act" in rows[0] else np.ones(t.size, bool))
    win = np.hanning(t.size)
    freq = np.fft.rfftfreq(t.size, dt)

    # Above the stick floor: the plain argmax of a joint's spectrum is the
    # operator on every joint the operator is driving, which made this column
    # read 0.1-0.3 Hz on a run that was ringing at 8.45 Hz.
    above = freq >= STICK_FLOOR_HZ

    def peak(x):
        spec = np.abs(np.fft.rfft((x - x.mean()) * win))
        spec[0] = 0.0
        if not above.any() or np.where(above, spec, 0.0).max() <= 0:
            return float("nan"), spec, 0
        k = int(np.argmax(np.where(above, spec, 0.0)))
        return freq[k], spec, k

    print(f"\n  oscillation (FFT over {t.size} live ticks at {1/dt:.0f} Hz, "
          f"Nyquist {0.5/dt:.0f} Hz)")
    print(f"  peaks are the strongest line above {STICK_FLOOR_HZ:g} Hz, "
          f"not the argmax (which is always the stick)")
    # The same population the verdict below uses. These columns used to be the
    # raw ~act mask while the verdict dropped QUIET_SETTLE_S of coast-down, so
    # on dtc_20260830_130541.csv the table read j2/j3 at 0.64 rad/s "quiet"
    # against an aggregate of 0.069 -- an order of magnitude apart, on the same
    # run, under the same word.
    quiet = _settled(~act, act, dt)
    peaks = []
    for j in range(n):
        f_tau, _, _ = peak(tau[:, j])
        f_qd, _, _ = peak(qd[:, j])
        peaks.append(f_qd)
        qq = (f"{q(np.abs(qd[quiet, j]), 95):10.3f}" if quiet.any() else "         -")
        tq = (f"{q(np.abs(tau[quiet, j]), 95):11.2f}" if quiet.any() else "          -")
        print(f"   j{j}      {f_tau:8.1f}       {f_qd:8.1f}       {qq}       {tq}")

    # Energy-weighted, not a median over joints: on the run this was written
    # for, three joints were barely moving and their peaks sat near DC, which
    # dragged a median from the true 24.5 Hz down to 19.4. Sum the per-joint
    # power first, then take the peak of the total -- the joints that are
    # actually shaking are the ones that should name the frequency.
    power = np.zeros(freq.size)
    for j in range(n):
        x = qd[:, j]
        power += np.abs(np.fft.rfft((x - x.mean()) * win)) ** 2
    power[0] = 0.0
    total = max(power.sum(), 1e-30)
    del peaks
    # Above the stick floor and below Nyquist. The upper edge is only there to
    # keep the very top of the band -- where the anti-alias story is worst and
    # a real mode would fold in from above -- from naming the peak.
    band = (freq >= STICK_FLOOR_HZ) & (freq <= min(40.0, 0.45 / dt))
    share = float(power[band].sum() / total)
    if band.any() and power[band].max() > 0:
        pk = float(freq[int(np.argmax(np.where(band, power, 0.0)))])
    else:
        pk = float("nan")
    print(f"\n  qdot power above {STICK_FLOOR_HZ:g} Hz (the stick cannot go "
          f"there): {share:.1%}, peaked at {pk:.2f} Hz")
    if share > 0.35:
        print("     \033[93mRINGING: most of the joint motion is at a frequency "
              "nobody commanded.\033[0m")
        phase = 360.0 * pk * 1.5 * dt
        print(f"     At {pk:.2f} Hz one loop delay T_d is {phase:.0f} deg of "
              f"phase (1.5 ticks of {1e3*dt:.1f} ms plus the republish "
              f"thread).")
        if phase > 90.0:
            # dtc_20260830_005006.csv: 24.5 Hz, 107 deg, D_ang 26.4. The damper
            # is the source there -- it needs no input to pump, it feeds on the
            # motion it made -- so cutting K does nothing and cutting D does.
            print("     Past 90 deg a damper is NEGATIVE damping: -D*v applied "
                  "that late adds energy instead of removing it, and it needs "
                  "no input to keep going.")
            print("     Lower D: --dtc-damping-max, or the two "
                  "--dtc-damping-ratio flags. Do NOT reach for fn here, and "
                  "raising --dtc-tau-max only raises the amplitude.")
        else:
            # dtc_20260830_123040.csv: 8.45 Hz, 37 deg, at --dtc-fn-lin 4.0.
            # The damper is still removing energy; the loop is simply asking
            # for a bandwidth its sample rate and delay cannot deliver.
            print("     Under 90 deg the damper is still damping -- this is an "
                  "under-damped closed loop, not a delayed damper, so the "
                  "lever is fn and not D.")
            print(f"     Lower --dtc-fn-lin (K = (2*pi*fn)^2). This loop's "
                  f"usable crossover is about f_s/12 = {1.0/(12*dt):.0f} Hz, "
                  f"and a second-order target has to sit well under that; "
                  f"raising --dtc-hz is the only way to move the ceiling.")
    else:
        print("     no ringing (the motion is where the operator put it).")

    held = quiet
    if not held.any():
        print(f"  (no quiet ticks past {QUIET_SETTLE_S:g} s of coast-down)")
        return
    moved = float(np.percentile(np.linalg.norm(qd[held], axis=1), 95))
    print(f"  |qdot| p95 with the stick QUIET and settled: "
          f"{moved:.3f} rad/s ({held.mean():.0%} of live ticks, "
          f"{QUIET_SETTLE_S:g} s of coast-down dropped)")
    if moved <= QUIET_HOLD_RADPS:
        return

    # Motion while quiet is only a *hold failure* if nothing was commanding it.
    # It usually is not. controller.hold_pose_at re-bases the position onto the
    # live pose but keeps the field orientation commanded (set_hold_rotation),
    # deliberately, so the wrist goes on converging on the field between stick
    # motions instead of being handed the whole angular error at the next
    # re-arm. On dtc_20260830_130541.csv that read 0.069 rad/s quiet and
    # printed "THE HOLD IS NOT HOLDING" while the loop was doing exactly what
    # it was built to do -- |W_ang| was live and err_ang was 14-16 deg at every
    # re-arm edge. Drift is motion under *no* wrench; convergence is motion
    # under one.
    wl_h = np.linalg.norm(W[live][held, 3:6], axis=1)
    wa_h = np.linalg.norm(W[live][held, 0:3], axis=1)
    ang = float(np.median(wa_h))
    lin = float(np.median(wl_h))
    print(f"     while quiet: |W_ang| median {ang:.2f} N.m  "
          f"|W_lin| median {lin:.2f} N")
    if ang > QUIET_WRENCH_ANG_NM or lin > QUIET_WRENCH_LIN_N:
        print("     COMMANDED, not drift: the idle hold re-bases position but "
              "keeps the field orientation commanded, so the wrist is still "
              "closing err_ang while the stick is quiet.")
        print("     Read it as convergence speed, not as a failed hold. It is "
              "a fault only if it never settles -- check err_ang at the "
              "re-arm edges below: falling edge-to-edge = converging.")
    else:
        print("     \033[93mTHE HOLD IS NOT HOLDING: the arm keeps moving "
              "long after the stick was released, under no commanded "
              "wrench.\033[0m")


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
        roll(ea, n_deg, on & ok)

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


def roll(ea, n_deg, mask):
    """Split the angular error into tilt (= conformity) and roll about tool Z.

    These two halves fail for opposite reasons and the norm hides it. ``n_deg``
    is the angle between live tool +Z and the commanded field normal, so it IS
    the tilt; whatever ``|err_ang|`` carries beyond it is rotation *about* tool
    Z, which the field pins through the tangent and which nothing else reports.

    Roll is the axis the shaped law starves. Under Lambda the commanded moment
    scales with apparent inertia, spinning the tool about its own axis only
    turns wrist 3's rotor, and breakaway friction does not scale with either --
    so on dtc_20260830_130541.csv the tilt converged to 0.5 deg while the roll
    sat at 17.8 deg untouched for the whole hold, on 0.109 N.m of commanded
    moment. See LAMBDA_MIN_ANG_KGM2. A conformity number that looks excellent
    next to an err_ang several times larger is this, every time.
    """
    if not mask.any():
        return
    tilt = float(np.median(n_deg[mask]))
    tot = float(np.median(ea[mask]))
    # Orthogonal by construction only in the small-angle limit, which is where
    # this matters; the point is the split, not the third decimal.
    res = float(np.sqrt(max(tot ** 2 - tilt ** 2, 0.0)))
    print(f"  err_ang split deg  tilt (= conformity) {tilt:6.2f}"
          f"   roll about tool Z {res:6.2f}")
    if res > 5.0 and res > 2.0 * max(tilt, 1e-6):
        print("     \033[93mROLL STARVED: the angular error is mostly rotation "
              "about the tool axis, which the field pins through the tangent "
              "and Lambda scales to near nothing.\033[0m")
        print("     Check lamfloor in the console line. If it is 0 the floor "
              "is inert -- raise LAMBDA_MIN_ANG_KGM2; if it is high and the "
              "roll still will not close, wrist 3 is friction-bound and the "
              "lever is --dtc-joint-hold-kp, not a task gain.")



if __name__ == "__main__":
    main()
