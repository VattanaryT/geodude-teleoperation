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

# Kept in step with torque_backend by hand: this script must run without
# gafropy (and therefore without importing the package) on a log-only box.
LAMBDA_MIN_ANG_KGM2 = 0.02


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
# A peak above this is the closed loop, not the operator -- a SpaceMouse cannot
# reach it even in a burst, so it needs far less of the total power to be worth
# calling. See the ringing verdict for the two runs that set these.
RING_FLOOR_HZ = 5.0
RING_MIN_SHARE = 0.15
# Ticks of coast-down to drop after the stick is released. The arm is still
# decelerating there, and that transient is not a limit cycle: on
# dtc_20260830_123245.csv the "quiet" |qdot| p95 read 1.111 rad/s, which looked
# like a hold failure and was entirely the first 0.1 s after each release
# (median 1.03 rad/s under 0.1 s, 0.0000 past 0.5 s -- the same shape as a run
# with a clean hold, which differed only in how many short stick bursts it had).
QUIET_SETTLE_S = 0.25
# Floor on the worst-quiet-window mean before it is worth printing. Healthy
# runs on this rig read 0.08-0.11 rad/s at the very start of the record;
# dtc_20260830_141034.csv's limit cycle reads 0.75.
QUIET_BURST_RADPS = 0.25
# Quiet-stick |qdot| p95 above this is worth explaining at all.
QUIET_HOLD_RADPS = 0.05
# ...and above either of these, a commanded wrench explains it. The idle hold
# keeps the field *orientation* commanded on purpose, so the angular bound is
# the one that fires; both sit well above the wrench a truly idle arm carries
# (|W| medians on the quiet ticks of a clean run are ~0).
QUIET_WRENCH_ANG_NM = 0.5
QUIET_WRENCH_LIN_N = 2.0

# --- release response (the step test) ---------------------------------------
# The cleanest discriminator this rig has between "excited" and
# "self-sustaining", and it needs no instrumentation the logs do not already
# carry: command a motion, let go of the stick, and watch the envelope. A ring
# the operator is exciting collapses within a couple of hundred ms of the
# release; a ring the loop is feeding does not care that the stick stopped.
# Everything else in this file averages that answer away -- the quiet-stick
# percentile drops QUIET_SETTLE_S of coast-down and then pools every quiet tick
# in the run, so one release that ran away is 1% of the population.
#
# Replayed over every dtc log on this rig the separation is total: every
# release of every run decays to <0.05 rad/s except t=49.63 of
# dtc_20260830_141034.csv, which goes 0.368 -> 0.922 rad/s with tau riding
# 40 -> 50 N.m at 32.7 Hz. That is the workstation run, and the release is
# 0.16 s BEFORE the event the rail guard catches.
REL_MIN_S = 0.30
# Windows after the release, in seconds. The first skips the release tick
# itself (the command is still being blended out) and the second starts where
# a healthy coast-down is already over.
REL_EARLY = (0.02, 0.15)
REL_LATE = (0.15, 0.40)
# Below this the arm was simply already still and the ratio is noise on noise.
#
# **The floor must sit clear of whatever threshold defined the release.** Under
# --dtc-push-test the "driven" mask is itself |qdot| > PUSH_ACTIVE_RADPS, so a
# release is *by construction* the moment motion crosses that value and the
# post-release envelope sits right on it: dtc_20260902_154438.csv duly reported
# one release "self-sustaining" at 0.0121 -> 0.0226 rad/s, which is noise
# against a known runaway of 0.368 -> 0.922. _rel_floor() lifts it to 2.5x the
# mask threshold there, landing on 0.05 rad/s -- the value that separates every
# release in this repo's history, healthy under it and the two runs that hurt
# far above it.
REL_FLOOR_RADPS = 0.02


def _rel_floor():
    if _PUSH_TEST or _ACT_DEAD:
        return max(REL_FLOOR_RADPS, 2.5 * PUSH_ACTIVE_RADPS)
    return REL_FLOOR_RADPS
REL_DECAY_FRAC = 0.5
REL_GROW_FRAC = 0.8


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


# --dtc-push-test has no stick, so the `act` column is 0 for the whole run and
# every metric keyed on it reads the wrong thing: "quiet stick" pools the
# deliberate hand pushes (one run read |qdot| p95 = 1.001 rad/s, which looks
# like a catastrophic hold failure and is just the operator moving the arm),
# "release response" finds 0 releases and skips the step test, and the stall
# report never runs. But a push test *is* a step test -- push, let go, watch the
# envelope -- so the fix is to derive the mask from motion instead: the hand is
# on the arm exactly when the arm is moving. Above this joint speed counts as
# driven.
PUSH_ACTIVE_RADPS = 0.02
# Motion must stay under the threshold this long before a push counts as
# released, so a momentary zero-crossing mid-push is not a release.
PUSH_RELEASE_HOLD_S = 0.10
# Set from the log's own argv line. Module-level because the three sites that
# build an act mask are called from main() with different signatures and this
# is a single-run script.
_PUSH_TEST = False
_ACT_DEAD = False
# Below this fraction of ticks high, the `act` column is not a usable mask.
ACT_DEAD_FRAC = 0.01


def _is_push_test(meta):
    return any("--dtc-push-test" in line for line in meta)


def _act_is_dead(rows):
    """True when the `act` column never goes high, so it cannot be the mask.

    --dtc-push-test is one way to get here. The other -- and it cost a hardware
    run -- is any session where the operator drives the arm *by hand* instead of
    by stick: dtc_20260902_171136.csv was launched with --dtc-compliance-hold
    and the SpaceMouse was never touched, so `act` was 0 for all 6755 live
    ticks, the step test reported "0 stick releases" and skipped itself, and
    "quiet stick" pooled every hand push into a 0.620 rad/s p95. The flag is not
    the signal; a dead column is.
    """
    if "act" not in rows[0]:
        return True
    hot = sum(1 for r in rows if r["act"] == "1")
    # Not "never": dtc_20260902_171136.csv registered **one** non-idle stick
    # sample in 9311 ticks -- a knock, or the operator brushing the puck -- and
    # a strict `any()` handed the whole run back to a column that is 0.01% high.
    # Below this the column cannot be the mask whatever produced the one tick.
    return hot < max(1, int(ACT_DEAD_FRAC * len(rows)))


def _act_array(rows, live, dt, qd=None):
    """The "a command source is driving" mask, and what it was derived from.

    Normally the ``act`` column (field-follow's ``input_active``). Under
    ``--dtc-push-test`` there is no stick, so it is derived from joint motion --
    see PUSH_ACTIVE_RADPS.
    """
    if (_PUSH_TEST or _ACT_DEAD) and qd is not None:
        moving = np.abs(qd).max(axis=1) > PUSH_ACTIVE_RADPS
        hold = max(1, int(round(PUSH_RELEASE_HOLD_S / dt)))
        # Dilate forward: stay "driven" until motion has been absent for hold.
        act = moving.copy()
        run = 0
        for i in range(act.size):
            run = 0 if moving[i] else run + 1
            act[i] = moving[i] or run < hold
        return act, ("motion (act never goes high, so it is derived from "
                     "|qdot| -- hand-driven run or push test)")
    if "act" in rows[0]:
        return np.array([r["act"] == "1" for r in rows])[live], "act column"
    return np.ones(int(live.sum()), bool), "no act column (all ticks treated as driven)"


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
    global _PUSH_TEST, _ACT_DEAD
    _PUSH_TEST = _is_push_test(meta)
    _ACT_DEAD = _act_is_dead(rows)
    n = sum(1 for k in rows[0] if k.startswith("tau"))
    print(f"{path}: {len(rows)} ticks, {n} joints")
    if _ACT_DEAD:
        why = ("--dtc-push-test" if _PUSH_TEST
               else "the act column is 0 for the whole run (hand-driven?)")
        print(f"  {why}: \"driven\" is derived from joint motion "
              f"(|qdot| > {PUSH_ACTIVE_RADPS:g} rad/s). Each push is a step test; "
              "read the release response below.")
    for line in meta:
        print(f"  {line}")

    phase = np.array([r["phase"] for r in rows])
    for p in sorted(set(phase)):
        print(f"  phase {p:<10} {int((phase == p).sum()):6d}  {(phase == p).mean():6.1%}")
    dt = np.array([float(r["dt_ms"]) for r in rows if r["dt_ms"]])
    print(f"  dt_ms  median {q(dt,50):6.2f}  p95 {q(dt,95):6.2f}  max {dt.max():6.2f}"
          f"   -> {1e3/max(q(dt,50),1e-9):.0f} Hz median")

    if all(f"q{j}" in rows[0] for j in range(n)):
        qm = np.degrees(np.array(
            [[float(r[f"q{j}"]) for j in range(n)] for r in rows]))
        lo, hi = qm.min(axis=0), qm.max(axis=0)
        print("  q deg  " + "  ".join(
            f"j{j}:{lo[j]:.0f}" + ("" if hi[j] - lo[j] < 1.0 else f"..{hi[j]:.0f}")
            for j in range(n)))
        # Two runs at identical gains are only comparable in the same arm
        # configuration, and "same pose" is easy to assume wrongly: 114126 was
        # run as a reproduce of 171136 and sat in the opposite wrist branch
        # (j3 -16 vs -207, j4 +78 vs -80, j5 +3 vs -180), never entering the
        # region where the burst happened. Print it so the next comparison
        # starts from the configuration, not from the flags.
        print("     compare runs only against the same branch: a flipped wrist "
              "is a different plant at the same gains.")

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

    # Before anything that averages over the run: a 1.4 s event that swings
    # the arm into a workstation is invisible in every percentile below.
    railed = rail_oscillation(rows, live, tau, n, meta)
    grew = release_response(rows, live, tau, n)
    burst_oscillation(rows, live, n, _tau_max_from_meta(meta, n),
                      w_lin=(np.array([[float(r[f"w{j}"]) for j in (3, 4, 5)]
                                       for r in rows])[live]
                             if "w3" in rows[0] else None))
    wrist_travel(rows, live, n)
    lambda_floor(rows, live)
    oscillation(rows, live, W, n, railed, grew)
    stall(rows, live, W, n)
    tracking(rows, live, W)


def lambda_floor(rows, live):
    """How often Lambda's angular eigenvalue floor was load-bearing.

    LAMBDA_MIN_ANG_KGM2 is sized against breakaway friction, not against any
    measured inertia, so "does it bind" is an empirical question and the answer
    belongs in the run record rather than in a 1 Hz console counter. It also
    sets the *bandwidth* of the axis it floors: a clamped Lambda is not that
    axis's real apparent inertia, so its true closed-loop frequency runs
    sqrt(floor / true) above the fn asked for -- ~3.2x on the roll axis here,
    which is why --dtc-fn-lin 4.0 rang at 14.43 Hz. See
    WRENCH_FN_ANG_MAX_LAMBDA.
    """
    if "lam_ang" not in rows[0]:
        return
    v = np.array([float(r["lam_ang"]) if r["lam_ang"] else np.nan
                  for r in rows])[live]
    ok = np.isfinite(v)
    if not ok.any():
        return
    v = v[ok]
    bound = float((v < LAMBDA_MIN_ANG_KGM2).mean())
    print(f"\n  Lambda angular min eig kg.m^2  median {q(v,50):.5f}"
          f"  p5 {q(v,5):.5f}   floor {LAMBDA_MIN_ANG_KGM2:g} bound on "
          f"{bound:.1%} of live ticks")
    if bound > 0.5:
        infl = float(np.sqrt(LAMBDA_MIN_ANG_KGM2 / max(q(v, 50), 1e-9)))
        print(f"     the floor is carrying the roll axis. Its real loop runs "
              f"~{infl:.1f}x the angular fn asked for, so read the peak "
              f"frequency below against that, not against fn.")
    elif bound < 0.05:
        print("     the floor is inert -- a stuck roll is not this, and "
              "raising it would change nothing.")


# Rail-to-rail sign alternation, the one signature a limit cycle has and a
# stall does not. Kept identical to RailOscillationGuard in torque_backend.py
# -- the loop's cut-out and this report must never disagree about what a
# limit cycle is.
RAIL_OSC_WINDOW_S = 0.3
RAIL_OSC_FLIPS = 12


def rail_oscillation(rows, live, tau, n, meta):
    """Did the arm throw itself at +/-tau_max, and when?

    Every other section here is a whole-run aggregate, and that is exactly how
    dtc_20260830_141034.csv passed. Over 51 s it read "no ringing" (33.2% of
    the qdot power, peaked at 15.03 Hz -- the operator's own band) and
    "|qdot| p95 with the stick QUIET: 0.000 rad/s", because 30 s of that run
    was a motionless idle that swamps any percentile. The event was 1.4 s long:
    from t=49.79 tau alternated +50/-50 N.m on j1 AND j2 -- the 50 N.m shoulder
    and elbow, which rail in no other run on this rig -- at **32.1 Hz**, with
    the stick quiet and the pose error at 0.0 mm, and the arm swung into the
    workstation. A percentile cannot see a 1.4 s event in a 51 s run. This
    section is time-resolved for that reason.

    Why sign alternation and not amplitude: a joint stalled against breakaway
    friction sits on *one* sign of tau_max for as long as the operator leans on
    it, which is correct. Reversing at the rail several times in 0.3 s is the
    loop fighting its own delay, and tau_max is then the only thing bounding
    how hard the arm is thrown.
    """
    tmax = _tau_max_from_meta(meta, n)
    t = np.array([float(r["t"]) for r in rows])
    print(f"\n  rail oscillation (sign flips at +/-tau_max within "
          f"{RAIL_OSC_WINDOW_S:g} s)")
    worst = 0
    worst_j = -1
    worst_t = float("nan")
    for j in range(n):
        idx = np.where(np.abs(tau[:, j]) >= 0.99 * tmax[j])[0]
        if idx.size < 2:
            print(f"   j{j}   {idx.size:5d} rail ticks       -")
            continue
        sign = np.sign(tau[idx, j])
        ft = t[idx[np.where(np.diff(sign) != 0)[0] + 1]]
        best, best_t = 0, float("nan")
        for x in ft:
            c = int(((ft >= x) & (ft < x + RAIL_OSC_WINDOW_S)).sum())
            if c > best:
                best, best_t = c, x
        flag = "  <-- LIMIT CYCLE" if best >= RAIL_OSC_FLIPS else ""
        print(f"   j{j}   {idx.size:5d} rail ticks   {int(ft.size):4d} sign flips"
              f"   worst {best:3d} in {RAIL_OSC_WINDOW_S:g} s at t={best_t:7.2f}{flag}")
        if best > worst:
            worst, worst_j, worst_t = best, j, best_t
    if worst >= RAIL_OSC_FLIPS:
        print(f"     \033[91mTHE LOOP THREW THE ARM: j{worst_j} reversed at "
              f"+/-tau_max {worst} times in {RAIL_OSC_WINDOW_S:g} s at "
              f"t={worst_t:.2f}.\033[0m")
        print("     tau_max is the ONLY bound on a limit cycle's amplitude: "
              "err_clip bounds the spring half of the wrench and the damping "
              "half -Lambda*D*v has no clip at all.")
        print("     Lower --dtc-damping-ratio first, then --dtc-fn-lin. Do NOT "
              "raise --dtc-damping-max, and do NOT read the ringing section "
              "below as permission to -- it averages over the whole run.")
        _rail_window(rows, live, t, worst_t)
    elif worst:
        print(f"     worst is {worst} on j{worst_j}, under the {RAIL_OSC_FLIPS} "
              f"the loop cuts out at. On this rig healthy runs top out at 9 and "
              f"only on the 6 N.m wrist joints.")
    else:
        print("     none: no joint ever reversed sign at its rail.")
    return worst


def _rail_window(rows, live, t, t0):
    """What the stick and the pose error were doing during the cycle."""
    m = (t >= t0) & (t < t0 + 1.0) & live
    if not m.any():
        return
    sm = np.array([[float(r[f"sm{i}"]) for i in range(6)] for r in rows])
    e = np.array([[float(r[f"e{a}"]) for a in "xyz"] for r in rows])
    qd = np.array([[float(r[f"qd{i}"]) for i in range(6)] for r in rows])
    quiet = np.linalg.norm(sm[m], axis=1) < 1e-9
    print(f"     in the second that followed: stick quiet on "
          f"{quiet.mean():.0%} of ticks, |e_lin| max "
          f"{1e3 * np.linalg.norm(e[m], axis=1).max():.1f} mm, |qdot| max "
          f"{np.linalg.norm(qd[m], axis=1).max():.2f} rad/s")
    if quiet.mean() > 0.5:
        print("     A cycle that runs with the stick quiet and the error at "
              "zero is the damping term feeding itself. It is NOT operator "
              "excitation and more D will make it worse.")


def _tau_max_from_meta(meta, n):
    """tau_max off the log's own gain header, or the shipped default."""
    for line in meta:
        if "tau_max=" in line:
            try:
                v = [float(x) for x in
                     line.split("tau_max=")[1].split()[0].split(",")]
                if len(v) >= n:
                    return np.array(v[:n], dtype=float)
            except Exception:
                pass
    return np.array([50.0] * 3 + [6.0, 6.0, 30.0])[:n]


def peak_q(freq, power, pk_idx, floor_hz):
    """Half-power Q of the spectral line at ``pk_idx``, or nan.

    Band share cannot A/B a damping change and has now proved it: at an
    identical D_lin = 12 and fn_lin = 4.0, dtc_20260830_134133.csv read 22.7%
    of the qdot power above 2 Hz and dtc_20260830_135527.csv read 11.8%. The
    difference is how hard the operator drove, because a *share* divides the
    ring by the total motion and the operator owns the denominator.

    A mode's Q does not: it is the width of the resonance, so it measures how
    damped the mode is rather than how hard it was hit. `Q = f_pk / df_3dB`
    and `zeta_mode ~= 1/(2Q)`, so more real damping at 15 Hz has to show up as
    a lower Q whether or not the run happened to be a busy one. That is the
    number to put beside a --dtc-damping-max change.

    Returns nan when the line does not fall to half power inside the band --
    a shoulder rather than a peak, which is not a resonance to measure.
    """
    if pk_idx <= 0 or pk_idx >= power.size - 1:
        return float("nan")
    half = power[pk_idx] / 2.0
    lo = pk_idx
    while lo > 0 and power[lo] > half and freq[lo] > floor_hz:
        lo -= 1
    hi = pk_idx
    while hi < power.size - 1 and power[hi] > half:
        hi += 1
    if power[lo] > half or power[hi] > half:
        return float("nan")

    def _cross(i0, i1):
        # Linear interpolation onto the half-power level between two bins.
        p0, p1 = power[i0], power[i1]
        if p1 == p0:
            return freq[i1]
        return freq[i0] + (freq[i1] - freq[i0]) * (half - p0) / (p1 - p0)

    f_lo = _cross(lo, lo + 1)
    f_hi = _cross(hi, hi - 1)
    width = f_hi - f_lo
    if not np.isfinite(width) or width <= 0:
        return float("nan")
    return float(freq[pk_idx] / width)


def release_response(rows, live, tau, n):
    """The step test: command a motion, let go, watch the envelope decay.

    Everything else in this file is a statistic over the whole run, and a
    statistic cannot answer the one question that decides which knob to turn.
    Two failures look identical while the operator is driving:

      - **excited** -- the loop is stable, the operator's own motion is
        exciting a real structural mode, and it bleeds off once they stop.
        The fix is MORE damping (--dtc-damping-max / --dtc-damping-ratio).
      - **self-sustaining** -- the loop is pumping its own delay and needs no
        input at all. The fix is a SLOWER loop (--dtc-fn-lin, K = (2*pi*fn)^2);
        adding damping to this one makes it worse, because past 90 deg of lag
        -D*v IS the source.

    Releasing the stick separates them in one motion, with no new
    instrumentation: after the falling edge of `act` the operator is
    contributing nothing, so whatever is left is the loop's own.

    Read the table as: env_early is the p95 of |qdot| over REL_EARLY after the
    release and env_late the same over REL_LATE. A ratio well under 1 is a
    decay; at or above REL_GROW_FRAC with real amplitude, the loop is feeding
    itself. tau alongside it, because a growing envelope riding tau_max is the
    shape of the run that swung the arm into the workstation.

    That run is why the windows are this short. dtc_20260830_141034.csv
    releases at t=49.63 and the cycle starts at t=49.79 -- 0.16 s later, inside
    a quiet stretch only 0.54 s long, because the operator pressed again at
    50.17. A 0.6 s minimum quiet window drops that release entirely and the
    section reports the run as clean; REL_MIN_S of 0.30 catches it, and it is
    the ONLY release in any log on this rig that does not decay (0.368 ->
    0.922 rad/s, tau 40 -> 50 N.m, 32.7 Hz). It also fires 0.56 s before the
    rail guard and 1.1 s before the 167 N peak.

    Returns the number of releases whose envelope did not decay.
    """
    if f"qd{n - 1}" not in rows[0] or ("act" not in rows[0] and not _PUSH_TEST):
        print("\n  (no qd*/act columns -- log predates them; no release report)")
        return 0
    t = np.array([float(r["t"]) for r in rows])[live]
    if t.size < 64:
        return 0
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return 0
    qd = np.array([[float(r[f"qd{j}"]) for j in range(n)] for r in rows])[live]
    tau = tau[live]
    act, act_src = _act_array(rows, live, dt, qd)
    speed = np.linalg.norm(qd, axis=1)
    tpk = np.abs(tau).max(axis=1)

    fall = np.where(act[:-1] & ~act[1:])[0] + 1
    what = "hand releases" if _PUSH_TEST else "stick releases"
    print(f"\n  release response ({fall.size} {what}; "
          f"the step test -- does the motion die when you let go?)")
    print(f"     driven mask: {act_src}")
    if _PUSH_TEST:
        print(f"     self-sustaining needs |qdot| late > {_rel_floor():.3f} rad/s "
              f"(2.5x the mask threshold), not just a rising ratio")
    if fall.size == 0:
        print("     (never released -- no step test in this run)")
        return 0
    print("     t_rel   quiet_s   f_pk Hz   |qdot| early    late   ratio"
          "    |tau| early   late   verdict")

    def env(sl, ts, lo, hi, x):
        m = (ts >= lo) & (ts < hi)
        return float(np.percentile(x[sl][m], 95)) if int(m.sum()) > 3 else float("nan")

    grew, shown, ratios, grew_f = 0, 0, [], []
    for i in fall:
        j = i
        while j < act.size and not act[j]:
            j += 1
        quiet_s = float(t[min(j, t.size - 1)] - t[i])
        if quiet_s < REL_MIN_S:
            continue
        sl = slice(i, j)
        ts = t[sl] - t[i]
        e0 = env(sl, ts, *REL_EARLY, speed)
        e1 = env(sl, ts, *REL_LATE, speed)
        t0 = env(sl, ts, *REL_EARLY, tpk)
        t1 = env(sl, ts, *REL_LATE, tpk)
        if not (np.isfinite(e0) and np.isfinite(e1)):
            continue
        seg = qd[sl]
        w = np.hanning(seg.shape[0])
        f2 = np.fft.rfftfreq(seg.shape[0], dt)
        P = np.zeros(f2.size)
        for k in range(n):
            x = seg[:, k]
            P += np.abs(np.fft.rfft((x - x.mean()) * w)) ** 2
        P[0] = 0.0
        b = (f2 >= RING_FLOOR_HZ) & (f2 <= min(40.0, 0.45 / dt))
        f_pk = (float(f2[int(np.argmax(np.where(b, P, 0.0)))])
                if b.any() and P[b].max() > 0 else float("nan"))
        ratio = e1 / e0 if e0 > 1e-9 else float("nan")
        # The floor first: a release from an already-still arm has a ratio of
        # noise over noise and must not be called a limit cycle.
        if e1 < _rel_floor() or (np.isfinite(ratio) and ratio < REL_DECAY_FRAC):
            verdict = "decays"
        elif np.isfinite(ratio) and ratio >= REL_GROW_FRAC:
            verdict = "\033[93mFLAT/GROWS\033[0m"
            grew += 1
            grew_f.append(f_pk)
        else:
            verdict = "slow"
        ratios.append(ratio)
        print(f"   {t[i]:8.2f}  {quiet_s:7.2f}  {f_pk:7.2f}   {e0:11.4f} "
              f"{e1:7.4f}  {ratio:6.2f}    {t0:9.2f}  {t1:6.2f}   {verdict}")
        shown += 1

    if shown == 0:
        print(f"     (no release held the stick quiet for {REL_MIN_S:g} s -- "
              "let go for half a second and re-run)")
        return 0
    if grew:
        print(f"     \033[93mSELF-SUSTAINING on {grew} of {shown} releases: "
              "the motion outlives the operator, so the loop is the "
              "source.\033[0m")
        # Which knob depends on where the runaway sits against f_90 = 1/(4T),
        # the frequency at which the damper's own output is a quarter cycle
        # late and stops opposing the motion. Above it, D is literally the
        # source and only D fixes it -- dtc_20260830_141034.csv ran away at
        # 32.7 Hz against an f_90 of 33 Hz, at D_lin 19.1. Below it the damper
        # is still damping and the loop is simply asking for a bandwidth its
        # delay cannot deliver, which is an fn problem.
        f90 = 1.0 / (4.0 * 1.5 * dt)
        hot = [f for f in grew_f if np.isfinite(f)]
        if hot and max(hot) > 0.7 * f90:
            print(f"     It ran away at {max(hot):.1f} Hz against "
                  f"f_90 = 1/(4T) = {f90:.0f} Hz, where -D*v is a quarter "
                  f"cycle late and becomes a SOURCE. Lower D "
                  f"(--dtc-damping-max); fn will not touch this one.")
        else:
            print("     Lower --dtc-fn-lin. Do NOT add damping -- a loop that "
                  "pumps itself is not short of D, and more D is more pump.")
    else:
        med = float(np.nanmedian(ratios))
        print(f"     EXCITED, not self-sustaining: every release decays "
              f"(median envelope ratio {med:.2f} over "
              f"{REL_LATE[0]:g}-{REL_LATE[1]:g} s). Whatever rings, rings "
              f"because the operator is driving it.")
        print("     So damping is the lever here, not fn -- but read the "
              "phase in the oscillation section first, and mind the measured "
              "ceiling (D_lin 15.08 clean, 19.1 catastrophic at 200 Hz).")
        print("     A clean step test is necessary, not sufficient: it says "
              "nothing about a cycle that only appears mid-drive, and it "
              "tests only the axes you actually moved. Repeat it as a pure "
              "rotation.")
    return grew


def oscillation(rows, live, W, n, railed=0, grew=0):
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
    act, act_src = _act_array(rows, live, dt, qd)
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
    # Needed by the ringing verdict below as well as by its own section: a ring
    # that survives the operator letting go and one that dies with them take
    # opposite fixes.
    moved = (float(np.percentile(np.linalg.norm(qd[quiet], axis=1), 95))
             if quiet.any() else float("nan"))
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
    # ...and again above RING_FLOOR_HZ, because the 2 Hz band edge is not a
    # wall. A SpaceMouse's spectrum does not stop at the floor, it leaks over
    # it, so the 2-40 Hz peak lands in the FIRST BIN above the floor on any
    # briskly-driven run -- dtc_20260830_150044.csv printed "35.0%, peaked at
    # 2.01 Hz", which reads as "that is just the operator", while 21.5% of its
    # motion sat at 12 Hz and the operator was watching the arm buzz. Worse,
    # `pk > RING_FLOOR_HZ` gated the Q measurement, so the one number that can
    # A/B a damping change was silently skipped on exactly the run that needed
    # it. Measured across every log here the 5-40 Hz share separates cleanly
    # where 2-40 does not: healthy runs 2.9-9.4% (including 132204, whose
    # 2-40 share is 15.0% and whose 5-40 share is 5.4%), ringing runs 15.1%
    # and up.
    ring = (freq >= RING_FLOOR_HZ) & (freq <= min(40.0, 0.45 / dt))
    ring_k = (int(np.argmax(np.where(ring, power, 0.0)))
              if ring.any() and power[ring].max() > 0 else -1)
    ring_share = float(power[ring].sum() / total)
    ring_pk = float(freq[ring_k]) if ring_k >= 0 else float("nan")
    print(f"  qdot power above {RING_FLOOR_HZ:g} Hz (nothing the operator "
          f"does lands here): {ring_share:.1%}, peaked at {ring_pk:.2f} Hz")
    is_ring = ring_share > RING_MIN_SHARE
    # The peak the rest of this section reasons about: the ring's, whenever
    # there is a ring, and the whole-band one otherwise.
    if is_ring:
        pk, band, k_pk = ring_pk, ring, ring_k
    else:
        k_pk = int(np.argmax(np.where(band, power, 0.0))) if band.any() else -1
    # Share is drive-dependent; Q is not. See peak_q.
    if np.isfinite(pk) and pk > RING_FLOOR_HZ and k_pk >= 0:
        qf = peak_q(freq, power, k_pk, STICK_FLOOR_HZ)
        if np.isfinite(qf):
            width = pk / qf
            print(f"     peak Q {qf:.1f} (half-power width {width:.2f} Hz), "
                  f"so the mode's own zeta is ~{1.0 / (2.0 * qf):.3f}. "
                  f"A/B a damping change on THIS, not on the share above: at "
                  f"an identical D_lin the share ran 22.7% and 11.8% on two "
                  f"runs, because the operator owns its denominator.")
            # A Hann window's own main lobe is 1.44/T wide at half power, so a
            # line narrower than that is measuring the record length, not the
            # mode. Say so rather than quoting a Q the run cannot support.
            res = 1.44 / (t.size * dt)
            if width < 2.0 * res:
                print(f"     ...but {width:.2f} Hz is within 2x the window's "
                      f"own {res:.2f} Hz resolution over {t.size * dt:.0f} s, "
                      f"so read that Q as a LOWER bound. A longer run "
                      f"separates them.")
    # Share alone is the wrong discriminator and it has now missed twice, in
    # both directions. A share threshold cannot separate "the operator drove
    # briskly" from "the arm has a line in it": dtc_20260830_132204.csv is
    # 15.0% peaked at 2.59 Hz (the stick, healthy) and dtc_20260830_132604.csv
    # is 33.7% peaked at **14.43 Hz** (a real 14 Hz line on j0/j4, with j3
    # clipping 20.4% by the last quartile) -- and the old `share > 0.35` called
    # the second one clean by 1.3 points. dtc_20260830_005006.csv, the run this
    # section was written for, was 34% and also squeaked under. What separates
    # them is WHERE the power is: nothing the operator does lands above
    # RING_FLOOR_HZ, so a peak up there with a real share is the loop talking
    # to itself no matter how much of the total it is.
    ringing = is_ring or share > 0.35
    if ringing:
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
                  "under-damped closed loop, not a delayed damper, so D is "
                  "still a lever here and not the disease.")
            # Excited vs self-sustaining. Under 90 deg the damper cannot
            # sustain anything on its own, so the quiet number says which one
            # this is -- and they take opposite fixes. dtc_20260830_134133.csv
            # rings at 14.99 Hz on 22.7% of the power and is 0.001 rad/s quiet:
            # it is stable, the operator is exciting it, and it responds to
            # MORE damping. dtc_20260830_132604.csv rang at 14.43 Hz and held
            # 0.215 rad/s quiet, which is a loop feeding itself.
            if railed >= RAIL_OSC_FLIPS:
                # This branch used to say "raise --dtc-damping-max first", and
                # following it is what swung the arm into the workstation. The
                # whole-run peak is the operator's band; the rail section
                # already found a cycle nobody commanded.
                print("     ...but the rail section above found a real limit "
                      "cycle in this run, and THAT is the fault. This peak is "
                      "a whole-run average and cannot see it. Lower D.")
            elif grew:
                # The release section watched the envelope after the operator
                # let go; this branch is about to average over every quiet tick
                # in the run, and one runaway release is 1% of that population.
                print("     ...but the release section above found "
                      f"{grew} release(s) whose motion did NOT decay. Believe "
                      "that over the pooled quiet percentile: lower "
                      "--dtc-fn-lin, do not add D.")
            elif np.isfinite(moved) and moved <= QUIET_HOLD_RADPS:
                print(f"     EXCITED, not self-sustaining: it dies when the "
                      f"stick stops ({moved:.3f} rad/s quiet). The peak is at "
                      f"{phase:.0f} deg, so D would still buy "
                      f"cos({phase:.0f}) = {np.cos(np.radians(phase)):.2f} of "
                      f"real damping there.")
                print("     Careful with that. The measured ceiling on this "
                      "rig at 200 Hz is D_lin 15.08 clean "
                      "(dtc_20260830_135043.csv) and 19.1 catastrophic "
                      "(dtc_20260830_141034.csv, a 32 Hz cycle at "
                      "+/-tau_max), and 32 Hz is f_90 for this loop -- raising "
                      "the rate sweeps f_90 up into new structure rather than "
                      "away from it. A small step, a short run, and read the "
                      "rail section before the next one.")
            else:
                print("     It survives the stick going quiet, so the loop is "
                      "feeding itself: lower --dtc-fn-lin (K = (2*pi*fn)^2) "
                      "rather than adding damping to it.")
            print(f"     This loop's usable crossover is about f_s/12 = "
                  f"{1.0/(12*dt):.0f} Hz and 90 deg of lag lands at "
                  f"{1.0/(4*1.5*dt):.0f} Hz; a second-order target has to sit "
                  f"well under the first. Raising --dtc-hz moves both.")
    else:
        print("     no ringing (the motion is where the operator put it).")

    held = quiet
    if not held.any():
        print(f"  (no quiet ticks past {QUIET_SETTLE_S:g} s of coast-down)")
        return
    print(f"  |qdot| p95 with the stick QUIET and settled: "
          f"{moved:.3f} rad/s ({held.mean():.0%} of live ticks, "
          f"{QUIET_SETTLE_S:g} s of coast-down dropped)")
    # A percentile over the whole run cannot see a short event, and that is not
    # hypothetical: dtc_20260830_141034.csv reads 0.000 rad/s here while
    # spending 1.4 s throwing the arm at +/-tau_max, because 30 s of its 51 s
    # was a motionless idle. Report the worst window too.
    t_all = np.array([float(r["t"]) for r in rows])[live]
    speed = np.linalg.norm(qd, axis=1)
    # Half a window's worth of ticks, so the first and last few samples of the
    # run cannot name a "burst" out of two points.
    need = max(3, int(0.5 * RAIL_OSC_WINDOW_S / max(dt, 1e-6)))
    burst, burst_t = 0.0, float("nan")
    for k in np.where(held)[0]:
        m = held & (t_all >= t_all[k]) & (t_all < t_all[k] + RAIL_OSC_WINDOW_S)
        if int(m.sum()) < need:
            continue
        v = float(speed[m].mean())
        if v > burst:
            burst, burst_t = v, t_all[k]
    # QUIET_BURST_RADPS, not QUIET_HOLD_RADPS: this is a mean over a window
    # rather than a percentile, and healthy runs on this rig show 0.08-0.11
    # rad/s of it at the edges of the record. The incident window is 0.75.
    if burst > max(QUIET_BURST_RADPS, 5.0 * moved):
        print(f"     ...but the worst quiet {RAIL_OSC_WINDOW_S:g} s window "
              f"averages {burst:.3f} rad/s at t={burst_t:.2f}. The percentile "
              f"above is hiding a short event -- go look at it.")
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
    elif ringing:
        # dtc_20260830_132604.csv reads both at once, and only one of them is a
        # bug you can fix at the hold: a ringing loop does not stop ringing
        # because the operator let go, so its quiet ticks are the oscillation
        # continuing, not the hold leaking. Fix the ring first and re-read.
        print("     this is the ringing above continuing through the quiet "
              f"ticks ({pk:.1f} Hz), not a leaking hold. Fix the ring first.")
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
    if f"qd{n - 1}" not in rows[0] or ("act" not in rows[0] and not _PUSH_TEST):
        print("\n  (no qd*/act columns -- log predates them; no stall report)")
        return
    qd = np.array([[float(r[f"qd{j}"]) for j in range(n)] for r in rows])
    t_all = np.array([float(r["t"]) for r in rows])
    dt_all = float(np.median(np.diff(t_all[live]))) if live.sum() > 2 else 0.005
    act_live, _ = _act_array(rows, live, dt_all, qd[live])
    act = np.zeros(len(rows), bool)
    act[np.where(live)[0]] = act_live
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
    n_joints = sum(1 for k in rows[0] if k.startswith("tau"))
    tl = t[live]
    dt = float(np.median(np.diff(tl))) if tl.size > 2 else 0.005
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.005
    qd_live = np.array(
        [[float(r[f"qd{k}"]) for k in range(n_joints)] for r in rows])[live]
    act_live, _ = _act_array(rows, live, dt, qd_live)
    act = np.zeros(len(rows), bool)
    act[np.where(live)[0]] = act_live
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
        print("     Check the wrist travel line above, and lamfloor in the "
              "console line. If lamfloor is 0 the floor is inert -- raise "
              "LAMBDA_MIN_ANG_KGM2. If it binds and the wrist still never "
              "moves, the command is already at the floor's ceiling and "
              "breakaway is above it: that is --dtc-stiction-assist-nm, a "
              "feedforward, NOT a task gain and NOT --dtc-joint-hold-kp.")


def q95(a):
    return q(a, 95)



# Sliding window for burst_oscillation. Long enough to resolve a 20 Hz line
# (10 cycles), short enough that a 1 s event is not averaged away.
BURST_WIN_S = 0.5
# A burst counts when this much of the window's qdot power sits above
# BURST_FLOOR_HZ *and* the arm is actually moving. Nothing an operator or a
# structure driven by one puts a third of its energy above 20 Hz.
BURST_MIN_SHARE = 0.30
BURST_FLOOR_HZ = 20.0
BURST_MIN_QDOT = 0.10


def burst_oscillation(rows, live, n, tau_max, w_lin=None):
    """Time-resolved limit-cycle detector that does NOT need the torque rail.

    ``rail_oscillation`` above only sees a cycle that reaches ``+/-tau_max``,
    which makes its sensitivity *inversely* tied to a number the operator
    raises for unrelated reasons. dtc_20260902_171136.csv is the case:
    ``--dtc-tau-max 140 140 140 30 30 30`` ran a 30 Hz cycle that peaked at
    **83.24 N.m on j1 and 1.33 rad/s** and never came near the 140 rail, so the
    guard read 0 flips, the aggregate 5-40 Hz share read 8.0% (under
    RING_MIN_SHARE) and this file printed "no ringing". The operator hit the
    emergency stop. At the previously-tested tau_max of 50 that same cycle
    would have railed j1 and cut out on its own.

    Every aggregate here is blind to a one-second event -- the same lesson as
    the workstation run, which is why rail_oscillation exists. This is the same
    idea with the rail taken out of it: slide a window, and ask how much of the
    motion is at a frequency no operator can produce.
    """
    if f"qd{n - 1}" not in rows[0]:
        return
    t = np.array([float(r["t"]) for r in rows])[live]
    if t.size < 128:
        return
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return
    qd = np.array([[float(r[f"qd{j}"]) for j in range(n)] for r in rows])[live]
    tau = np.array([[float(r[f"tau{j}"]) for j in range(n)] for r in rows])[live]
    w = int(round(BURST_WIN_S / dt))
    if w < 32 or w > t.size:
        return
    freq = np.fft.rfftfreq(w, dt)
    hi = freq > BURST_FLOOR_HZ
    if not hi.any():
        return
    win = np.hanning(w)
    f90 = 1.0 / (4.0 * 1.5 * dt)
    hits = []
    for s0 in range(0, t.size - w, max(1, w // 2)):
        seg = qd[s0:s0 + w]
        if np.abs(seg).max() < BURST_MIN_QDOT:
            continue
        P = np.zeros(freq.size)
        for k in range(n):
            P += np.abs(np.fft.rfft((seg[:, k] - seg[:, k].mean()) * win)) ** 2
        P[0] = 0.0
        tot = P.sum()
        if tot <= 0:
            continue
        share = float(P[hi].sum() / tot)
        if share < BURST_MIN_SHARE:
            continue
        f_pk = float(freq[hi][int(np.argmax(P[hi]))])
        seg_tau = np.abs(tau[s0:s0 + w])
        j = int(seg_tau.max(axis=0).argmax())
        hits.append((float(t[s0]), share, f_pk, float(np.abs(seg).max()),
                     float(seg_tau.max()), j))
    # Shoulder leverage: how much of J.T lands on the joint that carries this
    # rig's 30 Hz mode. Measured across four runs it orders them *exactly* by
    # whether they burst -- 0.206 and 0.312 clean, 0.382 and 0.438 burst -- so
    # it is the loop gain the instability actually scales with, and it is the
    # number that says whether a "clean" run was clean or merely gentle.
    wl = np.linalg.norm(w_lin, axis=1) if w_lin is not None else None
    if wl is not None:
        m = wl > 10.0
        if m.any():
            lev = np.abs(tau[m, 1]) / wl[m]
            print(f"\n  shoulder leverage |tau1|/|W_lin|  median {np.median(lev):.3f}"
                  f"  p95 {np.percentile(lev, 95):.3f} m")
            print(f"     the loop gain the 30 Hz mode scales with. Measured on "
                  f"this rig: 0.21/0.31 clean, 0.38/0.44 burst. A clean run "
                  f"under ~0.35 has not been tested where it fails.")
    print(f"\n  burst oscillation ({BURST_WIN_S:g} s windows, "
          f">{BURST_FLOOR_HZ:g} Hz share, rail-independent)")
    if not hits:
        print("     none: no window put a third of its motion above "
              f"{BURST_FLOOR_HZ:g} Hz while moving.")
        return
    print("        t    >20Hz    f_pk   |qdot|max   |tau|max  joint")
    for ts, share, f_pk, qm, tm, j in hits[:12]:
        print(f"   {ts:7.2f}  {share:6.1%}  {f_pk:6.1f}Hz   {qm:8.3f}   "
              f"{tm:8.2f}   j{j}")
    worst = max(hits, key=lambda h: h[4])
    tm_j = float(np.asarray(tau_max).reshape(-1)[worst[5]])
    print(f"\n     \033[93m{len(hits)} burst window(s): the arm was moving at a "
          f"frequency no operator produces.\033[0m")
    print(f"     worst: {worst[4]:.1f} N.m on j{worst[5]} at {worst[2]:.0f} Hz, "
          f"|qdot| {worst[3]:.2f} rad/s, against tau_max {tm_j:g} N.m "
          f"({worst[4] / max(tm_j, 1e-9):.0%} of the rail)")
    if worst[4] < 0.9 * tm_j:
        print("     \033[93mThis never reached the rail, so RailOscillationGuard "
              "could not fire and rail_oscillation above reads clean. Raising "
              "--dtc-tau-max raises the amplitude this can reach before "
              "anything cuts out.\033[0m")
    if worst[2] > 0.7 * f90:
        print(f"     {worst[2]:.0f} Hz is at f_90 = 1/(4T) = {f90:.0f} Hz, where "
              "the damper's own output is a quarter cycle late and contributes "
              "almost nothing either way (cos 83 deg = 0.13 at 30 Hz).")
        # Measured, and it overrules the phase argument that used to live here.
        # Eight runs on this rig separate PERFECTLY on shoulder leverage and not
        # at all on D: clean at 0.206/0.282/0.312/0.316/0.342 with D_lin
        # 8,9,10,12 mixed through them, and bursting at 0.382/0.386/0.438 with
        # D_lin 10 and 12. D changed the severity a little (83% -> 75% at the
        # same leverage) and never changed which side of the boundary a run
        # landed on. So "lower D" is not the advice; loop gain is
        # leverage * K, leverage is the arm's geometry, so K is the knob left.
        print("     \033[93mMeasured on this rig, D in 8..12 does NOT move the "
              "boundary -- eight runs separate on shoulder leverage alone "
              "(clean <=0.34, burst >=0.38) with D mixed through both sides. "
              "Loop gain is leverage * K and leverage is geometry, so the "
              "untested lever is K: lower --dtc-fn-lin.\033[0m")
    else:
        print(f"     {worst[2]:.0f} Hz is well under f_90 = {f90:.0f} Hz, so the "
              "damper is still damping and this is a bandwidth problem: lower "
              "--dtc-fn-lin/--dtc-fn-ang.")


def wrist_travel(rows, live, n):
    """Did the joint the roll needs actually move?

    This is the measurement that settled where the roll error was going, and it
    is one line: on dtc_20260830_151553/152659/154729 the last driven joint
    travelled **0.02, 0.07 and 0.49 deg end to end** while sitting at the
    Lambda floor's full 1.42 N.m ceiling, stalled on 99.7-100% of the ticks the
    spring was on. So the roll was not being served slowly, it was not being
    served at all, and every newton-metre the floor bought was spent against
    static friction.

    Two failures wear the same "roll starved" hat and the fix differs:

    - the joint IS moving but slowly -- the command is under-powered, and the
      Lambda floor (LAMBDA_MIN_ANG_KGM2) is the lever;
    - the joint never moves at its own ceiling -- breakaway is above anything
      the task wrench can command there, so the lever is feedforward
      (--dtc-stiction-assist-nm). Raising the floor to reach breakaway costs
      bandwidth on that axis by sqrt(new/old) and walks it into the 15 Hz mode.

    A percentile on |qdot| cannot tell them apart; end-to-end travel can.
    """
    j = n - 1
    if f"q{j}" not in rows[0]:
        return
    q = np.array([float(r[f"q{j}"]) for r in rows])[live]
    qd = np.abs(np.array([float(r[f"qd{j}"]) for r in rows])[live])
    tau = np.abs(np.array([float(r[f"tau{j}"]) for r in rows])[live])
    if q.size == 0:
        return
    t = np.array([float(r["t"]) for r in rows])[live]
    dt = float(np.median(np.diff(t))) if t.size > 2 else 0.005
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.005
    qd_all = np.array([[float(r[f"qd{k}"]) for k in range(n)] for r in rows])[live]
    act, _ = _act_array(rows, live, dt, qd_all)
    travel = float(np.degrees(q.max() - q.min()))
    on = act
    print(f"\n  wrist travel (j{j}, the roll axis)  end-to-end "
          f"{travel:6.2f} deg   |tau{j}| p95 {q95(tau):5.2f} max "
          f"{tau.max():5.2f} N.m")
    stalled = float((qd[on] < 0.01).mean()) if on.any() else float("nan")
    if not on.any():
        print("     (never driven -- no stall report)")
    else:
        print(f"     stalled (|qdot| < 0.01) on {stalled:.1%} of spring-on ticks")
    if travel < 1.0 and np.isfinite(stalled) and stalled > 0.9:
        print("     \033[93mIT NEVER MOVED. Whatever the roll error is, no "
              "part of it is being served -- and under Lambda an unservable "
              "demand does not vanish, it leaks onto the other angular blades "
              "through the cross terms.\033[0m")
        print("     Read |tau| max against the floor's ceiling "
              "(LAMBDA_MIN_ANG_KGM2 * K_ang * err_clip_rot, 1.42 N.m at the "
              "defaults). At the ceiling and still stalled = breakaway is "
              "above it: --dtc-stiction-assist-nm.")



if __name__ == "__main__":
    main()
