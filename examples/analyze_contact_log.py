#!/usr/bin/env python3
"""Summarize a --contact-log CSV: is the force loop settling, and if not, why?

    python examples/analyze_contact_log.py speedj_logs/contact_*.csv

The 0.5 s console lines alias everything interesting. This reads every tick and
answers the three questions that decide the next change:

  * is the loop period what the seek laws think it is (`dt`)?
  * how much of the force signal is real, and how much is tick-to-tick noise?
  * which guard is holding v_along, and what fraction of the time?
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# FieldFollow's lead clamp in hardware_object_contact_teleop (max_lead_m=0.005).
# The carrot is held at most this far ahead of the live TCP, so the pose error
# cannot exceed it -- and sitting at it is the lateral-stall signature. Not a
# CLI flag there, so it is a constant here; change both together.
MAX_LEAD_M = 0.005


def _col(rows, name):
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        try:
            out[i] = float(r.get(name, "nan"))
        except (TypeError, ValueError):
            pass
    return out


def _pct(mask):
    return 100.0 * float(np.count_nonzero(mask)) / max(1, mask.size)


def _windowed_rate(t, fs, window_s=0.02):
    """dFs/dt exactly as the spike guard measures it.

    The guard differences against the oldest sample still inside
    spike_rate_window_s, not against the previous tick. A single-tick estimate
    overstates it on a noisy signal, which is the number that matters when
    choosing spike_dF_dt_N_s.
    """
    ok = np.isfinite(t) & np.isfinite(fs)
    t, fs = t[ok], fs[ok]
    if t.size < 2:
        return np.zeros(0)
    j = np.searchsorted(t, t - window_s, side="left")
    j = np.minimum(j, np.arange(t.size))
    span = t - t[j]
    good = span > 1e-9
    return (fs[good] - fs[j][good]) / span[good]


def _angular_authority(rows, lead, k_xy):
    """Slide vs turn, in the same units, from the same spring.

    The pose spring commands ``v = dt*K/M * e`` on every axis (dt=0.001, M=1),
    so a 5 mm lateral error at K=500 is 2.5 mm/s while a 0.02 rad angular error
    at K=2000 is 0.04 rad/s -- 4 mm/s at the tool tip on a 10 cm lever. In
    contact the stick only writes xyz, so all of that rotation comes from the
    field: when turning outruns sliding the arm goes one way no matter which
    way the stick is pushed. Nothing printed the comparison, so it read as a
    bug rather than as a gain ratio.
    """
    erx, ery, erz = (_col(rows, k) for k in ("err_rx", "err_ry", "err_rz"))
    if not np.isfinite(erx).any():
        print("  (no err_r* columns: run predates angular-error logging)")
        return
    k_ang = _col(rows, "k_ang")
    ang = np.sqrt(erx ** 2 + ery ** 2 + erz ** 2)
    print("\n-- slide vs turn (contact stick writes xyz only; all turn is the field) --")
    print(f"  |e_ang|  median={np.degrees(np.nanmedian(ang)):5.2f}deg"
          f"  p95={np.degrees(np.nanpercentile(ang, 95)):5.2f}deg"
          f"  max={np.degrees(np.nanmax(ang)):5.2f}deg")
    if not (np.isfinite(k_ang).any() and np.isfinite(k_xy).any()):
        print("  (no k_xy/k_ang columns: cannot compare the two authorities)")
        return
    dt_m = 0.001  # AdmittanceTeleoperationController(dt=...), inertia = I
    kx, ka = np.nanmedian(k_xy), np.nanmedian(k_ang)
    v_lin = dt_m * kx * np.nanmedian(np.abs(lead))
    w = dt_m * ka * np.nanmedian(ang)
    print(f"  commanded slide  {v_lin * 1e3:5.2f} mm/s   (K_xy={kx:.0f} x "
          f"{np.nanmedian(np.abs(lead)) * 1e3:.2f}mm)")
    print(f"  commanded turn   {np.degrees(w):5.2f} deg/s  (K_ang={ka:.0f} x "
          f"{np.degrees(np.nanmedian(ang)):.2f}deg)"
          f"  = {w * 0.10 * 1e3:5.2f} mm/s at a 10 cm lever")
    tip = w * 0.10
    if tip > v_lin and v_lin > 0:
        print(f"  -> turning outruns sliding {tip / v_lin:.1f}x at that lever. "
              "The direction is the field's, not the stick's,\n"
              "     so the arm turns one way whichever axis is pushed. "
              "--ang-stiffness is the lever; keep it in\n"
              f"     proportion to --xy-stiffness ({ka:.0f}/{kx:.0f} today).")


def _dominant_hz(t, x, lo=0.2, hi=30.0):
    """Peak frequency of x, and its amplitude, over [lo, hi] Hz.

    Screening test for coupling, not proof of cause: if Fs and v_z peak at the
    same frequency the controller is part of that oscillation, so its gains and
    limits set the amplitude. Separate peaks mean something outside the loop is
    driving Fs and the loop is only riding along. A shared peak does NOT say
    which one started it -- the loop may simply be tracking a real disturbance
    at that rate.
    """
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    if t.size < 64:
        return float("nan"), float("nan")
    fs_hz = (t.size - 1) / (t[-1] - t[0])
    x = x - np.mean(x)
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win))
    freq = np.fft.rfftfreq(x.size, 1.0 / fs_hz)
    band = (freq >= lo) & (freq <= hi)
    if not band.any():
        return float("nan"), float("nan")
    k = np.argmax(spec[band])
    # Divide by the window's coherent gain (mean of hanning ~ 0.5), or every
    # amplitude reads half of true and the oscillations look benign.
    amp = 2.0 * spec[band][k] / (x.size * float(np.mean(win)))
    return float(freq[band][k]), float(amp)


def _band(fs, fd, width):
    ok = np.isfinite(fs) & np.isfinite(fd)
    return _pct(ok & (np.abs(fs - fd) <= width)) if ok.any() else 0.0


def summarize(path):
    all_rows = list(csv.DictReader(Path(path).open()))
    if not all_rows:
        print(f"{path}: empty")
        return

    # Ticks logged while the robot was safety-halted are not the loop's doing:
    # v_z was forced to zero and the arm was not moving. Report them, then
    # exclude them, so tracking and guard percentages describe the ticks the
    # loop was actually driving.
    t_all = _col(all_rows, "t")
    halted = np.array([r.get("mode") == "safety_halt" for r in all_rows])
    rows = [r for r, h in zip(all_rows, halted) if not h] or all_rows

    t = _col(rows, "t")
    dt = _col(rows, "dt")
    fs = _col(rows, "Fs_n_raw")
    if not np.isfinite(fs).any():          # older logs, or the legacy loop
        fs = _col(rows, "Fs")
    fd = _col(rows, "Fd")
    v_z = _col(rows, "v_z")
    gain = _col(rows, "gain")
    nominal = np.nanmedian(dt)

    print(f"\n=== {Path(path).name} ===")
    print(f"{len(all_rows)} ticks over {np.nanmax(t_all) - np.nanmin(t_all):.1f}s"
          f"   Fd={np.nanmedian(fd):.1f}N   loop={1.0 / nominal:.0f}Hz")
    if halted.any():
        t0 = np.nanmin(t_all)
        starts = np.flatnonzero(halted[1:] & ~halted[:-1]) + 1
        if halted[0]:
            starts = np.r_[0, starts]
        when = ", ".join(f"t+{t_all[i] - t0:.1f}s" for i in starts[:6])
        print(f"  !! robot safety-halted {starts.size}x, {_pct(halted):.1f}% of ticks"
              f"  [{when}] -- excluded from everything below")

    print("\n-- loop period (the seek laws assume update_period) --")
    print(f"  dt  median={nominal * 1e3:.2f}ms  p95={np.nanpercentile(dt, 95) * 1e3:.2f}ms"
          f"  max={np.nanmax(dt) * 1e3:.2f}ms")

    print("\n-- force signal --")
    d = np.diff(fs[np.isfinite(fs)])
    print(f"  Fs      median={np.nanmedian(fs):+.2f}N  p5={np.nanpercentile(fs, 5):+.2f}N"
          f"  p95={np.nanpercentile(fs, 95):+.2f}N")
    print(f"  |dFs|   median={np.median(np.abs(d)):.3f}N  p95={np.percentile(np.abs(d), 95):.3f}N"
          f"  max={np.max(np.abs(d)):.2f}N   (per tick)")
    rate = np.abs(_windowed_rate(t, fs))
    if rate.size:
        p99 = np.percentile(rate, 99)
        print(f"  dFs/dt over the guard's 20ms window: median={np.median(rate):.0f}"
              f"  p95={np.percentile(rate, 95):.0f}  p99={p99:.0f}"
              f"  max={rate.max():.0f} N/s")
        print(f"  -> a threshold near {50 * round(p99 * 1.5 / 50):.0f} N/s sits above"
              f" ordinary variation and below the {rate.max():.0f} N/s peak."
              "  (default is 250)")

    print("\n-- tracking --")
    for w in (0.5, 1.0, 2.0):
        print(f"  |Fs-Fd| <= {w:.1f}N : {_band(fs, fd, w):5.1f}% of ticks")
    print(f"  v_z == 0 exactly  : {_pct(v_z == 0.0):5.1f}%   "
          f"(a held guard, not a settled loop)")
    auth = np.nanmean(np.abs(v_z)) * 1e3
    print(f"  mean |v_z|        : {auth:5.2f} mm/s   "
          "<- normal authority. Compare to how fast you slide laterally: "
          "the surface has to not recede faster than this.")
    print(f"  gain_scale        : median={np.nanmedian(gain):.2f}"
          f"  at 1.0 for {_pct(gain >= 0.999):.1f}% of ticks")

    sig = _col(rows, "sigma_min")
    if np.isfinite(sig).any():
        clamp = _col(rows, "clamp_scale")
        qmax = _col(rows, "qdot_max")
        print("\n-- joint space (where C153 'could not follow path' lives) --")
        print(f"  sigma_min   median={np.nanmedian(sig):.4f}  p5={np.nanpercentile(sig, 5):.4f}"
              f"  min={np.nanmin(sig):.4f}"
              f"   ({_pct(sig < 0.02):.1f}% of ticks below 0.02)")
        print(f"  clamp_scale median={np.nanmedian(clamp):.3f}"
              f"   saturated (<1.0) on {_pct(clamp < 0.999):.1f}% of ticks"
              "   <- the press is scaled by this too, invisibly")
        print(f"  |qdot| max  median={np.nanmedian(qmax):.3f} rad/s"
              f"  p95={np.nanpercentile(qmax, 95):.3f}  max={np.nanmax(qmax):.3f}")

    ex, ey, ez = (_col(rows, k) for k in ("err_x", "err_y", "err_z"))
    if np.isfinite(ex).any():
        print("\n-- lateral pose error, per axis (which direction is prone?) --")
        for name, e in (("x", ex), ("y", ey), ("z", ez)):
            print(f"  err_{name}  rms={np.sqrt(np.nanmean(e ** 2)) * 1e3:6.2f}mm"
                  f"  p95|e|={np.nanpercentile(np.abs(e), 95) * 1e3:6.2f}mm"
                  f"  max|e|={np.nanmax(np.abs(e)) * 1e3:6.2f}mm")
        rms = [np.sqrt(np.nanmean(e ** 2)) for e in (ex, ey, ez)]
        worst = "xyz"[int(np.nanargmax(rms))]
        k_xy = _col(rows, "k_xy")
        k_note = (f"  (run used K_xy={np.nanmedian(k_xy):.0f} N/m)"
                  if np.isfinite(k_xy).any() else
                  "  (run predates the k_xy column; K_xy unknown)")
        print(f"  -> {worst} carries the most error; if the fault is directional "
              f"it should be this axis, and --xy-stiffness is the lever.{k_note}")
        # The follow carrot is held at most --max-lead ahead of the live TCP
        # (_clamp_lead), so the pose error cannot exceed it. Sitting *at* it is
        # the signature that matters: the target is at its ceiling and the TCP
        # is not closing the gap, which is a lateral stall, not a force fault.
        lead = np.sqrt(ex ** 2 + ey ** 2 + ez ** 2)
        ceiling = MAX_LEAD_M
        pinned = _pct(lead >= 0.98 * ceiling)
        print(f"  |e|     max={np.nanmax(lead) * 1e3:5.2f}mm   "
              f"{pinned:.1f}% of ticks at >=98% of the "
              f"{ceiling * 1e3:.0f} mm follow lead clamp")
        if pinned >= 20.0:
            push = (np.nanmedian(k_xy) * ceiling
                    if np.isfinite(k_xy).any() else float("nan"))
            push_s = f"{push:.1f} N" if np.isfinite(push) else "K_xy * e"
            print(f"  -> pinned at the field-follow lead clamp ({ceiling * 1e3:.0f} "
                  "mm): the carrot is at its ceiling and the TCP is not "
                  "closing it.\n"
                  f"     That is a lateral stall -- the spring shoves {push_s} "
                  "sideways for as long as it lasts, and the UR calls the\n"
                  "     accumulated deviation 'could not follow path'. Look here, "
                  "not at the force loop.")
        _angular_authority(rows, lead, k_xy)

    print("\n-- oscillation (friction grabs are real on this rig; do not filter) --")
    f_fs, a_fs = _dominant_hz(t, fs)
    f_vz, a_vz = _dominant_hz(t, v_z)
    print(f"  Fs   peaks at {f_fs:6.2f} Hz  (amplitude {a_fs:.2f} N)")
    print(f"  v_z  peaks at {f_vz:6.2f} Hz  (amplitude {a_vz * 1e3:.2f} mm/s)")
    if np.isfinite(f_fs) and np.isfinite(f_vz):
        same = abs(f_fs - f_vz) < 0.25 * max(f_fs, f_vz)
        print("  -> same frequency: the loop is part of the oscillation, so its"
              " gains/limits are in play"
              if same else
              "  -> different frequencies: Fs is driven from outside the loop"
              " (friction), the loop is only riding it")
    big = np.abs(fs - np.nanmedian(fs)) > 3.0
    events = int(np.count_nonzero(big[1:] & ~big[:-1]))
    span = np.nanmax(t) - np.nanmin(t)
    print(f"  excursions >3N off median: {events} in {span:.0f}s"
          f"  ({events / max(span, 1e-9):.2f}/s)")

    print("\n-- worst events (where to look for a pendant trip) --")
    t0 = np.nanmin(t)
    d_all = np.abs(np.diff(fs))
    if d_all.size:
        k = int(np.nanargmax(d_all)) + 1
        print(f"  biggest jump: {d_all[k - 1]:5.2f} N in one tick at t+{t[k] - t0:.1f}s"
              f"  (Fs {fs[k - 1]:+.1f} -> {fs[k]:+.1f} N)")
    for flag, label in (("fault_fs", "|Fs| fault"), ("wrong_sign", "wrong sign")):
        f = _col(rows, flag) > 0.5
        if not f.any():
            continue
        starts = np.flatnonzero(f[1:] & ~f[:-1]) + 1
        worst = np.nanmax(np.abs(fs[f])) if np.isfinite(fs[f]).any() else float("nan")
        when = ", ".join(f"t+{t[i] - t0:.1f}s" for i in starts[:6])
        more = "" if starts.size <= 6 else f", +{starts.size - 6} more"
        print(f"  {label}: {starts.size} episode(s), peak |Fs|={worst:.1f} N"
              f"  [{when}{more}]")

    print("\n-- guards --")
    for flag in ("spike", "osc", "wrong_sign", "fault_fs"):
        f = _col(rows, flag)
        if np.isfinite(f).any():
            print(f"  {flag:<11}: {_pct(f > 0.5):5.1f}% of ticks")


def main(argv):
    paths = argv[1:]
    if not paths:
        paths = sorted(Path("speedj_logs").glob("contact_*.csv"))[-1:]
    if not paths:
        print(__doc__)
        return 1
    for p in paths:
        summarize(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
