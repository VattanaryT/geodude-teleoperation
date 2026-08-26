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


def _dominant_hz(t, x, lo=0.2, hi=30.0):
    """Peak frequency of x, and its amplitude, over [lo, hi] Hz.

    Distinguishes a mechanical disturbance from one the loop is pumping: if Fs
    and v_z peak at the same frequency the controller is part of the
    oscillation, if only Fs does then something outside it (friction grabs) is
    driving and the loop is only riding along.
    """
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    if t.size < 64:
        return float("nan"), float("nan")
    fs_hz = (t.size - 1) / (t[-1] - t[0])
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freq = np.fft.rfftfreq(x.size, 1.0 / fs_hz)
    band = (freq >= lo) & (freq <= hi)
    if not band.any():
        return float("nan"), float("nan")
    k = np.argmax(spec[band])
    return float(freq[band][k]), float(2.0 * spec[band][k] / x.size)


def _band(fs, fd, width):
    ok = np.isfinite(fs) & np.isfinite(fd)
    return _pct(ok & (np.abs(fs - fd) <= width)) if ok.any() else 0.0


def summarize(path):
    rows = list(csv.DictReader(Path(path).open()))
    if not rows:
        print(f"{path}: empty")
        return
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
    print(f"{len(rows)} ticks over {np.nanmax(t) - np.nanmin(t):.1f}s"
          f"   Fd={np.nanmedian(fd):.1f}N   loop={1.0 / nominal:.0f}Hz")

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
