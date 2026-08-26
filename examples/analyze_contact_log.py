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
    print(f"  implied rate at p95: {np.percentile(np.abs(d), 95) / nominal:.0f} N/s"
          "   (spike_dF_dt_N_s is 40)")

    print("\n-- tracking --")
    for w in (0.5, 1.0, 2.0):
        print(f"  |Fs-Fd| <= {w:.1f}N : {_band(fs, fd, w):5.1f}% of ticks")
    print(f"  v_z == 0 exactly  : {_pct(v_z == 0.0):5.1f}%   "
          f"(a held guard, not a settled loop)")
    print(f"  gain_scale        : median={np.nanmedian(gain):.2f}"
          f"  at 1.0 for {_pct(gain >= 0.999):.1f}% of ticks")

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
