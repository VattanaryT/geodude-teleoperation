#!/usr/bin/env python3
"""Show what the FFT in analyze_contact_log.py is actually doing.

    python examples/plot_contact_spectrum.py                    # newest log
    python examples/plot_contact_spectrum.py <a.csv> <b.csv>    # compare runs
    python examples/plot_contact_spectrum.py --column v_z

Four panels:
  1. the signal in time            -- what you already see in the console
  2. the Hann window and its effect -- why we taper, and what it costs
  3. the spectrum                   -- the same data sorted by rate
  4. noise vs rhythm               -- the same plot for pure noise, for contrast

Writes a PNG next to the log and opens it if a display is available.
"""
from __future__ import annotations

import argparse
import csv
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


def load(path, column):
    rows = [r for r in csv.DictReader(Path(path).open())
            if r.get("mode") != "safety_halt"]
    if not rows:
        raise SystemExit(f"{path}: no commanded ticks")
    t = _col(rows, "t")
    y = _col(rows, column)
    if not np.isfinite(y).any():
        raise SystemExit(f"{path}: column {column!r} is empty")
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    return t - t[0], y


def spectrum(t, y):
    """Return (freq, amplitude) the same way analyze_contact_log does."""
    y = y - y.mean()
    fs_hz = (t.size - 1) / (t[-1] - t[0])
    win = np.hanning(y.size)
    spec = np.abs(np.fft.rfft(y * win))
    freq = np.fft.rfftfreq(y.size, 1.0 / fs_hz)
    amp = 2.0 * spec / (y.size * float(np.mean(win)))
    return freq, amp, fs_hz


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help="contact_*.csv (default: newest)")
    ap.add_argument("--column", default="Fs_n_raw",
                    help="which column to transform (Fs_n_raw, Fs, v_z, ...)")
    ap.add_argument("--hi", type=float, default=30.0, help="top of the plotted band, Hz")
    ap.add_argument("--zoom", type=float, default=3.0,
                    help="seconds shown in the panel-1 inset")
    ap.add_argument("-o", "--out", default=None, help="output PNG")
    args = ap.parse_args()

    logs = [Path(p) for p in args.logs]
    if not logs:
        logs = sorted(Path("speedj_logs").glob("contact_*.csv"))[-1:]
    if not logs:
        raise SystemExit("no contact_*.csv found; pass one explicitly")

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, figsize=(11, 12))
    fig.suptitle(f"What the FFT sees  --  column {args.column!r}", fontsize=13)

    # 1. the raw signal, with a zoom so the waveform is actually visible
    for p in logs:
        t, y = load(p, args.column)
        ax[0].plot(t, y, lw=0.6, label=p.name)
    ax[0].set(xlabel="time (s)", ylabel=args.column,
              title="1. the signal in time -- you can see it wobbles, not how often")
    ax[0].legend(fontsize=7, loc="upper right")
    ax[0].grid(alpha=0.3)
    inset = ax[0].inset_axes((0.06, 0.06, 0.34, 0.42))
    for p in logs:
        t, y = load(p, args.column)
        w = t <= min(args.zoom, t[-1])
        inset.plot(t[w], y[w], lw=0.8)
    inset.set_title(f"first {args.zoom:g}s", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.grid(alpha=0.3)

    # 2. the Hann taper, on the first log
    t, y = load(logs[0], args.column)
    win = np.hanning(y.size)
    ax[1].plot(t, y - y.mean(), lw=0.5, alpha=0.45, label="signal (mean removed)")
    ax[1].plot(t, (y - y.mean()) * win, lw=0.6, label="after Hann taper")
    scale = np.nanmax(np.abs(y - y.mean()))
    ax[1].plot(t, win * scale, "k--", lw=1.4,
               label=f"Hann window x{scale:.1f} (mean {win.mean():.3f})")
    ax[1].set(xlabel="time (s)",
              title="2. the Hann taper -- fades the ends so the chunk's edges do not "
                    "smear energy.\n   Costs a factor of ~0.5 in amplitude, which we "
                    "divide back out.")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)

    # 3. the spectrum
    for p in logs:
        t, y = load(p, args.column)
        f, a, fs_hz = spectrum(t, y)
        band = (f >= 0.2) & (f <= args.hi)
        ax[2].plot(f[band], a[band], lw=0.8, label=f"{p.name} ({fs_hz:.0f} Hz)")
        k = np.argmax(a[band])
        ax[2].annotate(f"  {f[band][k]:.2f} Hz, amp {a[band][k]:.2f}",
                       xy=(f[band][k], a[band][k]), fontsize=8, va="center",
                       xytext=(10, -2), textcoords="offset points")
        ax[2].plot(f[band][k], a[band][k], "o", ms=5)
    ax[2].margins(y=0.18)
    ax[2].set(xlabel="rate (Hz)", ylabel="amplitude",
              title="3. the same data sorted by rate -- a peak is a rhythm, "
                    "a flat spread is noise")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)

    # 4. noise vs rhythm, for calibration
    rng = np.random.default_rng(0)
    n = min(y.size, 8000)
    tt = np.linspace(0, n / 193.0, n)
    for label, sig in [("pure noise", rng.normal(0, 1, n)),
                       ("noise + 3 Hz rhythm", rng.normal(0, 1, n)
                        + 1.5 * np.sin(2 * np.pi * 3.0 * tt))]:
        f, a, _ = spectrum(tt, sig)
        band = (f >= 0.2) & (f <= args.hi)
        ax[3].plot(f[band], a[band], lw=0.8, label=label)
    ax[3].set(xlabel="rate (Hz)", ylabel="amplitude",
              title="4. for calibration: what noise looks like vs what a real "
                    "rhythm looks like")
    ax[3].legend(fontsize=7)
    ax[3].grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out or (logs[0].with_suffix("") .as_posix()
                            + f".{args.column}.spectrum.png"))
    fig.savefig(out, dpi=130)
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
