"""Commanded vs actual, for one deterministic-path run.

    python examples/plot_path_run.py                # newest dtc log + its contact log
    python examples/plot_path_run.py speedj_logs/dtc_20260906_120000.csv
    python examples/plot_path_run.py --no-show      # write the PNG only

The question this answers is "was the trajectory gradual and smooth, or does it
step". Every panel is therefore commanded *against* actual on one axis, in the
units the operator thinks in, with the plan's phases shaded behind them -- an
aggregate cannot show a 40 ms jump and a percentile actively hides it, which is
the same lesson the burst detector in analyze_dtc_log.py exists for.

Six rows:

1. **Tool position**, per axis: ``tgt_*`` (commanded) against ``ee_*`` (actual),
   base frame, in mm relative to the run's first sample.
2. **Position error** per axis and its norm. Under a P-D law a steady offset is
   expected (``e = F_friction/(Lambda K)``); a *step* is not.
3. **Tool orientation**: commanded and actual quaternion, plus ``n_deg`` (tool
   +Z against the field normal -- the conformity number) and the total angular
   error.
4. **Joints**: actual ``q`` against ``qc``, the resolved-rate estimate of the
   joints the pose error is asking for. There is no joint setpoint under a
   torque law, so ``qc`` is an estimate and is labelled as one -- but a jump in
   it is a jump in what the loop is asking for, which is the thing worth seeing.
5. **Joint velocity and torque.** Jitter lives here: ``qd`` shows it long before
   ``q`` does, and ``tau`` against ``tau_max`` says whether the clamp -- not the
   spring -- was deciding the motion.
6. **Force** (from the contact CSV, on its own clock): ``Fs`` against ``Fd``,
   the commanded press ``F_press`` and what actually reached the arm
   ``F_emit`` -- they differ silently whenever the contact axis is refused --
   plus the latch distance ``d_surf`` and the ``engaged`` bool it produced.
7. **Raw six-axis F/T**, untared, reference only. Its own panel because the
   free-space offset is several N of tool weight: the sensor is not gravity
   compensated and the speedj backend cannot issue ``zero_ftsensor``.
"""

import argparse
import csv
import glob
import pathlib
import re
import sys

import numpy as np

PHASE_COLORS = {
    "approach": "#dfeaff",
    "plunge": "#ffe9cc",
    "traverse": "#d8f3dc",
    "retract": "#f3d8e6",
}


def newest(pattern):
    hits = sorted(glob.glob(pattern), key=lambda p: pathlib.Path(p).stat().st_mtime)
    return hits[-1] if hits else None


def read_csv(path):
    """Rows plus the leading ``# ...`` header lines the backends write."""
    meta = []
    with open(path) as fh:
        lines = fh.readlines()
    while lines and lines[0].lstrip().startswith("#"):
        meta.append(lines.pop(0).lstrip().lstrip("#").strip())
    return list(csv.DictReader(lines)), meta


def col(rows, name, default=np.nan):
    out = np.full(len(rows), float(default))
    for i, r in enumerate(rows):
        v = r.get(name)
        if v in (None, ""):
            continue
        try:
            out[i] = float(v)
        except ValueError:
            pass
    return out


def scol(rows, name):
    return [(r.get(name) or "") for r in rows]


def contact_log_for(dtc_path):
    """The contact CSV from the same session, matched on the timestamp stem.

    Falls back to the newest one: a run that entered contact once has exactly
    one, and guessing wrong is visible immediately (the force panel will not
    line up with the phases).
    """
    stem = re.search(r"(\d{8}_\d{6})", pathlib.Path(dtc_path).name)
    if stem:
        exact = sorted(glob.glob(
            f"speedj_logs/contact_{stem.group(1)[:8]}_*.csv"))
        if exact:
            return exact[-1]
    return newest("speedj_logs/contact_*.csv")


def shade_phases(ax, t, phase):
    """Shade the plan's phases behind a panel, one span per contiguous run."""
    if not len(phase):
        return
    start = 0
    for i in range(1, len(phase) + 1):
        if i < len(phase) and phase[i] == phase[start]:
            continue
        name = phase[start]
        if name in PHASE_COLORS:
            ax.axvspan(t[start], t[min(i, len(t) - 1)],
                       color=PHASE_COLORS[name], zorder=0, lw=0)
        start = i


def _finite(a):
    return np.isfinite(a).any()


def _phase_on_dtc_clock(dtc_t, contact_rows):
    """The contact loop's ``phase`` column resampled onto the dtc tick clock.

    The two logs run on different threads at different rates, so nearest-sample
    is the only join that does not invent transitions. Both clocks start at
    their own t=0, which is close enough for shading; the force panel plots on
    its own clock so nothing quantitative depends on this.
    """
    if not contact_rows:
        return []
    ct = col(contact_rows, "elapsed")
    cp = scol(contact_rows, "phase")
    if not np.isfinite(ct).any():
        return []
    idx = np.clip(np.searchsorted(ct, dtc_t), 0, len(cp) - 1)
    return [cp[int(i)] for i in idx]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dtc_log", nargs="?", default=None,
                    help="dtc CSV (default: newest in speedj_logs/)")
    ap.add_argument("--contact-log", default=None,
                    help="contact CSV (default: the one from the same session)")
    ap.add_argument("--out", default=None, help="PNG path (default: next to the log)")
    ap.add_argument("--no-show", action="store_true",
                    help="write the PNG without opening a window")
    args = ap.parse_args(argv)

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = args.dtc_log or newest("speedj_logs/dtc_*.csv")
    if not path:
        sys.exit("no speedj_logs/dtc_*.csv found")
    rows, meta = read_csv(path)
    if not rows:
        sys.exit(f"{path}: empty")
    c_path = args.contact_log or contact_log_for(path)
    c_rows = read_csv(c_path)[0] if c_path and pathlib.Path(c_path).exists() else []

    t = col(rows, "t")
    n_joints = sum(1 for k in rows[0] if re.fullmatch(r"q\d+", k))
    phase = _phase_on_dtc_clock(t, c_rows)

    ee = np.stack([col(rows, f"ee_{a}") for a in "xyz"], axis=1)
    tgt = np.stack([col(rows, f"tgt_{a}") for a in "xyz"], axis=1)
    err = np.stack([col(rows, f"e{a}") for a in "xyz"], axis=1)
    q = np.stack([col(rows, f"q{i}") for i in range(n_joints)], axis=1)
    qc = np.stack([col(rows, f"qc{i}") for i in range(n_joints)], axis=1)
    qd = np.stack([col(rows, f"qd{i}") for i in range(n_joints)], axis=1)
    tau = np.stack([col(rows, f"tau{i}") for i in range(n_joints)], axis=1)
    n_deg = col(rows, "n_deg")
    err_ang = np.linalg.norm(
        np.stack([col(rows, f"er{a}") for a in ("x", "y", "z")], axis=1), axis=1)

    fig, axes = plt.subplots(7, 1, figsize=(13.5, 22.0), sharex=False)
    fig.suptitle(f"{pathlib.Path(path).name}"
                 + (f"   +   {pathlib.Path(c_path).name}" if c_rows else ""),
                 fontsize=10)

    # 1. tool position, commanded vs actual
    ax = axes[0]
    shade_phases(ax, t, phase)
    ref = np.where(np.isfinite(ee[0]), ee[0], 0.0)
    for i, a in enumerate("xyz"):
        line, = ax.plot(t, 1e3 * (tgt[:, i] - ref[i]), lw=1.6, ls="--",
                        label=f"cmd {a}")
        ax.plot(t, 1e3 * (ee[:, i] - ref[i]), lw=1.0, color=line.get_color(),
                alpha=0.85, label=f"act {a}")
    ax.set_ylabel("tool position [mm]\n(relative to t=0)")
    ax.legend(ncol=6, fontsize=7, loc="upper right")
    ax.set_title("commanded (dashed) vs actual (solid) TCP -- a step here is a "
                 "step in the command, not in the arm", fontsize=8, loc="left")

    # 2. position error
    ax = axes[1]
    shade_phases(ax, t, phase)
    for i, a in enumerate("xyz"):
        ax.plot(t, 1e3 * err[:, i], lw=1.0, label=f"e{a}")
    ax.plot(t, 1e3 * np.linalg.norm(err, axis=1), lw=1.6, color="k",
            label="|e_lin|")
    ax.set_ylabel("position error [mm]")
    ax.legend(ncol=4, fontsize=7, loc="upper right")

    # 3. orientation
    ax = axes[2]
    shade_phases(ax, t, phase)
    drew_quat = False
    for j, a in enumerate("wxyz"):
        tq = col(rows, f"tgt_q{a}")
        eq = col(rows, f"ee_q{a}")
        if not (_finite(tq) and _finite(eq)):
            continue
        drew_quat = True
        line, = ax.plot(t, tq, lw=1.4, ls="--", label=f"cmd q{a}")
        ax.plot(t, eq, lw=0.9, color=line.get_color(), alpha=0.85,
                label=f"act q{a}")
    ax.set_ylabel("tool orientation\n(base quaternion)")
    if not drew_quat:
        ax.text(0.01, 0.5, "no ee_q*/tgt_q* columns (log predates them)",
                transform=ax.transAxes, fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(t, n_deg, lw=1.2, color="tab:red", label="n_deg (tool Z vs normal)")
    ax2.plot(t, np.degrees(err_ang), lw=0.9, color="tab:purple", alpha=0.7,
             label="|e_ang|")
    ax2.set_ylabel("angle [deg]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, ncol=5, fontsize=7, loc="upper right")

    # 4. joints, actual vs the resolved-rate estimate of what is being asked for
    ax = axes[3]
    shade_phases(ax, t, phase)
    have_qc = _finite(qc)
    for i in range(n_joints):
        line, = ax.plot(t, np.degrees(q[:, i]), lw=1.1, label=f"q{i}")
        if have_qc:
            ax.plot(t, np.degrees(qc[:, i]), lw=1.4, ls="--", alpha=0.6,
                    color=line.get_color())
    ax.set_ylabel("joint angle [deg]")
    ax.legend(ncol=6, fontsize=7, loc="upper right")
    ax.set_title(
        "actual (solid) vs commanded (dashed, resolved-rate estimate: a torque "
        "law has no joint setpoint)" if have_qc
        else "actual joints (no qc columns: log predates them)",
        fontsize=8, loc="left")

    # 5. joint velocity and torque -- where jitter shows first
    ax = axes[4]
    shade_phases(ax, t, phase)
    for i in range(n_joints):
        ax.plot(t, qd[:, i], lw=0.9, label=f"qd{i}")
    ax.set_ylabel("joint velocity [rad/s]")
    ax.legend(ncol=6, fontsize=7, loc="upper right")
    ax3 = ax.twinx()
    ax3.plot(t, np.max(np.abs(tau), axis=1), lw=1.0, color="k", alpha=0.5,
             label="max |tau|")
    ax3.set_ylabel("|tau| [N.m]")
    ax3.legend(fontsize=7, loc="lower right")

    # 6. force, on the contact loop's own clock
    ax = axes[5]
    if c_rows:
        ct = col(c_rows, "elapsed")
        shade_phases(ax, ct, scol(c_rows, "phase"))
        ax.plot(ct, col(c_rows, "Fs"), lw=1.0, label="Fs (tared, along tool Z)")
        fs_n = col(c_rows, "Fs_n")
        if _finite(fs_n):
            ax.plot(ct, fs_n, lw=1.0, label="Fs_n (along the normal)")
        ax.plot(ct, col(c_rows, "Fd"), lw=1.2, ls="--", color="k", label="Fd")
        ax.plot(ct, col(c_rows, "F_press"), lw=1.2, label="F_press (commanded)")
        ax.plot(ct, col(c_rows, "F_emit"), lw=1.0, ls=":", label="F_emit (reached the arm)")
        ax.set_ylabel("force [N]")
        ax.set_xlabel("t [s]  (contact-loop clock)")
        ax.legend(ncol=4, fontsize=7, loc="upper right")
        d = col(c_rows, "d_surf")
        if _finite(d):
            ax4 = ax.twinx()
            ax4.plot(ct, 1e3 * d, lw=1.1, color="tab:brown",
                     label="d_surf (tool above the planned surface)")
            ax4.plot(ct, 20.0 * col(c_rows, "engaged"), lw=1.0, color="tab:green",
                     alpha=0.6, label="engaged (x20 mm)")
            ax4.axhline(0.0, lw=0.6, color="tab:brown", alpha=0.4)
            ax4.set_ylabel("distance [mm]")
            ax4.legend(fontsize=7, loc="lower right")
    else:
        ax.text(0.01, 0.5, "no contact CSV found -- run with --contact-log auto",
                transform=ax.transAxes, fontsize=9)
        ax.set_xlabel("t [s]")

    # 7. raw six-axis F/T, untared -- reference only, no loop reads it. Its own
    # panel because the free-space offset is several N of tool weight (the
    # sensor is not gravity compensated and the speedj backend cannot issue
    # zero_ftsensor), so plotting it next to a tared 15 N target squashes both.
    ax = axes[6]
    drew_ft = False
    if c_rows:
        ct = col(c_rows, "elapsed")
        shade_phases(ax, ct, scol(c_rows, "phase"))
        for name in ("ft_fx", "ft_fy", "ft_fz"):
            v = col(c_rows, name)
            if _finite(v):
                drew_ft = True
                ax.plot(ct, v, lw=0.9, label=name)
        axt = ax.twinx()
        for name in ("ft_tx", "ft_ty", "ft_tz"):
            v = col(c_rows, name)
            if _finite(v):
                drew_ft = True
                axt.plot(ct, v, lw=0.8, ls="--", alpha=0.7, label=name)
        ax.set_ylabel("raw F/T force [N]")
        axt.set_ylabel("raw F/T torque [N.m]")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axt.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, ncol=6, fontsize=7, loc="upper right")
        ax.set_title("raw ActualTCPForce, tool frame, UNTARED -- reference "
                     "only; the loop reads the tared value above",
                     fontsize=8, loc="left")
    if not drew_ft:
        ax.text(0.01, 0.5, "no ft_* columns (no 6-axis reader, or an older log)",
                transform=ax.transAxes, fontsize=9)

    for a in axes:
        a.set_xlabel("t [s]")
        a.grid(alpha=0.25)
    if meta:
        fig.text(0.01, 0.001, meta[-1][:220], fontsize=6, color="0.35")
    fig.tight_layout(rect=(0, 0.01, 1, 0.985))

    out = args.out or str(pathlib.Path(path).with_suffix("")) + "_path.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}", flush=True)
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
