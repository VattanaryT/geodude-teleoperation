"""Live plot of commanded speedJ (qdot, qddot, Δt). Not viser; not on the RTDE thread."""
from __future__ import annotations

from collections import deque, namedtuple
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
import os
import queue
import time

import numpy as np

SpeedJSample = namedtuple(
    "SpeedJSample", "t dt qdot qddot measured spring stick")


def sample_from_qdot(qdot, *, t, prev, measured=None, spring=False, stick=None):
    """Finite-difference one commanded velocity into a plot sample."""
    qdot = np.asarray(qdot, dtype=float).reshape(-1).copy()
    if measured is not None:
        measured = np.asarray(measured, dtype=float).reshape(-1).copy()
    if stick is None:
        stick = np.zeros(3)
    else:
        stick = np.asarray(stick, dtype=float).reshape(3).copy()
    if prev is None:
        dt = 0.0
        qddot = np.zeros_like(qdot)
    else:
        dt = float(t) - float(prev.t)
        prev_q = np.asarray(prev.qdot, dtype=float).reshape(-1)
        if dt <= 1e-9 or prev_q.shape != qdot.shape:
            qddot = np.zeros_like(qdot)
        else:
            qddot = (qdot - prev_q) / dt
    return SpeedJSample(
        float(t), dt, qdot, qddot, measured, bool(spring), stick)


def session_save_paths(directory, when=None):
    when = datetime.now() if when is None else when
    stamp = when.strftime("%Y%m%d_%H%M%S")
    d = Path(directory)
    return d / f"speedj_{stamp}.png", d / f"speedj_{stamp}.npz"


def _bin_picks(n, score, n_bins):
    n = int(n)
    n_bins = min(int(n_bins), n)
    if n <= n_bins:
        return np.arange(n)
    score = np.asarray(score, dtype=float).reshape(-1)
    chunk = n // n_bins
    n_use = chunk * n_bins
    pick = np.argmax(score[:n_use].reshape(n_bins, chunk), axis=1)
    return np.arange(n_bins) * chunk + pick


def downsample_maxabs(t, y, n_bins=4000):
    """Keep the max-|y| sample in each bin so a one-tick spike still plots."""
    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float)
    if t.size == 0:
        return t, y
    if y.ndim == 1:
        score = np.abs(y)
    else:
        score = np.max(np.abs(y), axis=1)
    picks = _bin_picks(t.size, score, n_bins)
    return t[picks], y[picks]


def write_speedj_session(path_png, *, t, qdot, qddot, dt, spring,
                         measured=None, stick=None, path_npz=None, n_bins=4000):
    """Write the full session: PNG (downsampled, spikes kept) and optional NPZ."""
    path_png = Path(path_png)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray(t, dtype=float).reshape(-1)
    qdot = np.asarray(qdot, dtype=float)
    qddot = np.asarray(qddot, dtype=float)
    dt = np.asarray(dt, dtype=float).reshape(-1)
    spring = np.asarray(spring, dtype=float).reshape(-1)
    n = min(t.size, qdot.shape[0], qddot.shape[0], dt.size, spring.size)
    if measured is not None:
        measured = np.asarray(measured, dtype=float)
        n = min(n, measured.shape[0])
    if stick is not None:
        stick = np.asarray(stick, dtype=float)
        if stick.ndim == 1:
            stick = stick.reshape(-1, 1)
        n = min(n, stick.shape[0])
    t, qdot, qddot, dt, spring = t[:n], qdot[:n], qddot[:n], dt[:n], spring[:n]
    if measured is not None:
        measured = measured[:n]
    if stick is not None:
        stick = stick[:n]
    if path_npz is not None:
        payload = dict(t=t, qdot=qdot, qddot=qddot, dt=dt, spring=spring)
        if measured is not None:
            payload["measured"] = measured
        if stick is not None:
            payload["stick"] = stick
        np.savez_compressed(path_npz, **payload)
    if t.size == 0:
        return
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        score = np.max(np.abs(qddot), axis=1) if qddot.ndim == 2 else np.abs(qddot)
        if stick is not None:
            score = np.maximum(score, np.max(np.abs(stick), axis=1))
        picks = _bin_picks(t.size, score, n_bins)
        t_p = t[picks]
        qdot_p = qdot[picks]
        qddot_p = qddot[picks]
        dt_p = dt[picks]
        spring_p = spring[picks]
        meas_p = None if measured is None else measured[picks]
        stick_p = None if stick is None else stick[picks]
        fig = _draw_speedj_figure(
            t_p - t_p[0], qdot_p, qddot_p, dt_p, spring_p, meas_p, stick_p,
            xlabel="seconds from session start",
            title="speedJ commanded velocity (full session)",
        )
        fig.savefig(path_png, dpi=120)
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as exc:
        print(
            f"speedj plot: png failed ({exc})"
            + (f"; npz is at {path_npz}" if path_npz is not None else ""),
            flush=True,
        )


def _draw_speedj_figure(x, qdot, qddot, dt, spring, measured=None, stick=None, *,
                        xlabel, title):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 9))
    fig.suptitle(title)
    ax_v, ax_a, ax_s, ax_dt = axes
    ax_v.set_ylabel("q̇ cmd [rad/s]")
    ax_a.set_ylabel("q̈ cmd [rad/s²]")
    ax_s.set_ylabel("stick [mm]")
    ax_dt.set_ylabel("Δt [ms]")
    ax_dt.set_xlabel(xlabel)
    ax_v.axhline(0.8, color="0.6", ls="--", lw=0.8, label="|q̇|max 0.8")
    ax_v.axhline(-0.8, color="0.6", ls="--", lw=0.8)
    ax_dt.axhline(2.0, color="0.6", ls="--", lw=0.8, label="2 ms (500 Hz)")
    ax_v.grid(True, alpha=0.3)
    ax_a.grid(True, alpha=0.3)
    ax_s.grid(True, alpha=0.3)
    ax_dt.grid(True, alpha=0.3)
    qdot = np.asarray(qdot, dtype=float)
    qddot = np.asarray(qddot, dtype=float)
    if qdot.ndim == 1:
        qdot = qdot.reshape(-1, 1)
        qddot = qddot.reshape(-1, 1)
    n_joints = qdot.shape[1]
    dt_ms = np.asarray(dt, dtype=float) * 1000.0
    for i in range(n_joints):
        color = f"C{i % 10}"
        ax_v.plot(x, qdot[:, i], color=color, lw=1.0, label=f"J{i}")
        ax_a.plot(x, qddot[:, i], color=color, lw=1.0)
        if measured is not None:
            m = np.asarray(measured, dtype=float)
            if m.ndim == 1:
                m = m.reshape(-1, 1)
            if m.shape[1] == n_joints:
                ax_v.plot(x, m[:, i], color=color, lw=0.8, ls="--", alpha=0.65)
    ax_v.plot(x, np.linalg.norm(qdot, axis=1), color="k", lw=1.6, label="|q̇|")
    if stick is not None:
        s = np.asarray(stick, dtype=float)
        if s.ndim == 1:
            s = s.reshape(-1, 1)
        s_mm = s * 1000.0
        labels = ("x", "y", "z")
        for i in range(min(3, s_mm.shape[1])):
            ax_s.plot(x, s_mm[:, i], color=f"C{i}", lw=1.1, label=labels[i])
        ax_s.plot(x, np.linalg.norm(s_mm, axis=1), color="k", lw=1.4, label="|lin|")
        ax_s.legend(loc="upper left", ncol=4, fontsize=8)
        ax_s.set_title(
            f"max |lin|={float(np.max(np.linalg.norm(s_mm, axis=1))):.2f} mm",
            loc="right", fontsize=9)
    ax_dt.plot(x, dt_ms, color="C0", lw=1.2, label="Δt")
    idle = np.asarray(spring, dtype=float).reshape(-1) < 0.5
    if np.any(idle):
        ymax = max(4.0, float(np.nanmax(dt_ms)) if dt_ms.size else 4.0)
        ax_dt.fill_between(
            x, 0.0, ymax, where=idle, color="0.75", alpha=0.35, step="post")
    ax_v.set_title(
        f"max |q̇|={float(np.max(np.abs(qdot))):.3f} rad/s", loc="right", fontsize=9)
    ax_a.set_title(
        f"max |q̈|={float(np.max(np.abs(qddot))):.1f} rad/s²", loc="right", fontsize=9)
    ax_dt.set_title(
        f"max Δt={float(np.max(dt_ms)):.2f} ms" if dt_ms.size else "",
        loc="right", fontsize=9)
    ax_v.legend(loc="upper left", ncol=4, fontsize=8)
    ax_dt.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def _enqueue_drop_oldest(sink, sample):
    put = getattr(sink, "put_nowait", None)
    if put is None:
        sink(sample)
        return
    try:
        put(sample)
        return
    except queue.Full:
        pass
    try:
        sink.get_nowait()
    except queue.Empty:
        pass
    try:
        put(sample)
    except queue.Full:
        pass


class SpeedJTrace:
    """Record commanded qdot without blocking the 500 Hz control thread."""

    def __init__(self, sink=None):
        self._sink = sink
        self._prev = None
        self._proc = None
        self._session = []
        self._save_png = None
        self._save_npz = None

    def record(self, qdot, *, t=None, measured=None, spring=False, stick=None):
        if t is None:
            t = time.perf_counter()
        sample = sample_from_qdot(
            qdot, t=t, prev=self._prev, measured=measured, spring=spring,
            stick=stick)
        self._prev = sample
        self._session.append(sample)
        if self._sink is not None:
            _enqueue_drop_oldest(self._sink, sample)
        return sample

    def start(self, *, window_s=8.0, port=8081, save_dir="speedj_logs"):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("speedj plot: matplotlib not installed; skipping", flush=True)
            return False
        png, npz = session_save_paths(Path(save_dir).resolve())
        png.parent.mkdir(parents=True, exist_ok=True)
        self._save_png = png
        self._save_npz = npz
        ctx = mp.get_context("spawn")
        self._sink = ctx.Queue(maxsize=512)
        self._prev = None
        self._session = []
        self._proc = ctx.Process(
            target=_plot_loop,
            args=(self._sink, float(window_s), int(port)),
            daemon=True,
            name="speedj-plot",
        )
        self._proc.start()
        print(
            f"speedj plot: session will save to {png.resolve()} (+ .npz) on exit",
            flush=True)
        return True

    def save_session(self, path_png=None, path_npz=None):
        path_png = Path(path_png or self._save_png or "speedj_logs/speedj.png")
        path_npz = Path(path_npz or self._save_npz or path_png.with_suffix(".npz"))
        if not self._session:
            print("speedj plot: no samples to save", flush=True)
            return None
        samples = list(self._session)
        n = int(samples[0].qdot.shape[0])
        measured_rows = []
        has_meas = False
        for s in samples:
            if s.measured is None:
                measured_rows.append(np.full(n, np.nan))
            else:
                has_meas = True
                measured_rows.append(np.asarray(s.measured, dtype=float).reshape(-1))
        write_speedj_session(
            path_png,
            t=np.array([s.t for s in samples], dtype=float),
            qdot=np.vstack([s.qdot for s in samples]),
            qddot=np.vstack([s.qddot for s in samples]),
            dt=np.array([s.dt for s in samples], dtype=float),
            spring=np.array([1.0 if s.spring else 0.0 for s in samples]),
            measured=(np.vstack(measured_rows) if has_meas else None),
            stick=np.vstack([s.stick for s in samples]),
            path_npz=path_npz,
        )
        if path_npz.is_file():
            print(f"speedj plot: saved {path_npz.resolve()}", flush=True)
        if path_png.is_file():
            print(f"speedj plot: saved {path_png.resolve()}", flush=True)
        else:
            print("speedj plot: png not written (npz still saved if listed above)",
                  flush=True)
        return path_png if path_png.is_file() else path_npz

    def stop(self):
        if self._save_png is not None:
            try:
                self.save_session()
            except BaseException as exc:
                # BaseException, not Exception: a second Ctrl+C during shutdown
                # landed inside savez_compressed and took the whole session's
                # data with it. We are already on the exit path, so swallowing
                # the interrupt here costs nothing and keeps whatever was
                # written. The per-tick contact CSV is the durable record.
                print(f"speedj plot: save failed ({exc!r})", flush=True)
        if self._sink is None:
            return
        try:
            self._sink.put_nowait(None)
        except Exception:
            pass
        proc = self._proc
        if proc is not None and proc.is_alive():
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=0.5)
        self._proc = None


def _choose_backend(port):
    import matplotlib

    display = os.environ.get("DISPLAY", "")
    if display:
        try:
            matplotlib.use("TkAgg", force=True)
            print("speedj plot: Tk window (commanded qdot / qddot / Δt)", flush=True)
            return "TkAgg"
        except Exception as exc:
            print(f"speedj plot: TkAgg failed ({exc}); trying WebAgg", flush=True)
    matplotlib.use("WebAgg", force=True)
    matplotlib.rcParams["webagg.port"] = int(port)
    matplotlib.rcParams["webagg.open_in_browser"] = False
    print(
        f"speedj plot: http://127.0.0.1:{int(port)}  "
        "(commanded qdot / qddot / Δt — open beside viser)",
        flush=True,
    )
    return "WebAgg"


def _plot_loop(in_q, window_s, port):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        _run_plot(in_q, window_s, port)
    except Exception as exc:
        print(f"speedj plot: stopped ({exc})", flush=True)


def _run_plot(in_q, window_s, port):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    _choose_backend(port)

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 9))
    try:
        fig.canvas.manager.set_window_title("speedJ command")
    except Exception:
        pass
    fig.suptitle("speedJ commanded velocity (parallel to viser, not in it)")
    ax_v, ax_a, ax_s, ax_dt = axes
    ax_v.set_ylabel("q̇ cmd [rad/s]")
    ax_a.set_ylabel("q̈ cmd [rad/s²]")
    ax_s.set_ylabel("stick [mm]")
    ax_dt.set_ylabel("Δt [ms]")
    ax_dt.set_xlabel("seconds (0 = now)")
    ax_v.axhline(0.8, color="0.6", ls="--", lw=0.8, label="|q̇|max 0.8")
    ax_v.axhline(-0.8, color="0.6", ls="--", lw=0.8)
    ax_dt.axhline(2.0, color="0.6", ls="--", lw=0.8, label="2 ms (500 Hz)")
    ax_v.grid(True, alpha=0.3)
    ax_a.grid(True, alpha=0.3)
    ax_s.grid(True, alpha=0.3)
    ax_dt.grid(True, alpha=0.3)

    ts = deque()
    qdots = deque()
    qddots = deque()
    dts = deque()
    springs = deque()
    measureds = deque()
    sticks = deque()
    lines_v = []
    lines_a = []
    lines_m = []
    line_dt, = ax_dt.plot([], [], color="C0", lw=1.2, label="Δt")
    line_norm, = ax_v.plot([], [], color="k", lw=1.6, label="|q̇|")
    line_sx, = ax_s.plot([], [], color="C0", lw=1.1, label="x")
    line_sy, = ax_s.plot([], [], color="C1", lw=1.1, label="y")
    line_sz, = ax_s.plot([], [], color="C2", lw=1.1, label="z")
    line_sn, = ax_s.plot([], [], color="k", lw=1.4, label="|lin|")
    ax_s.legend(loc="upper left", ncol=4, fontsize=8)
    fill_idle = None
    n_joints = 0

    def _ensure_joint_lines(n):
        nonlocal n_joints
        while n_joints < n:
            color = f"C{n_joints % 10}"
            lines_v.append(
                ax_v.plot([], [], color=color, lw=1.0, label=f"J{n_joints}")[0])
            lines_m.append(
                ax_v.plot([], [], color=color, lw=0.8, ls="--", alpha=0.65)[0])
            lines_a.append(
                ax_a.plot([], [], color=color, lw=1.0, label=f"J{n_joints}")[0])
            n_joints += 1
        if n_joints == n and n > 0:
            ax_v.legend(loc="upper left", ncol=4, fontsize=8)
            ax_dt.legend(loc="upper left", fontsize=8)

    def pump(_frame):
        nonlocal fill_idle
        while True:
            try:
                sample = in_q.get_nowait()
            except queue.Empty:
                break
            if sample is None:
                plt.close(fig)
                return
            ts.append(sample.t)
            qdots.append(sample.qdot)
            qddots.append(sample.qddot)
            dts.append(sample.dt)
            springs.append(1.0 if sample.spring else 0.0)
            measureds.append(sample.measured)
            stick = getattr(sample, "stick", None)
            sticks.append(
                np.zeros(3) if stick is None else np.asarray(stick, dtype=float).reshape(3))
        if not ts:
            return
        t_now = ts[-1]
        t_cut = t_now - float(window_s)
        while ts and ts[0] < t_cut:
            ts.popleft()
            qdots.popleft()
            qddots.popleft()
            dts.popleft()
            springs.popleft()
            measureds.popleft()
            sticks.popleft()
        if not ts:
            return
        _ensure_joint_lines(int(qdots[-1].shape[0]))
        x = np.asarray(ts, dtype=float) - t_now
        v = np.vstack(qdots)
        a = np.vstack(qddots)
        dt_ms = np.asarray(dts, dtype=float) * 1000.0
        for i, line in enumerate(lines_v):
            line.set_data(x, v[:, i])
        for i, line in enumerate(lines_a):
            line.set_data(x, a[:, i])
        line_norm.set_data(x, np.linalg.norm(v, axis=1))
        line_dt.set_data(x, dt_ms)
        s = np.vstack(sticks) * 1000.0
        line_sx.set_data(x, s[:, 0])
        line_sy.set_data(x, s[:, 1])
        line_sz.set_data(x, s[:, 2])
        line_sn.set_data(x, np.linalg.norm(s, axis=1))
        ax_s.set_title(
            f"max |lin|={float(np.max(np.linalg.norm(s, axis=1))):.2f} mm",
            loc="right", fontsize=9)
        if measureds[-1] is not None:
            stacked = []
            for meas in measureds:
                if meas is None or np.asarray(meas).reshape(-1).shape[0] != n_joints:
                    stacked.append(np.full(n_joints, np.nan))
                else:
                    stacked.append(np.asarray(meas, dtype=float).reshape(-1))
            m = np.vstack(stacked)
            for i, line in enumerate(lines_m):
                line.set_data(x, m[:, i])
        ax_v.set_title(
            f"max |q̇|={float(np.max(np.abs(v))):.3f} rad/s", loc="right", fontsize=9)
        ax_a.set_title(
            f"max |q̈|={float(np.max(np.abs(a))):.1f} rad/s²", loc="right", fontsize=9)
        ax_dt.set_title(
            f"max Δt={float(np.max(dt_ms)):.2f} ms", loc="right", fontsize=9)
        if fill_idle is not None:
            fill_idle.remove()
            fill_idle = None
        spring = np.asarray(springs, dtype=float)
        idle = spring < 0.5
        if np.any(idle):
            ymax = max(4.0, float(np.nanmax(dt_ms)) if dt_ms.size else 4.0)
            fill_idle = ax_dt.fill_between(
                x, 0.0, ymax, where=idle, color="0.75", alpha=0.35,
                step="post")
        ax_v.set_xlim(-float(window_s), 0.05)
        ax_v.relim()
        ax_v.autoscale_view(scalex=False, scaley=True)
        ax_a.relim()
        ax_a.autoscale_view(scalex=False, scaley=True)
        ax_s.relim()
        ax_s.autoscale_view(scalex=False, scaley=True)
        ax_dt.set_xlim(-float(window_s), 0.05)
        dt_max = max(6.0, float(np.nanmax(dt_ms)) * 1.2 if dt_ms.size else 6.0)
        ax_dt.set_ylim(0.0, dt_max)
        return lines_v + lines_a + lines_m + [
            line_dt, line_norm, line_sx, line_sy, line_sz, line_sn]

    anim = FuncAnimation(
        fig, pump, interval=100, blit=False, cache_frame_data=False)
    fig._speedj_anim = anim
    fig.tight_layout()
    plt.show()


__all__ = [
    "SpeedJSample",
    "SpeedJTrace",
    "downsample_maxabs",
    "sample_from_qdot",
    "session_save_paths",
    "write_speedj_session",
]
