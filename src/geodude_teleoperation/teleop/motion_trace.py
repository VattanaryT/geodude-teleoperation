"""Log / plot commanded q̇ and reference R_motion for wobble diagnosis.

Interpretation guide:
- Step discontinuity in ``R_motion`` aligned with a ``q̇`` spike → relock logic.
- Smooth ``R_motion`` with spiky ``q̇`` → DLS solve or host-loop timing instead.
- ``‖q̇‖_raw`` ≫ ``‖q̇‖`` (clipped) while TCP overshoots → gain / clamp saturation.
- Expected vs actual TCP diverge then ring → pose servo tuning / orientation snap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

# Linear keyboard keys → (axis index into twist linear xyz, sign).
_KEY_LINEAR = {
    "w": (0, 1.0),
    "s": (0, -1.0),
    "d": (1, 1.0),
    "a": (1, -1.0),
    "x": (2, 1.0),
    "y": (2, -1.0),
}


def tip_velocity_field_frame(
    v_world: Sequence[float], R_field: np.ndarray
) -> tuple[float, float, float]:
    """Project world linear velocity onto field ``t̂1``, ``t̂2``, outward ``n̂``.

    ``R_field`` columns are ``[t̂1, t̂2, approach]`` with approach **into** the
    surface; outward normal is ``n̂ = -R[:, 2]`` (same as line-tracking).
    """
    R = np.asarray(R_field, dtype=float).reshape(3, 3)
    v = np.asarray(v_world, dtype=float).reshape(3)

    def _unit(col: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(col))
        return col / n if n > 1e-12 else fallback

    t1 = _unit(R[:, 0], np.array([1.0, 0.0, 0.0]))
    t2 = _unit(R[:, 1], np.array([0.0, 1.0, 0.0]))
    n_out = _unit(-R[:, 2], np.array([0.0, 0.0, 1.0]))
    return float(np.dot(v, t1)), float(np.dot(v, t2)), float(np.dot(v, n_out))


def key_wasdxy_pulses(key: str | None) -> tuple[float, float, float, float, float, float]:
    """One-hot signed pulses for w,s,a,d,x,y (0 if not that key)."""
    pulses = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if not key:
        return tuple(pulses)
    order = ("w", "s", "a", "d", "x", "y")
    if key in order:
        pulses[order.index(key)] = 1.0
    return tuple(pulses)


def _pose6(T: np.ndarray | None) -> np.ndarray:
    if T is None:
        return np.full(6, np.nan)
    T = np.asarray(T, dtype=float).reshape(4, 4)
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()])


def _nan6() -> np.ndarray:
    return np.full(6, np.nan)


class MotionTraceLogger:
    """In-memory samples → CSV (+ optional matplotlib plot on save)."""

    HEADER = (
        "t_s,"
        "qdot0,qdot1,qdot2,qdot3,qdot4,qdot5,qdot_norm,"
        "qdot_raw0,qdot_raw1,qdot_raw2,qdot_raw3,qdot_raw4,qdot_raw5,qdot_unclipped_norm,qdot_clipped,"
        "tcp_x,tcp_y,tcp_z,tcp_rx,tcp_ry,tcp_rz,"
        "ref_x,ref_y,ref_z,ref_rx,ref_ry,ref_rz,"
        "err_pos_m,err_rot_rad,err_norm,"
        "key_w,key_s,key_a,key_d,key_x,key_y,"
        "twist_lx,twist_ly,twist_lz,"
        "v_t1,v_t2,v_n,"
        "rm_rx,rm_ry,rm_rz,rm_delta_rad,relocked,query_ms,loop_ms,"
        "scale_factor,distance_m,twist_align_rad,twist_align_motion_rad,"
        "tip_delta_m,dt_since_relock_s,"
        "Fd_N,Fs_N,Fe_N,n_freeze,force_sign,k_ori,"
        "ang_z_deg,phase_align,force_enable,"
        "a_t1,a_t2,a_n"
    )

    def __init__(self) -> None:
        self._t0: float | None = None
        self._R0: np.ndarray | None = None
        self._prev_t_mono: float | None = None
        self._prev_p: np.ndarray | None = None
        self._a_t1 = 0.0
        self._a_t2 = 0.0
        self._a_n = 0.0
        self._rows: list[list[float]] = []
        self._saw_contact = False

    def __len__(self) -> int:
        return len(self._rows)

    def append(
        self,
        *,
        t_mono: float,
        qdot: Sequence[float],
        R_motion: np.ndarray | None,
        relocked: bool = False,
        query_ms: float = 0.0,
        loop_ms: float = 0.0,
        qdot_raw: Sequence[float] | None = None,
        qdot_unclipped_norm: float = float("nan"),
        qdot_clipped: bool = False,
        tcp_pose: Sequence[float] | None = None,
        ref_pose: Sequence[float] | None = None,
        T_tcp: np.ndarray | None = None,
        T_ref: np.ndarray | None = None,
        err_pos_m: float = float("nan"),
        err_rot_rad: float = float("nan"),
        err_norm: float = float("nan"),
        key: str | None = None,
        twist_linear: Sequence[float] | None = None,
        scale_factor: float = float("nan"),
        distance_m: float = float("nan"),
        twist_align_rad: float = float("nan"),
        twist_align_motion_rad: float = float("nan"),
        tip_delta_m: float = float("nan"),
        dt_since_relock_s: float = float("nan"),
        relock_reason: str = "hold",
        v_tip_world: Sequence[float] | None = None,
        Fd_N: float = float("nan"),
        Fs_N: float = float("nan"),
        Fe_N: float = float("nan"),
        n_freeze: float = float("nan"),
        force_sign: float = float("nan"),
        k_ori: float = float("nan"),
        ang_z_deg: float = float("nan"),
        phase_align: float = float("nan"),
        force_enable: float = float("nan"),
        **_extra,
    ) -> None:
        del relock_reason  # logged via relocked flag; keep kw for call-site compat
        if self._t0 is None:
            self._t0 = float(t_mono)
        t_s = float(t_mono) - self._t0
        q = np.asarray(qdot, dtype=float).reshape(6)
        qn = float(np.linalg.norm(q))
        if qdot_raw is None:
            q_raw = np.full(6, np.nan)
            if not np.isnan(qdot_unclipped_norm) and qn > 1e-12:
                # Reconstruct direction from clipped command when raw vector absent.
                q_raw = q * (float(qdot_unclipped_norm) / qn)
            raw_n = float(qdot_unclipped_norm)
        else:
            q_raw = np.asarray(qdot_raw, dtype=float).reshape(6)
            raw_n = float(np.linalg.norm(q_raw))
            if np.isnan(qdot_unclipped_norm):
                qdot_unclipped_norm = raw_n

        if tcp_pose is not None:
            tcp6 = np.asarray(tcp_pose, dtype=float).reshape(6)
        else:
            tcp6 = _pose6(T_tcp)
        if ref_pose is not None:
            ref6 = np.asarray(ref_pose, dtype=float).reshape(6)
        else:
            ref6 = _pose6(T_ref)

        if np.isnan(err_pos_m) and not np.any(np.isnan(tcp6)) and not np.any(np.isnan(ref6)):
            err_pos_m = float(np.linalg.norm(tcp6[:3] - ref6[:3]))
            R_tcp = Rotation.from_rotvec(tcp6[3:6]).as_matrix()
            R_ref = Rotation.from_rotvec(ref6[3:6]).as_matrix()
            err_rot_rad = float(
                np.linalg.norm(Rotation.from_matrix(R_tcp @ R_ref.T).as_rotvec())
            )
            err_norm = float(np.hypot(err_pos_m, err_rot_rad))

        if R_motion is None:
            rotvec = np.zeros(3)
            delta = 0.0
            R = None
        else:
            R = np.asarray(R_motion, dtype=float).reshape(3, 3)
            if self._R0 is None:
                self._R0 = R.copy()
            rotvec = Rotation.from_matrix(R).as_rotvec()
            R_err = self._R0.T @ R
            delta = float(np.linalg.norm(Rotation.from_matrix(R_err).as_rotvec()))

        # Tip / TCP linear velocity in field frame (t̂1, t̂2, n̂_out).
        v_t1 = v_t2 = v_n = float("nan")
        p = tcp6[:3].copy()
        dt_s = float("nan")
        if v_tip_world is not None:
            v_world = np.asarray(v_tip_world, dtype=float).reshape(3)
            if self._prev_t_mono is not None:
                dt_s = float(t_mono) - float(self._prev_t_mono)
        elif (
            self._prev_p is not None
            and self._prev_t_mono is not None
            and not np.any(np.isnan(p))
        ):
            dt_s = float(t_mono) - float(self._prev_t_mono)
            v_world = (p - self._prev_p) / dt_s if dt_s > 1e-9 else np.full(3, np.nan)
        else:
            v_world = None
        if v_world is not None and R is not None and np.all(np.isfinite(v_world)):
            v_t1, v_t2, v_n = tip_velocity_field_frame(v_world, R)
            if np.isfinite(dt_s) and dt_s > 1e-9:
                self._a_t1 += float(v_t1) * dt_s
                self._a_t2 += float(v_t2) * dt_s
                self._a_n += float(v_n) * dt_s
        if not np.any(np.isnan(p)):
            self._prev_p = p
            self._prev_t_mono = float(t_mono)

        tw = (
            np.zeros(3)
            if twist_linear is None
            else np.asarray(twist_linear, dtype=float).reshape(3)
        )
        # If only a key was provided, synthesize a unit pulse on that linear axis.
        if twist_linear is None and key in _KEY_LINEAR:
            ax, sgn = _KEY_LINEAR[key]
            tw = np.zeros(3)
            tw[ax] = sgn

        kw, ks, ka, kd, kx, ky = key_wasdxy_pulses(key)

        if np.isfinite(Fd_N) or np.isfinite(Fs_N) or np.isfinite(Fe_N):
            self._saw_contact = True

        self._rows.append(
            [
                t_s,
                *q.tolist(),
                qn,
                *q_raw.tolist(),
                float(qdot_unclipped_norm),
                1.0 if qdot_clipped else 0.0,
                *tcp6.tolist(),
                *ref6.tolist(),
                float(err_pos_m),
                float(err_rot_rad),
                float(err_norm),
                kw,
                ks,
                ka,
                kd,
                kx,
                ky,
                float(tw[0]),
                float(tw[1]),
                float(tw[2]),
                float(v_t1),
                float(v_t2),
                float(v_n),
                float(rotvec[0]),
                float(rotvec[1]),
                float(rotvec[2]),
                delta,
                1.0 if relocked else 0.0,
                float(query_ms),
                float(loop_ms),
                float(scale_factor),
                float(distance_m),
                float(twist_align_rad),
                float(twist_align_motion_rad),
                float(tip_delta_m),
                float(dt_since_relock_s),
                float(Fd_N),
                float(Fs_N),
                float(Fe_N),
                float(n_freeze),
                float(force_sign),
                float(k_ori),
                float(ang_z_deg),
                float(phase_align),
                float(force_enable),
                float(self._a_t1),
                float(self._a_t2),
                float(self._a_n),
            ]
        )

    def save_csv(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(self.HEADER + "\n")
            for row in self._rows:
                f.write(",".join(f"{v:.9g}" for v in row) + "\n")
        return path

    def as_arrays(self) -> dict[str, np.ndarray]:
        names = [c.strip() for c in self.HEADER.split(",")]
        if not self._rows:
            empty = np.zeros(0)
            out: dict[str, np.ndarray] = {n: empty.copy() for n in names}
            out["qdot"] = np.zeros((0, 6))
            out["qdot_raw"] = np.zeros((0, 6))
            out["tcp_pose"] = np.zeros((0, 6))
            out["ref_pose"] = np.zeros((0, 6))
            out["rm_rotvec"] = np.zeros((0, 3))
            out["twist_lin"] = np.zeros((0, 3))
            out["keys"] = np.zeros((0, 6))
            out["v_field"] = np.zeros((0, 3))
            out["a_field"] = np.zeros((0, 3))
            return out
        arr = np.asarray(self._rows, dtype=float)
        col = {n: i for i, n in enumerate(names)}

        def _c(name: str) -> np.ndarray:
            return arr[:, col[name]]

        return {
            "t_s": _c("t_s"),
            "qdot": arr[:, col["qdot0"] : col["qdot5"] + 1],
            "qdot_norm": _c("qdot_norm"),
            "qdot_raw": arr[:, col["qdot_raw0"] : col["qdot_raw5"] + 1],
            "qdot_unclipped_norm": _c("qdot_unclipped_norm"),
            "qdot_clipped": _c("qdot_clipped"),
            "tcp_pose": arr[:, col["tcp_x"] : col["tcp_rz"] + 1],
            "ref_pose": arr[:, col["ref_x"] : col["ref_rz"] + 1],
            "err_pos_m": _c("err_pos_m"),
            "err_rot_rad": _c("err_rot_rad"),
            "err_norm": _c("err_norm"),
            "keys": arr[:, col["key_w"] : col["key_y"] + 1],
            "twist_lin": arr[:, col["twist_lx"] : col["twist_lz"] + 1],
            "v_field": arr[:, col["v_t1"] : col["v_n"] + 1],
            "a_field": arr[:, col["a_t1"] : col["a_n"] + 1],
            "rm_rotvec": arr[:, col["rm_rx"] : col["rm_rz"] + 1],
            "rm_delta_rad": _c("rm_delta_rad"),
            "relocked": _c("relocked"),
            "query_ms": _c("query_ms"),
            "loop_ms": _c("loop_ms"),
            "scale_factor": _c("scale_factor"),
            "distance_m": _c("distance_m"),
            "twist_align_rad": _c("twist_align_rad"),
            "twist_align_motion_rad": _c("twist_align_motion_rad"),
            "tip_delta_m": _c("tip_delta_m"),
            "dt_since_relock_s": _c("dt_since_relock_s"),
            "Fd_N": _c("Fd_N"),
            "Fs_N": _c("Fs_N"),
            "Fe_N": _c("Fe_N"),
            "n_freeze": _c("n_freeze"),
            "force_sign": _c("force_sign"),
            "k_ori": _c("k_ori"),
            "ang_z_deg": _c("ang_z_deg"),
            "phase_align": _c("phase_align"),
            "force_enable": _c("force_enable"),
            "a_t1": _c("a_t1"),
            "a_t2": _c("a_t2"),
            "a_n": _c("a_n"),
            # flat aliases for tests
            **{n: _c(n) for n in names},
        }

    def prefers_contact_plot(self) -> bool:
        return bool(self._saw_contact)

    def plot(
        self,
        path: Path | str | None = None,
        *,
        show: bool = False,
        style: str = "auto",
    ) -> Path | None:
        """Plot free-space or in-contact motion trace.

        ``style``: ``free`` | ``contact`` | ``auto`` (contact if Fd/Fs/Fe logged).
        """
        mode = str(style or "auto").lower()
        if mode == "auto":
            mode = "contact" if self.prefers_contact_plot() else "free"
        if mode == "contact":
            return self.plot_contact(path, show=show)
        return self.plot_free(path, show=show)

    def plot_free(self, path: Path | str | None = None, *, show: bool = False) -> Path | None:
        """Five-panel free-space: q̇, R_motion, TCP, field tip velocity, WASDXY."""
        try:
            import matplotlib.pyplot as plt
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "matplotlib is required for --plot-motion; "
                "pip install --user 'matplotlib>=3.8' "
                "(system matplotlib often breaks on NumPy 2.x)"
            ) from exc

        data = self.as_arrays()
        t = data["t_s"]
        if t.size == 0:
            return None

        fig, axes = plt.subplots(5, 1, sharex=True, figsize=(11, 12))
        ax0, ax1, ax2, ax_v, ax3 = axes

        for i in range(6):
            ax0.plot(t, data["qdot"][:, i], lw=0.8, label=f"q{i}")
        ax0.plot(t, data["qdot_norm"], "k-", lw=1.2, label="‖q̇‖ clipped")
        ax0.plot(
            t,
            data["qdot_unclipped_norm"],
            "k--",
            lw=1.0,
            alpha=0.8,
            label="‖q̇‖ raw",
        )
        clipped_t = t[data["qdot_clipped"] > 0.5]
        if clipped_t.size:
            ymax = float(np.nanmax(data["qdot_unclipped_norm"]))
            if not np.isfinite(ymax):
                ymax = float(np.nanmax(data["qdot_norm"]))
            ax0.plot(
                clipped_t,
                np.full(clipped_t.shape, ymax * 0.95 if ymax > 0 else 1.0),
                "r|",
                ms=6,
                label="clip",
            )
        ax0.set_ylabel("q̇ (rad/s)")
        ax0.legend(loc="upper right", ncol=4, fontsize=7)
        ax0.grid(True, alpha=0.3)
        ax0.set_title("Joint rates: raw (pre-clamp) vs clipped / sent")

        ax1.plot(t, data["rm_delta_rad"], "C0", lw=1.2, label="∠(R_motion, R0)")
        for i, name in enumerate(("rx", "ry", "rz")):
            ax1.plot(t, data["rm_rotvec"][:, i], lw=0.8, alpha=0.7, label=name)
        relock_t = t[data["relocked"] > 0.5]
        if relock_t.size:
            for ax in (ax0, ax1, ax2, ax_v, ax3):
                for tr in relock_t:
                    ax.axvline(tr, color="C3", alpha=0.2, lw=0.7)
            ax1.plot(
                relock_t,
                np.interp(relock_t, t, data["rm_delta_rad"]),
                "C3o",
                ms=3,
                label="relock",
            )
        ax1.set_ylabel("R_motion (rad)")
        ax1.legend(loc="upper right", ncol=3, fontsize=7)
        ax1.grid(True, alpha=0.3)

        for i, name in enumerate(("x", "y", "z")):
            ax2.plot(t, data["tcp_pose"][:, i], lw=1.0, label=f"tcp {name}")
            ax2.plot(
                t,
                data["ref_pose"][:, i],
                lw=1.0,
                ls="--",
                label=f"ref {name}",
            )
        ax2.plot(t, data["err_pos_m"], "k-", lw=1.1, label="‖Δp‖")
        ax2.plot(t, data["err_rot_rad"], "k--", lw=1.0, alpha=0.8, label="‖ΔR‖")
        ax2.set_ylabel("TCP (m / rad)")
        ax2.legend(loc="upper right", ncol=4, fontsize=7)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Expected (ref) vs actual TCP + pose error")

        ax_v.plot(t, data["v_t1"], lw=1.2, label="v·t̂1")
        ax_v.plot(t, data["v_t2"], lw=1.2, label="v·t̂2")
        ax_v.plot(t, data["v_n"], lw=1.2, label="v·n̂ (out)")
        ax_v.axhline(0.0, color="k", lw=0.6, alpha=0.4)
        ax_v.set_ylabel("v_tip (m/s)")
        ax_v.legend(loc="upper right", ncol=3, fontsize=7)
        ax_v.grid(True, alpha=0.3)
        ax_v.set_title(
            "Tip velocity in field frame (Δp/Δt on TCP; n̂ = −approach)"
        )

        key_names = ("w", "s", "a", "d", "x", "y")
        for i, name in enumerate(key_names):
            ax3.step(
                t,
                data["keys"][:, i] * (1.0 + 0.05 * i),
                where="post",
                lw=1.0,
                label=name,
            )
        for i, name in enumerate(("lx", "ly", "lz")):
            ax3.plot(
                t,
                data["twist_lin"][:, i],
                lw=1.1,
                ls="--",
                alpha=0.85,
                label=f"twist {name}",
            )
        ax3.set_xlabel("t (s)")
        ax3.set_ylabel("keys / twist")
        ax3.legend(loc="upper right", ncol=5, fontsize=7)
        ax3.grid(True, alpha=0.3)
        ax3.set_title("WASDXY keypresses + linear twist command")

        fig.suptitle(
            "Motion trace — clip / TCP lag → tuning; "
            "field v_t1/v_t2/v_n → slide vs plunge; key spikes → command path"
        )
        fig.tight_layout()
        return _savefig(fig, path, show=show)

    def plot_contact(
        self, path: Path | str | None = None, *, show: bool = False
    ) -> Path | None:
        """Six-panel in-contact: force, a_t*, v_t*, R_motion, keys, q̇."""
        try:
            import matplotlib.pyplot as plt
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "matplotlib is required for --plot-motion; "
                "pip install --user 'matplotlib>=3.8' "
                "(system matplotlib often breaks on NumPy 2.x)"
            ) from exc

        data = self.as_arrays()
        t = data["t_s"]
        if t.size == 0:
            return None

        fig, axes = plt.subplots(6, 1, sharex=True, figsize=(11, 14))
        ax_f, ax_a, ax_v, ax_r, ax_k, ax_q = axes

        ax_f.plot(t, data["Fd_N"], "k--", lw=1.4, label="Fd")
        ax_f.plot(t, data["Fs_N"], "C0", lw=1.2, label="Fs")
        ax_f.plot(t, data["Fe_N"], "C3", lw=1.0, alpha=0.9, label="Fe=Fd−Fs")
        ax_f.axhline(0.0, color="k", lw=0.5, alpha=0.35)
        freeze_t = t[np.isfinite(data["n_freeze"]) & (data["n_freeze"] > 0.5)]
        if freeze_t.size:
            for tr in freeze_t[:: max(1, len(freeze_t) // 40)]:
                ax_f.axvline(tr, color="C4", alpha=0.15, lw=0.8)
        ax_f.set_ylabel("Force (N)")
        ax_f.legend(loc="upper right", ncol=3, fontsize=7)
        ax_f.grid(True, alpha=0.3)
        ax_f.set_title("Desired vs measured normal force (Fs along n*, force_sign applied)")

        ax_a.plot(t, data["a_t1"], lw=1.2, label="a_t1 (∫v·t̂1)")
        ax_a.plot(t, data["a_t2"], lw=1.2, label="a_t2 (∫v·t̂2)")
        ax_a.plot(t, data["a_n"], lw=1.2, label="a_n (∫v·n̂_out)")
        ax_a.axhline(0.0, color="k", lw=0.5, alpha=0.35)
        ax_a.set_ylabel("a (m)")
        ax_a.legend(loc="upper right", ncol=3, fontsize=7)
        ax_a.grid(True, alpha=0.3)
        ax_a.set_title("Cumulative tip translation in field frame")

        ax_v.plot(t, data["v_t1"], lw=1.2, label="v·t̂1")
        ax_v.plot(t, data["v_t2"], lw=1.2, label="v·t̂2")
        ax_v.plot(t, data["v_n"], lw=1.2, label="v·n̂ (out)")
        ax_v.axhline(0.0, color="k", lw=0.5, alpha=0.35)
        ax_v.set_ylabel("v (m/s)")
        ax_v.legend(loc="upper right", ncol=3, fontsize=7)
        ax_v.grid(True, alpha=0.3)
        ax_v.set_title("Tip velocity in field frame")

        ax_r.plot(t, data["rm_delta_rad"], "C0", lw=1.2, label="∠(R_motion, R0)")
        if np.any(np.isfinite(data.get("ang_z_deg", np.array([np.nan])))):
            ax_r.plot(
                t,
                np.deg2rad(data["ang_z_deg"]),
                "C2",
                lw=1.3,
                label="ang_z tool→n* (rad)",
            )
        for i, name in enumerate(("rx", "ry", "rz")):
            ax_r.plot(t, data["rm_rotvec"][:, i], lw=0.8, alpha=0.75, label=name)
        if np.any(np.isfinite(data["k_ori"])):
            ax_r2 = ax_r.twinx()
            ax_r2.plot(t, data["k_ori"], "C5:", lw=1.0, alpha=0.8, label="k_ori")
            ax_r2.set_ylabel("k_ori", color="C5")
            ax_r2.tick_params(axis="y", labelcolor="C5")
        seek_mask = np.isfinite(data.get("force_enable", np.array([np.nan]))) & (
            data["force_enable"] > 0.5
        )
        if np.any(seek_mask):
            ax_r.plot(
                t[seek_mask],
                np.full(int(np.count_nonzero(seek_mask)), np.nanmax(data["rm_delta_rad"]) * 1.05),
                "C3.",
                ms=2,
                alpha=0.3,
                label="seek",
            )
        relock_t = t[data["relocked"] > 0.5]
        if relock_t.size:
            for tr in relock_t:
                ax_r.axvline(tr, color="C3", alpha=0.2, lw=0.7)
        ax_r.set_ylabel("orientation (rad)")
        ax_r.legend(loc="upper left", ncol=4, fontsize=7)
        ax_r.grid(True, alpha=0.3)
        ax_r.set_title("Align: ang_z = tool-Z vs surface n* (should fall before seek)")

        key_names = ("w", "s", "a", "d", "x", "y")
        for i, name in enumerate(key_names):
            ax_k.step(
                t,
                data["keys"][:, i] * (1.0 + 0.05 * i),
                where="post",
                lw=1.0,
                label=name,
            )
        for i, name in enumerate(("lx", "ly", "lz")):
            ax_k.plot(
                t,
                data["twist_lin"][:, i],
                lw=1.1,
                ls="--",
                alpha=0.85,
                label=f"twist {name}",
            )
        ax_k.set_ylabel("keys / twist")
        ax_k.legend(loc="upper right", ncol=5, fontsize=7)
        ax_k.grid(True, alpha=0.3)
        ax_k.set_title("Keyboard (tangent w/s a/d in contact) + field twist")

        ax_q.plot(t, data["qdot_norm"], "k-", lw=1.2, label="‖q̇‖")
        for i in range(6):
            ax_q.plot(t, data["qdot"][:, i], lw=0.6, alpha=0.55, label=f"q{i}")
        ax_q.set_xlabel("t (s)")
        ax_q.set_ylabel("q̇ (rad/s)")
        ax_q.legend(loc="upper right", ncol=4, fontsize=7)
        ax_q.grid(True, alpha=0.3)
        ax_q.set_title("Joint rates (contact DualLoop → speedJ)")

        fig.suptitle(
            "In-contact motion — Fd/Fs/Fe tracking; a_t1/a_t2 slide vs a_n plunge; "
            "magenta freeze markers on force panel when n_freeze=1"
        )
        fig.tight_layout()
        return _savefig(fig, path, show=show)


def _savefig(fig, path: Path | str | None, *, show: bool) -> Path | None:
    import matplotlib.pyplot as plt

    out: Path | None = None
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def save_and_maybe_plot(
    logger: MotionTraceLogger,
    csv_path: Path | str,
    *,
    plot: bool = False,
    plot_path: Path | str | None = None,
    show: bool = False,
    style: str = "auto",
) -> tuple[Path, Path | None]:
    csv = logger.save_csv(csv_path)
    png: Path | None = None
    if plot:
        if plot_path is None:
            plot_path = Path(csv_path).with_suffix(".png")
        try:
            png = logger.plot(plot_path, show=show, style=style)
        except Exception as exc:
            raise RuntimeError(
                f"wrote {csv} but plot failed ({exc}); "
                "fix with: pip install --user 'matplotlib>=3.8'"
            ) from exc
    return csv, png
