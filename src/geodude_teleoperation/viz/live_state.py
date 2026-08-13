"""Thread-safe TCP / VF snapshot for live Polyscope updates."""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def latency_table_rows(
    snap: Mapping[str, Any],
    *,
    now_mono: float | None = None,
) -> list[tuple[str, str, bool]]:
    """Format latency snapshot into ``(name, value, warn)`` rows for Polyscope.

    ``warn`` is True when ``loop`` or ``query`` exceeds ``target_dt_ms``.
    ``viz age`` is ``(now_mono - publish_mono)`` in ms when both are set.
    """

    def _fmt(v) -> str:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return "n/a"
        if not np.isfinite(x):
            return "n/a"
        return f"{x:.1f}"

    target = snap.get("target_dt_ms", float("nan"))
    loop = snap.get("loop_ms", float("nan"))
    query = snap.get("query_ms", float("nan"))
    relock = snap.get("relock_age_ms", float("nan"))
    reason = str(snap.get("relock_reason") or "").strip()
    sm = snap.get("sm_age_ms")
    pub_t = snap.get("publish_mono")
    viz_age = float("nan")
    if now_mono is not None and pub_t is not None:
        try:
            viz_age = (float(now_mono) - float(pub_t)) * 1000.0
        except (TypeError, ValueError):
            viz_age = float("nan")

    def _over_target(v) -> bool:
        try:
            vv = float(v)
            tt = float(target)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(vv) and np.isfinite(tt) and vv > tt + 1e-9)

    relock_s = _fmt(relock)
    if reason:
        relock_s = f"{relock_s}  {reason}"
    sm_s = "n/a" if sm is None else _fmt(sm)
    return [
        ("target dt", _fmt(target), False),
        ("loop", _fmt(loop), _over_target(loop)),
        ("query (WoS)", _fmt(query), _over_target(query)),
        ("relock age", relock_s, False),
        ("spacemouse age", sm_s, False),
        ("viz age", _fmt(viz_age), False),
    ]


class LiveVizState:
    """Thread-safe latest TCP / VF snapshot for Polyscope live updates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tcp_pose: np.ndarray | None = None
        self._R_field_tcp: np.ndarray | None = None
        self._signed_d: float | None = None
        self._in_contact: bool | None = None
        self._seq: int = 0
        self._viz_open = True
        self.loop_done = threading.Event()
        self.loop_exit_code = 0
        self.loop_error: BaseException | None = None
        # Shared with args.object_offset_t (same ndarray) when bound.
        self.object_offset_t: np.ndarray | None = None
        self.laser_offset_t: np.ndarray | None = None
        self._offset_seq: int = 0
        # Polyscope → RTDE keyboard bridge (stdin is dead while the GUI has focus).
        self._teleop_keys: queue.Queue[str] = queue.Queue()
        # Continuous hold (WASD/X/Y) sampled each Polyscope frame.
        self._held_teleop: frozenset[str] = frozenset()
        self._latency: dict[str, Any] = {}
        self.show_latency_table: bool = False

    @property
    def viz_open(self) -> bool:
        with self._lock:
            return self._viz_open

    def mark_viz_closed(self) -> None:
        with self._lock:
            self._viz_open = False

    def bump_offset_seq(self) -> None:
        with self._lock:
            self._offset_seq += 1

    def offset_seq(self) -> int:
        with self._lock:
            return self._offset_seq

    def push_teleop_key(self, key: str) -> None:
        """Queue a teleop key from the Polyscope thread (Enter, t/v/f/q, …)."""
        if not key:
            return
        self._teleop_keys.put(str(key))

    def drain_teleop_keys(self) -> list[str]:
        """Pop all pending Polyscope one-shot keys for the RTDE keyboard handler."""
        out: list[str] = []
        while True:
            try:
                out.append(self._teleop_keys.get_nowait())
            except queue.Empty:
                break
        return out

    def set_held_teleop_keys(self, keys: Iterable[str]) -> None:
        """Update which movement keys are currently held in the Polyscope window."""
        with self._lock:
            self._held_teleop = frozenset(str(k) for k in keys if k)

    def held_teleop_keys(self) -> frozenset[str]:
        with self._lock:
            return self._held_teleop

    def publish(
        self,
        tcp_pose: Sequence[float],
        *,
        R_field_tcp: np.ndarray | None = None,
        signed_d: float | None = None,
        in_contact: bool | None = None,
    ) -> None:
        with self._lock:
            self._tcp_pose = np.asarray(tcp_pose, dtype=float).reshape(6).copy()
            if R_field_tcp is not None:
                self._R_field_tcp = np.asarray(R_field_tcp, dtype=float).reshape(3, 3).copy()
            if signed_d is not None:
                self._signed_d = float(signed_d)
            if in_contact is not None:
                self._in_contact = bool(in_contact)
            self._seq += 1

    def snapshot(
        self,
    ) -> tuple[int, np.ndarray | None, np.ndarray | None, float | None, bool | None]:
        with self._lock:
            tcp = None if self._tcp_pose is None else self._tcp_pose.copy()
            R = None if self._R_field_tcp is None else self._R_field_tcp.copy()
            return self._seq, tcp, R, self._signed_d, self._in_contact

    def publish_latency(
        self,
        *,
        loop_ms: float,
        query_ms: float,
        relock_age_ms: float,
        relock_reason: str = "",
        sm_age_ms: float | None = None,
        target_dt_ms: float,
        publish_mono: float | None = None,
    ) -> None:
        with self._lock:
            self._latency = {
                "loop_ms": float(loop_ms),
                "query_ms": float(query_ms),
                "relock_age_ms": float(relock_age_ms),
                "relock_reason": str(relock_reason or ""),
                "sm_age_ms": None if sm_age_ms is None else float(sm_age_ms),
                "target_dt_ms": float(target_dt_ms),
                "publish_mono": float(
                    time.monotonic() if publish_mono is None else publish_mono
                ),
            }

    def latency_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latency)
