"""Teleop mode flags, command gating, and status line formatting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class TeleopState:
    mode: str = "calibrate"
    motion_enabled: bool = False
    teleop_active: bool = False
    vf_enabled: bool = True

    def enable(self) -> None:
        self.motion_enabled = True
        self.teleop_active = True
        self.mode = "teleop"

    def commands_allowed(self, *, dry_run: bool, fault: bool, tf_ok: bool) -> bool:
        if dry_run or fault or not tf_ok:
            return False
        return self.motion_enabled and self.teleop_active


def _flag(value: bool) -> str:
    return "1" if value else "0"


def _format_signed_d(signed_d: float) -> str:
    return f"{signed_d:+.4f}"


def should_emit_status(last_emit: float, now: float, *, status_hz: float) -> bool:
    """Throttle status-line prints. ``status_hz <= 0`` → every tick."""
    hz = float(status_hz)
    if hz <= 0.0:
        return True
    return (float(now) - float(last_emit)) >= (1.0 / hz)


def format_status_line(
    state: TeleopState,
    vf_status: Mapping[str, Any],
    *,
    qdot_norm: float,
    tf_age_ms: float,
    tag_vs_tf_mm: float | None = None,
) -> str:
    signed_d = float(vf_status.get("signed_d", 0.0))
    in_contact = bool(vf_status.get("in_contact", False))
    used_wos = bool(vf_status.get("used_wos", False))
    tip_nn = float(vf_status.get("tip_nn", 0.0))
    query_ms = float(vf_status.get("query_ms", 0.0))

    tag_vs_tf = "N/A" if tag_vs_tf_mm is None else f"{tag_vs_tf_mm:.2f}"

    parts = [
        f"mode={state.mode}",
        f"motion_enabled={_flag(state.motion_enabled)}",
        f"teleop_active={_flag(state.teleop_active)}",
        f"d={_format_signed_d(signed_d)}",
        f"in_contact={_flag(in_contact)}",
        f"wos={_flag(used_wos)}",
        f"tip_nn={tip_nn:.4f}",
        f"tag_vs_tf_mm={tag_vs_tf}",
        f"query_ms={query_ms:.2f}",
        f"qdot_norm={qdot_norm:.4f}",
        f"tf_age_ms={tf_age_ms:.2f}",
    ]
    return "  ".join(parts)
