"""Force/torque sensor zero and bias sampling helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from geodude_teleoperation.calibration.io import validate_wrench


def build_ft_section(
    bias_wrench: list[float],
    sample_count: int,
    note: str = "",
    zeroed_at: Optional[str] = None,
) -> dict[str, Any]:
    bias_wrench = validate_wrench(bias_wrench)
    if sample_count <= 0:
        raise ValueError(f"sample_count must be > 0, got {sample_count}")
    if zeroed_at is None:
        zeroed_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    return {
        "zeroed_at": zeroed_at,
        "bias_wrench": bias_wrench,
        "sample_count": int(sample_count),
        "note": note,
    }


def sample_ft_bias(session: Any, samples: int) -> list[float]:
    return validate_wrench(session.average_tcp_force(samples=samples))


def apply_ft_zero(session: Any) -> None:
    session.zero_ft_sensor()
