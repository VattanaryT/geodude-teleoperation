"""Payload calibration section builders and apply helpers."""

from __future__ import annotations

from typing import Any

from geodude_teleoperation.calibration.io import validate_payload


def build_payload_section(
    mass_kg: float, cog: list[float], source: str = "manual"
) -> dict[str, Any]:
    mass_kg, cog = validate_payload(mass_kg, cog)
    return {
        "mass_kg": mass_kg,
        "cog": cog,
        "source": source,
    }


def apply_payload(session: Any, mass_kg: float, cog: list[float]) -> None:
    mass_kg, cog = validate_payload(mass_kg, cog)
    session.set_payload(mass_kg, cog)
