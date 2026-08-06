"""TCP calibration section builders and apply helpers."""

from __future__ import annotations

from typing import Any

from geodude_teleoperation.calibration.io import validate_tcp_pose


def build_tcp_section(pose: list[float], source: str = "manual") -> dict[str, Any]:
    pose = validate_tcp_pose(pose)
    return {
        "pose": pose,
        "frame": "tool0",
        "source": source,
    }


def apply_tcp(session: Any, pose: list[float]) -> None:
    session.set_tcp(validate_tcp_pose(pose))
