"""YAML IO and validators for UR TCP / payload / FT calibration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

VALID_ARMS = ("left", "right")


def default_calib_dir() -> Path:
    """Repo-root ``calibration/`` (parent of ``src/``)."""
    # .../geodude-teleoperation/src/geodude_teleoperation/calibration/io.py
    return Path(__file__).resolve().parents[3] / "calibration"


def arm_yaml_path(calib_dir: Path, arm: str) -> Path:
    arm = _require_arm(arm)
    return Path(calib_dir) / f"{arm}.yaml"


def parse_float_list(text: str, expected_len: int) -> list[float]:
    parts = [p.strip() for p in str(text).split(",") if p.strip() != ""]
    if len(parts) != expected_len:
        raise ValueError(
            f"expected {expected_len} comma-separated numbers, got {len(parts)}: {text!r}"
        )
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"non-numeric value in list: {text!r}") from exc


def validate_tcp_pose(pose: list[float]) -> list[float]:
    values = [float(v) for v in pose]
    if len(values) != 6:
        raise ValueError(f"tcp pose must have length 6 [x,y,z,rx,ry,rz], got {len(values)}")
    return values


def validate_payload(mass_kg: float, cog: list[float]) -> tuple[float, list[float]]:
    mass = float(mass_kg)
    if mass <= 0.0:
        raise ValueError(f"payload mass_kg must be > 0, got {mass}")
    cog_vals = [float(v) for v in cog]
    if len(cog_vals) != 3:
        raise ValueError(f"payload cog must have length 3, got {len(cog_vals)}")
    return mass, cog_vals


def validate_wrench(wrench: list[float]) -> list[float]:
    values = [float(v) for v in wrench]
    if len(values) != 6:
        raise ValueError(f"wrench must have length 6 [Fx,Fy,Fz,Tx,Ty,Tz], got {len(values)}")
    return values


def load_arm_calib(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"calibration file must be a mapping: {path}")
    return data


def save_arm_calib(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def merge_section(
    data: dict[str, Any],
    arm: str,
    section: str,
    section_body: dict[str, Any],
    robot_ip: Optional[str] = None,
) -> dict[str, Any]:
    """Return a copy of ``data`` with ``section`` replaced and metadata updated."""
    arm = _require_arm(arm)
    out = dict(data)
    out["arm"] = arm
    out["updated_at"] = _now_iso()
    if robot_ip is not None:
        out["robot_ip"] = str(robot_ip)
    out[section] = dict(section_body)
    return out


def _require_arm(arm: str) -> str:
    arm = str(arm).strip().lower()
    if arm not in VALID_ARMS:
        raise ValueError(f"arm must be one of {VALID_ARMS}, got {arm!r}")
    return arm


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
