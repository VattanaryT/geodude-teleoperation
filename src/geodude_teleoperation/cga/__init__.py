"""Conformal GA contact control (pygafro): line tracking + Eq. 56 torque law."""

from __future__ import annotations

from geodude_teleoperation.cga.tactile_law import (
    Eq56Gains,
    assemble_eq56_tau,
    default_eq56_gains,
)
from geodude_teleoperation.cga.line_tracker import (
    alignment_twist_se3,
    ee_line_from_tip,
    target_line_from_approach,
)

__all__ = [
    "Eq56Gains",
    "alignment_twist_se3",
    "assemble_eq56_tau",
    "default_eq56_gains",
    "ee_line_from_tip",
    "target_line_from_approach",
]
