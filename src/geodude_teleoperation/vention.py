"""Hardware does not drive or sense the vention prismatic rails.

left_ee / right_ee kinematic chains still include those joints (task-space
DoF 7 = rail + 6 UR). With no driver, RobotInterface reports them at 0, which
is the bottom of the 0–0.5 m slide — both arms render ~25 cm too low against
the vention frame. Overlay the physical rail height onto a copy of q for
visualization (and any FK that should match that drawing). Do not write the
overlaid q into a hold-pose control target: the C++ loop still samples
unpatched hardware state, and a 0.25 m z-error would command the UR joints up.
"""

from __future__ import annotations

import numpy as np

LEFT_VENTION_JOINT = "left_arm_linear_vention"
RIGHT_VENTION_JOINT = "right_arm_linear_vention"

# Matches geodude.yaml keyframe "ready" and the examples' DEFAULT_VENTION_HEIGHT.
DEFAULT_VENTION_HEIGHT = 0.25
# Prismatic origin z in geodude.yaml. Calibrated T_*_base_vention t_z is 1.65 m;
# rail q = 0.25 then leaves the UR mount 280 mm low vs that frame.
VENTION_PRISMATIC_ORIGIN_Z = 1.12
VENTION_Q_MAX = 0.5


def vention_q_for_base_z(z_vention: float) -> float:
    """Rail q so yaml UR mount z matches a calibrated vention z, clamped to limits."""
    return float(np.clip(
        float(z_vention) - VENTION_PRISMATIC_ORIGIN_Z, 0.0, VENTION_Q_MAX))


def vention_dof_indices(system) -> dict[str, int]:
    """Actuated-DoF indices of the two vention prismatic joints."""
    return {
        joint.get_name(): joint.get_index()
        for joint in system.get_joints()
        if joint.is_actuated() and joint.get_name() in (LEFT_VENTION_JOINT, RIGHT_VENTION_JOINT)
    }


def apply_vention_heights(system, q, left: float, right: float) -> np.ndarray:
    """Return a copy of q with both rail joints set. Does not modify q."""
    indices = vention_dof_indices(system)
    out = np.asarray(q, dtype=float).copy()
    out[indices[LEFT_VENTION_JOINT]] = left
    out[indices[RIGHT_VENTION_JOINT]] = right
    return out
