"""Map gafro right-arm FK onto bag ``right_base`` TCP.

UR ``right_base`` is gafro ``right_ur5e/base`` yawed 180° about Z. For
bags_08_16, measured TCP **position** is the chamfer tip and **orientation**
is the flange / tool0 (not the later taught TCP rotvec).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from geodude_teleoperation.ee_viz import (
    chamfer_tcp_world_from_wrist3,
    flange_from_wrist3,
)

MODEL = Path(__file__).resolve().parent.parent.parent / "assets" / "robots" / "geodude" / "geodude.yaml"
RIGHT_BASE = "right_ur5e/base"
RIGHT_WRIST3 = "right_ur5e/wrist_3_link"
RIGHT_EE = "right_ur5e/endeffector_link"

# gafro right_ur5e/base_fixed is 180° about Z vs UR right_base.
GAFRO_BASE_TO_RIGHT_BASE = np.diag([-1.0, -1.0, 1.0])

BAG_JOINT_TO_GAFRO = {
    "right_shoulder_pan_joint": "right_ur5e/shoulder_pan_joint",
    "right_shoulder_lift_joint": "right_ur5e/shoulder_lift_joint",
    "right_elbow_joint": "right_ur5e/elbow_joint",
    "right_wrist_1_joint": "right_ur5e/wrist_1_joint",
    "right_wrist_2_joint": "right_ur5e/wrist_2_joint",
    "right_wrist_3_joint": "right_ur5e/wrist_3_joint",
}

UR6_ORDER = (
    "right_shoulder_pan_joint",
    "right_shoulder_lift_joint",
    "right_elbow_joint",
    "right_wrist_1_joint",
    "right_wrist_2_joint",
    "right_wrist_3_joint",
)


def _link_T(fk, name: str) -> np.ndarray:
    return np.asarray(fk.get_link_pose(name).to_transformation_matrix(), dtype=float)


def apply_right_ur6(system, q: np.ndarray, q_ur6: np.ndarray) -> np.ndarray:
    """Copy 6 UR joints (``UR6_ORDER``) into a full gafro configuration."""
    out = np.asarray(q, dtype=float).copy()
    q6 = np.asarray(q_ur6, dtype=float).reshape(6)
    by_name = {j.get_name(): j.get_index() for j in system.get_joints() if j.is_actuated()}
    for bag_name, val in zip(UR6_ORDER, q6):
        out[by_name[BAG_JOINT_TO_GAFRO[bag_name]]] = float(val)
    return out


def poses_in_right_base(system, q_full: np.ndarray) -> dict[str, np.ndarray]:
    """Chamfer / flange / yaml-EE poses in UR ``right_base``."""
    fk = system.compute_forward_kinematics(np.asarray(q_full, dtype=float))
    T_base = _link_T(fk, RIGHT_BASE)
    T_wrist = _link_T(fk, RIGHT_WRIST3)
    T_ee = _link_T(fk, RIGHT_EE)
    inv_base = np.linalg.inv(T_base)
    R_fix = GAFRO_BASE_TO_RIGHT_BASE

    def to_right_base(T_world: np.ndarray) -> np.ndarray:
        T = inv_base @ np.asarray(T_world, dtype=float).reshape(4, 4)
        out = np.eye(4)
        out[:3, :3] = R_fix @ T[:3, :3]
        out[:3, 3] = R_fix @ T[:3, 3]
        return out

    T_flange_w = T_wrist @ flange_from_wrist3()
    return {
        "chamfer": to_right_base(chamfer_tcp_world_from_wrist3(T_wrist)),
        "flange": to_right_base(T_flange_w),
        "yaml_ee": to_right_base(T_ee),
    }


def _yaml_joint_block(q6: np.ndarray) -> str:
    return "\n".join(
        f"          - {float(v)}" for v in np.asarray(q6, dtype=float).reshape(6)
    )


def write_geodude_model_with_right_ur6(
    q_ur6: np.ndarray,
    dest: Path,
    *,
    src: Path = MODEL,
) -> Path:
    """Copy ``src`` with ready/home right-UR6 joints set to ``q_ur6``.

    MuJoCo boots from those keyframes. ``go_to_joint_position`` deadlocks the
    Python GIL against the integrate thread, so the start pose has to be in
    the model file.
    """
    src = Path(src)
    dest = Path(dest)
    text = src.read_text()
    import gafropy as ga

    system = ga.SystemSerialization.load(str(src))
    ready = np.asarray(system.get_keyframe("ready").qpos, dtype=float)
    by_name = {
        j.get_name(): j.get_index()
        for j in system.get_joints() if j.is_actuated()
    }
    old_q6 = np.array(
        [ready[by_name[BAG_JOINT_TO_GAFRO[n]]] for n in UR6_ORDER], dtype=float
    )
    old_block = _yaml_joint_block(old_q6)
    if old_block not in text:
        raise ValueError(f"right UR6 keyframe block not found in {src}")
    n = text.count(old_block)
    if n < 2:
        raise ValueError(f"expected ready+home right UR6 blocks, found {n} in {src}")
    text = text.replace(old_block, _yaml_joint_block(q_ur6))
    if dest.parent.resolve() != src.parent.resolve():
        mesh = (src.parent / "assets").resolve()
        text = text.replace("meshdir: assets", f"meshdir: {mesh}", 1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    return dest


def rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    from scipy.spatial.transform import Rotation

    rel = Rotation.from_matrix(
        np.asarray(R_a, dtype=float).reshape(3, 3).T
        @ np.asarray(R_b, dtype=float).reshape(3, 3)
    )
    return float(np.degrees(np.linalg.norm(rel.as_rotvec())))
