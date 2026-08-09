"""AprilTag vs live-TF calibrate cross-check helpers (pure math + optional ROS)."""

from __future__ import annotations

from typing import Any

import numpy as np


def apriltag_T_tag0_base(
    T_EE_base: np.ndarray, T_cam_EE: np.ndarray, T_tag_cam: np.ndarray
) -> np.ndarray:
    """T_tag0^base = T_EE^base @ T_cam^EE @ T_tag^cam (same as ee_object_calib)."""
    return (
        np.asarray(T_EE_base, dtype=float)
        @ np.asarray(T_cam_EE, dtype=float)
        @ np.asarray(T_tag_cam, dtype=float)
    )


def tag_vs_tf_translation_mm(T_tag_tf: np.ndarray, T_tag_april: np.ndarray) -> float:
    """Translation disagreement between two tag_0-in-base poses (millimetres)."""
    delta = np.linalg.inv(np.asarray(T_tag_tf, dtype=float)) @ np.asarray(
        T_tag_april, dtype=float
    )
    return float(np.linalg.norm(delta[:3, 3]) * 1000.0)


def compute_tag_vs_tf_mm(
    T_tag_tf: np.ndarray | None,
    T_EE_base: np.ndarray | None,
    T_cam_EE: np.ndarray | None,
    T_tag_cam: np.ndarray | None,
) -> float | None:
    """Return disagreement in mm when TF and AprilTag chain inputs are all present."""
    if T_tag_tf is None or T_EE_base is None or T_cam_EE is None or T_tag_cam is None:
        return None
    T_april = apriltag_T_tag0_base(T_EE_base, T_cam_EE, T_tag_cam)
    return tag_vs_tf_translation_mm(T_tag_tf, T_april)


def load_T_cam_EE(path) -> np.ndarray | None:
    """Load wrist-camera extrinsics from ee_object_calib mark meta JSON."""
    if path is None:
        return None
    try:
        import json
        from pathlib import Path

        meta_path = Path(path)
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        return np.asarray(meta["T_cam_EE"], dtype=float).reshape(4, 4)
    except (OSError, KeyError, ValueError, TypeError):
        return None


def tcp_pose_to_T_EE_base(tcp_pose) -> np.ndarray:
    """UR getActualTCPPose [x,y,z,rx,ry,rz] axis-angle → 4x4 T_EE^base."""
    from scipy.spatial.transform import Rotation

    p = np.asarray(tcp_pose[:3], dtype=float)
    R = Rotation.from_rotvec(np.asarray(tcp_pose[3:6], dtype=float)).as_matrix()
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def try_detect_T_tag_cam(
    rgb: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None,
    *,
    tag_size_m: float,
    tag_id: int,
    family: str = "tag36h11",
) -> np.ndarray | None:
    """Detect origin AprilTag in an RGB frame; return T_tag^cam or None."""
    try:
        import cv2
    except ImportError:
        return None

    try:
        from bimanual.scripts.ee_object_calib_geom import estimate_tag_pose_opencv
    except ImportError:
        return None

    family_map = {
        "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
        "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    }
    dict_id = family_map.get(str(family).lower())
    if dict_id is None:
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.ndim == 3 else rgb
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None

    wanted = int(tag_id)
    for corner_set, detected_id in zip(corners, ids.reshape(-1)):
        if int(detected_id) != wanted:
            continue
        return estimate_tag_pose_opencv(
            corner_set.reshape(4, 2),
            float(tag_size_m),
            camera_matrix,
            dist_coeffs,
        )
    return None


def resolve_tag_vs_tf_mm(
    args: Any,
    *,
    T_tag_tf: np.ndarray | None,
    tcp_pose,
    rgb: np.ndarray | None,
    camera_matrix: np.ndarray | None,
    dist_coeffs: np.ndarray | None,
) -> float | None:
    """Best-effort AprilTag vs TF disagreement for calibrate status line."""
    T_cam_EE = load_T_cam_EE(getattr(args, "calib_meta", None))
    if T_cam_EE is None:
        return None
    T_EE_base = tcp_pose_to_T_EE_base(tcp_pose)
    tag_size_m = float(getattr(args, "tag_size_m", 0.04))
    tag_id = int(getattr(args, "tag_id", 0))
    family = str(getattr(args, "tag_family", "tag36h11"))
    T_tag_cam = None
    if rgb is not None and camera_matrix is not None:
        T_tag_cam = try_detect_T_tag_cam(
            rgb,
            camera_matrix,
            dist_coeffs,
            tag_size_m=tag_size_m,
            tag_id=tag_id,
            family=family,
        )
    return compute_tag_vs_tf_mm(T_tag_tf, T_EE_base, T_cam_EE, T_tag_cam)
