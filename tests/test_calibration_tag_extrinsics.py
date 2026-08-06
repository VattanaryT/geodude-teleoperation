"""Unit tests for tag extrinsics validation (synthetic SE3 marks)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geodude_teleoperation.calibration import tag_extrinsics as te


def _T(R=None, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def _rotz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _write_mark(
    session: Path,
    mark_id: str,
    cam: str,
    *,
    T_EE_base: np.ndarray,
    T_cam_EE: np.ndarray,
    T_tag_cam: np.ndarray,
    tag_id: int = 0,
) -> None:
    cam_dir = session / "marked" / mark_id / cam
    cam_dir.mkdir(parents=True, exist_ok=True)
    T_tag_base = te.compose_T_tag_base(T_EE_base, T_cam_EE, T_tag_cam)
    meta = {
        "ee_frame": "right_tool0",
        "cam_frame": "wrist_cam_0_color_optical_frame",
        "T_EE_base": T_EE_base.tolist(),
        "T_cam_EE": T_cam_EE.tolist(),
        "T_cam_base": (T_EE_base @ T_cam_EE).tolist(),
    }
    tags = {
        "origin_tag_id": tag_id,
        "tags": [
            {
                "id": tag_id,
                "T_tag_cam": T_tag_cam.tolist(),
                "T_tag_base": T_tag_base.tolist(),
            }
        ],
    }
    (cam_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (cam_dir / "tags.json").write_text(json.dumps(tags), encoding="utf-8")


def test_multiview_consistent_passes(tmp_path: Path):
    cam = "wrist_cam_0"
    T_cam_EE = _T(t=(0.05, 0.0, 0.08))
    # Fixed tag in base; two EE poses looking at it.
    T_tag_base = _T(t=(0.4, 0.1, 0.05))
    samples_spec = []
    for i, yaw in enumerate((0.0, 25.0)):
        T_EE = _T(R=_rotz(yaw), t=(0.3 + 0.02 * i, 0.0, 0.2))
        T_tag_cam = te.invert_SE3(T_EE @ T_cam_EE) @ T_tag_base
        mid = f"{i:04d}"
        _write_mark(
            tmp_path, mid, cam, T_EE_base=T_EE, T_cam_EE=T_cam_EE, T_tag_cam=T_tag_cam
        )
        samples_spec.append(mid)
    samples = te.load_samples(tmp_path, cam, 0)
    result = te.score_multiview(samples, max_trans=1e-6, max_rot_deg=1e-3)
    assert result.passed
    assert result.n_views == 2
    assert result.max_trans_m < 1e-9


def test_multiview_detects_inconsistency(tmp_path: Path):
    cam = "wrist_cam_0"
    T_cam_EE = _T(t=(0.05, 0.0, 0.08))
    T_tag_base = _T(t=(0.4, 0.1, 0.05))
    for i, yaw in enumerate((0.0, 20.0)):
        T_EE = _T(R=_rotz(yaw), t=(0.3, 0.0, 0.2))
        T_tag_cam = te.invert_SE3(T_EE @ T_cam_EE) @ T_tag_base
        if i == 1:
            T_tag_cam = T_tag_cam.copy()
            T_tag_cam[:3, 3] += np.array([0.02, 0.0, 0.0])  # 2 cm error
        _write_mark(
            tmp_path,
            f"{i:04d}",
            cam,
            T_EE_base=T_EE,
            T_cam_EE=T_cam_EE,
            T_tag_cam=T_tag_cam,
        )
    samples = te.load_samples(tmp_path, cam, 0)
    result = te.score_multiview(samples, max_trans=0.005, max_rot_deg=2.0)
    assert not result.passed
    assert result.max_trans_m > 0.005


def test_tip_contact_perfect(tmp_path: Path):
    cam = "wrist_cam_0"
    T_cam_EE = _T(t=(0.02, -0.01, 0.12))
    # Tip on tag: tag origin = EE origin in base; orientations may differ.
    T_EE = _T(R=_rotz(10.0), t=(0.5, 0.2, 0.1))
    T_tag_base = _T(R=_rotz(-5.0), t=T_EE[:3, 3])  # same origin
    T_tag_cam = te.invert_SE3(T_EE @ T_cam_EE) @ T_tag_base
    _write_mark(
        tmp_path, "0000", cam, T_EE_base=T_EE, T_cam_EE=T_cam_EE, T_tag_cam=T_tag_cam
    )
    samples = te.load_samples(tmp_path, cam, 0)
    result = te.score_tip_contact(samples, max_tip_tag=1e-6, max_cam=1e-6)
    assert result.passed
    assert result.max_tip_tag_m < 1e-9
    assert result.max_cam_err_m < 1e-9
    assert abs(result.per_view[0]["cam_to_tag_m"] - result.per_view[0]["cam_to_tip_m"]) < 1e-9


def test_tip_contact_fails_when_offset(tmp_path: Path):
    cam = "wrist_cam_0"
    T_cam_EE = _T(t=(0.0, 0.0, 0.1))
    T_EE = _T(t=(0.5, 0.0, 0.1))
    T_tag_base = _T(t=(0.5, 0.0, 0.13))  # 3 cm above tip
    T_tag_cam = te.invert_SE3(T_EE @ T_cam_EE) @ T_tag_base
    _write_mark(
        tmp_path, "0000", cam, T_EE_base=T_EE, T_cam_EE=T_cam_EE, T_tag_cam=T_tag_cam
    )
    samples = te.load_samples(tmp_path, cam, 0)
    result = te.score_tip_contact(samples, max_tip_tag=0.005, max_cam=0.01)
    assert not result.passed
    assert result.max_tip_tag_m == pytest.approx(0.03, abs=1e-9)


def test_run_validation_both(tmp_path: Path):
    cam = "wrist_cam_0"
    T_cam_EE = _T(t=(0.05, 0.0, 0.08))
    T_tag_base = _T(t=(0.4, 0.1, 0.05))
    for i, yaw in enumerate((0.0, 15.0)):
        T_EE = _T(R=_rotz(yaw), t=(0.3, 0.05 * i, 0.2))
        T_tag_cam = te.invert_SE3(T_EE @ T_cam_EE) @ T_tag_base
        _write_mark(
            tmp_path,
            f"{i:04d}",
            cam,
            T_EE_base=T_EE,
            T_cam_EE=T_cam_EE,
            T_tag_cam=T_tag_cam,
        )
    # Contact mark: tip on tag
    T_EE_c = _T(t=T_tag_base[:3, 3])
    T_tag_cam_c = te.invert_SE3(T_EE_c @ T_cam_EE) @ T_tag_base
    _write_mark(
        tmp_path,
        "0002",
        cam,
        T_EE_base=T_EE_c,
        T_cam_EE=T_cam_EE,
        T_tag_cam=T_tag_cam_c,
    )
    mv, tip, passed = te.run_validation(
        tmp_path,
        cam=cam,
        tag_id=0,
        mode="both",
        contact_marks=["0002"],
        max_trans=1e-6,
        max_rot_deg=1e-3,
        max_tip_tag=1e-6,
        max_cam=1e-6,
    )
    assert mv is not None and tip is not None
    assert mv.n_views == 3
    assert tip.n_views == 1
    assert passed
