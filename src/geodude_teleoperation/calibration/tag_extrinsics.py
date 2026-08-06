"""Validate tag extrinsics from ee_object_calib session marks.

Two checks (see design doc):
  1. Multi-view consistency of T_tag_base across free-space marks.
  2. Tip-contact: EE/TCP origin vs tag when tip touches the tag.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


def compose_T_tag_base(
    T_EE_base: np.ndarray, T_cam_EE: np.ndarray, T_tag_cam: np.ndarray
) -> np.ndarray:
    return np.asarray(T_EE_base, dtype=float) @ np.asarray(T_cam_EE, dtype=float) @ np.asarray(
        T_tag_cam, dtype=float
    )


def rotation_angle_deg(R: np.ndarray) -> float:
    """Geodesic angle (deg) of a rotation matrix."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_theta = float((np.trace(R) - 1.0) * 0.5)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


def mean_rotation_svd(rotations: Sequence[np.ndarray]) -> np.ndarray:
    """Project average of rotation matrices onto SO(3) via SVD."""
    if not rotations:
        raise ValueError("need at least one rotation")
    M = np.zeros((3, 3), dtype=float)
    for R in rotations:
        M += np.asarray(R, dtype=float).reshape(3, 3)
    M /= len(rotations)
    U, _, Vt = np.linalg.svd(M)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U = U.copy()
        U[:, -1] *= -1.0
        R_mean = U @ Vt
    return R_mean


def invert_SE3(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=float)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


@dataclass
class MarkSample:
    mark_id: str
    cam: str
    T_EE_base: np.ndarray
    T_cam_EE: np.ndarray
    T_tag_cam: np.ndarray
    T_tag_base: np.ndarray
    tag_id: int
    ee_frame: str = ""
    cam_frame: str = ""


@dataclass
class MultiviewResult:
    n_views: int
    rms_trans_m: float
    max_trans_m: float
    rms_rot_deg: float
    max_rot_deg: float
    per_view: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TipContactResult:
    n_views: int
    max_tip_tag_m: float
    rms_tip_tag_m: float
    max_cam_err_m: float
    rms_cam_err_m: float
    per_view: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_matrix(obj: Any) -> np.ndarray:
    T = np.asarray(obj, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"expected 4x4 matrix, got shape {T.shape}")
    return T


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def list_mark_ids(session_dir: Path) -> list[str]:
    marked = Path(session_dir) / "marked"
    if not marked.is_dir():
        raise FileNotFoundError(f"missing marked/ under {session_dir}")
    ids = sorted(p.name for p in marked.iterdir() if p.is_dir())
    if not ids:
        raise FileNotFoundError(f"no mark directories under {marked}")
    return ids


def load_mark_sample(
    session_dir: Path,
    mark_id: str,
    cam: str,
    tag_id: int,
) -> Optional[MarkSample]:
    cam_dir = Path(session_dir) / "marked" / mark_id / cam
    meta_path = cam_dir / "meta.json"
    tags_path = cam_dir / "tags.json"
    if not meta_path.is_file() or not tags_path.is_file():
        return None
    meta = _load_json(meta_path)
    tags_doc = _load_json(tags_path)
    T_EE_base = _as_matrix(meta["T_EE_base"])
    T_cam_EE = _as_matrix(meta["T_cam_EE"])
    tag_entry = None
    for t in tags_doc.get("tags", []):
        if int(t["id"]) == int(tag_id):
            tag_entry = t
            break
    if tag_entry is None:
        return None
    T_tag_cam = _as_matrix(tag_entry["T_tag_cam"])
    if "T_tag_base" in tag_entry:
        T_tag_base = _as_matrix(tag_entry["T_tag_base"])
    else:
        T_tag_base = compose_T_tag_base(T_EE_base, T_cam_EE, T_tag_cam)
    return MarkSample(
        mark_id=mark_id,
        cam=cam,
        T_EE_base=T_EE_base,
        T_cam_EE=T_cam_EE,
        T_tag_cam=T_tag_cam,
        T_tag_base=T_tag_base,
        tag_id=int(tag_id),
        ee_frame=str(meta.get("ee_frame", "")),
        cam_frame=str(meta.get("cam_frame", "")),
    )


def load_samples(
    session_dir: Path,
    cam: str,
    tag_id: int,
    mark_ids: Optional[Sequence[str]] = None,
) -> list[MarkSample]:
    ids = list(mark_ids) if mark_ids is not None else list_mark_ids(session_dir)
    samples: list[MarkSample] = []
    for mid in ids:
        sample = load_mark_sample(session_dir, mid, cam, tag_id)
        if sample is not None:
            samples.append(sample)
    if not samples:
        raise ValueError(
            f"no marks with cam={cam!r} tag_id={tag_id} under {session_dir}/marked"
        )
    return samples


def score_multiview(
    samples: Sequence[MarkSample],
    *,
    max_trans: float,
    max_rot_deg: float,
) -> MultiviewResult:
    if len(samples) < 2:
        raise ValueError(f"multiview needs >= 2 marks, got {len(samples)}")
    translations = np.stack([s.T_tag_base[:3, 3] for s in samples], axis=0)
    rotations = [s.T_tag_base[:3, :3] for s in samples]
    t_ref = translations.mean(axis=0)
    R_ref = mean_rotation_svd(rotations)

    per_view: list[dict[str, Any]] = []
    dt_list: list[float] = []
    dr_list: list[float] = []
    for s, t, R in zip(samples, translations, rotations):
        dt = float(np.linalg.norm(t - t_ref))
        dr = rotation_angle_deg(R_ref.T @ R)
        dt_list.append(dt)
        dr_list.append(dr)
        per_view.append(
            {
                "mark_id": s.mark_id,
                "delta_trans_m": dt,
                "delta_rot_deg": dr,
                "t_tag_base": t.tolist(),
            }
        )
    rms_t = float(np.sqrt(np.mean(np.square(dt_list))))
    rms_r = float(np.sqrt(np.mean(np.square(dr_list))))
    max_t = float(max(dt_list))
    max_r = float(max(dr_list))
    return MultiviewResult(
        n_views=len(samples),
        rms_trans_m=rms_t,
        max_trans_m=max_t,
        rms_rot_deg=rms_r,
        max_rot_deg=max_r,
        per_view=per_view,
        passed=(max_t <= max_trans and max_r <= max_rot_deg),
    )


def score_tip_contact(
    samples: Sequence[MarkSample],
    *,
    max_tip_tag: float,
    max_cam: float,
) -> TipContactResult:
    if not samples:
        raise ValueError("tip-contact needs >= 1 mark")
    per_view: list[dict[str, Any]] = []
    tip_errs: list[float] = []
    cam_errs: list[float] = []
    for s in samples:
        p_tip = s.T_EE_base[:3, 3]
        p_tag = s.T_tag_base[:3, 3]
        d_tip = float(np.linalg.norm(p_tag - p_tip))
        T_EE_cam = invert_SE3(s.T_cam_EE)
        t_pred = T_EE_cam[:3, 3]
        t_meas = s.T_tag_cam[:3, 3]
        d_cam = float(np.linalg.norm(t_meas - t_pred))
        tip_errs.append(d_tip)
        cam_errs.append(d_cam)
        per_view.append(
            {
                "mark_id": s.mark_id,
                "delta_tip_tag_m": d_tip,
                "delta_cam_m": d_cam,
                "cam_to_tag_m": float(np.linalg.norm(t_meas)),
                "cam_to_tip_m": float(np.linalg.norm(t_pred)),
                "t_tag_cam_meas": t_meas.tolist(),
                "t_tag_cam_pred": t_pred.tolist(),
            }
        )
    max_tip = float(max(tip_errs))
    max_c = float(max(cam_errs))
    return TipContactResult(
        n_views=len(samples),
        max_tip_tag_m=max_tip,
        rms_tip_tag_m=float(np.sqrt(np.mean(np.square(tip_errs)))),
        max_cam_err_m=max_c,
        rms_cam_err_m=float(np.sqrt(np.mean(np.square(cam_errs)))),
        per_view=per_view,
        passed=(max_tip <= max_tip_tag and max_c <= max_cam),
    )


def build_tag_extrinsics_section(
    *,
    session: str,
    cam: str,
    tag_id: int,
    mode: str,
    multiview: Optional[MultiviewResult],
    tip_contact: Optional[TipContactResult],
) -> dict[str, Any]:
    passed_parts = []
    if multiview is not None:
        passed_parts.append(multiview.passed)
    if tip_contact is not None:
        passed_parts.append(tip_contact.passed)
    return {
        "validated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "session": session,
        "cam": cam,
        "tag_id": int(tag_id),
        "mode": mode,
        "passed": bool(passed_parts) and all(passed_parts),
        "multiview": None if multiview is None else multiview.to_dict(),
        "tip_contact": None if tip_contact is None else tip_contact.to_dict(),
        "note": (
            "Tip = EE origin (pendant TCP in mark ee_frame). "
            "Multi-view checks free-space T_tag_base consistency; "
            "tip-contact checks tip↔tag when touching."
        ),
    }


def parse_mark_list(text: Optional[str]) -> Optional[list[str]]:
    if text is None or str(text).strip() == "":
        return None
    return [p.strip() for p in str(text).split(",") if p.strip()]


def run_validation(
    session_dir: Path,
    *,
    cam: str,
    tag_id: int,
    mode: str = "both",
    mark_ids: Optional[Sequence[str]] = None,
    contact_marks: Optional[Sequence[str]] = None,
    max_trans: float = 0.005,
    max_rot_deg: float = 2.0,
    max_tip_tag: float = 0.005,
    max_cam: float = 0.01,
) -> tuple[Optional[MultiviewResult], Optional[TipContactResult], bool]:
    mode = str(mode).strip().lower()
    if mode not in ("both", "multiview", "tip-contact"):
        raise ValueError(f"mode must be both|multiview|tip-contact, got {mode!r}")

    all_samples = load_samples(session_dir, cam, tag_id, mark_ids=mark_ids)
    multiview = None
    tip = None

    if mode in ("both", "multiview"):
        multiview = score_multiview(
            all_samples, max_trans=max_trans, max_rot_deg=max_rot_deg
        )

    if mode in ("both", "tip-contact"):
        if contact_marks is not None:
            contact_set = set(contact_marks)
            contact_samples = [s for s in all_samples if s.mark_id in contact_set]
            if not contact_samples:
                raise ValueError(
                    f"no contact marks matched {sorted(contact_set)} among loaded samples"
                )
        else:
            contact_samples = list(all_samples)
        tip = score_tip_contact(
            contact_samples, max_tip_tag=max_tip_tag, max_cam=max_cam
        )

    passed = True
    if multiview is not None:
        passed = passed and multiview.passed
    if tip is not None:
        passed = passed and tip.passed
    return multiview, tip, passed
