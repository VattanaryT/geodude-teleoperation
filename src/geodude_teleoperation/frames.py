"""SE(3) helpers. T_tag0_base = pose of tag_0 expressed in base (crop_meta convention)."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np


def load_T_tag0_base(path: Path | None) -> np.ndarray:
    if path is None:
        return np.eye(4, dtype=float)
    path = Path(path)
    if path.suffix == ".json":
        with path.open() as f:
            meta = json.load(f)
        T = np.asarray(meta["T_tag0_base"] if "T_tag0_base" in meta else meta, dtype=float)
    else:
        T = np.load(path)
        if isinstance(T, np.lib.npyio.NpzFile):
            T = T["T_tag0_base"] if "T_tag0_base" in T.files else T[T.files[0]]
        T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T_tag0_base must be 4x4, got {T.shape}")
    return T


def base_to_tag0_point(p_base: np.ndarray, T_tag0_base: np.ndarray) -> np.ndarray:
    R = T_tag0_base[:3, :3]
    t = T_tag0_base[:3, 3]
    return R.T @ (np.asarray(p_base, dtype=float).reshape(3) - t)


def tag0_rot_to_base(R_tag0: np.ndarray, T_tag0_base: np.ndarray) -> np.ndarray:
    return T_tag0_base[:3, :3] @ np.asarray(R_tag0, dtype=float).reshape(3, 3)


def tip_from_tcp(p_tcp, R_tcp, tip_offset) -> np.ndarray:
    return np.asarray(p_tcp, dtype=float).reshape(3) + (
        np.asarray(R_tcp, dtype=float).reshape(3, 3)
        @ np.asarray(tip_offset, dtype=float).reshape(3)
    )


def warn_tip_offset_xy(tip_offset: np.ndarray) -> str | None:
    tip = np.asarray(tip_offset, dtype=float).reshape(3)
    if abs(tip[0]) > 1e-9 or abs(tip[1]) > 1e-9:
        return (
            "WARNING: tip-offset TX/TY non-zero; prefer TX=TY=0 and only TZ "
            f"(got {tip.tolist()})"
        )
    return None
