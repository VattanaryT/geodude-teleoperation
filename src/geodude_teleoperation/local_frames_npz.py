"""Load CGA local-frame NPZ episodes (right_base, 10 Hz).

Convention (stored in NPZ meta):
    a_local = R_field^T @ (x[t+1] - x[t])
    R_field columns = [t1, t2, n]

``a_local`` is ``R_field^T @ (x[t+1]-x[t])`` for the path named in
``meta["action_source"]`` (``desired`` or ``actual``). Reconstruct with
``integrate_desired_xyz`` from ``episode.action_xyz``. Force / 6-D action
live in the sibling training NPZ.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_DEFAULT_CONTACT_THRESHOLD_M = 0.01


@dataclass(frozen=True)
class CgaLocalEpisode:
    timestamps: np.ndarray
    poses: np.ndarray
    poses_desired_xyz: np.ndarray
    a_local: np.ndarray
    R_field: np.ndarray
    signed_distance: np.ndarray
    in_contact: np.ndarray
    meta: dict[str, Any]
    action: np.ndarray | None = None
    f_n: np.ndarray | None = None
    f_t1: np.ndarray | None = None
    f_t2: np.ndarray | None = None

    @property
    def pose_frame(self) -> str:
        return str(self.meta.get("pose_frame", ""))

    @property
    def action_source(self) -> str:
        return str(self.meta.get("action_source", "desired"))

    @property
    def action_xyz(self) -> np.ndarray:
        """XYZ path that ``a_local`` was built from."""
        if self.action_source == "actual":
            return np.asarray(self.poses[:, :3], dtype=float)
        return np.asarray(self.poses_desired_xyz, dtype=float)

    @property
    def hz(self) -> int:
        dt = np.diff(np.asarray(self.timestamps, dtype=float))
        if dt.size == 0:
            return 0
        return int(round(1.0 / float(dt.mean())))


def world_deltas_from_local(R_field: np.ndarray, a_local: np.ndarray) -> np.ndarray:
    """World Δp[t] = R_field[t] @ a_local[t]. ``R_field`` may be length N or N-1."""
    R = np.asarray(R_field, dtype=float)
    a = np.asarray(a_local, dtype=float).reshape(-1, 3)
    if R.ndim != 3 or R.shape[-2:] != (3, 3):
        raise ValueError(f"R_field must be (N,3,3), got {R.shape}")
    if R.shape[0] == a.shape[0] + 1:
        R = R[:-1]
    if R.shape[0] != a.shape[0]:
        raise ValueError(f"R_field length {R.shape[0]} != a_local {a.shape[0]}")
    return np.einsum("nij,nj->ni", R, a)


def local_actions_from_xyz(R_field: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """a_local[t] = R_field[t].T @ (xyz[t+1] - xyz[t])."""
    R = np.asarray(R_field, dtype=float)
    p = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if R.shape[0] != p.shape[0]:
        raise ValueError(f"R_field length {R.shape[0]} != xyz {p.shape[0]}")
    return np.einsum("nji,nj->ni", R[:-1], np.diff(p, axis=0))


def write_cga_actual_tcp(src: Path | str, dst: Path | str) -> Path:
    """Copy a CGA NPZ, replacing a_local with actual-TCP deltas. Does not modify src."""
    src = Path(src)
    dst = Path(dst)
    if dst.resolve() == src.resolve():
        raise ValueError("refusing to overwrite the source NPZ")
    z = np.load(src, allow_pickle=True)
    a_actual = local_actions_from_xyz(z["R_field_cga"], z["poses"][:, :3])
    meta = _load_meta(z["meta"]) if "meta" in z.files else {}
    meta = dict(meta)
    meta["action_source"] = "actual"
    meta["derived_from"] = str(src)
    meta["convention"] = (
        "a_local = R_field^T (x_actual[t+1]-x_actual[t]); columns [t1, t2, n]; "
        "R_field is the stored CGA field (queried at actual TCP)"
    )
    payload = {k: z[k] for k in z.files if k not in ("a_local_cga", "meta")}
    payload["a_local_cga"] = a_actual
    payload["a_local_cga_desired"] = np.asarray(z["a_local_cga"])
    payload["meta"] = json.dumps(meta)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **payload)
    return dst


def integrate_desired_xyz(
    start_xyz: np.ndarray, R_field: np.ndarray, a_local: np.ndarray
) -> np.ndarray:
    start = np.asarray(start_xyz, dtype=float).reshape(3)
    deltas = world_deltas_from_local(R_field, a_local)
    out = np.empty((deltas.shape[0] + 1, 3), dtype=float)
    out[0] = start
    out[1:] = start + np.cumsum(deltas, axis=0)
    return out


def replay_arrays(episode: CgaLocalEpisode) -> dict[str, np.ndarray]:
    """Time-aligned series for timestep plots (actions are one sample shorter)."""
    t = np.asarray(episode.timestamps, dtype=float)
    t = t - t[0]
    t_act = t[:-1]
    a = np.asarray(episode.a_local, dtype=float)
    out: dict[str, np.ndarray] = {
        "t_s": t,
        "t_action_s": t_act,
        "a_t1": a[:, 0],
        "a_t2": a[:, 1],
        "a_n": a[:, 2],
        "in_contact": np.asarray(episode.in_contact[: len(t_act)], dtype=bool),
        "signed_distance": np.asarray(episode.signed_distance[: len(t)], dtype=float),
    }
    if episode.f_n is not None:
        out["f_n"] = np.asarray(episode.f_n, dtype=float)[: len(t_act)]
    if episode.f_t1 is not None:
        out["f_t1"] = np.asarray(episode.f_t1, dtype=float)[: len(t_act)]
    if episode.f_t2 is not None:
        out["f_t2"] = np.asarray(episode.f_t2, dtype=float)[: len(t_act)]
    return out


def _load_meta(raw: np.ndarray) -> dict[str, Any]:
    val: Any = raw.item() if getattr(raw, "shape", None) == () else raw
    if isinstance(val, bytes):
        val = val.decode()
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            return {"convention": val}
    if not isinstance(val, Mapping):
        return {}
    return dict(val)


def _contact_from_signed(signed: np.ndarray, threshold_m: float) -> np.ndarray:
    return np.abs(np.asarray(signed, dtype=float)) <= float(threshold_m)


def load_cga_episode(
    path: Path | str, *, training_path: Path | str | None = None
) -> CgaLocalEpisode:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    meta = _load_meta(z["meta"]) if "meta" in z.files else {}
    a_local = np.asarray(z["a_local_cga"], dtype=float)
    R_field = np.asarray(z["R_field_cga"], dtype=float)
    poses = np.asarray(z["poses"], dtype=float)
    desired = np.asarray(z["poses_desired"], dtype=float)
    if desired.ndim == 1:
        desired = desired.reshape(-1, 3)
    signed = np.asarray(z["signed_cga"], dtype=float)
    timestamps = np.asarray(z["timestamps"], dtype=float)
    threshold = float(meta.get("contact_threshold", _DEFAULT_CONTACT_THRESHOLD_M))
    in_contact = _contact_from_signed(signed, threshold)

    action = f_n = f_t1 = f_t2 = None
    if training_path is not None:
        train = Path(training_path)
        if not train.is_file():
            raise FileNotFoundError(train)
        packed = np.load(train, allow_pickle=True)
        action = np.asarray(packed["action"], dtype=float)
        obs = np.asarray(packed["obs"], dtype=float)
        ic = np.asarray(packed["in_contact"]).astype(bool)
        in_contact = np.zeros(len(signed), dtype=bool)
        in_contact[: len(ic)] = ic
        if ic[-1] and len(in_contact) > len(ic):
            in_contact[-1] = ic[-1]
        f_t1, f_t2, f_n = obs[:, 7], obs[:, 8], obs[:, 9]

    return CgaLocalEpisode(
        timestamps=timestamps,
        poses=poses,
        poses_desired_xyz=desired,
        a_local=a_local,
        R_field=R_field,
        signed_distance=signed,
        in_contact=in_contact,
        meta=meta,
        action=action,
        f_n=f_n,
        f_t1=f_t1,
        f_t2=f_t2,
    )
