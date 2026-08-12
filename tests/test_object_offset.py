"""Tests for per-session object translation offset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from geodude_teleoperation.teleop.object_offset import (
    apply_args_object_offset,
    apply_object_t_offset,
    bind_object_offset_to_args,
    delta_viz_to_vention,
    load_object_offset,
    resolve_object_offset_path,
    save_object_offset,
    sidecar_path_for,
)


def test_sidecar_path_for_crop_meta():
    p = Path("/data/processed/crop_meta.json")
    assert sidecar_path_for(p) == Path("/data/processed/crop_meta.object_offset.json")


def test_save_load_roundtrip(tmp_path: Path):
    path = tmp_path / "crop_meta.object_offset.json"
    save_object_offset(path, [0.01, -0.002, 0.003], step_m=0.002)
    t, step = load_object_offset(path)
    assert np.allclose(t, [0.01, -0.002, 0.003])
    assert step == 0.002
    meta = json.loads(path.read_text())
    assert "t_vention_m" in meta


def test_apply_object_t_offset():
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    out = apply_object_t_offset(T, [0.1, 0.0, -0.2])
    assert np.allclose(out[:3, 3], [1.1, 2.0, 2.8])
    assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])  # copy


def test_delta_viz_to_vention_rtde():
    T_rv = np.eye(4)
    # 90deg about Z: rtde +x → vention +y
    T_rv[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    d = delta_viz_to_vention(
        [0.01, 0.0, 0.0],
        viz_frame="rtde",
        T_tag0_vention=np.eye(4),
        T_rtde_base_vention=T_rv,
    )
    assert np.allclose(d, [0.0, 0.01, 0.0])


def test_bind_object_offset_to_args_loads_sidecar(tmp_path: Path):
    crop = tmp_path / "crop_meta.json"
    crop.write_text("{}")
    side = sidecar_path_for(crop)
    assert side is not None
    save_object_offset(side, [0.005, 0.0, 0.0], step_m=0.001)

    class _Args:
        T_tag0_base = crop
        object_offset = None
        no_object_offset = False

    args = _Args()
    t, step, path = bind_object_offset_to_args(args)
    assert path == side
    assert np.allclose(t, [0.005, 0.0, 0.0])
    assert step == 0.001
    assert np.allclose(args.object_offset_t, [0.005, 0.0, 0.0])


def test_resolve_path_no_object_offset():
    class _Args:
        T_tag0_base = Path("/tmp/crop_meta.json")
        object_offset = None
        no_object_offset = True

    assert resolve_object_offset_path(_Args()) is None


def test_apply_args_object_offset_vention_parent():
    class _Args:
        object_offset_t = np.array([0.1, 0.0, 0.0])
        base_frame = None
        no_rtde_base_vention = False
        T_rtde_base_vention = None

    T = np.eye(4)
    T[:3, 3] = [1.0, 0.0, 0.0]
    out = apply_args_object_offset(T, _Args())
    assert np.allclose(out[:3, 3], [1.1, 0.0, 0.0])
