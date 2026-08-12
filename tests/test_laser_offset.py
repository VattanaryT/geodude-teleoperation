"""Laser vention-frame offset sidecar."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from geodude_teleoperation.teleop.laser_offset import (
    apply_laser_t_offset,
    bind_laser_offset_to_args,
    load_laser_offset,
    resolve_laser_offset_path,
    save_laser_offset,
    sidecar_path_for_laser,
)


def test_sidecar_path_for_laser():
    p = Path("/data/processed/laser_profilometer/merged.ply")
    assert sidecar_path_for_laser(p) == Path(
        "/data/processed/laser_profilometer/merged.laser_offset.json"
    )


def test_save_load_laser_offset(tmp_path: Path):
    path = tmp_path / "merged.laser_offset.json"
    save_laser_offset(path, [0.01, -0.002, 0.003], step_m=0.002)
    t, step = load_laser_offset(path)
    assert np.allclose(t, [0.01, -0.002, 0.003])
    assert step == 0.002


def test_apply_laser_t_offset():
    pts = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    out = apply_laser_t_offset(pts, [0.1, 0.0, -0.2])
    assert np.allclose(out, [[1.1, 2.0, 2.8], [0.1, 0.0, -0.2]])


def test_bind_laser_offset_to_args(tmp_path: Path):
    ply = tmp_path / "merged.ply"
    ply.write_text("ply\n")
    side = tmp_path / "merged.laser_offset.json"
    save_laser_offset(side, [0.005, 0.0, 0.0], step_m=0.001)
    args = SimpleNamespace(
        laser_ply=ply,
        laser_offset=None,
        no_laser_offset=False,
    )
    t, step, path = bind_laser_offset_to_args(args)
    assert path == side
    assert np.allclose(t, [0.005, 0.0, 0.0])
    assert np.allclose(args.laser_offset_t, [0.005, 0.0, 0.0])


def test_resolve_path_no_laser_offset():
    args = SimpleNamespace(
        laser_ply=None,
        laser_offset=None,
        no_laser_offset=True,
    )
    assert resolve_laser_offset_path(args) is None
