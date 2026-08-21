"""Object / laser vention-frame sidecars (loaded every RTDE session)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from geodude_teleoperation.teleop.laser_offset import (
    apply_laser_t_offset,
    save_laser_offset,
    sidecar_path_for_laser,
    load_laser_offset,
)
from geodude_teleoperation.teleop.object_offset import (
    apply_object_t_offset,
    load_object_offset,
    save_object_offset,
    sidecar_path_for,
)


def test_object_and_laser_offset_roundtrip(tmp_path: Path):
    obj = tmp_path / "crop_meta.object_offset.json"
    save_object_offset(obj, [0.01, -0.002, 0.003], step_m=0.002)
    t, step = load_object_offset(obj)
    assert np.allclose(t, [0.01, -0.002, 0.003]) and step == 0.002
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(apply_object_t_offset(T, [0.1, 0.0, -0.2])[:3, 3], [1.1, 2.0, 2.8])
    assert sidecar_path_for(Path("/data/crop_meta.json")).name == "crop_meta.object_offset.json"

    laser = tmp_path / "merged.laser_offset.json"
    save_laser_offset(laser, [0.01, -0.002, 0.003], step_m=0.002)
    t, step = load_laser_offset(laser)
    assert np.allclose(t, [0.01, -0.002, 0.003])
    pts = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(apply_laser_t_offset(pts, [0.1, 0.0, -0.2]), [[1.1, 2.0, 2.8]])
    assert sidecar_path_for_laser(Path("/data/merged.ply")).name == "merged.laser_offset.json"
