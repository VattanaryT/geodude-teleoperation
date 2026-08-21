"""Line tracking: don't fight contact depth (kill normal + standoff)."""
from __future__ import annotations

import numpy as np

from geodude_teleoperation.teleop.line_tracking import (
    LineTrackingConfig,
    apply_line_tracking_pose_error,
    combine_normal_feedforward,
    kill_normal_linear_error,
    outward_normal_from_R_field,
)


def test_kill_normal_and_standoff_into_surface():
    n = np.array([0.0, 0.0, 1.0])
    err = np.array([0.1, -0.2, 0.5, 0.0, 0.0, 0.1])
    out = kill_normal_linear_error(err, n)
    assert abs(out[2]) < 1e-9
    assert abs(out[0] - 0.1) < 1e-9

    R = np.eye(3)
    R[:, 2] = [0.0, 0.0, -1.0]
    assert np.allclose(outward_normal_from_R_field(R), [0, 0, 1])
    killed = apply_line_tracking_pose_error(np.array([0.0, 0.0, 0.4, 0, 0, 0]), R, enabled=True)
    assert abs(killed[2]) < 1e-12

    cfg = LineTrackingConfig(enabled=True, standoff_enabled=True, standoff_gain=1.0)
    v = combine_normal_feedforward(
        cfg=cfg,
        signed_d=0.01,
        R_field=R,
        keyboard_v_world=np.array([0.0, 0.0, -0.01]),
    )
    assert v[2] < 0
