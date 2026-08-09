import numpy as np
from geodude_teleoperation.controller import qdot_from_error


def test_qdot_scales_and_clamps():
    err = np.ones(6)
    J = np.eye(6)
    q = qdot_from_error(err, J, gain=1.0, damping=0.0, max_joint_velocity=0.5)
    assert np.linalg.norm(q) <= 0.5 + 1e-9
