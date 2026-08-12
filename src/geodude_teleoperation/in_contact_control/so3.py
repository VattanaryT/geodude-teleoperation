"""SO(3) helpers: orientation error as rotation vector (axis-angle, rad)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def orientation_error_rotvec(R_des: np.ndarray, R_meas: np.ndarray) -> np.ndarray:
    """
    Orientation error e_R ∈ ℝ³ (rad) such that a positive spring
    K e_R tends to rotate meas → des when integrated as ω ≈ K e_R.

    Uses body/base-consistent relative rotation:

        R_err = R_des.T @ R_meas
        e_R   = Log(R_err)   (rotvec of R_err)

    Then ω_base ≈ R_des @ (gain * e_R) is one mapping; for the simple
    pose-hold demo we apply gains in base as ω_base ≈ gain * vee(R_meas
    related). See pose_hold for the chosen mapping.
    """
    R_d = np.asarray(R_des, dtype=np.float64).reshape(3, 3)
    R_m = np.asarray(R_meas, dtype=np.float64).reshape(3, 3)
    R_err = R_d.T @ R_m
    # rotvec is Log(R_err); for small angles ≈ axis*angle of meas relative to des
    e = Rotation.from_matrix(R_err).as_rotvec()
    return np.asarray(e, dtype=np.float64).reshape(3)


def orientation_error_rotvec_base(R_des: np.ndarray, R_meas: np.ndarray) -> np.ndarray:
    """
    Orientation error expressed so that ω_base ≈ −K * e drives meas → des.

    e = Log(R_meas.T @ R_des) mapped to base: R_meas @ e_body
    equivalently scipy rotvec of relative rotation from meas to des in base:

        R_delta = R_des @ R_meas.T   # rotation taking meas frame axes to des
        e_base  = Log(R_delta)

    Spring: ω_base = K * e_base (with sign: we use e = Log(R_des @ R_meas.T)).
    """
    R_d = np.asarray(R_des, dtype=np.float64).reshape(3, 3)
    R_m = np.asarray(R_meas, dtype=np.float64).reshape(3, 3)
    R_delta = R_d @ R_m.T
    e = Rotation.from_matrix(R_delta).as_rotvec()
    return np.asarray(e, dtype=np.float64).reshape(3)
