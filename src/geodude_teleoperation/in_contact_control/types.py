"""Pose / twist types for the geometric admittance sandbox.

Convention (matches Kinova TCP + this repo):
  - Position: meters in robot **base**
  - Orientation: R ∈ SO(3), **EE → base** (columns = tool X,Y,Z in base)
  - Built from Kinova Euler xyz degrees: scipy Rotation.from_euler(\"xyz\", …)
  - Twist command: linear m/s, angular **deg/s**, base frame (SendTwistCommand)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PoseSO3:
    """Cartesian pose: point + rotation matrix (EE→base)."""

    p: np.ndarray  # (3,) m
    R: np.ndarray  # (3, 3) EE→base

    def __post_init__(self) -> None:
        object.__setattr__(self, "p", np.asarray(self.p, dtype=np.float64).reshape(3))
        R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        object.__setattr__(self, "R", R)

    @property
    def tool_z_base(self) -> np.ndarray:
        """Unit tool +Z in base (Kinova EE frame third column)."""
        return self.R[:, 2].copy()

    def copy(self) -> "PoseSO3":
        return PoseSO3(self.p.copy(), self.R.copy())

    @classmethod
    def from_kinova_xyz_euler_deg(
        cls,
        x: float,
        y: float,
        z: float,
        theta_x_deg: float,
        theta_y_deg: float,
        theta_z_deg: float,
    ) -> "PoseSO3":
        """From Kinova BaseCyclic tool_pose (m + deg, xyz extrinsic Euler)."""
        R = Rotation.from_euler(
            "xyz",
            [theta_x_deg, theta_y_deg, theta_z_deg],
            degrees=True,
        ).as_matrix()
        return cls(np.array([x, y, z], dtype=np.float64), R)

    def to_euler_xyz_deg(self) -> Tuple[float, float, float]:
        e = Rotation.from_matrix(self.R).as_euler("xyz", degrees=True)
        return float(e[0]), float(e[1]), float(e[2])


@dataclass
class TwistCommand:
    """Cartesian twist for Kortex SendTwistCommand (base frame)."""

    linear_x: float = 0.0  # m/s
    linear_y: float = 0.0
    linear_z: float = 0.0
    angular_x: float = 0.0  # deg/s
    angular_y: float = 0.0
    angular_z: float = 0.0

    def as_linear(self) -> np.ndarray:
        return np.array(
            [self.linear_x, self.linear_y, self.linear_z], dtype=np.float64
        )

    def as_angular_deg(self) -> np.ndarray:
        return np.array(
            [self.angular_x, self.angular_y, self.angular_z], dtype=np.float64
        )

    @classmethod
    def from_vectors(
        cls, v_m_s: Sequence[float], w_deg_s: Sequence[float]
    ) -> "TwistCommand":
        v = np.asarray(v_m_s, dtype=np.float64).reshape(3)
        w = np.asarray(w_deg_s, dtype=np.float64).reshape(3)
        return cls(
            linear_x=float(v[0]),
            linear_y=float(v[1]),
            linear_z=float(v[2]),
            angular_x=float(w[0]),
            angular_y=float(w[1]),
            angular_z=float(w[2]),
        )

    def zero(self) -> "TwistCommand":
        return TwistCommand()
