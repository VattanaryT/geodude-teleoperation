"""
First-order / second-order geometric admittance (spring–damper).

  M * a + B * v + K * e = 0
  a = M^{-1} (-B v - K e)
  v ← v + dt * a

Diagonal M, B, K (6,) for pose-hold: [lin×3, ang×3].
Linear units: e_p m, v m/s, K in (N/m)/kg equiv. → 1/s² style gains.
Angular units: e_R rad, v rad/s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


def _as6(x: Sequence[float] | np.ndarray | float, default: float) -> np.ndarray:
    if isinstance(x, (int, float)):
        return np.full(6, float(x), dtype=np.float64)
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size == 1:
        return np.full(6, float(a[0]), dtype=np.float64)
    if a.size != 6:
        raise ValueError(f"expected scalar or 6-vector, got shape {a.shape}")
    return a.copy()


@dataclass
class SpringDamperGains:
    """Diagonal second-order gains for 6-vector geometric error."""

    # Soft free-space defaults (safe to start low on real robot)
    M: Sequence[float] | float = (1.0, 1.0, 1.0, 0.05, 0.05, 0.05)
    B: Sequence[float] | float = (40.0, 40.0, 40.0, 2.0, 2.0, 2.0)
    K: Sequence[float] | float = (80.0, 80.0, 80.0, 4.0, 4.0, 4.0)

    def M_diag(self) -> np.ndarray:
        return _as6(self.M, 1.0)

    def B_diag(self) -> np.ndarray:
        return _as6(self.B, 20.0)

    def K_diag(self) -> np.ndarray:
        return _as6(self.K, 50.0)


class SpringDamperState:
    """Integrates task-space velocity from geometric error."""

    def __init__(self, gains: Optional[SpringDamperGains] = None) -> None:
        self.gains = gains or SpringDamperGains()
        self.v = np.zeros(6, dtype=np.float64)  # [v_xyz m/s, ω_xyz rad/s]

    def reset(self) -> None:
        self.v[:] = 0.0

    def step(self, e: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        One integration step.

        Parameters
        ----------
        e : (6,) geometric error [Δp (m), e_R (rad)]
        dt : s

        Returns
        -------
        v_lin : (3,) m/s
        w_rad : (3,) rad/s
        """
        e6 = np.asarray(e, dtype=np.float64).reshape(6)
        K = self.gains.K_diag()
        # e := desired − measured (points toward goal).
        # Virtual spring:  M a + B v = K e
        return self.step_from_force(K * e6, dt)

    def step_from_force(
        self, f_virt: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate with an explicit virtual wrench f_virt (6,).

        M a + B v = f_virt  →  a = M^{-1}(f_virt - B v)
        """
        f6 = np.asarray(f_virt, dtype=np.float64).reshape(6)
        dt = float(max(dt, 1e-6))
        M = self.gains.M_diag()
        B = self.gains.B_diag()
        a = (f6 - B * self.v) / np.maximum(M, 1e-9)
        self.v = self.v + dt * a
        return self.v[:3].copy(), self.v[3:].copy()
