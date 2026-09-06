"""Where "the tool is on the part" comes from, as a swappable source.

The press latch decides three things at once and they must never disagree --
whether the pose spring keeps the contact normal (``free_along_normal``),
whether the press commands any force at all (``_press_force_n``), and whether
field-follow strips the normal out of a stick step (``tangent=``). Gating them
on different signals is what deadlocked contact mode once already, so there is
exactly one bool and this module is where it is produced.

Three sources, same output:

``ForceLatch``     hysteresis on the measured ``Fs``. This is what
                   ``ContactTeleoperationController`` does internally and stays
                   the default for SpaceMouse teleop -- it needs no plan.
``DistanceLatch``  hysteresis on the signed distance from the tool to the
                   planned surface point, along the surface normal. Needs no
                   force sensor at all, which is the point: on this rig
                   ``getActualTCPForce`` is noisy enough at the 1-3 N scale that
                   latching on it near a 15 N target chatters.
``PhaseLatch``     the deterministic plan's own phase.

``AllOf`` ANDs them. The deterministic experiment runs
``AllOf([PhaseLatch(), DistanceLatch()])``: the plan says *when* pressing is
allowed and the geometry says *whether the tool has actually arrived*.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class LatchState:
    """Everything any latch may read on a tick. Fields may be ``None``."""

    tcp_xyz: Optional[Sequence[float]] = None
    surface_xyz: Optional[Sequence[float]] = None
    normal: Optional[Sequence[float]] = None   # outward unit surface normal
    phase_press: bool = False                  # the plan allows pressing here
    fs: Optional[float] = None                 # measured along-normal force, N


class PressLatch:
    """One bool with hysteresis. ``update`` is called once per control tick."""

    def update(self, state: LatchState) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def reset(self) -> None:
        pass

    @property
    def engaged(self) -> bool:  # pragma: no cover - trivial
        return bool(getattr(self, "_engaged", False))


class ForceLatch(PressLatch):
    """``Fs >= engage_frac*Fd`` on, ``Fs < release_frac*Fd`` off."""

    def __init__(self, target_force_n, *, engage_frac=0.5, release_frac=0.15):
        self.fd = abs(float(target_force_n))
        self.engage_frac = float(np.clip(engage_frac, 0.0, 1.0))
        self.release_frac = float(np.clip(release_frac, 0.0, self.engage_frac))
        self._engaged = False

    def reset(self):
        self._engaged = False

    def update(self, state: LatchState) -> bool:
        fs = state.fs
        if fs is None or not np.isfinite(fs):
            return self._engaged
        if fs >= self.engage_frac * self.fd:
            self._engaged = True
        elif fs < self.release_frac * self.fd:
            self._engaged = False
        return self._engaged


class DistanceLatch(PressLatch):
    """Signed distance to the planned surface point, along its normal.

    ``d = n . (tcp - surface)`` -- positive above the part. Engages at
    ``d <= engage_gap_m`` and releases at ``d > release_gap_m``, so the two gaps
    are the hysteresis and ``release > engage`` is required.

    Both terms come from forward kinematics and the plan, so this works with a
    dead, noisy or unzeroed F/T sensor. It is wrong in exactly one way the force
    latch is not: if the surface has moved since the plan was made, the tool
    reaches ``d = 0`` without touching anything. The press is bounded by
    ``press_max_n`` for that reason, and the measured ``Fs`` is still logged.
    """

    def __init__(self, *, engage_gap_m=0.002, release_gap_m=0.005):
        self.engage_gap_m = float(engage_gap_m)
        self.release_gap_m = float(max(release_gap_m, engage_gap_m))
        self._engaged = False
        self.last_distance_m = float("nan")

    def reset(self):
        self._engaged = False
        self.last_distance_m = float("nan")

    def update(self, state: LatchState) -> bool:
        if state.tcp_xyz is None or state.surface_xyz is None or state.normal is None:
            return self._engaged
        tcp = np.asarray(state.tcp_xyz, dtype=float).reshape(3)
        surf = np.asarray(state.surface_xyz, dtype=float).reshape(3)
        n = np.asarray(state.normal, dtype=float).reshape(3)
        nn = float(np.linalg.norm(n))
        if not np.isfinite(nn) or nn < 1e-12:
            return self._engaged
        d = float(np.dot(n / nn, tcp - surf))
        if not np.isfinite(d):
            return self._engaged
        self.last_distance_m = d
        if d <= self.engage_gap_m:
            self._engaged = True
        elif d > self.release_gap_m:
            self._engaged = False
        return self._engaged


class PhaseLatch(PressLatch):
    """The plan's own phase. No hysteresis -- the plan is already monotone."""

    def __init__(self):
        self._engaged = False

    def reset(self):
        self._engaged = False

    def update(self, state: LatchState) -> bool:
        self._engaged = bool(state.phase_press)
        return self._engaged


class AllOf(PressLatch):
    """AND of several latches. Every member is updated every tick.

    Updating all of them (rather than short-circuiting) keeps each member's
    hysteresis honest: a distance latch that stopped being ticked while the
    phase said no would come back with stale state at the next plunge.
    """

    def __init__(self, latches: Iterable[PressLatch]):
        self.latches: List[PressLatch] = list(latches)
        self._engaged = False

    def reset(self):
        for latch in self.latches:
            latch.reset()
        self._engaged = False

    def update(self, state: LatchState) -> bool:
        results = [bool(latch.update(state)) for latch in self.latches]
        self._engaged = bool(results) and all(results)
        return self._engaged
