"""Frame-aware contact / Walk-on-Spheres orientation virtual fixture."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from geodude_teleoperation.field.field_query import query_field_basis
from geodude_teleoperation.frames import base_to_tag0_point, tag0_rot_to_base, tip_from_tcp


def _orient_approach_into_surface(local_basis: np.ndarray) -> np.ndarray:
    """Match virtual_fixture._orient_approach_into_surface (z = approach into surface)."""
    approach_basis = local_basis.copy()
    approach_basis[:, 1] *= -1
    approach_basis[:, 2] *= -1
    return approach_basis


def slerp_R(Ra: np.ndarray, Rb: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear blend of rotations; ``t=0`` → ``Ra``, ``t=1`` → ``Rb``."""
    from scipy.spatial.transform import Rotation, Slerp

    t = float(np.clip(t, 0.0, 1.0))
    if t <= 1e-12:
        return np.asarray(Ra, dtype=float).reshape(3, 3).copy()
    if t >= 1.0 - 1e-12:
        return np.asarray(Rb, dtype=float).reshape(3, 3).copy()
    q0 = Rotation.from_matrix(np.asarray(Ra, dtype=float).reshape(3, 3)).as_quat()
    q1 = Rotation.from_matrix(np.asarray(Rb, dtype=float).reshape(3, 3)).as_quat()
    if np.dot(q0, q1) < 0:
        q1 = -q1
    return Slerp([0.0, 1.0], Rotation.from_quat(np.vstack([q0, q1])))([t])[0].as_matrix()


def smooth_R_field(
    R_prev: np.ndarray | None, R_new: np.ndarray, *, smoothing: float
) -> np.ndarray:
    """EMA on SO(3): ``smoothing`` is weight on previous (match OrientationField 0.75)."""
    R_new = np.asarray(R_new, dtype=float).reshape(3, 3)
    if R_prev is None or float(smoothing) <= 0.0:
        return R_new.copy()
    # OrientationField: nlerp(last, new, 1 - smoothing) → blend toward new.
    return slerp_R(R_prev, R_new, 1.0 - float(smoothing))


# Shared with Polyscope --viz-teleop-keys and live RTDE VF.
DEFAULT_R_FIELD_SMOOTHING = 0.75
DEFAULT_RELOCK_DIST_M = 0.012
DEFAULT_RELOCK_PERIOD_S = 0.10
DEFAULT_MOTION_TRACK = 0.55


def tip_tag0_from_tcp(
    p_tcp: np.ndarray,
    R_tcp: np.ndarray,
    tip_offset: np.ndarray,
    T_tag0_base: np.ndarray,
) -> np.ndarray:
    """TCP pose in base → tip position in tag_0 (pure-numpy helper)."""
    p_tip_base = tip_from_tcp(p_tcp, R_tcp, tip_offset)
    return base_to_tag0_point(p_tip_base, T_tag0_base)


def _continuity_flip(R_new: np.ndarray, R_last: np.ndarray | None) -> np.ndarray:
    """Pick the quaternion hemisphere continuous with the previous basis."""
    R_new = np.asarray(R_new, dtype=float).reshape(3, 3)
    if R_last is not None:
        q_old = Rotation.from_matrix(R_last).as_quat()
        q_new = Rotation.from_matrix(R_new).as_quat()
        if np.dot(q_old, q_new) < 0:
            R_new = Rotation.from_quat(-q_new).as_matrix()
    return R_new


def _rotor_to_matrix(rotor) -> np.ndarray:
    if hasattr(rotor, "as_matrix"):
        return np.asarray(rotor.as_matrix(), dtype=float).reshape(3, 3)
    from diffused_fields.manifold._rotor_interop import as_scipy_rotation

    return as_scipy_rotation(rotor).as_matrix()


def _tcp_pose_from_motor(ee_motor) -> tuple[np.ndarray, np.ndarray]:
    translator = ee_motor.get_translator()
    p_tcp = np.array([translator.x(), translator.y(), translator.z()], dtype=float)
    R_tcp = _rotor_to_matrix(ee_motor.get_rotor())
    return p_tcp, R_tcp


class _FallbackMotor:
    """Minimal motor when gafropy is unavailable (CI / tests)."""

    def __init__(self, translator, R: np.ndarray):
        self._translator = translator
        self._R = np.asarray(R, dtype=float).reshape(3, 3)

    def get_translator(self):
        return self._translator

    def get_rotor(self):
        return Rotation.from_matrix(self._R)


def _motor_with_field_orientation(proposed_reference, R_field_base: np.ndarray):
    """Keep proposed translator; replace orientation with R_field_base."""
    translator = proposed_reference.get_translator()
    try:
        import gafropy

        rotor = gafropy.Rotor.from_matrix(R_field_base)
        return gafropy.Motor(translator, rotor)
    except ImportError:
        return _FallbackMotor(translator, R_field_base)


@dataclass
class VFStatus:
    signed_d: float
    in_contact: bool
    tip_nn: float
    used_wos: bool
    query_ms: float
    R_field_base: np.ndarray
    R_motion_base: np.ndarray
    relocked: bool = False
    tip_delta_m: float = 0.0
    dt_since_relock_s: float = 0.0
    relock_reason: str = "hold"

    def as_dict(self) -> dict[str, Any]:
        return {
            "signed_d": self.signed_d,
            "in_contact": self.in_contact,
            "tip_nn": self.tip_nn,
            "used_wos": self.used_wos,
            "query_ms": self.query_ms,
            # Teleop / project use motion frame (relock-gated).
            "R_field_base": self.R_motion_base,
            "R_field_raw_smoothed": self.R_field_base,
            "R_motion_base": self.R_motion_base,
            "relocked": bool(self.relocked),
            "tip_delta_m": float(self.tip_delta_m),
            "dt_since_relock_s": float(self.dt_since_relock_s),
            "relock_reason": self.relock_reason,
        }


def twist_field_alignment_rad(
    cmd_dir_world: np.ndarray | None, R_field: np.ndarray | None
) -> float:
    """Min angle (rad) between commanded world linear dir and field t̂1 / t̂2.

    Large values while holding a/d (field ±y → t̂2) mean the composition frame
    is misaligned with the live field at the tip.
    """
    if cmd_dir_world is None or R_field is None:
        return float("nan")
    v = np.asarray(cmd_dir_world, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return float("nan")
    v = v / n
    R = np.asarray(R_field, dtype=float).reshape(3, 3)
    best = 0.0
    for col in (0, 1):
        t = R[:, col]
        tn = float(np.linalg.norm(t))
        if tn < 1e-12:
            continue
        c = float(np.clip(abs(np.dot(v, t / tn)), 0.0, 1.0))
        ang = float(np.arccos(c))
        if col == 0:
            best = ang
        else:
            best = min(best, ang)
    return best


class ContactWosOrientationField:
    """Query surface/WoS orientation; EMA-smooth + distance/time motion re-lock.

    Matches Polyscope ``--viz-teleop-keys`` defaults: smooth ``R_field``, then
    only advance the teleop/project frame (``R_motion``) every
    ``relock_dist_m`` of tip travel or ``relock_period_s``.
    """

    def __init__(
        self,
        pcloud,
        wos,
        *,
        tip_offset: np.ndarray | None = None,
        contact_threshold: float = 0.001,
        T_tag0_base: np.ndarray | None = None,
        r_field_smoothing: float = DEFAULT_R_FIELD_SMOOTHING,
        relock_dist_m: float = DEFAULT_RELOCK_DIST_M,
        relock_period_s: float = DEFAULT_RELOCK_PERIOD_S,
        motion_track: float = DEFAULT_MOTION_TRACK,
        **query_kwargs,
    ):
        self._pcloud = pcloud
        self._wos = wos
        self._tip_offset = (
            np.zeros(3, dtype=float)
            if tip_offset is None
            else np.asarray(tip_offset, dtype=float).reshape(3)
        )
        self._contact_threshold = float(contact_threshold)
        self._T_tag0_base = (
            np.eye(4, dtype=float)
            if T_tag0_base is None
            else np.asarray(T_tag0_base, dtype=float).reshape(4, 4)
        )
        self._query_kwargs = dict(query_kwargs)
        self._r_field_smoothing = float(r_field_smoothing)
        self._relock_dist_m = float(relock_dist_m)
        self._relock_period_s = float(relock_period_s)
        self._motion_track = float(np.clip(motion_track, 0.0, 1.0))

        self._last_R_tag0: np.ndarray | None = None
        self._last_R_field_base: np.ndarray | None = None
        self._R_motion: np.ndarray | None = None
        self._last_tip_tag0: np.ndarray | None = None
        self._tip_at_relock: np.ndarray | None = None
        self._last_relock_t: float = 0.0
        self._last_ee_motor = None
        self._cache_warm = False
        self._status: VFStatus | None = None

    def set_T_tag0_base(self, T: np.ndarray) -> None:
        self._T_tag0_base = np.asarray(T, dtype=float).reshape(4, 4)

    def motion_basis(self) -> np.ndarray:
        """Orientation used for teleop twists / ``project`` (relock-gated)."""
        if self._R_motion is not None:
            return self._R_motion.copy()
        if self._last_R_field_base is not None:
            return self._last_R_field_base.copy()
        return np.eye(3)

    def _relock_decision(
        self, tip_tag0: np.ndarray, now: float
    ) -> tuple[bool, str, float, float]:
        """Return ``(do_query, reason, tip_delta_m, dt_since_relock_s)``.

        Units: tip positions are metres (tag_0); times are seconds (monotonic).

        ``relock_period_s`` is a **hard minimum** between queries. Without that
        floor, fast teleop (‖q̇‖~1 with adaptive scale) moves the tip >12 mm
        every host tick, so the distance gate alone re-fires every sample and
        never throttles. After the min period, re-query if tip travel ≥
        ``relock_dist_m`` **or** the period has elapsed (stationary refresh).
        """
        if self._R_motion is None or self._tip_at_relock is None:
            return True, "cold", 0.0, 0.0
        dist = float(np.linalg.norm(tip_tag0 - self._tip_at_relock))
        dt = float(now - self._last_relock_t)
        want_dist = dist >= self._relock_dist_m
        want_time = dt >= self._relock_period_s
        if not (want_dist or want_time):
            return False, "hold", dist, dt
        # Hard rate limit: never query more often than relock_period_s.
        if dt < self._relock_period_s:
            return False, "rate_limit", dist, dt
        if want_dist and want_time:
            return True, "dist+time", dist, dt
        if want_dist:
            return True, "dist", dist, dt
        return True, "time", dist, dt

    def update_from_state(self, ee_motor, T_tag0_base: np.ndarray | None = None) -> None:
        if T_tag0_base is not None:
            self.set_T_tag0_base(T_tag0_base)

        t0 = time.perf_counter()
        now = time.monotonic()
        p_tcp, R_tcp = _tcp_pose_from_motor(ee_motor)
        p_tip_tag0 = tip_tag0_from_tcp(p_tcp, R_tcp, self._tip_offset, self._T_tag0_base)
        self._last_tip_tag0 = p_tip_tag0.copy()

        if not self._cache_warm:
            do_query, reason, tip_delta_m, dt_relock = True, "cold", 0.0, 0.0
        else:
            do_query, reason, tip_delta_m, dt_relock = self._relock_decision(
                p_tip_tag0, now
            )
        used_wos = False
        nearest_idx = 0
        query_ms = 0.0
        relocked = bool(do_query)

        if do_query:
            try:
                R_tag0, signed_d, in_contact, diag = query_field_basis(
                    self._pcloud,
                    self._wos,
                    p_tip_tag0,
                    contact_threshold=self._contact_threshold,
                    **self._query_kwargs,
                )
                used_wos = bool(diag.get("used_wos", False))
                nearest_idx = int(diag.get("nearest_idx", 0))
            except Exception:
                R_tag0 = (
                    self._last_R_tag0 if self._last_R_tag0 is not None else np.eye(3)
                )
                if self._status is not None:
                    signed_d = self._status.signed_d
                    in_contact = self._status.in_contact
                    used_wos = self._status.used_wos
                else:
                    signed_d = 0.0
                    in_contact = False

            R_tag0 = _continuity_flip(R_tag0, self._last_R_tag0)
            self._last_R_tag0 = R_tag0.copy()

            R_base = tag0_rot_to_base(R_tag0, self._T_tag0_base)
            R_field_raw = _orient_approach_into_surface(R_base)
            R_field_base = smooth_R_field(
                self._last_R_field_base,
                R_field_raw,
                smoothing=self._r_field_smoothing,
            )
            self._last_R_field_base = R_field_base.copy()

            if self._R_motion is None:
                self._R_motion = R_field_base.copy()
            else:
                self._R_motion = slerp_R(
                    self._R_motion, R_field_base, self._motion_track
                )
            self._tip_at_relock = p_tip_tag0.copy()
            self._last_relock_t = now
            query_ms = (time.perf_counter() - t0) * 1000.0
        else:
            assert self._status is not None and self._last_R_field_base is not None
            signed_d = self._status.signed_d
            in_contact = self._status.in_contact
            used_wos = self._status.used_wos
            R_field_base = self._last_R_field_base
            nearest_idx = int(
                np.linalg.norm(self._pcloud.vertices - p_tip_tag0.reshape(1, 3), axis=1).argmin()
            )
            query_ms = (time.perf_counter() - t0) * 1000.0

        tip_nn = float(
            np.linalg.norm(p_tip_tag0 - self._pcloud.vertices[nearest_idx])
        )
        assert self._R_motion is not None
        self._status = VFStatus(
            signed_d=float(signed_d),
            in_contact=bool(in_contact),
            tip_nn=tip_nn,
            used_wos=used_wos,
            query_ms=query_ms,
            R_field_base=R_field_base,
            R_motion_base=self._R_motion.copy(),
            relocked=relocked,
            tip_delta_m=float(tip_delta_m),
            dt_since_relock_s=float(dt_relock),
            relock_reason=reason,
        )
        self._last_ee_motor = ee_motor
        self._cache_warm = True

    def get_distance(self, position) -> float:
        """Return cached abs(signed_d) from the last tip query."""
        if not self._cache_warm:
            if self._last_ee_motor is not None:
                self.update_from_state(self._last_ee_motor)
            else:
                return float("nan")
        assert self._status is not None
        return abs(self._status.signed_d)

    def project(self, proposed_reference):
        if not self._cache_warm:
            self.update_from_state(proposed_reference)
        R = self.motion_basis()
        return _motor_with_field_orientation(proposed_reference, R)

    @property
    def status(self) -> dict[str, Any]:
        if self._status is None:
            I = np.eye(3)
            return {
                "signed_d": 0.0,
                "in_contact": False,
                "tip_nn": 0.0,
                "used_wos": False,
                "query_ms": 0.0,
                "R_field_base": I,
                "R_field_raw_smoothed": I,
                "R_motion_base": I,
                "relocked": False,
                "tip_delta_m": 0.0,
                "dt_since_relock_s": 0.0,
                "relock_reason": "hold",
            }
        return self._status.as_dict()


class FixedOrientationField:
    """Constant orientation fixture (wobble diag: bypass live field / relock).

    Locks ``R_motion`` to the EE orientation at first ``update_from_state``
    (or an explicit ``R_fixed``), then never re-queries the cloud.
    """

    def __init__(self, R_fixed: np.ndarray | None = None):
        self._R = (
            None
            if R_fixed is None
            else np.asarray(R_fixed, dtype=float).reshape(3, 3).copy()
        )
        self._status: VFStatus | None = None
        self._cache_warm = False

    def set_T_tag0_base(self, T: np.ndarray) -> None:
        pass

    def motion_basis(self) -> np.ndarray:
        if self._R is None:
            return np.eye(3)
        return self._R.copy()

    def update_from_state(self, ee_motor, T_tag0_base: np.ndarray | None = None) -> None:
        del T_tag0_base
        t0 = time.perf_counter()
        _, R_tcp = _tcp_pose_from_motor(ee_motor)
        if self._R is None:
            self._R = np.asarray(R_tcp, dtype=float).reshape(3, 3).copy()
        query_ms = (time.perf_counter() - t0) * 1000.0
        self._status = VFStatus(
            signed_d=0.0,
            in_contact=False,
            tip_nn=0.0,
            used_wos=False,
            query_ms=query_ms,
            R_field_base=self._R.copy(),
            R_motion_base=self._R.copy(),
            relocked=not self._cache_warm,
            tip_delta_m=0.0,
            dt_since_relock_s=0.0,
            relock_reason="cold" if not self._cache_warm else "hold",
        )
        self._cache_warm = True

    def get_distance(self, position) -> float:
        del position
        return 0.0

    def project(self, proposed_reference):
        if not self._cache_warm:
            self.update_from_state(proposed_reference)
        assert self._R is not None
        return _motor_with_field_orientation(proposed_reference, self._R)

    @property
    def status(self) -> dict[str, Any]:
        if self._status is None:
            I = np.eye(3)
            return {
                "signed_d": 0.0,
                "in_contact": False,
                "tip_nn": 0.0,
                "used_wos": False,
                "query_ms": 0.0,
                "R_field_base": I,
                "R_field_raw_smoothed": I,
                "R_motion_base": I,
                "relocked": False,
                "tip_delta_m": 0.0,
                "dt_since_relock_s": 0.0,
                "relock_reason": "hold",
            }
        return self._status.as_dict()
