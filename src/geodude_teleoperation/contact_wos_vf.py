"""Frame-aware contact / Walk-on-Spheres orientation virtual fixture."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from geodude_teleoperation.field_query import query_field_basis
from geodude_teleoperation.frames import base_to_tag0_point, tag0_rot_to_base, tip_from_tcp


def _orient_approach_into_surface(local_basis: np.ndarray) -> np.ndarray:
    """Match virtual_fixture._orient_approach_into_surface (z = approach into surface)."""
    approach_basis = local_basis.copy()
    approach_basis[:, 1] *= -1
    approach_basis[:, 2] *= -1
    return approach_basis


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


@dataclass
class VFStatus:
    signed_d: float
    in_contact: bool
    tip_nn: float
    used_wos: bool
    query_ms: float
    R_field_base: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "signed_d": self.signed_d,
            "in_contact": self.in_contact,
            "tip_nn": self.tip_nn,
            "used_wos": self.used_wos,
            "query_ms": self.query_ms,
            "R_field_base": self.R_field_base,
        }


class ContactWosOrientationField:
    """Query surface/WoS orientation in tag_0; apply approach convention in base."""

    def __init__(
        self,
        pcloud,
        wos,
        *,
        tip_offset: np.ndarray | None = None,
        contact_threshold: float = 0.001,
        T_tag0_base: np.ndarray | None = None,
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

        self._last_R_tag0: np.ndarray | None = None
        self._last_R_field_base: np.ndarray | None = None
        self._cache_warm = False
        self._status: VFStatus | None = None

    def set_T_tag0_base(self, T: np.ndarray) -> None:
        self._T_tag0_base = np.asarray(T, dtype=float).reshape(4, 4)

    def update_from_state(self, ee_motor, T_tag0_base: np.ndarray | None = None) -> None:
        if T_tag0_base is not None:
            self.set_T_tag0_base(T_tag0_base)

        t0 = time.perf_counter()
        p_tcp, R_tcp = _tcp_pose_from_motor(ee_motor)
        p_tip_tag0 = tip_tag0_from_tcp(p_tcp, R_tcp, self._tip_offset, self._T_tag0_base)

        used_wos = False
        nearest_idx = 0
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
            R_tag0 = self._last_R_tag0 if self._last_R_tag0 is not None else np.eye(3)
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
        R_field_base = _orient_approach_into_surface(R_base)
        self._last_R_field_base = R_field_base.copy()

        tip_nn = float(
            np.linalg.norm(p_tip_tag0 - self._pcloud.vertices[nearest_idx])
        )
        query_ms = (time.perf_counter() - t0) * 1000.0

        self._status = VFStatus(
            signed_d=float(signed_d),
            in_contact=bool(in_contact),
            tip_nn=tip_nn,
            used_wos=used_wos,
            query_ms=query_ms,
            R_field_base=R_field_base,
        )
        self._cache_warm = True

    def get_distance(self, position) -> float:
        """Return cached abs(signed_d) from the last tip query (no re-query)."""
        del position  # matches controller call site; distance comes from cache
        if self._status is None:
            return 0.0
        return abs(self._status.signed_d)

    def project(self, proposed_reference):
        import gafropy

        if not self._cache_warm:
            self.update_from_state(proposed_reference)
        assert self._last_R_field_base is not None
        translator = proposed_reference.get_translator()
        rotor = gafropy.Rotor.from_matrix(self._last_R_field_base)
        return gafropy.Motor(translator, rotor)

    @property
    def status(self) -> dict[str, Any]:
        if self._status is None:
            return {
                "signed_d": 0.0,
                "in_contact": False,
                "tip_nn": 0.0,
                "used_wos": False,
                "query_ms": 0.0,
                "R_field_base": np.eye(3),
            }
        return self._status.as_dict()
