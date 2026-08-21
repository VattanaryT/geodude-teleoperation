"""CGA motor / rotor sandwiches for local-frame teleop."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("pygafro")

from geodude_teleoperation.cga.motors import (
    rotor_from_rotation_matrix,
    sandwich_direction,
)
from geodude_teleoperation.teleop.contact_teleop import (
    ContactTeleopBridge,
    ContactTeleopConfig,
    field_tangent_world,
)
from geodude_teleoperation.teleop.spacemouse import contact_tangent_linear_from_spacemouse


def test_rotor_sandwich_matches_R_times_v():
    R = Rotation.from_euler("xyz", [25.0, -40.0, 15.0], degrees=True).as_matrix()
    for v in (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.3, -0.7, 0.2]),
    ):
        out = sandwich_direction(R, v)
        assert np.allclose(out, R @ v, atol=1e-12)


def test_rotor_roundtrip_to_matrix():
    R = Rotation.from_euler("zyx", [-10.0, 33.0, 5.0], degrees=True).as_matrix()
    rotor = rotor_from_rotation_matrix(R)
    Rm = np.asarray(rotor.toRotationMatrix(), dtype=float).reshape(3, 3)
    assert np.allclose(Rm, R, atol=1e-12)


def test_field_tangent_world_uses_cga_local_frame():
    R = Rotation.from_euler("xyz", [10.0, 20.0, -30.0], degrees=True).as_matrix()
    # SpaceMouse / WASD: +local-x only.
    v_cga, used = field_tangent_world([1.0, 0.0, 0.0], R, scale=1.0, use_cga_motor=True)
    v_se3, used_off = field_tangent_world(
        [1.0, 0.0, 0.0], R, scale=1.0, use_cga_motor=False
    )
    assert used is True
    assert used_off is False
    assert np.allclose(v_cga, v_se3, atol=1e-12)
    # Local +x → R column 0, with n* (col 2) killed (already ⊥ if orthonormal).
    assert np.allclose(v_cga, R[:, 0], atol=1e-12)
    assert abs(float(np.dot(v_cga, R[:, 2]))) < 1e-12


def test_spacemouse_coeffs_map_through_cga_local_frame():
    """SM xy → field linear → CGA sandwich into world tangent (⊥ n*)."""
    # Blade order [ang, lin]; contact helper keeps xy only.
    coeffs = np.array([0.0, 0.0, 0.0, 0.02, -0.01, 0.05])
    lin = contact_tangent_linear_from_spacemouse(coeffs)
    assert abs(lin[2]) < 1e-15
    assert np.allclose(lin[:2], [0.02, -0.01], atol=1e-12)

    R = Rotation.from_euler("xyz", [5.0, -12.0, 40.0], degrees=True).as_matrix()
    bridge = ContactTeleopBridge(
        ContactTeleopConfig(actuation="speedj", align_first=False, tangent_scale=1.0)
    )
    bridge.arm([0, 0, 0], R, R)
    bridge.push_tangent_field_linear(lin, R, n_approach=R[:, 2], continuous=True)
    v = bridge._consume_tangent(0.02, aligning=False)
    expect, used = field_tangent_world(
        lin, R, scale=1.0, n_approach=R[:, 2], use_cga_motor=True
    )
    assert used is True
    assert bridge._last_cga_tan is True
    assert np.allclose(v, expect, atol=1e-12)
    assert abs(float(np.dot(v, R[:, 2]))) < 1e-12
