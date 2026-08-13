"""SpaceMouse adapter: buttons, axis→twist, in-contact tangent, handle_state."""

from types import SimpleNamespace

import numpy as np

from geodude_teleoperation.teleop.keyboard import dispatch_key
from geodude_teleoperation.teleop.spacemouse import (
    contact_tangent_linear_from_spacemouse,
    dispatch_spacemouse_button,
    spacemouse_axes_to_twist_coeffs,
)


def test_spacemouse_button_0_enable_does_not_change_keyboard_t():
    assert dispatch_spacemouse_button(0) == ("enable", None)
    # Keyboard t remains teleop toggle (button 0); SpaceMouse btn0 is Enter, not t.
    assert dispatch_key("t", step=0.01) == ("button", 0)
    assert dispatch_key("v", step=0.01) == ("button", 1)
    assert dispatch_key("\n", step=0.01)[0] == "enable"


def test_spacemouse_button_1_is_teleop_toggle():
    assert dispatch_spacemouse_button(1) == ("button", 0)
    assert dispatch_spacemouse_button(2) == ("ignore", None)


def test_spacemouse_axes_blade_order_and_deadzone():
    coeffs = spacemouse_axes_to_twist_coeffs(
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        linear_scale=0.05,
        angular_scale=0.5,
        deadzone=0.05,
    )
    assert coeffs is not None
    # [e12,e13,e23, e1i,e2i,e3i] = [angular, linear]
    assert np.allclose(coeffs, [0, 0, 0, 0.05, 0, 0])

    yaw = spacemouse_axes_to_twist_coeffs(
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        linear_scale=0.05,
        angular_scale=0.5,
        deadzone=0.05,
    )
    assert np.allclose(yaw, [0, 0, 0.5, 0, 0, 0])

    tiny = spacemouse_axes_to_twist_coeffs(
        0.01, 0.0, 0.0, 0.0, 0.0, 0.0,
        linear_scale=0.05,
        angular_scale=0.5,
        deadzone=0.05,
    )
    assert tiny is None


def test_spacemouse_axis_sign_inverts():
    flipped = spacemouse_axes_to_twist_coeffs(
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        linear_scale=0.05,
        angular_scale=0.5,
        deadzone=0.05,
        axis_sign=(-1, 1, 1, 1, 1, -1),
    )
    assert np.allclose(flipped, [0, 0, -0.5, -0.05, 0, 0])


def test_contact_tangent_drops_z_keeps_xy():
    coeffs = np.array([0.1, 0.2, 0.3, 0.04, -0.05, 0.06])
    lin = contact_tangent_linear_from_spacemouse(coeffs)
    assert np.allclose(lin, [0.04, -0.05, 0.0])


def test_handle_state_enable_pause_and_twist():
    from geodude_teleoperation.teleop.spacemouse import SpaceMouseTeleop

    events = []
    sm = SpaceMouseTeleop(linear_scale=0.05, angular_scale=0.5, deadzone=0.05)
    sm._running.set()
    sm._on_twist = lambda tw: events.append(
        ("twist", np.asarray(tw.get_linear()).tolist())
    )
    sm._on_button = lambda i: events.append(("button", i))
    sm._on_enable = lambda: events.append(("enable", None))

    idle = SimpleNamespace(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, buttons=(0, 0))
    sm.handle_state(idle)
    assert events == []

    sm.handle_state(
        SimpleNamespace(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, buttons=(1, 0))
    )
    sm.handle_state(
        SimpleNamespace(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, buttons=(1, 1))
    )
    sm.handle_state(
        SimpleNamespace(x=1.0, y=0, z=0, roll=0, pitch=0, yaw=0, buttons=(0, 0))
    )
    assert events[0] == ("enable", None)
    assert events[1] == ("button", 0)
    assert events[2][0] == "twist"
    assert np.allclose(events[2][1], [0.05, 0.0, 0.0])
