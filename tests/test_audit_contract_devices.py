"""Contract audit for the two input devices.

`SpaceMouseTeleop` and `KeyboardTeleop` are used interchangeably by every
example (`--device keyboard|spacemouse`), so they must expose the same
interface *and the same behaviour* for the callbacks a caller registers.
"""
import ast
import inspect
import pathlib
import sys
import types

import numpy as np
import pytest

gafropy = pytest.importorskip("gafropy")

from geodude_teleoperation.keyboard import KeyboardTeleop  # noqa: E402
from geodude_teleoperation.spacemouse import (  # noqa: E402
    SpaceMouseTeleop,
    is_idle_twist,
    rising_button_indices,
)

PKG = pathlib.Path(
    sys.modules["geodude_teleoperation"].__file__).resolve().parent
DEVICES = [SpaceMouseTeleop, KeyboardTeleop]


@pytest.mark.parametrize("cls", DEVICES, ids=lambda c: c.__name__)
def test_device_exposes_the_start_stop_interface(cls):
    for name in ("start", "stop"):
        assert callable(getattr(cls, name, None)), \
            f"{cls.__name__} is used interchangeably but has no {name}()"


@pytest.mark.parametrize("cls", DEVICES, ids=lambda c: c.__name__)
def test_device_start_accepts_on_twist_and_on_button(cls):
    """Every example calls `device.start(on_twist, on_button)`."""
    params = list(inspect.signature(cls.start).parameters)
    assert params[:3] == ["self", "on_twist", "on_button"], (
        f"{cls.__name__}.start signature is {params}; callers pass "
        "(on_twist, on_button) positionally")


def _self_attr_calls(path, attr):
    """Number of times `self.<attr>` is *called* in a module."""
    tree = ast.parse(path.read_text(), filename=str(path))
    calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == attr \
                and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "self":
            calls += 1
    return calls


def test_keyboard_teleop_actually_invokes_the_on_button_callback():
    """KeyboardTeleop.start documents `on_button(index) fires on 't' (0) / 'v' (1)`.

    Every example registers a button handler and drives a real mode change from
    it (hardware_object_contact_teleop's approach<->contact toggle). If the
    keyboard adapter stores the callback but never calls it, `--device keyboard`
    silently loses that mode toggle with no error anywhere.
    """
    called = _self_attr_calls(PKG / "keyboard.py", "_on_button")
    stored = "self._on_button" in (PKG / "keyboard.py").read_text()
    assert stored, "precondition: KeyboardTeleop stores an _on_button callback"
    assert called > 0, (
        "KeyboardTeleop stores on_button but never calls it -- the documented "
        "'t' (0) / 'v' (1) buttons, and therefore --device keyboard's mode "
        "toggle, are dead. SpaceMouseTeleop does call its on_button.")


def test_spacemouse_teleop_actually_invokes_the_on_button_callback():
    assert _self_attr_calls(PKG / "spacemouse.py", "_on_button") > 0


def _fake_pyspacemouse(reading):
    """Minimal stand-in for the HID package (no device, no hidapi)."""
    class _State:
        t = 0.0
        x = y = z = roll = pitch = yaw = 0.0
        buttons = (0, 0)

    state = _State()
    for key, value in reading.items():
        setattr(state, key, value)

    class _Device:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return state

    module = types.ModuleType("pyspacemouse")
    module.open_by_path = lambda path: _Device()
    module.open = lambda: _Device()
    module.get_connected_devices_by_path = lambda: {"/dev/hidraw0": "fake"}
    return module


def test_verify_spacemouse_connection_with_zero_wait_never_verifies(monkeypatch):
    """`SpaceMouseTeleop._run` calls this with `wait_s=0.0`.

    With wait_s=0 the deadline has already passed, so the polling loop body
    never runs and the function reports "no motion" no matter what the device
    is doing -- the startup check is a no-op that always prints its warning.
    """
    monkeypatch.setitem(
        sys.modules, "pyspacemouse",
        _fake_pyspacemouse({"x": 1.0, "y": 1.0, "z": 1.0}))
    from geodude_teleoperation.spacemouse import verify_spacemouse_connection

    saw_motion, peak = verify_spacemouse_connection("/dev/hidraw0", wait_s=0.0)
    assert (saw_motion, peak) == (True, 1.0), (
        "verify_spacemouse_connection(wait_s=0.0) reported "
        f"(saw_motion={saw_motion}, peak={peak}) for a device at full "
        "deflection. SpaceMouseTeleop._run passes wait_s=0.0, so its "
        "connection check can never succeed and always warns.")


@pytest.mark.parametrize("previous,buttons,expected", [
    (None, (0, 0), ()),
    (None, (1, 0), ()),          # held at start must not fire
    ((0, 0), (1, 0), (0,)),
    ((1, 0), (1, 0), ()),        # still held: no repeat
    ((1, 0), (0, 0), ()),        # release is not a rising edge
    ((0, 0), (1, 1), (0, 1)),
    ((0,), (1, 1), (0,)),        # ragged widths clamp to the shorter tuple
])
def test_rising_button_indices(previous, buttons, expected):
    _, edges = rising_button_indices(previous, buttons)
    assert edges == expected


def _twist(angular, linear):
    return gafropy.Twist.from_array(np.concatenate([angular, linear]))


def test_is_idle_twist_treats_rest_bias_as_idle():
    # Just above the deadzone on every axis, at SpaceMouseTeleop's own defaults.
    rest = _twist(np.full(3, 0.06 * 0.1), np.full(3, 0.06 * 0.01))
    assert is_idle_twist(rest) is True


def test_is_idle_twist_rejects_a_real_push():
    push = _twist(np.zeros(3), np.array([0.5 * 0.01, 0.0, 0.0]))
    assert is_idle_twist(push) is False


def test_is_idle_twist_reads_the_twist_blade_order_the_devices_write():
    """Both adapters build the array as [angular(3), linear(3)]; is_idle_twist
    must slice it the same way or the two scales get swapped."""
    big_angular_only = _twist(np.full(3, 10.0), np.zeros(3))
    big_linear_only = _twist(np.zeros(3), np.full(3, 10.0))
    assert is_idle_twist(big_angular_only) is False
    assert is_idle_twist(big_linear_only) is False


def test_keyboard_step_maps_to_the_documented_twist_axes():
    """_KEY_TWIST indexes [e12, e13, e23, e1i, e2i, e3i]; 'w' must be +linear-x."""
    from geodude_teleoperation.keyboard import _KEY_TWIST

    index, sign = _KEY_TWIST["w"]
    coeffs = np.zeros(6)
    coeffs[index] = sign * 0.01
    twist = gafropy.Twist.from_array(coeffs)
    np.testing.assert_allclose(np.asarray(twist.get_linear()).ravel(),
                               [0.01, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(twist.get_angular()).ravel(),
                               np.zeros(3), atol=1e-12)


def test_keyboard_help_text_matches_the_actual_key_map():
    """The help banner advertises 'q: quit'; an operator will press it."""
    from geodude_teleoperation.keyboard import KEYBOARD_HELP, _KEY_TWIST

    advertised = {"t", "v", "q"}
    handled = set(_KEY_TWIST) | {"t", "T", "\n", "\r"}  # see KeyboardTeleop._run
    unhandled = sorted(k for k in advertised if k in KEYBOARD_HELP
                       and k not in handled)
    assert unhandled == [], (
        f"KEYBOARD_HELP advertises keys the adapter ignores: {unhandled}")
