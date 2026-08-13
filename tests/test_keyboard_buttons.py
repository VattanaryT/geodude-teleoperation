import numpy as np

from geodude_teleoperation.teleop.keyboard import dispatch_key, make_twist


def test_enable_enter():
    assert dispatch_key("\n", step=0.01)[0] == "enable"


def test_handle_key_enable_and_twist():
    from geodude_teleoperation.teleop.keyboard import KeyboardTeleop

    events = []
    kb = KeyboardTeleop(step=0.01)
    kb._running.set()
    kb._on_twist = lambda tw: events.append(("twist", tw.get_linear().tolist()))
    kb._on_enable = lambda: events.append(("enable", None))
    kb.handle_key("\n")
    kb.handle_key("w")
    assert events[0][0] == "enable"
    assert events[1][0] == "twist"
    assert np.allclose(events[1][1], [0.01, 0.0, 0.0])


def test_t_v_buttons():
    assert dispatch_key("t", step=0.01) == ("button", 0)
    assert dispatch_key("v", step=0.01) == ("button", 1)
    assert dispatch_key("f", step=0.01) == ("button", 2)


def test_contact_tangent_helpers():
    from geodude_teleoperation.teleop.keyboard import (
        contact_field_linear_from_twist_coeffs,
        is_contact_tangent_key,
    )

    assert is_contact_tangent_key("w")
    assert is_contact_tangent_key("a")
    assert not is_contact_tangent_key("x")
    assert not is_contact_tangent_key("n")
    _, w = dispatch_key("w", step=0.01)
    assert np.allclose(contact_field_linear_from_twist_coeffs(w), [0.01, 0, 0])


def test_twist_blade_order_wasd():
    """Keyboard payload is [e12,e13,e23,e1i,e2i,e3i] = [angular(3), linear(3)]."""
    kind, w = dispatch_key("w", step=0.01)
    assert kind == "twist"
    assert np.allclose(w, [0, 0, 0, 0.01, 0, 0])

    _, a = dispatch_key("a", step=0.02)
    assert np.allclose(a, [0, 0, 0, 0, -0.02, 0])

    _, d = dispatch_key("d", step=0.02)
    assert np.allclose(d, [0, 0, 0, 0, 0.02, 0])

    _, n = dispatch_key("n", step=0.03)
    assert np.allclose(n, [0.03, 0, 0, 0, 0, 0])

    tw = make_twist(d)
    assert np.allclose(tw.get_angular(), [0, 0, 0])
    assert np.allclose(tw.get_linear(), [0, 0.02, 0])
