"""Modular teleop input startup: keyboard-only vs keyboard + SpaceMouse."""

from types import SimpleNamespace

from geodude_teleoperation.teleop.inputs import (
    spacemouse_config_from_args,
    start_keyboard_teleop,
    start_spacemouse_teleop,
    start_teleop_inputs,
    stop_teleop_inputs,
)


class _FakeKeyboard:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.kwargs = None

    def start(self, on_twist, on_button=None, on_enable=None, enable_key="\n"):
        self.started = True
        self.kwargs = {
            "on_twist": on_twist,
            "on_button": on_button,
            "on_enable": on_enable,
            "enable_key": enable_key,
        }

    def stop(self):
        self.stopped = True


class _FakeSpaceMouse:
    def __init__(self, **_kwargs):
        self.started = False
        self.stopped = False
        self.kwargs = None

    def start(self, on_twist, on_button=None, on_enable=None):
        self.started = True
        self.kwargs = {
            "on_twist": on_twist,
            "on_button": on_button,
            "on_enable": on_enable,
        }

    def stop(self):
        self.stopped = True


def test_spacemouse_config_from_args_uses_dedicated_hz_and_signs():
    args = SimpleNamespace(
        spacemouse_linear_scale=0.01,
        spacemouse_angular_scale=0.2,
        spacemouse_deadzone=0.08,
        spacemouse_hz=40.0,
        spacemouse_axis_sign=(-1, 1, 1, 1, -1, 1),
        spacemouse_path="/dev/hidraw4",
        control_hz=500.0,
    )
    cfg = spacemouse_config_from_args(args)
    assert cfg["linear_scale"] == 0.01
    assert cfg["angular_scale"] == 0.2
    assert cfg["deadzone"] == 0.08
    assert cfg["frequency"] == 40.0
    assert cfg["axis_sign"] == (-1.0, 1.0, 1.0, 1.0, -1.0, 1.0)
    assert cfg["path"] == "/dev/hidraw4"


def test_spacemouse_config_hz_falls_back_to_control_hz():
    args = SimpleNamespace(
        spacemouse_linear_scale=0.05,
        spacemouse_angular_scale=0.5,
        spacemouse_deadzone=0.05,
        spacemouse_hz=None,
        spacemouse_axis_sign=None,
        spacemouse_path=None,
        control_hz=125.0,
    )
    cfg = spacemouse_config_from_args(args)
    assert cfg["frequency"] == 125.0
    assert cfg["axis_sign"] == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert cfg["path"] is None


def test_start_keyboard_teleop_alone():
    kb = _FakeKeyboard()
    on_twist = object()
    out = start_keyboard_teleop(
        step=0.01,
        on_twist=on_twist,
        on_button=None,
        on_enable=None,
        enable_key="\n",
        keyboard=kb,
    )
    assert out is kb
    assert kb.started
    assert kb.kwargs["on_twist"] is on_twist
    assert kb.kwargs["enable_key"] == "\n"


def test_start_teleop_inputs_keyboard_only_when_spacemouse_off():
    args = SimpleNamespace(
        keyboard_step=0.01,
        enable_key="\n",
        spacemouse=False,
    )
    kb = _FakeKeyboard()
    sm_unused = _FakeSpaceMouse()
    out_kb, out_sm = start_teleop_inputs(
        args,
        on_twist=lambda t: None,
        on_button=lambda i: None,
        on_enable=lambda: None,
        keyboard=kb,
        spacemouse=sm_unused,
    )
    assert out_kb is kb
    assert kb.started
    assert out_sm is None
    assert not sm_unused.started


def test_start_teleop_inputs_adds_spacemouse_without_replacing_keyboard():
    args = SimpleNamespace(
        keyboard_step=0.02,
        enable_key="e",
        spacemouse=True,
        spacemouse_linear_scale=0.05,
        spacemouse_angular_scale=0.5,
        spacemouse_deadzone=0.05,
        control_hz=50.0,
    )
    kb = _FakeKeyboard()
    sm = _FakeSpaceMouse()
    on_twist = lambda t: None
    on_sm_twist = lambda t: None
    on_button = lambda i: None
    on_enable = lambda: None
    out_kb, out_sm = start_teleop_inputs(
        args,
        on_twist=on_twist,
        on_button=on_button,
        on_enable=on_enable,
        keyboard=kb,
        spacemouse=sm,
        on_spacemouse_twist=on_sm_twist,
    )
    assert out_kb is kb and kb.started
    assert kb.kwargs["on_twist"] is on_twist
    assert kb.kwargs["enable_key"] == "e"
    assert out_sm is sm and sm.started
    assert sm.kwargs["on_twist"] is on_sm_twist
    assert sm.kwargs["on_button"] is on_button
    assert sm.kwargs["on_enable"] is on_enable


def test_stop_teleop_inputs_stops_both():
    kb = _FakeKeyboard()
    sm = _FakeSpaceMouse()
    stop_teleop_inputs(kb, sm)
    assert kb.stopped
    assert sm.stopped
    stop_teleop_inputs(kb, None)
    assert kb.stopped
