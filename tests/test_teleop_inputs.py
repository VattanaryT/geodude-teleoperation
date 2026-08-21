"""Keyboard stays; SpaceMouse is additive."""

from types import SimpleNamespace

from geodude_teleoperation.teleop.inputs import (
    spacemouse_config_from_args,
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
        self.kwargs = {"on_twist": on_twist, "enable_key": enable_key}

    def stop(self):
        self.stopped = True


class _FakeSpaceMouse:
    def __init__(self, **_kwargs):
        self.started = False
        self.stopped = False
        self.kwargs = None

    def start(self, on_twist, on_button=None, on_enable=None):
        self.started = True
        self.kwargs = {"on_twist": on_twist, "on_enable": on_enable}

    def stop(self):
        self.stopped = True


def test_spacemouse_config_hz_and_signs():
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
    assert cfg["frequency"] == 40.0
    assert cfg["axis_sign"] == (-1.0, 1.0, 1.0, 1.0, -1.0, 1.0)
    args.spacemouse_hz = None
    args.spacemouse_axis_sign = None
    args.control_hz = 125.0
    cfg = spacemouse_config_from_args(args)
    assert cfg["frequency"] == 125.0


def test_spacemouse_adds_without_replacing_keyboard():
    args = SimpleNamespace(
        keyboard_step=0.02,
        enable_key="e",
        spacemouse=True,
        spacemouse_linear_scale=0.05,
        spacemouse_angular_scale=0.5,
        spacemouse_deadzone=0.05,
        control_hz=50.0,
    )
    kb, sm = _FakeKeyboard(), _FakeSpaceMouse()
    on_kb, on_sm = object(), object()
    out_kb, out_sm = start_teleop_inputs(
        args,
        on_twist=on_kb,
        on_button=None,
        on_enable=lambda: None,
        keyboard=kb,
        spacemouse=sm,
        on_spacemouse_twist=on_sm,
    )
    assert out_kb is kb and kb.started and kb.kwargs["on_twist"] is on_kb
    assert out_sm is sm and sm.started and sm.kwargs["on_twist"] is on_sm
    off = SimpleNamespace(keyboard_step=0.01, enable_key="\n", spacemouse=False)
    kb2, sm2 = _FakeKeyboard(), _FakeSpaceMouse()
    _, out = start_teleop_inputs(off, on_twist=lambda t: None, keyboard=kb2, spacemouse=sm2)
    assert out is None and not sm2.started
    stop_teleop_inputs(kb, sm)
    assert kb.stopped and sm.stopped
