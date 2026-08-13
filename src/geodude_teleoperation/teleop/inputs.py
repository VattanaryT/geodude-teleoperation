"""Start / stop teleop input devices. Keyboard is always available; SpaceMouse
is an optional add-on that does not replace keyboard bindings.
"""

from __future__ import annotations

from typing import Any, Callable

from geodude_teleoperation.teleop.keyboard import KeyboardTeleop
from geodude_teleoperation.teleop.spacemouse import SpaceMouseTeleop


def start_keyboard_teleop(
    *,
    step: float,
    on_twist: Callable,
    on_button: Callable | None = None,
    on_enable: Callable | None = None,
    enable_key: str = "\n",
    keyboard: Any | None = None,
):
    """Start keyboard teleop. Pass ``keyboard=`` to inject a fake in tests."""
    kb = keyboard if keyboard is not None else KeyboardTeleop(step=step)
    kb.start(
        on_twist,
        on_button=on_button,
        on_enable=on_enable,
        enable_key=enable_key,
    )
    return kb


def spacemouse_config_from_args(args) -> dict:
    """SSoT / Namespace → SpaceMouseTeleop constructor kwargs."""
    hz = getattr(args, "spacemouse_hz", None)
    if hz is None or (isinstance(hz, (int, float)) and float(hz) <= 0):
        frequency = float(getattr(args, "control_hz", 50.0))
    else:
        frequency = float(hz)
    signs = getattr(args, "spacemouse_axis_sign", None)
    if signs is None:
        axis_sign = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    else:
        axis_sign = tuple(float(x) for x in signs)
    path = getattr(args, "spacemouse_path", None)
    if path == "":
        path = None
    return {
        "linear_scale": float(getattr(args, "spacemouse_linear_scale", 0.05)),
        "angular_scale": float(getattr(args, "spacemouse_angular_scale", 0.5)),
        "deadzone": float(getattr(args, "spacemouse_deadzone", 0.05)),
        "frequency": frequency,
        "axis_sign": axis_sign,
        "path": path,
    }


def start_spacemouse_teleop(
    *,
    on_twist: Callable,
    on_button: Callable | None = None,
    on_enable: Callable | None = None,
    linear_scale: float = 0.05,
    angular_scale: float = 0.5,
    deadzone: float = 0.05,
    frequency: float = 50.0,
    axis_sign=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    path=None,
    spacemouse: Any | None = None,
):
    """Start SpaceMouse teleop. Pass ``spacemouse=`` to inject a fake in tests."""
    sm = (
        spacemouse
        if spacemouse is not None
        else SpaceMouseTeleop(
            linear_scale=linear_scale,
            angular_scale=angular_scale,
            deadzone=deadzone,
            frequency=frequency,
            axis_sign=axis_sign,
            path=path,
        )
    )
    sm.start(on_twist, on_button=on_button, on_enable=on_enable)
    return sm


def spacemouse_enabled(args) -> bool:
    """True only for a real bool; MagicMock / missing attr → False."""
    v = getattr(args, "spacemouse", False)
    return v if isinstance(v, bool) else False


def start_teleop_inputs(
    args,
    *,
    on_twist: Callable,
    on_button: Callable | None = None,
    on_enable: Callable | None = None,
    keyboard: Any | None = None,
    spacemouse: Any | None = None,
    on_spacemouse_twist: Callable | None = None,
):
    """Always start keyboard. Also start SpaceMouse when ``args.spacemouse``.

    ``on_spacemouse_twist`` defaults to ``on_twist``. Keyboard callbacks and
    key codes are unchanged. Returns ``(keyboard, spacemouse_or_none)``.
    """
    kb = start_keyboard_teleop(
        step=float(getattr(args, "keyboard_step", 0.01)),
        on_twist=on_twist,
        on_button=on_button,
        on_enable=on_enable,
        enable_key=getattr(args, "enable_key", "\n"),
        keyboard=keyboard,
    )
    sm = None
    if spacemouse_enabled(args):
        sm_twist = on_spacemouse_twist if on_spacemouse_twist is not None else on_twist
        sm = start_spacemouse_teleop(
            on_twist=sm_twist,
            on_button=on_button,
            on_enable=on_enable,
            spacemouse=spacemouse,
            **spacemouse_config_from_args(args),
        )
    return kb, sm


def stop_teleop_inputs(keyboard, spacemouse=None) -> None:
    if spacemouse is not None:
        stop = getattr(spacemouse, "stop", None)
        if callable(stop):
            stop()
    if keyboard is not None:
        stop = getattr(keyboard, "stop", None)
        if callable(stop):
            stop()
