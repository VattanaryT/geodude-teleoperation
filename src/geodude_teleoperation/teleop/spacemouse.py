"""SpaceMouse input adapter: polls a 3Dconnexion device on a background thread and
turns each reading into a body-frame twist, mirroring the independent
device-poll-thread design of the original ufactory_demos teleoperation node (there:
a dedicated ROS2 node at the device's own rate; here: a plain Python thread).

Keyboard bindings stay in ``keyboard.py``. This adapter is an optional extra:
button 0 = enable (Enter), button 1 = teleop pause (``t``). VF ``v`` / freeze
``f`` / quit ``q`` remain keyboard-only.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback

import numpy as np

from geodude_teleoperation.teleop.keyboard import (
    contact_field_linear_from_twist_coeffs,
    make_twist,
)

SPACEMOUSE_HELP = """\
SpaceMouse teleoperation (optional; keyboard still works):
  Translate / rotate the cap for analog 6-DOF twist (same blade order as WASD).
  Button 0: enable motion (same as Enter; live locks orientation, then
  press again after settle to approve teleop)
  Button 1: toggle teleoperation pause (same as t)
  v / f / q stay on the keyboard (or Polyscope window).
  f holds pose + freezes VF; press again to resume and unlock fields.
  Idle (within deadzone) holds current TCP pose.
  In-contact: only field-tangent xy (continuous while held); z and angular ignored.
"""


def dispatch_spacemouse_button(index: int) -> tuple[str, object]:
    """Return (kind, payload) for a SpaceMouse button rising edge.

    Button 0 → enable (Enter). Button 1 → teleop toggle (same payload as
    keyboard ``t``). Other indices ignored. Does not change ``dispatch_key``.
    """
    if index == 0:
        return ("enable", None)
    if index == 1:
        return ("button", 0)
    return ("ignore", None)


def spacemouse_axes_to_twist_coeffs(
    x,
    y,
    z,
    roll,
    pitch,
    yaw,
    *,
    linear_scale: float,
    angular_scale: float,
    deadzone: float,
    axis_sign=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
) -> np.ndarray | None:
    """Map device axes to blade-order twist coeffs, or None if all deadzoned.

    Blade order is ``[e12, e13, e23, e1i, e2i, e3i]`` = ``[angular, linear]``.
    ``axis_sign`` is ``[x, y, z, roll, pitch, yaw]`` (``+1`` / ``-1``).
    """
    axes = np.array(
        [float(x), float(y), float(z), float(roll), float(pitch), float(yaw)],
        dtype=float,
    )
    signs = np.asarray(axis_sign, dtype=float).reshape(6)
    axes = axes * signs
    axes = np.where(np.abs(axes) < float(deadzone), 0.0, axes)
    linear = axes[:3] * float(linear_scale)
    angular = axes[3:] * float(angular_scale)
    coeffs = np.concatenate([angular, linear])
    if float(np.linalg.norm(coeffs)) < 1e-12:
        return None
    return coeffs


def contact_tangent_linear_from_spacemouse(coeffs) -> np.ndarray:
    """Field-frame linear ``[lx, ly, 0]`` — drop z so Force PID owns the normal."""
    lin = contact_field_linear_from_twist_coeffs(coeffs)
    lin[2] = 0.0
    return lin


class SpaceMouseTeleop:
    def __init__(
        self,
        linear_scale=0.05,
        angular_scale=0.5,
        deadzone=0.05,
        frequency=50.0,
        axis_sign=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        path=None,
    ):
        self._linear_scale = linear_scale
        self._angular_scale = angular_scale
        self._deadzone = deadzone
        self._axis_sign = tuple(float(x) for x in np.asarray(axis_sign, dtype=float).reshape(6))
        self._path = None if path in (None, "") else str(path)
        self._period = 1.0 / max(float(frequency), 1e-6)
        self._thread = None
        self._running = threading.Event()
        self._on_twist = None
        self._on_button = None
        self._on_enable = None
        self._previous_buttons = None
        self.last_twist_coeffs: np.ndarray | None = None

    def start(self, on_twist, on_button=None, on_enable=None):
        """``on_twist(twist)`` on non-zero readings; ``on_button(index)`` on
        rising edges after SpaceMouse→keyboard mapping (button 1 → index 0);
        ``on_enable()`` on button 0. Zero / deadzoned readings are skipped so
        idle holds pose. Open failure prints a warning and exits the thread."""
        self._on_twist = on_twist
        self._on_button = on_button
        self._on_enable = on_enable
        self._previous_buttons = None
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def handle_state(self, state) -> None:
        """Process one pyspacemouse-like state. Thread-safe enough for tests."""
        if not self._running.is_set():
            return
        try:
            coeffs = spacemouse_axes_to_twist_coeffs(
                getattr(state, "x", 0.0),
                getattr(state, "y", 0.0),
                getattr(state, "z", 0.0),
                getattr(state, "roll", 0.0),
                getattr(state, "pitch", 0.0),
                getattr(state, "yaw", 0.0),
                linear_scale=self._linear_scale,
                angular_scale=self._angular_scale,
                deadzone=self._deadzone,
                axis_sign=self._axis_sign,
            )
            on_twist = self._on_twist
            on_button = self._on_button
            on_enable = self._on_enable
            if coeffs is not None and on_twist is not None:
                self.last_twist_coeffs = np.asarray(coeffs, dtype=float).copy()
                on_twist(make_twist(coeffs))

            buttons = tuple(getattr(state, "buttons", ()) or ())
            prev = self._previous_buttons
            if prev is not None:
                for index, (was, now) in enumerate(zip(prev, buttons)):
                    if now and not was:
                        kind, payload = dispatch_spacemouse_button(index)
                        if kind == "enable" and on_enable is not None:
                            on_enable()
                        elif kind == "button" and on_button is not None:
                            on_button(payload)
            self._previous_buttons = buttons
        except Exception:
            print("SpaceMouseTeleop: on_twist/on_button raised, continuing:")
            traceback.print_exc()

    def _run(self):
        # Imported lazily: pyspacemouse pulls in a HID backend that needs actual
        # device access, which is not available (and not needed) elsewhere.
        try:
            import pyspacemouse
        except Exception as exc:
            print(
                f"warning: pyspacemouse import failed ({exc}); keyboard-only",
                file=sys.stderr,
            )
            return
        try:
            open_kw = {"path": self._path} if self._path else {}
            with pyspacemouse.open(**open_kw) as device:
                print(SPACEMOUSE_HELP, flush=True)
                while self._running.is_set():
                    state = device.read()
                    if state is not None:
                        self.handle_state(state)
                    time.sleep(self._period)
        except Exception as exc:
            print(
                f"warning: SpaceMouse unavailable ({exc}); keyboard-only",
                file=sys.stderr,
            )
