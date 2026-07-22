"""SpaceMouse input adapter: polls a 3Dconnexion device on a background thread and
turns each reading into a body-frame gafropy.Twist, mirroring the independent
device-poll-thread design of the original ufactory_demos teleoperation node (there:
a dedicated ROS2 node at the device's own rate; here: a plain Python thread).
"""

import threading
import time

import gafropy
import numpy as np


class SpaceMouseTeleop:
    def __init__(self, linear_scale=0.05, angular_scale=0.5, deadzone=0.05, frequency=50.0):
        self._linear_scale = linear_scale
        self._angular_scale = angular_scale
        self._deadzone = deadzone
        self._period = 1.0 / frequency
        self._thread = None
        self._running = threading.Event()
        self._on_twist = None
        self._on_button = None

    def start(self, on_twist, on_button=None):
        """on_twist(twist: gafropy.Twist) is called at `frequency` Hz with the latest
        reading (zero twist included -- the caller decides what a zero twist means).
        on_button(index) fires once per button press (rising edge only)."""
        self._on_twist = on_twist
        self._on_button = on_button
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self):
        # Imported lazily: pyspacemouse pulls in a HID backend that needs actual device
        # access, which is not available (and not needed) anywhere else in this package.
        import pyspacemouse

        with pyspacemouse.open() as device:
            previous_buttons = None

            while self._running.is_set():
                state = device.read()

                axes = np.array([state.x, state.y, state.z, state.roll, state.pitch, state.yaw])
                axes = np.where(np.abs(axes) < self._deadzone, 0.0, axes)

                linear = axes[:3] * self._linear_scale
                angular = axes[3:] * self._angular_scale
                # Twist blade order is [e12, e13, e23, e1i, e2i, e3i] = [angular, linear].
                twist = gafropy.Twist.from_array(np.concatenate([angular, linear]))
                self._on_twist(twist)

                if self._on_button is not None:
                    buttons = tuple(state.buttons)
                    if previous_buttons is not None:
                        for index, (prev, cur) in enumerate(zip(previous_buttons, buttons)):
                            if cur and not prev:
                                self._on_button(index)
                    previous_buttons = buttons

                time.sleep(self._period)
