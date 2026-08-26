"""Keyboard input adapter: alternative to SpaceMouseTeleop for when no physical
device is available. Ports the key bindings from ufactory_demos'
gafro_teleoperation/TeleoperationKeyboard.cpp (WASD/XY/UIJK/NM for the six twist
axes, terminal raw mode via termios, one fixed-size step per keypress -- holding
a key relies on the terminal's own key-repeat for continuous motion, exactly as
the original did) -- plus 't'/'v' toggle keys, since that original device
predates the teleop/virtual-fixture enable toggles (only its 3D-mouse sibling had
buttons; this class's counterpart, SpaceMouseTeleop, needs both).
"""

import _thread
import select
import sys
import termios
import threading
import traceback
import tty

import gafropy
import numpy as np

# key -> (index into the [e12, e13, e23, e1i, e2i, e3i] twist array, sign).
_KEY_TWIST = {
    "w": (3, 1.0),   # linear +x
    "s": (3, -1.0),  # linear -x
    "d": (4, 1.0),   # linear +y
    "a": (4, -1.0),  # linear -y
    "x": (5, 1.0),   # linear +z
    "y": (5, -1.0),  # linear -z
    "n": (0, 1.0),   # angular e12 +
    "m": (0, -1.0),  # angular e12 -
    "j": (1, 1.0),   # angular e13 +
    "k": (1, -1.0),  # angular e13 -
    "u": (2, 1.0),   # angular e23 +
    "i": (2, -1.0),  # angular e23 -
}

# key -> button index handed to ``on_button``. Mirrors SpaceMouseTeleop, whose
# button 0 is the approach<->contact toggle every example registers; without
# these `--device keyboard` had no mode toggle at all.
_KEY_BUTTON = {
    "t": 0,
    "T": 0,
    "v": 1,
    "V": 1,
}

# Keys offered to ``on_key`` before the button map sees them, so the startup
# align (which owns 't' and Enter while it is still waiting) keeps priority.
# ``on_key`` returns truthy to claim the key; anything it declines falls
# through, which is how 't' becomes button 0 once align has finished.
_KEY_PASSTHROUGH = ("t", "T", "\n", "\r")

_KEY_QUIT = ("q", "Q")

KEYBOARD_HELP = """\
Keyboard teleoperation:
  w/s: +x/-x    a/d: -y/+y    y/x: -z/+z   (linear)
  n/m: e12 +/-  j/k: e13 +/-  u/i: e23 +/-  (angular)
  t: toggle teleoperation    v: toggle virtual fixture    q: quit
"""


class KeyboardTeleop:
    def __init__(self, step=0.01):
        self._step = step
        self._thread = None
        self._running = threading.Event()
        self._on_twist = None
        self._on_button = None
        self._on_key = None
        # Seam: _dispatch is unit-tested off the main thread's back, and a real
        # interrupt_main() there would land KeyboardInterrupt in the test runner.
        self._interrupt = _thread.interrupt_main

    def start(self, on_twist, on_button=None, on_key=None):
        """on_twist(twist: gafropy.Twist) is called once per movement keypress
        with a fixed-size step; on_button(index) fires on 't' (0) / 'v' (1).
        on_key(key) is offered 't'/Enter first and returns truthy to claim
        them (startup align); 'q' raises KeyboardInterrupt in the main thread,
        the same shutdown path as Ctrl+C."""
        self._on_twist = on_twist
        self._on_button = on_button
        self._on_key = on_key
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            print(KEYBOARD_HELP)
            while self._running.is_set():
                # A short select timeout (not a blocking read) so stop() -- called from
                # another thread -- doesn't have to wait for one more keypress to notice
                # _running was cleared.
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                key = sys.stdin.read(1)

                try:
                    if not self._dispatch(key):
                        break
                except Exception:
                    # Without this, an exception here (e.g. a numerical issue in the
                    # controller) kills this background thread silently -- keypresses
                    # would then just stop doing anything, with no indication why, since
                    # the main viser loop keeps running regardless. Print and keep going
                    # rather than let one bad update take down keyboard input entirely.
                    print("KeyboardTeleop: on_twist/on_button raised, continuing:")
                    traceback.print_exc()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _dispatch(self, key):
        """Route one key. Returns False when the adapter should stop reading.

        Split out of ``_run`` so the key map is testable without a tty. Order
        matters: ``on_key`` gets first refusal on 't'/Enter and claims them by
        returning truthy, so startup align keeps 't' while it is still waiting
        and hands it back to button 0 afterwards.
        """
        if (self._on_key is not None
                and key in _KEY_PASSTHROUGH
                and self._on_key(key)):
            return True
        if key in _KEY_QUIT:
            # Advertised by KEYBOARD_HELP, so an operator will press it. Hand
            # the main thread the same KeyboardInterrupt Ctrl+C would, rather
            # than inventing a second shutdown path the examples do not know.
            self._running.clear()
            self._interrupt()
            return False
        if key in _KEY_BUTTON:
            if self._on_button is not None:
                self._on_button(_KEY_BUTTON[key])
            return True
        mapping = _KEY_TWIST.get(key)
        if mapping is not None:
            index, sign = mapping
            coeffs = np.zeros(6)
            coeffs[index] = sign * self._step
            self._on_twist(gafropy.Twist.from_array(coeffs))
        return True
