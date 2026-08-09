"""Keyboard input adapter: alternative to SpaceMouseTeleop for when no physical
device is available. Ports the key bindings from ufactory_demos'
gafro_teleoperation/TeleoperationKeyboard.cpp (WASD/XY/UIJK/NM for the six twist
axes, terminal raw mode via termios, one fixed-size step per keypress -- holding
a key relies on the terminal's own key-repeat for continuous motion, exactly as
the original did) -- plus 't'/'v' toggle keys, since that original device
predates the teleop/virtual-fixture enable toggles (only its 3D-mouse sibling had
buttons; this class's counterpart, SpaceMouseTeleop, needs both).
"""

import select
import sys
import termios
import threading
import traceback
import tty

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

KEYBOARD_HELP = """\
Keyboard teleoperation:
  w/s: +x/-x    a/d: -y/+y    y/x: -z/+z   (linear)
  n/m: e12 +/-  j/k: e13 +/-  u/i: e23 +/-  (angular)
  Enter: enable motion (calibrate → teleop)
  t: toggle teleoperation    v: toggle virtual fixture    q: quit
"""


def dispatch_key(key: str, *, step, enable_key="\n") -> tuple[str, object]:
    """Return (kind, payload) where kind in twist|button|enable|quit|ignore."""
    if key == enable_key:
        return ("enable", None)
    if key == "t":
        return ("button", 0)
    if key == "v":
        return ("button", 1)
    if key == "q":
        return ("quit", None)
    mapping = _KEY_TWIST.get(key)
    if mapping is None:
        return ("ignore", None)
    index, sign = mapping
    coeffs = np.zeros(6)
    coeffs[index] = sign * step
    return ("twist", coeffs)


class KeyboardTeleop:
    def __init__(self, step=0.01):
        self._step = step
        self._thread = None
        self._running = threading.Event()
        self._on_twist = None
        self._on_button = None
        self._on_enable = None
        self._enable_key = "\n"

    def start(self, on_twist, on_button=None, on_enable=None, enable_key="\n"):
        """on_twist(twist: gafropy.Twist) is called once per movement keypress
        with a fixed-size step; on_button(index) fires on 't' (0) / 'v' (1);
        on_enable() fires once per enable_key press (default Enter)."""
        self._on_twist = on_twist
        self._on_button = on_button
        self._on_enable = on_enable
        self._enable_key = enable_key
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self):
        import gafropy

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
                    kind, payload = dispatch_key(
                        key, step=self._step, enable_key=self._enable_key
                    )
                    if kind == "ignore":
                        continue
                    if kind == "quit":
                        self._running.clear()
                        continue
                    if kind == "enable":
                        if self._on_enable is not None:
                            self._on_enable()
                        continue
                    if kind == "button":
                        if self._on_button is not None:
                            self._on_button(payload)
                        continue
                    if kind == "twist":
                        self._on_twist(gafropy.Twist.from_array(payload))
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
