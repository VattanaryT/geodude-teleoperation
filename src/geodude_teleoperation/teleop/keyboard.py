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
  Blade order [e12,e13,e23, e1i,e2i,e3i] = [angular, linear]
  w/s: +x/-x    a/d: -y/+y    y/x: -z/+z   (linear)
  n/m: e12 +/-  j/k: e13 +/-  u/i: e23 +/-  (angular)
  With VF on: twist axes are field / R_field (not TCP body); each keypress
  re-locks orientation to R_field. Idle (no keys) holds current TCP pose.
  Enter: enable motion (calibrate → teleop); live RTDE also locks
  TCP orientation to R_field with a startup vel/accel ramp.
  Live: Enter again after lock settles to approve teleop.
  t: toggle teleoperation    v: toggle virtual fixture
  f: hold pose + freeze VF; again = resume + unlock fields (or n* in-contact)
  q: quit

With --viz (live Polyscope): Enter / w/s/a/d/x/y / t/v/f / q also work in the
  Polyscope window (stdin is unused while the GUI has focus). Object offset
  remains j/l i/k u/o in-window; angular teleop n/m from the terminal only.
  --dry-run --viz: after Enter, tip moves in Polyscope only (sim); VF still
  applies unless you press v. Robot is never commanded.

In-contact mode (--in-contact):
  Enter: Align tool-Z → n* first, then Force seeks/holds Fd along n*.
  w/s a/d: tangent slide in VF local frame (R_motion x/y), not tool body.
  y/x and angular ignored.  f: freeze/unfreeze n*    t: pause    q: quit
"""


def dispatch_key(key: str, *, step, enable_key="\n") -> tuple[str, object]:
    """Return (kind, payload) where kind in twist|button|enable|quit|ignore."""
    if key == enable_key:
        return ("enable", None)
    if key == "t":
        return ("button", 0)
    if key == "v":
        return ("button", 1)
    if key == "f":
        return ("button", 2)
    if key == "q":
        return ("quit", None)
    mapping = _KEY_TWIST.get(key)
    if mapping is None:
        return ("ignore", None)
    index, sign = mapping
    coeffs = np.zeros(6)
    coeffs[index] = sign * step
    return ("twist", coeffs)


def is_contact_tangent_key(key: str) -> bool:
    """True for field-tangent linear keys used in --in-contact mode."""
    return key in ("w", "s", "a", "d")


def contact_field_linear_from_twist_coeffs(coeffs) -> np.ndarray:
    """Extract field-frame linear [lx, ly, lz] from blade-order twist array."""
    arr = np.asarray(coeffs, dtype=float).reshape(6)
    return arr[3:6].copy()


def make_twist(payload):
    """Build gafropy.Twist when usable; otherwise NumpyTwist (same blade order)."""
    from geodude_teleoperation.gafropy_compat import gafropy_importable

    if gafropy_importable():
        import gafropy

        return gafropy.Twist.from_array(payload)
    from geodude_teleoperation.se3 import NumpyTwist

    return NumpyTwist.from_array(payload)


class KeyboardTeleop:
    def __init__(self, step=0.01):
        self._step = step
        self._thread = None
        self._running = threading.Event()
        self._on_twist = None
        self._on_button = None
        self._on_enable = None
        self._enable_key = "\n"
        self._lock = threading.Lock()
        self.last_key = ""
        self.last_twist_coeffs: np.ndarray | None = None

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

    def handle_key(self, key: str) -> None:
        """Process one key from stdin or Polyscope inject. Thread-safe."""
        if not key or not self._running.is_set():
            return
        try:
            with self._lock:
                kind, payload = dispatch_key(
                    key, step=self._step, enable_key=self._enable_key
                )
                on_enable = self._on_enable
                on_button = self._on_button
                on_twist = self._on_twist
                if kind == "twist":
                    self.last_key = key
                    self.last_twist_coeffs = np.asarray(payload, dtype=float).copy()
                    twist = make_twist(payload)
                else:
                    twist = None
            if kind == "ignore":
                return
            if kind == "quit":
                self._running.clear()
                return
            if kind == "enable":
                if on_enable is not None:
                    on_enable()
                return
            if kind == "button":
                if on_button is not None:
                    on_button(payload)
                return
            if kind == "twist" and on_twist is not None and twist is not None:
                on_twist(twist)
        except Exception:
            # Without this, an exception here (e.g. a numerical issue in the
            # controller) kills this background thread silently -- keypresses
            # would then just stop doing anything, with no indication why, since
            # the main viser loop keeps running regardless. Print and keep going
            # rather than let one bad update take down keyboard input entirely.
            print("KeyboardTeleop: on_twist/on_button raised, continuing:")
            traceback.print_exc()

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
                self.handle_key(key)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
