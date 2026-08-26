"""SpaceMouse input adapter: polls a 3Dconnexion device on a background thread and
turns each reading into a body-frame gafropy.Twist, mirroring the independent
device-poll-thread design of the original ufactory_demos teleoperation node (there:
a dedicated ROS2 node at the device's own rate; here: a plain Python thread).
"""

import pathlib
import threading
import time

import gafropy
import numpy as np

_BY_ID_GLOB = "/dev/input/by-id/usb-3Dconnexion_*"


def list_spacemouse_paths() -> dict[str, str]:
    """Return ``{hidraw_path: device_name}`` for connected SpaceMice."""
    import pyspacemouse

    return dict(pyspacemouse.get_connected_devices_by_path())


def resolve_spacemouse_path(
    preferred: str | None = None,
    *,
    device_index: int | None = None,
) -> str:
    """Pick a hidraw path when multiple SpaceMice are connected.

    Resolution order:
    1. Explicit ``preferred`` path (must exist in ``list_spacemouse_paths()``).
    2. ``device_index`` into the sorted path list.
    3. Hidraw node matching a ``/dev/input/by-id/usb-3Dconnexion_*`` symlink.
    4. Newest hidraw node by mtime (most recently plugged/unplugged).
    """
    paths = list_spacemouse_paths()
    if not paths:
        raise RuntimeError(
            "no SpaceMouse hidraw device found — check USB cable and "
            "'ls -l /dev/input/by-id/usb-3Dconnexion_*'"
        )

    sorted_paths = sorted(paths)
    if preferred is not None:
        preferred = str(pathlib.Path(preferred).resolve())
        if preferred not in paths:
            known = ", ".join(sorted_paths)
            raise ValueError(
                f"SpaceMouse path {preferred!r} not connected "
                f"(available: {known or 'none'})"
            )
        return preferred

    if device_index is not None:
        if device_index < 0 or device_index >= len(sorted_paths):
            raise ValueError(
                f"device_index {device_index} out of range "
                f"(0..{len(sorted_paths) - 1}, {len(sorted_paths)} connected)"
            )
        return sorted_paths[device_index]

    by_id = sorted(pathlib.Path(p) for p in pathlib.Path("/dev/input/by-id").glob(
        "usb-3Dconnexion_*"
    ) if not p.name.endswith("-mouse"))
    for link in by_id:
        event = link.resolve()
        hidraw = _hidraw_for_input_event(event)
        if hidraw is not None and hidraw in paths:
            return hidraw

    return max(sorted_paths, key=lambda p: pathlib.Path(p).stat().st_mtime)


def _hidraw_for_input_event(event_path: pathlib.Path) -> str | None:
    """Map ``/dev/input/eventN`` to its ``/dev/hidrawM`` sibling, if any."""
    sys_name = event_path.name  # event19
    for hidraw in sorted(pathlib.Path("/sys/class/hidraw").glob("hidraw*")):
        for event_node in hidraw.glob("device/input/input*/event*"):
            if event_node.name == sys_name:
                return f"/dev/{hidraw.name}"
    return None


def verify_spacemouse_connection(
    path: str,
    *,
    wait_s: float = 2.0,
    poll_hz: float = 50.0,
    move_threshold: float = 0.02,
) -> tuple[bool, float]:
    """Poll ``path`` briefly; return ``(saw_motion, peak_abs_axis)``."""
    import pyspacemouse

    peak = 0.0
    period = 1.0 / poll_hz
    deadline = time.monotonic() + max(wait_s, period)
    with pyspacemouse.open_by_path(path) as device:
        while True:
            state = device.read()
            peak = max(
                peak,
                abs(state.x),
                abs(state.y),
                abs(state.z),
                abs(state.roll),
                abs(state.pitch),
                abs(state.yaw),
            )
            if peak >= move_threshold:
                return True, peak
            if time.monotonic() >= deadline:
                break
            time.sleep(period)
    return False, peak


def format_spacemouse_paths() -> str:
    paths = list_spacemouse_paths()
    if not paths:
        return "none"
    parts = []
    for path in sorted(paths):
        age = time.strftime(
            "%Y-%m-%d %H:%M",
            time.localtime(pathlib.Path(path).stat().st_mtime),
        )
        parts.append(f"{path} ({paths[path]}, mtime {age})")
    return "; ".join(parts)


def is_idle_twist(twist, linear_scale=0.01, angular_scale=0.1, deadzone=0.05,
                  rest_margin=4.0) -> bool:
    """True when the reading is hands-off, including typical rest bias.

    SpaceMouseTeleop already zeros raw axes below `deadzone`, then scales. Rest
    bias just above that deadzone still becomes a nonzero Twist (default
    angular_scale 0.1 turns 0.1 raw yaw into 0.01). Callers that replace
    orientation on every update_command (approach-mode virtual fixtures) must
    treat that as idle, or the wrist servos with no operator input.

    `rest_margin * deadzone * scale` is the per-axis idle cap, using the same
    defaults as SpaceMouseTeleop. Linear and angular are separate because the
    two scales differ by 10x: a shared atol large enough to swallow rest yaw
    would also swallow a full-deflection translation.
    """
    values = np.asarray(twist.to_array(), dtype=float)
    angular = values[:3]
    linear = values[3:]
    return bool(
        np.all(np.abs(linear) <= rest_margin * deadzone * linear_scale)
        and np.all(np.abs(angular) <= rest_margin * deadzone * angular_scale)
    )


def rising_button_indices(previous, buttons):
    """Rising edges in ``buttons``. ``previous is None`` → no fire (held at start).

    Call on every HID report, including backlog drain. Waiting until after drain
    drops a click that is pressed and released in the queued reports.
    """
    cur = tuple(buttons)
    if previous is None:
        return cur, ()
    prev = tuple(previous)
    n = min(len(prev), len(cur))
    return cur, tuple(i for i in range(n) if cur[i] and not prev[i])


class SpaceMouseTeleop:
    def __init__(
        self,
        linear_scale=0.01,
        angular_scale=0.1,
        deadzone=0.05,
        frequency=50.0,
        *,
        device_path: str | None = None,
        device_index: int | None = None,
        verify_connection: bool = True,
    ):
        self._linear_scale = linear_scale
        self._angular_scale = angular_scale
        self._deadzone = deadzone
        self._period = 1.0 / frequency
        self._device_path = device_path
        self._device_index = device_index
        self._verify_connection = verify_connection
        self._thread = None
        self._running = threading.Event()
        self._on_twist = None
        self._on_button = None
        self._opened_path: str | None = None

    @property
    def device_path(self) -> str | None:
        return self._opened_path

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

        all_paths = list_spacemouse_paths()
        path = resolve_spacemouse_path(
            self._device_path, device_index=self._device_index)
        self._opened_path = path
        if len(all_paths) > 1:
            print(
                "spacemouse: multiple devices — "
                f"using {path}; others: "
                f"{', '.join(p for p in sorted(all_paths) if p != path)}. "
                "Pass --spacemouse-path or --spacemouse-index to override.",
                flush=True,
            )
        else:
            print(f"spacemouse: opened {path} ({all_paths[path]})", flush=True)

        if self._verify_connection:
            saw_motion, peak = verify_spacemouse_connection(path, wait_s=0.0)
            if not saw_motion:
                print(
                    "spacemouse: no axis motion yet "
                    f"(peak={peak:.3f}); move the cap now. "
                    f"Connected: {format_spacemouse_paths()}",
                    flush=True,
                )

        open_device = pyspacemouse.open_by_path(path)
        with open_device as device:
            previous_buttons = None

            while self._running.is_set():
                try:
                    state = device.read()
                    pressed = []
                    previous_buttons, edges = rising_button_indices(
                        previous_buttons, state.buttons)
                    pressed.extend(edges)

                    # Drain any backlog: keep reading while the HID queue still has new reports
                    # waiting (state.t only advances inside pyspacemouse's _process(), i.e. only
                    # when read() actually consumed new data -- not just returned the cached
                    # state), so on_twist always acts on the freshest report rather than a
                    # queued-but-stale one. Without this, if the device emits reports faster than
                    # this loop polls, a backlog can build up and playback lags behind real-time --
                    # visible as the controlled object still moving for a while after the user
                    # releases the device. Bounded (not `while True`) so a device streaming faster
                    # than we can ever fully drain can't starve this loop of reaching on_twist/sleep.
                    # Button edges are taken per report so a short click in the backlog is not lost.
                    for _ in range(64):
                        previous_t = state.t
                        state = device.read()
                        if state.t == previous_t:
                            break
                        previous_buttons, edges = rising_button_indices(
                            previous_buttons, state.buttons)
                        pressed.extend(edges)

                    axes = np.array([-state.x, state.y, -state.z, state.yaw, -state.roll, state.pitch])
                    axes = np.where(np.abs(axes) < self._deadzone, 0.0, axes)

                    linear = axes[:3] * self._linear_scale
                    angular = axes[3:] * self._angular_scale
                    # Twist blade order is [e12, e13, e23, e1i, e2i, e3i] = [angular, linear].
                    twist = gafropy.Twist.from_array(np.concatenate([angular, linear]))
                    self._on_twist(twist)

                    if self._on_button is not None:
                        for index in pressed:
                            self._on_button(index)

                    time.sleep(self._period)
                except Exception as exc:
                    print(f"spacemouse: read failed ({exc}); retrying", flush=True)
                    time.sleep(self._period)
