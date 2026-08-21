"""Attach --viz to a local X11 session when Cursor/SSH left DISPLAY empty."""
from __future__ import annotations

import os
from pathlib import Path

X11_UNIX_DIR = Path("/tmp/.X11-unix")


def local_x11_displays(*, sock_dir: Path | None = None) -> list[str]:
    """``:N`` names for sockets in ``/tmp/.X11-unix``, uid-owned first."""
    d = X11_UNIX_DIR if sock_dir is None else Path(sock_dir)
    if not d.is_dir():
        return []
    uid = os.getuid()
    owned: list[tuple[int, str]] = []
    other: list[tuple[int, str]] = []
    for p in d.iterdir():
        name = p.name
        if not (name.startswith("X") and name[1:].isdigit()):
            continue
        n = int(name[1:])
        disp = f":{n}"
        try:
            st = p.stat()
        except OSError:
            continue
        (owned if st.st_uid == uid else other).append((n, disp))
    owned.sort()
    other.sort()
    return [disp for _, disp in owned] + [disp for _, disp in other]


def ensure_gui_display(*, sock_dir: Path | None = None) -> str | None:
    """Keep an existing DISPLAY/Wayland; else set DISPLAY to a local X11 socket.

    Returns the display string to use, or None if none is available.
    """
    existing = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if existing:
        return existing
    displays = local_x11_displays(sock_dir=sock_dir)
    if not displays:
        return None
    chosen = displays[0]
    os.environ["DISPLAY"] = chosen
    if not os.environ.get("XAUTHORITY"):
        uid = os.getuid()
        for cand in (
            Path(f"/run/user/{uid}/gdm/Xauthority"),
            Path.home() / ".Xauthority",
        ):
            if cand.is_file():
                os.environ["XAUTHORITY"] = str(cand)
                break
    return chosen


def viz_missing_display_message() -> str:
    found = ", ".join(local_x11_displays()) or "none"
    return (
        "error: --viz needs a graphical display (DISPLAY is unset).\n"
        f"  Local X11 sockets: {found}\n"
        "  On this machine's desktop: export DISPLAY=:1\n"
        "  Remote: ssh -X …   or omit --viz for a terminal-only run."
    )
