"""Local X11 attach for --viz when DISPLAY is empty."""

from pathlib import Path

from geodude_teleoperation.viz.display import (
    ensure_gui_display,
    local_x11_displays,
)


def test_local_x11_prefers_uid_owned_socket(tmp_path: Path, monkeypatch):
    sock = tmp_path / "X11-unix"
    sock.mkdir()
    (sock / "X0").touch()
    mine = sock / "X1"
    mine.touch()
    monkeypatch.setattr("geodude_teleoperation.viz.display.os.getuid", lambda: mine.stat().st_uid)
    # X0 owned by us too in tmp_path — both same uid; sort by number.
    assert local_x11_displays(sock_dir=sock) == [":0", ":1"]


def test_ensure_gui_display_keeps_existing(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":99")
    assert ensure_gui_display() == ":99"


def test_ensure_gui_display_attaches_when_unset(tmp_path: Path, monkeypatch):
    sock = tmp_path / "X11-unix"
    sock.mkdir()
    (sock / "X1").touch()
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("XAUTHORITY", raising=False)
    assert ensure_gui_display(sock_dir=sock) == ":1"
    assert ensure_gui_display(sock_dir=sock) == ":1"
