"""Read UR ``ActualTCPForce`` for contact force feedback.

gafropy's ``URRobot::sampleState`` currently copies position/velocity/torque only —
it never calls ``SystemState::setWrench``, even though the RTDE recipe includes
``actual_TCP_force``. ``interface.get_state().get_wrench()`` is therefore always
zero on hardware, and any force PID that trusts it presses open-loop.

This helper opens a receive-only ``RTDEReceiveInterface``. On many controllers only
one RTDE client is allowed (gafropy's urcl already holds it), so ``connect`` may
fail — callers must treat that as "no force feedback" and refuse open-loop press.
"""
from __future__ import annotations

from typing import Callable, Sequence


# UR safety modes. 1/2 are the only ones where it is safe to keep streaming
# motion; everything else means the controller has stopped the robot and our
# commands are being discarded until someone clears it on the pendant.
SAFETY_MODE_NORMAL = 1
SAFETY_MODE_REDUCED = 2
SAFETY_MODE_NAMES = {
    1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY",
    5: "SAFEGUARD_STOP", 6: "SYSTEM_EMERGENCY_STOP", 7: "ROBOT_EMERGENCY_STOP",
    8: "VIOLATION", 9: "FAULT",
}


def safety_mode_name(mode) -> str:
    try:
        mode = int(mode)
    except (TypeError, ValueError):
        return str(mode)
    return SAFETY_MODE_NAMES.get(mode, f"mode {mode}")


def is_safe_to_command(mode) -> bool:
    """True only in NORMAL/REDUCED. Unknown (None) reads as safe: no reader is
    not evidence of a stop, and refusing to press on a missing signal would be
    a worse failure than the one this guards against."""
    if mode is None:
        return True
    try:
        return int(mode) in (SAFETY_MODE_NORMAL, SAFETY_MODE_REDUCED)
    except (TypeError, ValueError):
        return True


def connect_tcp_force_reader(robot_ip: str):
    """Return ``(read_tcp_force, close)`` for ``robot_ip``.

    ``read_tcp_force()`` -> length-6 sequence ``[Fx, Fy, Fz, Tx, Ty, Tz]`` (UR tool
    frame, Newtons / N·m), same order as ``getActualTCPForce()``.

    ``read_tcp_force.safety_mode()`` is attached to the same receive interface
    and returns the UR safety mode, or None if this firmware does not expose
    it. Reusing the one socket matters: most controllers allow a single RTDE
    client and gafropy's urcl already holds one.
    """
    try:
        from rtde_receive import RTDEReceiveInterface
    except ImportError as exc:
        raise ImportError(
            "rtde_receive is required for ActualTCPForce feedback "
            "(pip install ur-rtde)"
        ) from exc

    receive = RTDEReceiveInterface(str(robot_ip))

    def read() -> Sequence[float]:
        return list(receive.getActualTCPForce())

    def close() -> None:
        disconnect = getattr(receive, "disconnect", None)
        if callable(disconnect):
            try:
                disconnect()
            except Exception:
                pass

    def safety_mode():
        getter = getattr(receive, "getSafetyMode", None)
        if not callable(getter):
            return None
        try:
            return int(getter())
        except Exception:
            return None

    read.safety_mode = safety_mode
    # The live interface itself, for callers that need more than force and
    # safety mode off this one socket -- DTC reads getActualQ/getActualQd here
    # rather than opening a *second* client on 30004. Only one RTDE client is
    # allowed on most controllers, so sharing this is not an optimization.
    read.receive = receive

    # Smoke-read once so a dead/contested RTDE socket fails at connect time.
    _ = read()
    return read, close


def fz_from_tcp_force(tcp_force: Sequence[float]) -> float:
    """Tool-frame linear Z from UR ``ActualTCPForce`` (index 2)."""
    return float(tcp_force[2])


def force_from_tcp_force(tcp_force: Sequence[float]) -> tuple[float, float, float]:
    """Tool-frame linear force ``(Fx, Fy, Fz)`` from UR ``ActualTCPForce``.

    UR's order is ``[Fx, Fy, Fz, Rx, Ry, Rz]``, so the linear part is just the
    first three. The press loop needs all of it once the press axis is a field
    normal rather than tool Z; ``fz_from_tcp_force`` stays for the paths that
    only ever look along tool Z.
    """
    return (float(tcp_force[0]), float(tcp_force[1]), float(tcp_force[2]))
