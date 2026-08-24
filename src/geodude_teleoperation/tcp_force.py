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


def connect_tcp_force_reader(robot_ip: str) -> tuple[Callable[[], Sequence[float]], Callable[[], None]]:
    """Return ``(read_tcp_force, close)`` for ``robot_ip``.

    ``read_tcp_force()`` -> length-6 sequence ``[Fx, Fy, Fz, Tx, Ty, Tz]`` (UR tool
    frame, Newtons / N·m), same order as ``getActualTCPForce()``.
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

    # Smoke-read once so a dead/contested RTDE socket fails at connect time.
    _ = read()
    return read, close


def fz_from_tcp_force(tcp_force: Sequence[float]) -> float:
    """Tool-frame linear Z from UR ``ActualTCPForce`` (index 2)."""
    return float(tcp_force[2])
