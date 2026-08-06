"""Thin RTDE wrapper for TCP / payload / FT apply and wrench sampling."""

from __future__ import annotations

import time
from typing import Any, Optional

from geodude_teleoperation.calibration.io import validate_payload, validate_tcp_pose, validate_wrench


def _import_rtde():
    try:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
    except ImportError as exc:
        raise ImportError(
            "ur_rtde is required for robot apply / FT sampling; "
            "install with: pip install 'geodude-teleoperation[ur]'"
        ) from exc
    return RTDEControlInterface, RTDEReceiveInterface


class RtdeSession:
    """Context-managed RTDE control + receive pair for one robot IP."""

    def __init__(self, ip: str, control: Any = None, receive: Any = None):
        self.ip = str(ip)
        self._control = control
        self._receive = receive
        self._owns = control is None and receive is None

    def __enter__(self) -> "RtdeSession":
        if self._control is None or self._receive is None:
            RTDEControlInterface, RTDEReceiveInterface = _import_rtde()
            self._control = RTDEControlInterface(self.ip)
            self._receive = RTDEReceiveInterface(self.ip)
            self._owns = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if not self._owns:
            return
        for iface in (self._control, self._receive):
            if iface is None:
                continue
            disconnect = getattr(iface, "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect()
                except Exception:
                    pass
        self._control = None
        self._receive = None

    def set_tcp(self, pose: list[float]) -> None:
        pose = validate_tcp_pose(pose)
        self._require_control().setTcp(pose)

    def set_payload(self, mass_kg: float, cog: list[float]) -> None:
        mass_kg, cog = validate_payload(mass_kg, cog)
        self._require_control().setPayload(mass_kg, cog)

    def zero_ft_sensor(self) -> None:
        self._require_control().zeroFtSensor()

    def average_tcp_force(self, samples: int, settle_s: float = 0.01) -> list[float]:
        if samples <= 0:
            raise ValueError(f"samples must be > 0, got {samples}")
        receive = self._require_receive()
        acc = [0.0] * 6
        for _ in range(samples):
            wrench = validate_wrench(list(receive.getActualTCPForce()))
            for i in range(6):
                acc[i] += wrench[i]
            if settle_s > 0:
                time.sleep(settle_s)
        return [v / samples for v in acc]

    def _require_control(self) -> Any:
        if self._control is None:
            raise RuntimeError("RtdeSession is not connected; use as a context manager")
        return self._control

    def _require_receive(self) -> Any:
        if self._receive is None:
            raise RuntimeError("RtdeSession is not connected; use as a context manager")
        return self._receive
