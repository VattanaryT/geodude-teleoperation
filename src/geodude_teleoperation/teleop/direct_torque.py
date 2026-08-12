"""Phase 2: Direct Torque Control (DTC) + 500 Hz verification.

UR Direct Torque Control (PolyScope ≥ 5.23 / PolyScope X ≥ 10.10) accepts
``direct_torque(tau)`` / ur_rtde ``directTorque(tau)`` at the robot's **500 Hz**
timestep. Gravity is compensated on-controller — do **not** include gravity in
``tau``. The call must land every robot timestep or the arm leaves torque mode.

This module:
- documents the bridge contract
- verifies host ``--control-hz`` vs 500 Hz
- optionally parses PolyScope version for readiness
- can **command** zero-torque DTC on a live UR and measure achieved rate
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

UR_SYSTEM_HZ = 500.0
DTC_MIN_POLYSCOPE = (5, 23, 0)
DTC_MIN_POLYSCOPE_X = (10, 10, 0)


@dataclass
class ControlRateReport:
    requested_hz: float
    target_hz: float = UR_SYSTEM_HZ
    ok: bool = False
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "requested_hz": self.requested_hz,
            "target_hz": self.target_hz,
            "ok": self.ok,
            "warnings": list(self.warnings),
        }


def verify_control_hz(
    requested_hz: float, *, target_hz: float = UR_SYSTEM_HZ, tol: float = 1.0
) -> ControlRateReport:
    """Check host / DTC loop rate against the UR system update frequency."""
    hz = float(requested_hz)
    warnings: list[str] = []
    if hz <= 0:
        warnings.append(f"control rate must be positive, got {hz}")
        return ControlRateReport(
            requested_hz=hz, target_hz=target_hz, ok=False, warnings=tuple(warnings)
        )
    if abs(hz - target_hz) > tol:
        warnings.append(
            f"requested {hz:g} Hz; UR Direct Torque / system update is {target_hz:g} Hz. "
            "For DTC, call ur_rtde directTorque() every robot timestep (~2 ms) — "
            "Phase 1/1.5 speedJ teleop may stay at 50 Hz."
        )
        ok = False
    else:
        ok = True
    if hz < 100.0:
        warnings.append(
            f"{hz:g} Hz is fine for Phase 1/1.5 speedJ teleop; DTC verification "
            "uses a separate 500 Hz directTorque loop on the control interface."
        )
    return ControlRateReport(
        requested_hz=hz, target_hz=target_hz, ok=ok, warnings=tuple(warnings)
    )


def parse_polyscope_version(version: str) -> tuple[int, ...] | None:
    """Parse ``'5.23.0'`` / ``'10.10.0'`` → tuple; None if unparseable."""
    if not version:
        return None
    cleaned = str(version).strip().lstrip("vV")
    parts: list[int] = []
    for token in cleaned.split("."):
        digits = ""
        for ch in token:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else None


def polyscope_supports_dtc(version: str | tuple[int, ...] | None) -> bool:
    """True if version meets DTC minimum (5.23+ or 10.10+)."""
    if version is None:
        return False
    ver = parse_polyscope_version(version) if isinstance(version, str) else tuple(version)
    if not ver:
        return False
    major = ver[0]
    if major >= 10:
        return ver >= DTC_MIN_POLYSCOPE_X[: len(ver)] + (0,) * max(
            0, len(DTC_MIN_POLYSCOPE_X) - len(ver)
        ) and ver >= DTC_MIN_POLYSCOPE_X
    padded = ver + (0,) * max(0, 3 - len(ver))
    return padded[:3] >= DTC_MIN_POLYSCOPE


def query_polyscope_version(robot_ip: str, timeout_s: float = 2.0) -> str | None:
    """Best-effort PolyScope version via DashboardClient (port 29999)."""
    _ = timeout_s
    try:
        from dashboard_client import DashboardClient
    except ImportError:
        return None
    client = None
    try:
        client = DashboardClient(str(robot_ip))
        connect = getattr(client, "connect", None)
        if callable(connect):
            connect()
        ver = client.polyscopeVersion()
        return str(ver).strip() if ver else None
    except Exception:
        return None
    finally:
        if client is not None:
            for name in ("disconnect", "close"):
                fn = getattr(client, name, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass


@dataclass
class DirectTorqueBridge:
    """Contract for DTC (Phase 2): latch τ for URScript / ur_rtde consumer.

    Prefer calling ``RTDEControlInterface.directTorque(tau)`` every 2 ms from a
    dedicated 500 Hz host loop (see ``verify_direct_torque_rate``). Gravity is
    compensated by the controller — do **not** include gravity in ``tau``.
    """

    enabled: bool = False
    friction_comp: bool = True
    last_tau: np.ndarray | None = None

    def publish_torques(self, tau: Sequence[float]) -> np.ndarray:
        """Validate and latch torques for the external URScript / RTDE consumer."""
        arr = np.asarray(tau, dtype=float).reshape(6)
        if not np.isfinite(arr).all():
            raise ValueError("direct_torque command contains non-finite values")
        self.last_tau = arr.copy()
        return self.last_tau

    def urscript_snippet(self) -> str:
        return (
            "# Phase 2 DTC skeleton — run on robot (500 Hz) OR use ur_rtde:\n"
            "#   ctrl.directTorque(tau, friction_comp=True) every timestep\n"
            "# Stream target_tau[0:6] via RTDE input registers / socket if URScript.\n"
            "while (True):\n"
            "  tau = get_target_tau_from_host()  # replace with RTDE read\n"
            f"  direct_torque(tau, friction_comp={'True' if self.friction_comp else 'False'})\n"
            "  sync()\n"
        )


@dataclass
class DtcRateVerificationResult:
    """Outcome of commanding ``directTorque`` on a live UR and measuring rate."""

    commanded: bool
    target_hz: float = UR_SYSTEM_HZ
    duration_s: float = 0.0
    n_calls: int = 0
    measured_hz: float = float("nan")
    period_mean_ms: float = float("nan")
    period_p95_ms: float = float("nan")
    period_max_ms: float = float("nan")
    ok: bool = False
    polyscope_version: str | None = None
    polyscope_ok: bool | None = None
    messages: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "commanded": self.commanded,
            "target_hz": self.target_hz,
            "duration_s": self.duration_s,
            "n_calls": self.n_calls,
            "measured_hz": self.measured_hz,
            "period_mean_ms": self.period_mean_ms,
            "period_p95_ms": self.period_p95_ms,
            "period_max_ms": self.period_max_ms,
            "ok": self.ok,
            "polyscope_version": self.polyscope_version,
            "polyscope_ok": self.polyscope_ok,
            "messages": list(self.messages),
            "errors": list(self.errors),
        }


def verify_direct_torque_rate(
    control: Any,
    *,
    duration_s: float = 2.0,
    target_hz: float = UR_SYSTEM_HZ,
    friction_comp: bool = True,
    hz_tol: float = 25.0,
    tau: Sequence[float] | None = None,
) -> DtcRateVerificationResult:
    """Command DTC on ``control`` (ur_rtde RTDEControlInterface) and measure rate.

    Sends ``directTorque(tau)`` (default zeros — gravity compensated hold) for
    ``duration_s``, using ``initPeriod``/``waitPeriod`` when available so the
    host tracks the robot's 500 Hz clock. Ends by leaving torque mode via
    ``speedStop`` / ``stopJ`` when present.
    """
    messages: list[str] = []
    errors: list[str] = []
    dt_cmd = 1.0 / float(target_hz)
    duration_s = float(max(duration_s, dt_cmd))
    tau6 = (
        np.zeros(6, dtype=float)
        if tau is None
        else np.asarray(tau, dtype=float).reshape(6)
    )
    if not np.isfinite(tau6).all():
        return DtcRateVerificationResult(
            commanded=False,
            duration_s=duration_s,
            errors=("tau contains non-finite values",),
        )

    direct = getattr(control, "directTorque", None)
    if not callable(direct):
        return DtcRateVerificationResult(
            commanded=False,
            duration_s=duration_s,
            errors=(
                "control interface has no directTorque(); "
                "upgrade ur_rtde / PolyScope for DTC",
            ),
        )

    init_period = getattr(control, "initPeriod", None)
    wait_period = getattr(control, "waitPeriod", None)
    use_period = callable(init_period) and callable(wait_period)

    periods: list[float] = []
    n_calls = 0
    t_end = time.perf_counter() + duration_s
    t_prev = time.perf_counter()
    try:
        while time.perf_counter() < t_end:
            t0 = time.perf_counter()
            t_mark = init_period() if use_period else None
            ok_call = direct(tau6.tolist(), bool(friction_comp))
            if ok_call is False:
                errors.append("directTorque returned False")
                break
            if use_period:
                wait_period(t_mark)
            else:
                remain = dt_cmd - (time.perf_counter() - t0)
                if remain > 0:
                    time.sleep(remain)
            t_now = time.perf_counter()
            periods.append(t_now - t_prev)
            t_prev = t_now
            n_calls += 1
    except Exception as exc:
        errors.append(f"directTorque loop failed: {exc}")
    finally:
        for name, args in (
            ("speedStop", ()),
            ("stopJ", ([0.2] * 6,)),
        ):
            fn = getattr(control, name, None)
            if callable(fn):
                try:
                    fn(*args)
                    messages.append(f"exited torque mode via {name}()")
                    break
                except Exception as exc:
                    errors.append(f"{name}() failed: {exc}")

    if n_calls < 2 or not periods:
        return DtcRateVerificationResult(
            commanded=n_calls > 0,
            duration_s=duration_s,
            n_calls=n_calls,
            ok=False,
            messages=tuple(messages),
            errors=tuple(errors) if errors else ("too few DTC cycles measured",),
        )

    arr = np.asarray(periods[1:], dtype=float)
    mean_p = float(np.mean(arr))
    p95 = float(np.percentile(arr, 95))
    max_p = float(np.max(arr))
    measured_hz = (1.0 / mean_p) if mean_p > 1e-9 else float("nan")
    ok = (
        not errors
        and np.isfinite(measured_hz)
        and abs(measured_hz - float(target_hz)) <= float(hz_tol)
    )
    if ok:
        messages.append(
            f"measured {measured_hz:.1f} Hz over {n_calls} directTorque calls "
            f"(target {target_hz:g} ±{hz_tol:g})"
        )
    else:
        messages.append(
            f"measured {measured_hz:.1f} Hz (target {target_hz:g} ±{hz_tol:g}) — "
            "check RT priority, network, External Control / remote mode"
        )
    return DtcRateVerificationResult(
        commanded=True,
        target_hz=float(target_hz),
        duration_s=duration_s,
        n_calls=n_calls,
        measured_hz=measured_hz,
        period_mean_ms=mean_p * 1e3,
        period_p95_ms=p95 * 1e3,
        period_max_ms=max_p * 1e3,
        ok=ok,
        messages=tuple(messages),
        errors=tuple(errors),
    )


def print_control_rate_report(report: ControlRateReport, *, prefix: str = "dtc") -> None:
    status = "OK" if report.ok else "WARN"
    print(
        f"[{prefix}] control_hz={report.requested_hz:g} "
        f"target={report.target_hz:g} [{status}]"
    )
    for w in report.warnings:
        print(f"[{prefix}] {w}")


def print_dtc_rate_verification(
    result: DtcRateVerificationResult, *, prefix: str = "dtc-live"
) -> None:
    status = "OK" if result.ok else ("SKIP" if not result.commanded else "FAIL")
    print(
        f"[{prefix}] commanded={int(result.commanded)} calls={result.n_calls} "
        f"measured_hz={result.measured_hz:.2f} "
        f"period_ms mean={result.period_mean_ms:.3f} "
        f"p95={result.period_p95_ms:.3f} max={result.period_max_ms:.3f} [{status}]",
        flush=True,
    )
    if result.polyscope_version is not None:
        ps = (
            "supports DTC"
            if result.polyscope_ok
            else "DOES NOT support DTC (need ≥5.23 or ≥10.10)"
        )
        print(f"[{prefix}] PolyScope {result.polyscope_version}: {ps}", flush=True)
    for m in result.messages:
        print(f"[{prefix}] {m}", flush=True)
    for e in result.errors:
        print(f"[{prefix}] ERROR: {e}", flush=True)
