"""UR Direct Torque Control (DTC) bridge + 500 Hz verification.

Ported from ``exp/in-contact-dtc``. UR DTC accepts
``direct_torque(tau)`` / ur_rtde ``directTorque(tau)`` at the robot's **500 Hz**
timestep. Gravity is compensated on-controller — do **not** include gravity in
``tau``. The call must land every robot timestep or the arm leaves torque mode.

Stock ur_rtde applies **one** ``direct_torque`` per host ``directTorque()``
(unlike ``servoJ``, which has an on-robot latch thread). Host RTDE rarely
sustains 500 Hz, so this module patches the control script to latch τ on-robot
(same pattern as ``servo_thread``). Host then publishes τ at ``control.hz``.

This module:
- documents the bridge contract
- patches / uploads the DTC-latch control script
- verifies host ``--control-hz`` vs 500 Hz
- can **command** zero-torque DTC on a live UR and measure achieved rate
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

UR_SYSTEM_HZ = 500.0
# ur_rtde directTorque friction scale defaults (PolyScope DTC).
DEFAULT_VISCOUS_SCALE = (0.9, 0.9, 0.8, 0.9, 0.9, 0.9)
DEFAULT_COULOMB_SCALE = (0.8, 0.8, 0.7, 0.8, 0.8, 0.8)

# Marker embedded in patched URScript (idempotency + diagnostics).
DTC_LATCH_MARKER = "# geodude DTC latch (on-robot 500 Hz thread)"

# How often the publisher may repeat the stale-tau warning. Long enough that
# a startup hiccup stays a single line, short enough that a loop which cannot
# hold its period says so while you are still watching.
STALE_REPORT_WINDOW_S = 5.0

# Module-scope defaults read these, so they are library data and ship in the
# wheel (pyproject packages ``data/**/*``). assets/ belongs to examples/ and is
# not packaged, which is where these lived on exp/in-contact-dtc.
_URSCRIPT_DIR = Path(__file__).resolve().parent / "data" / "urscript"
DEFAULT_STOCK_RTDE_SCRIPT = _URSCRIPT_DIR / "rtde_control.script"
DEFAULT_DTC_LATCH_SCRIPT = _URSCRIPT_DIR / "rtde_control_dtc_latch.script"

_DTC_GLOBALS = """\
    global dtc_target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    global dtc_friction_comp = True
    global dtc_thrd = 0
    global is_dtc = 0
"""

_DTC_THREAD = f"""\
    {DTC_LATCH_MARKER}
    thread dtc_thread():
        while (True):
            enter_critical
            tau = dtc_target
            fc = dtc_friction_comp
            exit_critical
$5.23         direct_torque(tau, fc)
        end
    end
"""

_DTC_CMD66 = """\
        elif cmd == 66:
              # direct_torque — latched on-robot @ 500 Hz (geodude DTC latch)
              dummy_op = 1
$5.23         torque = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
$5.23         torque = q_from_input_float_registers(0)
$5.23         friction_comp = True
$5.23         friction_comp_val = read_input_integer_reg(1)
$5.23         if friction_comp_val == 1:
$5.23             friction_comp = True
$5.23         elif friction_comp_val == 0:
$5.23             friction_comp = False
$5.23         end
$5.23         enter_critical
$5.23         dtc_target = torque
$5.23         dtc_friction_comp = friction_comp
$5.23         exit_critical
$5.23         if is_dtc == 0:
$5.23             is_dtc = 1
$5.23             if dtc_thrd == 0:
$5.23                 global dtc_thrd = run dtc_thread()
$5.23             end
$5.23         end
"""


def patch_rtde_control_script_for_dtc_latch(source: str) -> str:
    """Patch stock ur_rtde ``rtde_control.script`` so cmd 66 latches like servoJ.

    Host ``directTorque`` then only updates ``dtc_target``; ``dtc_thread`` calls
    ``direct_torque`` every robot timestep (~500 Hz).
    """
    if DTC_LATCH_MARKER in source:
        return source
    text = source

    needle_globals = "    global speed_thrd = 0\n"
    if needle_globals not in text:
        raise ValueError("stock script missing global speed_thrd (unexpected layout)")
    if "global dtc_thrd" not in text:
        text = text.replace(
            needle_globals, needle_globals + _DTC_GLOBALS, 1
        )

    # Insert thread after servo_thread() definition.
    servo_end = (
        "            servoj(q, t=dt, lookahead_time=lh_time, gain=g)\n"
        "        end\n"
        "    end\n"
    )
    if "thread dtc_thread():" not in text:
        if servo_end not in text:
            raise ValueError("stock script missing servo_thread body (unexpected layout)")
        text = text.replace(servo_end, servo_end + "\n" + _DTC_THREAD, 1)

    # Replace one-shot cmd == 66 block with latch+thread start.
    start = text.find("        elif cmd == 66:")
    if start < 0:
        raise ValueError("stock script missing cmd == 66 (direct_torque)")
    end = text.find("        elif cmd == 67:", start)
    if end < 0:
        raise ValueError("stock script missing cmd == 67 after direct_torque")
    text = text[:start] + _DTC_CMD66 + "\n" + text[end:]

    # Kill DTC latch on speed_stop / servo_stop (same as leaving torque mode).
    speed_kill = (
        "            is_speeding = 0\n"
        "            kill speed_thrd\n"
        "            speed_thrd = 0\n"
    )
    speed_kill_with_dtc = (
        "            is_speeding = 0\n"
        "            kill speed_thrd\n"
        "            speed_thrd = 0\n"
        "            is_dtc = 0\n"
        "            kill dtc_thrd\n"
        "            dtc_thrd = 0\n"
    )
    if "kill dtc_thrd" not in text.split("speed_stop")[1].split("servo_stop")[0]:
        if speed_kill not in text:
            raise ValueError("stock script missing speed_stop kill block")
        text = text.replace(speed_kill, speed_kill_with_dtc, 1)

    servo_kill = (
        "            is_servoing = 0\n"
        "            kill servo_thrd\n"
        "            kill servoc_thrd\n"
        "            servo_thrd = 0\n"
        "            servoc_thrd = 0\n"
    )
    servo_kill_with_dtc = (
        "            is_servoing = 0\n"
        "            kill servo_thrd\n"
        "            kill servoc_thrd\n"
        "            servo_thrd = 0\n"
        "            servoc_thrd = 0\n"
        "            is_dtc = 0\n"
        "            kill dtc_thrd\n"
        "            dtc_thrd = 0\n"
    )
    if servo_kill not in text:
        raise ValueError("stock script missing servo_stop kill block")
    # Only first occurrence (servo_stop); avoid double-patch if re-run failed earlier.
    if text.count("kill dtc_thrd") < 2:
        text = text.replace(servo_kill, servo_kill_with_dtc, 1)

    if DTC_LATCH_MARKER not in text:
        raise RuntimeError("DTC latch patch failed to embed marker")
    return text


def wrap_rtde_control_script(body: str) -> str:
    """Wrap script body in ``def rtde_control(): … end``.

    ur_rtde only auto-wraps the *compiled-in* script. ``setCustomScriptFile``
    uploads the file bytes as-is — a bare body (stock ``rtde_control.script``)
    is invalid URScript and the control program never runs, so every
    ``directTorque`` returns False.
    """
    text = body.strip("\n") + "\n"
    stripped = text.lstrip()
    if stripped.startswith("def rtde_control("):
        if not stripped.rstrip().endswith("end"):
            raise ValueError("rtde_control script missing trailing end")
        return text if text.endswith("\n") else text + "\n"
    return f"def rtde_control():\n{text}end\n"


def resolve_stock_rtde_script(
    stock_script: str | Path | None = None,
) -> Path:
    """Return path to stock ``rtde_control.script`` (bundled or explicit)."""
    if stock_script is not None:
        path = Path(stock_script)
        if not path.is_file():
            raise FileNotFoundError(f"stock RTDE script not found: {path}")
        return path
    if DEFAULT_STOCK_RTDE_SCRIPT.is_file():
        return DEFAULT_STOCK_RTDE_SCRIPT
    raise FileNotFoundError(
        f"bundled stock RTDE script missing: {DEFAULT_STOCK_RTDE_SCRIPT}"
    )


def install_dtc_latch_control_script(
    control: Any,
    *,
    stock_script: str | Path | None = None,
    out_dir: str | Path | None = None,
    ready_timeout_s: float = 5.0,
) -> Path:
    """Patch stock control script and ``setCustomScriptFile`` on ``control``.

    Returns the written patched script path. Raises if the control program is
    not running after upload (invalid script / pendant stop).
    """
    set_fn = getattr(control, "setCustomScriptFile", None)
    if not callable(set_fn):
        raise RuntimeError(
            "control interface has no setCustomScriptFile(); "
            "cannot install on-robot DTC latch (upgrade ur_rtde)"
        )

    if isinstance(stock_script, str) and "\n" in stock_script:
        source = stock_script
        stock_path = None
    else:
        stock_path = resolve_stock_rtde_script(stock_script)
        source = stock_path.read_text(encoding="utf-8")

    # Strip an existing wrapper so patching sees the stock body layout.
    src_body = source
    if src_body.lstrip().startswith("def rtde_control("):
        lines = src_body.strip().splitlines()
        if lines and lines[0].startswith("def rtde_control(") and lines[-1].strip() == "end":
            src_body = "\n".join(lines[1:-1]) + "\n"

    patched = wrap_rtde_control_script(
        patch_rtde_control_script_for_dtc_latch(src_body)
    )
    if out_dir is not None:
        out_path = Path(out_dir) / "rtde_control_dtc_latch.script"
    elif stock_path is not None:
        out_path = stock_path.with_name("rtde_control_dtc_latch.script")
    else:
        out_path = DEFAULT_DTC_LATCH_SCRIPT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(patched, encoding="utf-8")
    set_fn(str(out_path))

    is_running = getattr(control, "isProgramRunning", None)
    if callable(is_running):
        deadline = time.perf_counter() + float(ready_timeout_s)
        while time.perf_counter() < deadline:
            try:
                if bool(is_running()):
                    break
            except Exception:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(
                "DTC latch script uploaded but RTDE control program not running "
                f"(check teach pendant / script syntax). path={out_path}"
            )
    return out_path


_DTC_SIGNATURE_LOGGED = False
# Which directTorque binding actually took, and whether friction compensation
# actually reached the robot. Module-scope so the dtc log header can record it:
# a startup line in scrollback is not evidence anyone can go back to, and the
# fallback here is SILENT. The bool path sets input int register 1, which the
# on-robot latch reads into dtc_friction_comp; the scale path does not. Joint
# breakaway on a UR5e is several N.m -- comparable to the whole command at the
# velocities this loop works at -- so which path bound decides whether the arm
# moves at all.
DTC_BOUND_SIGNATURE = ""
DTC_BOUND_FRICTION_COMP = None


def _log_dtc_signature(signature: str, friction_comp: bool) -> None:
    """Say once which directTorque binding took, and whether friction comp is on.

    The two paths are not equivalent and the fallback is silent: the bool path
    sets input int register 1, which is what the on-robot latch reads into
    ``dtc_friction_comp``; the scale path does not. Joint breakaway friction on
    a UR5e is several N.m, comparable to the whole command at small commanded
    velocities, so "which path bound" decides whether the arm moves at all.
    """
    global _DTC_SIGNATURE_LOGGED, DTC_BOUND_SIGNATURE, DTC_BOUND_FRICTION_COMP
    DTC_BOUND_SIGNATURE = signature
    DTC_BOUND_FRICTION_COMP = bool(friction_comp)
    if _DTC_SIGNATURE_LOGGED:
        return
    _DTC_SIGNATURE_LOGGED = True
    print(
        f"dtc: bound {signature}, friction_comp={bool(friction_comp)}",
        flush=True,
    )


def call_direct_torque(
    control: Any,
    tau: Sequence[float],
    *,
    friction_comp: bool = True,
    viscous_scale: Sequence[float] | None = None,
    coulomb_scale: Sequence[float] | None = None,
) -> bool:
    """Call ur_rtde ``directTorque``.

    Prefer ``directTorque(tau, friction_comp=bool)`` (PolyScope ≥5.23 script).
    Fall back to viscous/coulomb scales (PolyScope ≥5.26 / newer ur_rtde).
    """
    direct = getattr(control, "directTorque", None)
    if not callable(direct):
        raise RuntimeError(
            "control interface has no directTorque(); "
            "upgrade ur_rtde / PolyScope for DTC"
        )
    tau6 = [float(v) for v in np.asarray(tau, dtype=float).reshape(6)]
    if not np.isfinite(tau6).all():
        raise ValueError("direct_torque command contains non-finite values")
    # Bool friction_comp matches ur_rtde control script cmd==66 on ≥5.23.
    try:
        ok = direct(tau6, bool(friction_comp))
        _log_dtc_signature("directTorque(tau, friction_comp: bool)", friction_comp)
        return ok is not False
    except TypeError:
        pass
    if friction_comp:
        vs = list(
            DEFAULT_VISCOUS_SCALE
            if viscous_scale is None
            else np.asarray(viscous_scale, dtype=float).reshape(6)
        )
        cs = list(
            DEFAULT_COULOMB_SCALE
            if coulomb_scale is None
            else np.asarray(coulomb_scale, dtype=float).reshape(6)
        )
    else:
        vs = [0.0] * 6
        cs = [0.0] * 6
    ok = direct(tau6, vs, cs)
    _log_dtc_signature("directTorque(tau, viscous_scale, coulomb_scale)", friction_comp)
    return ok is not False



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
            "# Phase 2 DTC — prefer patched rtde_control (geodude DTC latch):\n"
            "#   host directTorque() latches dtc_target; dtc_thread @ 500 Hz\n"
            "#   calls direct_torque(tau) every timestep (see assets/urscript/).\n"
            "while (True):\n"
            "  enter_critical\n"
            "  tau = dtc_target\n"
            "  exit_critical\n"
            "  direct_torque(tau)\n"
        )


class HostDirectTorqueStreamer:
    """Background ``directTorque`` loop that streams a latched τ.

    Prefer ``robot_latch=True`` (default): on-robot ``dtc_thread`` applies τ at
    500 Hz; this host loop only republishes at ``target_hz`` (e.g. 50–125).

    With ``robot_latch=False`` (legacy / verify), the host must call near
    **500 Hz** or the arm leaves torque mode. Pace on ``sendCommand``'s
    ready-wait inside ``directTorque`` — do **not** also ``waitPeriod``.
    """

    def __init__(
        self,
        control: Any,
        *,
        friction_comp: bool = True,
        target_hz: float = UR_SYSTEM_HZ,
        robot_latch: bool = True,
        stale_tau_s: float | None = None,
        stale_zero_s: float | None = None,
    ):
        self._control = control
        self._friction_comp = bool(friction_comp)
        self._target_hz = float(target_hz)
        self._robot_latch = bool(robot_latch)
        # The on-robot latch holds the last τ forever if the host stops calling
        # set_tau(), which is the *safe* failure for a dropped packet but the
        # wrong one for a stalled control loop: a constant torque under a moving
        # arm keeps accelerating it against a plan nobody is computing any more.
        # Past this age the publisher decays τ towards zero (gravity-compensated
        # float) instead of holding the stale value. None keeps the raw hold,
        # which is what the rate-verification path wants.
        self._stale_tau_s = None if stale_tau_s is None else float(stale_tau_s)
        # ...and it *decays*, it does not step. A hard step from a live τ to 0
        # and back one publish later is itself an impulse: measured on this rig
        # the control tick jittered to 25 ms against a 16 ms bound, so the
        # publisher zeroed ~19 times per 5 s and the on-robot latch applied
        # every edge at 500 Hz. On the wrist (I ~= 0.05 kg.m^2) a 10 N.m pulse
        # 22 ms wide is d(omega) = tau*dt/I = 4.4 rad/s -- that is the arm
        # moving when nothing asked it to, which is the C283A2 the pendant
        # reported. Full zero still arrives, at stale_zero_s, so a dead loop
        # floats exactly as before.
        self._stale_zero_s = (
            None if stale_zero_s is None else float(stale_zero_s))
        if (self._stale_tau_s is not None and self._stale_zero_s is not None
                and self._stale_zero_s <= self._stale_tau_s):
            raise ValueError(
                f"stale_zero_s ({self._stale_zero_s}) must be greater than "
                f"stale_tau_s ({self._stale_tau_s})")
        self._n_zeroed = 0
        self._tau = np.zeros(6, dtype=float)
        self._tau_t = time.perf_counter()
        self._n_stale = 0
        self._stale_logged = False
        self._stale_report_t = 0.0
        self._stale_report_n = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._n_ok = 0
        self._n_fail = 0
        self._last_error: str | None = None
        self._active = False
        self._t_rate0: float | None = None
        self._n_ok_rate0 = 0
        self._measured_hz = float("nan")
        self._rate_warn_logged = False

    @property
    def active(self) -> bool:
        return self._active and self._thread is not None and self._thread.is_alive()

    @property
    def robot_latch(self) -> bool:
        return self._robot_latch

    def set_tau(self, tau: Sequence[float]) -> None:
        arr = np.asarray(tau, dtype=float).reshape(6)
        if not np.isfinite(arr).all():
            arr = np.zeros(6, dtype=float)
        with self._lock:
            self._tau = arr.copy()
            self._tau_t = time.perf_counter()

    def start(self) -> None:
        if self.active:
            return
        with self._lock:
            tau0 = self._tau.copy()
        # Leave any prior speedJ / jog mode before entering DTC.
        for name, args in (("speedStop", ()), ("speedStop", (1.0,))):
            fn = getattr(self._control, name, None)
            if callable(fn):
                try:
                    fn(*args)
                    break
                except TypeError:
                    continue
                except Exception:
                    break
        self._stop.clear()
        self._n_ok = 0
        self._n_fail = 0
        self._last_error = None
        # Synchronous first latch — thread start alone left a gap after speedStop
        # with τ≈0 under contact → UR "position deviation from path".
        try:
            ok0 = call_direct_torque(
                self._control, tau0, friction_comp=self._friction_comp
            )
            if ok0 is False:
                self._n_fail = 1
                self._last_error = "directTorque returned False (sync start)"
            else:
                self._n_ok = 1
        except Exception as exc:
            self._n_fail = 1
            self._last_error = f"{type(exc).__name__}: {exc}"
        self._t_rate0 = None
        self._n_ok_rate0 = 0
        self._measured_hz = float("nan")
        self._rate_warn_logged = False
        self._n_stale = 0
        self._n_zeroed = 0
        self._stale_logged = False
        self._stale_report_t = 0.0
        self._stale_report_n = 0
        with self._lock:
            self._tau_t = time.perf_counter()
        thr_name = (
            "host-dtc-latch-pub" if self._robot_latch else "host-dtc-500hz"
        )
        self._thread = threading.Thread(
            target=self._run, name=thr_name, daemon=True
        )
        self._active = True
        self._thread.start()
        if self._robot_latch:
            print(
                "\033[93mWARNING: DTC latch publisher started "
                f"(host publish {self._target_hz:g} Hz → on-robot 500 Hz "
                "dtc_thread; dtc_hz in status is host publish rate)."
                "\033[0m",
                flush=True,
            )
        else:
            print(
                "\033[93mWARNING: host DTC streamer started "
                f"(target {self._target_hz:g} Hz; no on-robot latch — "
                "watch dtc_hz; <400 Hz → arm leaves torque mode)."
                "\033[0m",
                flush=True,
            )

    def stop(self, *, zero: bool = True) -> None:
        self._stop.set()
        if zero:
            self.set_tau(np.zeros(6))
        thr = self._thread
        if thr is not None and thr.is_alive():
            thr.join(timeout=1.0)
        self._thread = None
        self._active = False
        # Leave torque mode cleanly.
        for name, args in (("speedStop", ()), ("stopJ", ([0.2] * 6,))):
            fn = getattr(self._control, name, None)
            if callable(fn):
                try:
                    fn(*args)
                    break
                except Exception:
                    continue

    def _zero_age(self) -> float:
        """Age at which τ reaches full float. Defaults to 2x the stale bound."""
        if self._stale_tau_s is None:
            return float("inf")
        if self._stale_zero_s is not None:
            return self._stale_zero_s
        return 2.0 * self._stale_tau_s

    def _stale_scale(self, age: float) -> float:
        """1.0 while fresh, ramping linearly to 0.0 at :meth:`_zero_age`."""
        if self._stale_tau_s is None:
            return 1.0
        span = self._zero_age() - self._stale_tau_s
        if span <= 0.0:
            return 0.0
        return float(np.clip(1.0 - (age - self._stale_tau_s) / span, 0.0, 1.0))

    def stats(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "n_ok": self._n_ok,
            "n_fail": self._n_fail,
            "last_error": self._last_error,
            "measured_hz": float(self._measured_hz),
            "robot_latch": bool(self._robot_latch),
            # Publish-side staleness: how many publishes were decayed because
            # the control tick was late. This is NOT the backend's own n_stale
            # (a dead RTDE *state read*, bounded at stale_after_s) -- the 1 Hz
            # console line printed that one as 0 while this one was counting 34.
            "n_stale": int(self._n_stale),
            "n_zeroed": int(self._n_zeroed),
            "stale_tau_ms": (
                None if self._stale_tau_s is None
                else 1e3 * self._stale_tau_s),
        }

    def _run(self) -> None:
        # robot_latch: pace host publishes with sleep (on-robot thread holds τ).
        # legacy: pace only on failure — directTorque ready-wait is the clock.
        dt_pub = 1.0 / max(self._target_hz, 1.0)
        dt_fail = dt_pub
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self._lock:
                tau = self._tau.copy()
                age = t0 - self._tau_t
            if self._stale_tau_s is not None and age > self._stale_tau_s:
                tau = tau * self._stale_scale(age)
                self._n_stale += 1
                if not np.any(tau):
                    self._n_zeroed += 1
                # Deliberately NOT a one-shot latch. A single warning at startup
                # (viser and the SpaceMouse coming up behind the control thread)
                # and a control loop that cannot hold its period print exactly
                # the same line, and only recurrence tells them apart -- so
                # re-report on a window, carrying the count since the last one.
                # A true one-shot still prints once and then stays quiet.
                if not self._stale_logged:
                    self._stale_logged = True
                    self._stale_report_t = t0
                    self._stale_report_n = self._n_stale
                    print(
                        "\033[93mWARNING: DTC τ stale by "
                        f"{age * 1e3:.0f} ms (> {self._stale_tau_s * 1e3:.0f} ms); "
                        "decaying torque towards zero (full float at "
                        f"{self._zero_age() * 1e3:.0f} ms) until the control "
                        "loop catches up.\033[0m",
                        flush=True,
                    )
                elif t0 - self._stale_report_t >= STALE_REPORT_WINDOW_S:
                    dn = self._n_stale - self._stale_report_n
                    self._stale_report_t = t0
                    self._stale_report_n = self._n_stale
                    print(
                        "\033[93mWARNING: DTC τ stale again -- "
                        f"{dn} decayed publishes in the last "
                        f"{STALE_REPORT_WINDOW_S:.0f} s "
                        f"({self._n_stale} total, {self._n_zeroed} of them at "
                        f"full float), age now {age * 1e3:.0f} ms. "
                        "The control loop is not holding its period.\033[0m",
                        flush=True,
                    )
            try:
                ok = call_direct_torque(
                    self._control, tau, friction_comp=self._friction_comp
                )
                if ok is False:
                    self._n_fail += 1
                    self._last_error = "directTorque returned False"
                    time.sleep(dt_fail)
                else:
                    self._n_ok += 1
                    now = time.perf_counter()
                    if self._t_rate0 is None:
                        self._t_rate0 = now
                        self._n_ok_rate0 = self._n_ok
                    else:
                        elapsed = now - float(self._t_rate0)
                        if elapsed >= 0.1:
                            dn = self._n_ok - int(self._n_ok_rate0)
                            self._measured_hz = float(dn) / elapsed
                            self._t_rate0 = now
                            self._n_ok_rate0 = self._n_ok
                            if (
                                not self._robot_latch
                                and not self._rate_warn_logged
                                and self._measured_hz < 400.0
                                and self._n_ok > 50
                            ):
                                self._rate_warn_logged = True
                                print(
                                    "\033[93mWARNING: DTC streamer measured "
                                    f"{self._measured_hz:.0f} Hz (need ~500). "
                                    "Arm leaves torque mode between calls — "
                                    "no motion. Install DTC latch script or "
                                    "check network / RTDE load."
                                    "\033[0m",
                                    flush=True,
                                )
                    if self._robot_latch:
                        remain = dt_pub - (time.perf_counter() - t0)
                        if remain > 0.0:
                            time.sleep(remain)
            except Exception as exc:
                self._n_fail += 1
                self._last_error = str(exc)
                time.sleep(dt_fail)
        self._active = False



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
            ok_call = call_direct_torque(
                control, tau6, friction_comp=bool(friction_comp)
            )
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
    for m in result.messages:
        print(f"[{prefix}] {m}", flush=True)
    for e in result.errors:
        print(f"[{prefix}] ERROR: {e}", flush=True)
