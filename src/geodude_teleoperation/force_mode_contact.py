"""UR ``force_mode`` contact via native URScript (host does not need Fz).

Matches ``ur_controllers/ForceModeController`` semantics, not tool_contact:
``force_mode()`` every cycle (same as ur_rtde ``force_thread``), type 2, Z
compliant. Compliant-axis limits are max TCP *speed*; non-compliant are max
*deviation*. Force on Z overwrites any motion command on Z — do not speedL
there. Combined execution speed is ``speed_scaling * target_speed_fraction``.
"""
from __future__ import annotations

import threading
import time
from typing import Sequence

import numpy as np


# ForceModeController packs limits as: compliant → speed (m/s), else deviation (m).
# Official example uses Z speed 1.0 m/s (free-air demo). Contact press stays slow.
DEFAULT_SELECTION = [0, 0, 1, 0, 0, 0]
DEFAULT_SPEED_LIMITS = [0.0, 0.0, 0.005, 0.0, 0.0, 0.0]
DEFAULT_DEVIATION_LIMITS = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
# URScript: damping 0 = none, 1 = full. ForceModeController default 0.025;
# official example and this stack use 0.1. 0.80 kissed at ~1 N.
DEFAULT_DAMPING = 0.10
# Official example 0.8. Docs: >1 can make force_mode unstable on a hard surface.
DEFAULT_GAIN_SCALING = 0.8
DEFAULT_SPEEDL_ACC = 0.3
DEFAULT_SPEEDL_CMD_T = 0.1
# speedL into contact faults: "speedl unable to find inverse kinematics
# solution. verify force mode is being used." The press path is one async
# moveL along task Z; force_mode is allowed to miss that pose on Z.
DEFAULT_PRESS_DISTANCE_M = 0.012
DEFAULT_PRESS_VEL = 0.005
DEFAULT_PRESS_ACC = 0.02
DEFAULT_SEEK_TRAVEL_LIMIT_M = 0.015
DEFAULT_MAX_FORCE_N = 50.0
# Kept for tests / documentation of the infeasible speedL P-seek.
DEFAULT_SEEK_KP = 0.0006
DEFAULT_SEEK_VMAX = 0.008
DEFAULT_SEEK_DEADBAND_N = 1.0
DEFAULT_RTDE_RELEASE_SETTLE_S = 3.0
UR_SCRIPT_PORT = 30002
UR_DASHBOARD_PORT = 29999
RUNTIME_STOPPED = 1
RUNTIME_PLAYING = 2


def _pose_rotation(tcp_pose: Sequence[float]) -> np.ndarray:
    """Axis-angle TCP pose [x,y,z,rx,ry,rz] → 3x3 rotation (base ← tool)."""
    rx, ry, rz = (float(tcp_pose[3]), float(tcp_pose[4]), float(tcp_pose[5]))
    angle = float(np.sqrt(rx * rx + ry * ry + rz * rz))
    if angle < 1e-9:
        return np.eye(3)
    axis = np.array([rx, ry, rz], dtype=float) / angle
    kx, ky, kz = axis
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=float)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _force_along_task_z(
    tcp_force: Sequence[float],
    task_frame: Sequence[float],
    current_pose: Sequence[float],
) -> float:
    """Signed linear force along fixed task +Z (UR force_mode type 2).

    ``getActualTCPForce()`` is the wrench **on the TCP** (environment → tool).
    ``force_mode`` wrench is the wrench **on the environment** (tool → part).
    Those differ by a minus sign: cmd_Fz=+15 and meas≈−15 is a loaded press,
    not a sign bug. Do not flip ``--force-sign`` if the arm already moves into
    the part.
    """
    f_tool = np.asarray(tcp_force, dtype=float).reshape(6)[:3]
    R_cur = _pose_rotation(current_pose)
    R_task = _pose_rotation(task_frame)
    f_base = R_cur @ f_tool
    return float(np.dot(f_base, R_task[:, 2]))


def twist_to_speedl_base(
    linear_tool: Sequence[float],
    angular_tool: Sequence[float],
    tcp_pose: Sequence[float],
    *,
    max_linear: float = 0.02,
    max_angular: float = 0.15,
    task_frame: Sequence[float] | None = None,
) -> np.ndarray:
    """Map tool-frame twist → base-frame speedL; strip velocity along task +Z.

    Force_mode owns the compliant normal. Commanding speedL along that axis
    (including idle zeros) fights contact seek — leave task Z free.
    """
    lin = np.asarray(linear_tool, dtype=float).reshape(3).copy()
    ang = np.asarray(angular_tool, dtype=float).reshape(3).copy()
    R = _pose_rotation(tcp_pose)
    v = np.zeros(6, dtype=float)
    v[:3] = R @ lin
    v[3:] = R @ ang
    if task_frame is not None:
        z_task = _pose_rotation(task_frame)[:, 2]
        v[:3] = v[:3] - float(np.dot(v[:3], z_task)) * z_task
    ln = float(np.linalg.norm(v[:3]))
    if ln > max_linear > 0.0:
        v[:3] *= max_linear / ln
    an = float(np.linalg.norm(v[3:]))
    if an > max_angular > 0.0:
        v[3:] *= max_angular / an
    return v


def seek_speedl_task_z(
    cmd_fz: float,
    meas_fz: float,
    task_frame: Sequence[float],
    *,
    kp: float = DEFAULT_SEEK_KP,
    vmax: float = DEFAULT_SEEK_VMAX,
    deadband_n: float = DEFAULT_SEEK_DEADBAND_N,
) -> np.ndarray:
    """P-only speedL along fixed task +Z so meas_fz tracks cmd_fz.

    force_mode stays up (Z compliant, wrench = Fd). This does not re-enter
    forceMode. Positive ``cmd_fz - meas_fz`` commands motion along task +Z.
    """
    error = float(cmd_fz) - float(meas_fz)
    v = np.zeros(6, dtype=float)
    if abs(error) <= float(deadband_n):
        return v
    v_n = float(np.clip(kp * error, -float(vmax), float(vmax)))
    v[:3] = v_n * _pose_rotation(task_frame)[:, 2]
    return v


def pose_offset_along_task_z(pose: Sequence[float], distance_m: float) -> list[float]:
    """Same orientation; XYZ shifted along this pose's +Z (task frame at entry)."""
    out = [float(x) for x in pose]
    z = _pose_rotation(pose)[:, 2]
    dist = float(distance_m)
    for i in range(3):
        out[i] += dist * float(z[i])
    return out


def _ur_list(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(x):.6f}" for x in values) + "]"


def pack_force_mode_limits(
    selection: Sequence[int],
    speed_limits: Sequence[float],
    deviation_limits: Sequence[float],
) -> list[float]:
    """Same packing as ``ur_controllers/ForceModeController::setForceMode``."""
    out: list[float] = []
    for i in range(6):
        if int(selection[i]):
            out.append(float(speed_limits[i]))
        else:
            out.append(float(deviation_limits[i]))
    return out


DEFAULT_LIMITS = pack_force_mode_limits(
    DEFAULT_SELECTION, DEFAULT_SPEED_LIMITS, DEFAULT_DEVIATION_LIMITS)


def _urscript_bytes(script: str) -> bytes:
    payload = script if script.endswith("\n") else script + "\n"
    return payload.encode("utf-8")


def send_urscript(robot_ip: str, script: str, *, timeout: float = 5.0) -> bool:
    """Send URScript on the secondary interface (port 30002). Same as V2."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((str(robot_ip), UR_SCRIPT_PORT))
            sock.sendall(_urscript_bytes(script))
            try:
                sock.recv(1024)
            except socket.timeout:
                pass
        return True
    except OSError as exc:
        print(f"send_urscript: failed ({exc})", flush=True)
        return False


def dashboard_command(robot_ip: str, command: str, *, timeout: float = 5.0) -> str:
    """One Dashboard-server command (port 29999)."""
    import socket

    sock = socket.create_connection((str(robot_ip), UR_DASHBOARD_PORT), timeout=timeout)
    try:
        sock.recv(4096)
        sock.sendall((command + "\n").encode("utf-8"))
        time.sleep(0.2)
        return sock.recv(4096).decode(errors="replace")
    finally:
        sock.close()


def dashboard_stop_program(robot_ip: str) -> str:
    """Stop leftover External Control so a 30002 script can become primary."""
    try:
        reply = dashboard_command(robot_ip, "stop")
        print(f"dashboard stop: {reply.strip()}", flush=True)
        return reply
    except OSError as exc:
        print(f"dashboard stop: failed ({exc})", flush=True)
        return ""


def set_speed_slider_fraction(robot_ip: str, fraction: float = 1.0) -> bool:
    """Force the speed overlay to 100%. gafropy/urcl can leave it at 0."""
    try:
        from rtde_io import RTDEIOInterface

        io = RTDEIOInterface(str(robot_ip))
        ok = bool(io.setSpeedSlider(float(fraction)))
        try:
            io.disconnect()
        except Exception:
            pass
        return ok
    except Exception as exc:
        print(f"set_speed_slider_fraction: failed ({exc})", flush=True)
        return False


def wait_runtime_state(rtde_r, want: int, timeout_s: float) -> int:
    """Poll RTDE runtime_state until it matches ``want`` (or timeout)."""
    deadline = time.monotonic() + float(timeout_s)
    last = -1
    while time.monotonic() < deadline:
        try:
            last = int(rtde_r.getRuntimeState())
        except Exception:
            last = -1
        if last == want:
            return last
        time.sleep(0.05)
    return last


def build_force_hold_script(
    *,
    wrench: Sequence[float],
    limits: Sequence[float],
    damping: float,
    gain_scaling: float,
    zero_ft_sensor: bool,
    max_force_n: float = DEFAULT_MAX_FORCE_N,
    max_travel_m: float = DEFAULT_SEEK_TRAVEL_LIMIT_M,
) -> str:
    """Native URScript: re-call force_mode every cycle (ur_rtde force_thread).

    Calling force_mode once then sync() is not what the ROS driver / ur_rtde
    control script do; that pattern kisses and floats with frozen dZ.
    """
    zero = "    zero_ftsensor()\n    sleep(0.3)\n" if zero_ft_sensor else ""
    return f"""def force_hold_press():
    textmsg("FORCE_HOLD_PRESS_START")
{zero}    p0 = get_actual_tcp_pose()
    force_mode_set_damping({float(damping):.6f})
    force_mode_set_gain_scaling({float(gain_scaling):.6f})
    sleep(0.02)
    textmsg("FORCE_HOLD_PRESS_ACTIVE")
    while True:
        force_mode(p0, [0, 0, 1, 0, 0, 0], {_ur_list(wrench)}, 2, {_ur_list(limits)})
        p = get_actual_tcp_pose()
        w = get_tcp_force()
        f_norm = norm([w[0], w[1], w[2]])
        d = norm([p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]])
        if (f_norm > {float(max_force_n):.6f}):
            textmsg("FORCE_HOLD_ABORT_F")
            stopl(1.0)
            end_force_mode()
            halt
        end
        if (d > {float(max_travel_m):.6f}):
            textmsg("FORCE_HOLD_ABORT_TRAVEL")
            stopl(1.0)
            end_force_mode()
            halt
        end
        sync()
    end
end
force_hold_press()
"""


def release_ur_rtde_control(robot_ip: str, *, settle_s: float = DEFAULT_RTDE_RELEASE_SETTLE_S) -> None:
    """Best-effort stop of force_mode / leftover program before urcl reconnect."""
    send_urscript(str(robot_ip), "end_force_mode()\nstopl(2.0)\n")
    dashboard_stop_program(str(robot_ip))
    if settle_s > 0.0:
        time.sleep(float(settle_s))


class ForceModeContactSession:
    """Owns RTDE control/receive for one arm during contact."""

    def __init__(
        self,
        robot_ip: str,
        *,
        target_force: float = 15.0,
        force_sign: float = -1.0,
        force_ramp_s: float = 2.0,
        zero_ft_sensor: bool = True,
        limits: Sequence[float] | None = None,
        damping: float = DEFAULT_DAMPING,
        gain_scaling: float = DEFAULT_GAIN_SCALING,
        speedl_acc: float = DEFAULT_SPEEDL_ACC,
        speedl_cmd_t: float = DEFAULT_SPEEDL_CMD_T,
        max_linear: float = 0.02,
        max_angular: float = 0.15,
        update_hz: float = 50.0,
    ):
        self.robot_ip = str(robot_ip)
        self._target_force = float(target_force)
        self._force_sign = float(force_sign)
        self._force_ramp_s = max(0.0, float(force_ramp_s))
        self._zero_ft_sensor = bool(zero_ft_sensor)
        self._limits = list(limits) if limits is not None else list(DEFAULT_LIMITS)
        self._damping = float(damping)
        self._gain_scaling = float(gain_scaling)
        self._speedl_acc = float(speedl_acc)
        self._speedl_cmd_t = float(speedl_cmd_t)
        self._max_linear = float(max_linear)
        self._max_angular = float(max_angular)
        self._period = 1.0 / float(update_hz)

        self._rtde_c = None
        self._rtde_r = None
        self._lock = threading.Lock()
        self._cmd_linear = np.zeros(3, dtype=float)
        self._cmd_angular = np.zeros(3, dtype=float)
        self._stop = threading.Event()
        self._thread = None
        self._t_start = None
        self._task_frame: list[float] | None = None
        self._warned_low_load = False
        self._warned_not_on_surface = False
        self._warned_seek_travel = False
        self._warned_runtime = False
        self._speedl_streaming = False
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def enter(self, *, settle_s: float = 0.35) -> None:
        """Enable native URScript force_mode. Caller must already stop gafropy."""
        if self._active:
            raise RuntimeError("ForceModeContactSession already active")

        from rtde_receive import RTDEReceiveInterface

        if settle_s > 0.0:
            time.sleep(settle_s)

        # Do not connect RTDEControlInterface: it uploads a control script that
        # replaces the native force_mode+sync() program and never seeks.
        # Stop leftover External Control first. A 30002 script sent while that
        # program is PLAYING runs as secondary; force_mode is then a no-op.
        dashboard_stop_program(self.robot_ip)
        time.sleep(0.3)
        set_speed_slider_fraction(self.robot_ip, 1.0)

        self._rtde_c = None
        self._rtde_r = RTDEReceiveInterface(self.robot_ip)
        stopped = wait_runtime_state(self._rtde_r, RUNTIME_STOPPED, 2.0)
        if stopped != RUNTIME_STOPPED:
            dashboard_stop_program(self.robot_ip)
            stopped = wait_runtime_state(self._rtde_r, RUNTIME_STOPPED, 2.0)
        if stopped != RUNTIME_STOPPED:
            raise RuntimeError(
                f"could not stop leftover External Control "
                f"(runtime_state={stopped}, want STOPPED={RUNTIME_STOPPED}). "
                "On the pendant: Remote, then Stop, then retry the toggle."
            )

        self._task_frame = list(self._rtde_r.getActualTCPPose())
        self._warned_low_load = False
        self._warned_not_on_surface = False
        self._warned_seek_travel = False
        self._warned_runtime = False
        self._speedl_streaming = False
        wrench = self._target_wrench()
        script = build_force_hold_script(
            wrench=wrench,
            limits=self._limits,
            damping=self._damping,
            gain_scaling=self._gain_scaling,
            zero_ft_sensor=self._zero_ft_sensor,
        )
        if not send_urscript(self.robot_ip, script):
            raise RuntimeError("failed to send force_hold URScript on port 30002")
        playing = wait_runtime_state(self._rtde_r, RUNTIME_PLAYING, 3.0)
        if playing != RUNTIME_PLAYING:
            raise RuntimeError(
                f"force_hold URScript did not start (runtime_state={playing}, "
                f"want PLAYING={RUNTIME_PLAYING}). Check Remote Control and "
                "pendant log for FORCE_HOLD_PRESS_ACTIVE."
            )

        with self._lock:
            self._cmd_linear[:] = 0.0
            self._cmd_angular[:] = 0.0
        self._t_start = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._active = True
        print(
            f"force_mode contact: native URScript force_mode+sync()  "
            f"force_sign={self._force_sign:g}  Fd={wrench[2]:+.1f}N task Z  "
            f"damping={self._damping:.2f}  gain={self._gain_scaling:.2f}  "
            f"Z-limit={self._limits[2]}m/s seek cap  runtime=PLAYING  "
            "speed_slider=1.0  (no ur_rtde speedL/moveL)",
            flush=True,
        )

    def set_twist(self, linear_tool: Sequence[float], angular_tool: Sequence[float]) -> None:
        """Body/tool-frame command from SpaceMouse; tool-Z linear is ignored."""
        lin = np.asarray(linear_tool, dtype=float).reshape(3).copy()
        ang = np.asarray(angular_tool, dtype=float).reshape(3).copy()
        lin[2] = 0.0
        with self._lock:
            self._cmd_linear[:] = lin
            self._cmd_angular[:] = ang

    def clear_twist(self) -> None:
        """Zero teleop command (hands-off Space must be called on idle, or last speedL sticks."""
        with self._lock:
            self._cmd_linear[:] = 0.0
            self._cmd_angular[:] = 0.0

    def _target_wrench(self) -> list[float]:
        fz = self._force_sign * self._target_force
        return [0.0, 0.0, float(fz), 0.0, 0.0, 0.0]

    def _stream_speedl(self, speed: np.ndarray | None) -> None:
        """Stream speedL, or speedStop once when the command is exactly idle."""
        assert self._rtde_c is not None
        if speed is None:
            if self._speedl_streaming:
                try:
                    self._rtde_c.speedStop(0.5)
                except Exception:
                    pass
                self._speedl_streaming = False
            return
        self._rtde_c.speedL(speed.tolist(), self._speedl_acc, self._speedl_cmd_t)
        self._speedl_streaming = True

    def _halt_press(self, reason: str) -> None:
        print(f"\033[91mforce_mode: {reason}\033[0m", flush=True)
        send_urscript(self.robot_ip, "end_force_mode()\nstopl(1.0)\n")
        dashboard_stop_program(self.robot_ip)

    def _run(self) -> None:
        assert self._rtde_r is not None
        assert self._task_frame is not None
        wrench = self._target_wrench()
        last_log = 0.0
        z0 = float(np.dot(
            np.asarray(self._task_frame[:3], dtype=float),
            _pose_rotation(self._task_frame)[:, 2],
        ))
        while not self._stop.is_set():
            pose = list(self._rtde_r.getActualTCPPose())
            ft = self._rtde_r.getActualTCPForce()
            meas_task = _force_along_task_z(ft, self._task_frame, pose)
            z_now = float(np.dot(
                np.asarray(pose[:3], dtype=float),
                _pose_rotation(self._task_frame)[:, 2],
            ))
            dz = z_now - z0
            f_norm = float(np.linalg.norm(np.asarray(ft, dtype=float)[:3]))
            try:
                runtime = int(self._rtde_r.getRuntimeState())
            except Exception:
                runtime = -1
            try:
                speed_scaling = float(self._rtde_r.getSpeedScaling())
            except Exception:
                speed_scaling = -1.0
            try:
                speed_combined = float(self._rtde_r.getSpeedScalingCombined())
            except Exception:
                speed_combined = speed_scaling

            if runtime != RUNTIME_PLAYING and not self._warned_runtime:
                self._warned_runtime = True
                print(
                    f"\033[91mforce_mode: runtime_state={runtime} "
                    f"(want PLAYING={RUNTIME_PLAYING}); force_mode is not the "
                    "robot program\033[0m",
                    flush=True,
                )

            # Never speedL during a press. speedL into contact kills the RTDE
            # script (IK). SpaceMouse XY is ignored until that is solved.
            if f_norm > DEFAULT_MAX_FORCE_N:
                self._halt_press(
                    f"force watchdog |F|={f_norm:.1f}N > {DEFAULT_MAX_FORCE_N}N")
                break
            if (
                abs(dz) > DEFAULT_SEEK_TRAVEL_LIMIT_M
                and abs(meas_task) < 0.35 * abs(wrench[2] or 15.0)
            ):
                self._halt_press(
                    f"travel watchdog |dZ|={abs(dz):.3f}m with meas="
                    f"{meas_task:+.1f}N (force_mode not loading)")
                break

            now = time.monotonic()
            if now - last_log >= 0.5:
                last_log = now
                print(
                    f"force_mode: cmd_Fz={wrench[2]:+.2f}N  "
                    f"gain={self._gain_scaling:.2f}  damping={self._damping:.2f}  "
                    f"meas_task_Fz={meas_task:+.2f}N  "
                    f"tool_Fz={float(ft[2]):+.2f}N  "
                    f"dZ_task={dz:+.4f}m  runtime={runtime}  "
                    f"speed_scaling={speed_scaling:.2f}  "
                    f"speed_combined={speed_combined:.2f}",
                    flush=True,
                )
            time.sleep(self._period)

    def exit(self, *, settle_s: float = DEFAULT_RTDE_RELEASE_SETTLE_S) -> None:
        """end_force_mode → stopl. Safe to call if already inactive."""
        if not self._active and self._rtde_r is None:
            return

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        c = self._rtde_c
        r = self._rtde_r
        robot_ip = self.robot_ip
        self._rtde_c = None
        self._rtde_r = None
        self._active = False

        if c is not None:
            try:
                c.speedStop(0.5)
            except Exception:
                pass
            time.sleep(0.15)
            try:
                c.forceModeStop()
            except Exception:
                pass
            time.sleep(0.1)
            try:
                c.stopScript()
            except Exception:
                pass
            try:
                c.disconnect()
            except Exception:
                pass
        if r is not None:
            try:
                r.disconnect()
            except Exception:
                pass

        release_ur_rtde_control(robot_ip, settle_s=float(settle_s))
        print("force_mode contact: exited", flush=True)


def shutdown_gafropy_for_handoff(interface, *, settle_s: float = 0.4) -> None:
    """Stop gafropy/urcl so ``ur_rtde`` can own the arm (smooth: settle first)."""
    time.sleep(settle_s)
    interface.shutdown()
    time.sleep(0.3)


def reopen_gafropy(
    model: str,
    hardware_config: str,
    *,
    robot_ip: str | None = None,
    retries: int = 6,
):
    """Fresh ``RobotInterface`` after force_mode session (external_control re-uploaded)."""
    from gafropy import robot_interface as ri

    if robot_ip is not None:
        release_ur_rtde_control(str(robot_ip), settle_s=1.0)

    last_exc: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            config = ri.Configuration(model, "hardware", hardware_config)
            interface = ri.RobotInterface(config)
            interface.initialize()
            time.sleep(0.3)
            return interface
        except Exception as exc:
            last_exc = exc
            wait_s = 3.0 * (attempt + 1)
            print(
                f"\033[93mreopen_gafropy: attempt {attempt + 1}/{retries} failed "
                f"({exc}); retrying in {wait_s:.0f}s\033[0m",
                flush=True,
            )
            if robot_ip is not None and "speed_slider_mask" in str(exc):
                release_ur_rtde_control(str(robot_ip), settle_s=2.0)
            time.sleep(wait_s)
    assert last_exc is not None
    raise last_exc
