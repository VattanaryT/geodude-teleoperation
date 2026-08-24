"""UR ``force_mode`` contact session via ``ur_rtde`` (host does not need Fz).

Free-space teleop stays on gafropy speedJ. For contact, the arm is handed to
``RTDEControlInterface``: UR holds tool-Z force while the host streams ``speedL``
with tool-Z velocity forced to zero (SpaceMouse slides on the surface).

Handoff is intentionally conservative — settle, soft limits, force ramp, no
orientation snap — matching the bring-up style of ``UR-5E/force_mode/...``.
"""
from __future__ import annotations

import threading
import time
from typing import Sequence

import numpy as np


# Compliant Z seek / slide caps (m/s, rad/s). Z is force-compliant seek speed.
DEFAULT_LIMITS = [0.05, 0.05, 0.015, 0.2, 0.2, 0.2]
DEFAULT_DAMPING = 0.80
DEFAULT_GAIN_SCALING = 0.70
DEFAULT_GAIN_RAMP_MIN = 0.05
DEFAULT_SPEEDL_ACC = 0.3
DEFAULT_SPEEDL_CMD_T = 0.1
DEFAULT_RTDE_RELEASE_SETTLE_S = 3.0
UR_SCRIPT_PORT = 30002


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

    ``getActualTCPForce()`` is in the current tool frame; the task frame is the
    TCP pose captured at ``forceMode`` entry and does not move with the arm.
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


def release_ur_rtde_control(robot_ip: str, *, settle_s: float = DEFAULT_RTDE_RELEASE_SETTLE_S) -> None:
    """Best-effort URScript nudge so ur_rtde releases RTDE input fields before urcl reconnect."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(3.0)
            sock.connect((str(robot_ip), UR_SCRIPT_PORT))
            sock.sendall(b"end_force_mode()\nstopl(2.0)\n")
            try:
                sock.recv(1024)
            except socket.timeout:
                pass
    except OSError as exc:
        print(f"release_ur_rtde_control: script send failed ({exc})", flush=True)
    if settle_s > 0.0:
        time.sleep(float(settle_s))


class ForceModeContactSession:
    """Owns RTDE control/receive for one arm during contact."""

    def __init__(
        self,
        robot_ip: str,
        *,
        target_force: float = 5.0,
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
        self._speedl_streaming = False
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def enter(self, *, settle_s: float = 0.35) -> None:
        """Connect RTDE and enable force_mode. Caller must already stop gafropy streaming."""
        if self._active:
            raise RuntimeError("ForceModeContactSession already active")

        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface

        if settle_s > 0.0:
            time.sleep(settle_s)

        self._rtde_c = RTDEControlInterface(self.robot_ip)
        self._rtde_r = RTDEReceiveInterface(self.robot_ip)

        # One more settle after script takeover so residual speedJ dies out.
        time.sleep(0.25)

        if self._zero_ft_sensor:
            self._rtde_c.zeroFtSensor()
            time.sleep(0.35)

        # Type-2 task frame is fixed at entry (see UR-5E/force_mode reference scripts).
        self._task_frame = list(self._rtde_r.getActualTCPPose())
        self._warned_low_load = False
        self._warned_not_on_surface = False
        wrench = self._target_wrench()
        # One forceMode call only — re-entering each ramp step resets seek (~3 mm cap).
        # Ramp contact force via gain_scaling instead (matches V2 single force_mode() call).
        self._rtde_c.forceMode(
            self._task_frame,
            [0, 0, 1, 0, 0, 0],
            wrench,
            2,
            self._limits,
        )
        self._rtde_c.forceModeSetDamping(self._damping)
        self._rtde_c.forceModeSetGainScaling(DEFAULT_GAIN_RAMP_MIN)
        # Do NOT stream speedL(0) here — that pins TCP velocity and blocks Z seek.
        # speedL is only used while the SpaceMouse commands a non-zero XY slide.
        self._speedl_streaming = False

        with self._lock:
            self._cmd_linear[:] = 0.0
            self._cmd_angular[:] = 0.0
        self._t_start = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._active = True
        print(
            f"force_mode contact: force_sign={self._force_sign:g}  "
            f"Fd→{wrench[2]:+.1f}N task Z (gain ramp {self._force_ramp_s:.1f}s); "
            "SpaceMouse slides XY only (no idle speedL)",
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

    def _ramped_gain(self) -> float:
        if self._t_start is None or self._force_ramp_s <= 0.0:
            return self._gain_scaling
        ramp = min(1.0, (time.monotonic() - self._t_start) / self._force_ramp_s)
        return DEFAULT_GAIN_RAMP_MIN + ramp * (self._gain_scaling - DEFAULT_GAIN_RAMP_MIN)

    def _apply_gain_ramp(self) -> float:
        assert self._rtde_c is not None
        gain = self._ramped_gain()
        self._rtde_c.forceModeSetGainScaling(gain)
        return gain

    def _stream_speedl(self, speed: np.ndarray | None) -> None:
        """Stream tangential speedL, or release it so force_mode can own Z.

        Idle ``speedL(0)`` every tick fights compliant seek (reference free-air
        scripts never call speedL). On idle we ``speedStop`` once, then leave
        the velocity channel alone.
        """
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

    def _run(self) -> None:
        assert self._rtde_c is not None and self._rtde_r is not None
        assert self._task_frame is not None
        wrench = self._target_wrench()
        last_log = 0.0
        z0 = float(np.dot(
            np.asarray(self._task_frame[:3], dtype=float),
            _pose_rotation(self._task_frame)[:, 2],
        ))
        while not self._stop.is_set():
            pose = list(self._rtde_r.getActualTCPPose())
            gain = self._apply_gain_ramp()

            with self._lock:
                lin = self._cmd_linear.copy()
                ang = self._cmd_angular.copy()
            idle = (
                float(np.linalg.norm(lin)) < 1e-9
                and float(np.linalg.norm(ang)) < 1e-9
            )
            if idle:
                speed = None
                self._stream_speedl(None)
            else:
                speed = twist_to_speedl_base(
                    lin, ang, pose,
                    max_linear=self._max_linear,
                    max_angular=self._max_angular,
                    task_frame=self._task_frame,
                )
                self._stream_speedl(speed)

            now = time.monotonic()
            if now - last_log >= 0.5:
                last_log = now
                ft = self._rtde_r.getActualTCPForce()
                meas_task = _force_along_task_z(ft, self._task_frame, pose)
                z_now = float(np.dot(
                    np.asarray(pose[:3], dtype=float),
                    _pose_rotation(self._task_frame)[:, 2],
                ))
                dz = z_now - z0
                v_slide = 0.0 if speed is None else float(np.linalg.norm(speed[:3]))
                ramp = (
                    1.0
                    if self._force_ramp_s <= 0.0 or self._t_start is None
                    else min(1.0, (now - self._t_start) / self._force_ramp_s)
                )
                print(
                    f"force_mode: cmd_Fz={wrench[2]:+.2f}N  gain={gain:.2f}  "
                    f"meas_task_Fz={meas_task:+.2f}N  "
                    f"tool_Fz={float(ft[2]):+.2f}N  "
                    f"dZ_task={dz:+.4f}m  "
                    f"|v_slide|={v_slide:.4f}m/s",
                    flush=True,
                )
                target = abs(self._force_sign * self._target_force)
                if (
                    ramp >= 1.0
                    and target > 0.5
                    and abs(meas_task) < 0.35 * target
                    and not self._warned_low_load
                ):
                    self._warned_low_load = True
                    alt = -self._force_sign
                    print(
                        f"\033[93mforce_mode: low contact load "
                        f"(|meas|={abs(meas_task):.2f}N vs Fd={target:.1f}N). "
                        f"Approach until touching before toggling contact, or flip "
                        f"--force-sign (try {alt:g}; you ran {self._force_sign:g}). "
                        f"dZ_task>0 with cmd_Fz>0 = motion along task +Z.\033[0m",
                        flush=True,
                    )
                if (
                    ramp >= 1.0
                    and abs(dz) < 0.008
                    and abs(meas_task) < 0.5
                    and not self._warned_not_on_surface
                ):
                    self._warned_not_on_surface = True
                    print(
                        "\033[93mforce_mode: dZ_task stalled with no load — likely "
                        "not on the surface yet. Slide in approach until light touch, "
                        "then toggle contact.\033[0m",
                        flush=True,
                    )
            time.sleep(self._period)

    def exit(self, *, settle_s: float = DEFAULT_RTDE_RELEASE_SETTLE_S) -> None:
        """speedStop → forceModeStop → disconnect. Safe to call if already inactive."""
        if not self._active and self._rtde_c is None:
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
