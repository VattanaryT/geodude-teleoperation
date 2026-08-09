"""RTDE receive-only dry-run and live speedJ host loop for stringer teleop.

Kinematics v1 (Task 9):
- **Dry-run / VF status:** ``getActualTCPPose`` → minimal EE motor for
  ``vf.update_from_state``; joint commands stay zero (receive-only).
- **Live speedJ:** when gafropy is importable, a kinematics-only MuJoCo
  ``RobotInterface`` (geodude model) supplies FK and Jacobian for
  ``TeleoperationController.compute_control_command``. RTDE ``getActualQ``
  is copied into the interface state each tick (see ``_GafropyKinematicsBridge``).
  Without gafropy or if joint sync fails, motion is refused (``speedJ(0)``) with a
  one-time warning; TCP-based VF queries still run.

Live TF via ``lookup_T_tag0_base_tf`` when ROS tf2 is available; otherwise fall
back to an explicit ``--T-tag0-base`` file (warn once, ``tf_ok=True``). If live TF
is requested but unavailable and ``--T-tag0-base`` was not provided (implicit
identity), ``T`` is identity but ``tf_ok=False`` so motion is refused.
``tf_age_ms`` is ``nan`` when live lookup fails. Calibrate mode may show
``tag_vs_tf_mm`` when live TF and AprilTag chain inputs are both available
(enable stays manual).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent.parent
GEODUDE_MODEL = ROOT / "assets" / "robots" / "geodude" / "geodude.yaml"
TASK_SPACE_NAME = "right_ee"


def _import_rtde():
    try:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
    except ImportError as exc:
        raise ImportError(
            "ur_rtde is required for RTDE teleop; "
            "install with: pip install 'geodude-teleoperation[ur]'"
        ) from exc
    return RTDEControlInterface, RTDEReceiveInterface


def _runtime_fault(receive: Any) -> bool:
    if getattr(receive, "isEmergencyStopped", lambda: False)():
        return True
    if getattr(receive, "isProtectiveStopped", lambda: False)():
        return True
    mode = getattr(receive, "getRobotMode", lambda: 7)()
    return int(mode) != 7


class RtdeTeleopSession:
    """RTDE receive (+ optional control) session for one robot IP."""

    def __init__(
        self,
        ip: str,
        receive: Any,
        control: Any | None,
        *,
        dry_run: bool,
        accel: float = 0.5,
        speedj_time: float = 0.02,
    ):
        self.ip = str(ip)
        self._receive = receive
        self._control = control
        self._dry_run = bool(dry_run)
        self._accel = float(accel)
        self._speedj_time = float(speedj_time)

    @property
    def control_open(self) -> bool:
        return self._control is not None

    def read_state(self) -> dict[str, Any]:
        return {
            "q": [float(v) for v in self._receive.getActualQ()],
            "tcp_pose": [float(v) for v in self._receive.getActualTCPPose()],
            "fault": _runtime_fault(self._receive),
        }

    def speedj(self, qdot: Sequence[float], accel: float) -> None:
        if self._dry_run or self._control is None:
            raise RuntimeError("speedj unavailable in dry-run (receive-only)")
        qdot_list = [float(v) for v in qdot]
        self._control.speedJ(qdot_list, float(accel), self._speedj_time)

    def close(self) -> None:
        if self._control is not None:
            try:
                self._control.speedJ([0.0] * 6, self._accel, 0.5)
            except Exception:
                pass
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


def open(
    ip: str,
    *,
    dry_run: bool,
    accel: float = 0.5,
    speedj_time: float = 0.02,
) -> RtdeTeleopSession:
    RTDEControlInterface, RTDEReceiveInterface = _import_rtde()
    receive = RTDEReceiveInterface(str(ip))
    control = None if dry_run else RTDEControlInterface(str(ip))
    return RtdeTeleopSession(
        ip,
        receive,
        control,
        dry_run=dry_run,
        accel=accel,
        speedj_time=speedj_time,
    )


def ee_motor_from_tcp_pose(tcp_pose: Sequence[float]):
    """UR ``getActualTCPPose`` [x,y,z,rx,ry,rz] axis-angle → minimal EE motor."""
    from geodude_teleoperation.contact_wos_vf import _FallbackMotor

    p = np.asarray(tcp_pose[:3], dtype=float)
    R = Rotation.from_rotvec(np.asarray(tcp_pose[3:6], dtype=float)).as_matrix()

    class _Translator:
        def x(self):
            return float(p[0])

        def y(self):
            return float(p[1])

        def z(self):
            return float(p[2])

    return _FallbackMotor(_Translator(), R)


_live_tf_fallback_warned = False
_ros_tf_unavailable_warned = False


class _RosTfContext:
    """Optional ROS tf2 buffer for live ``T_tag0_base`` lookup."""

    def __init__(self):
        self.buffer = None
        self._node = None
        self._listener = None
        try:
            import rclpy
            from tf2_ros import Buffer, TransformListener
        except ImportError:
            return
        try:
            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node("geodude_stringer_teleop_tf")
            self.buffer = Buffer()
            self._listener = TransformListener(self.buffer, self._node)
        except Exception:
            self.buffer = None
            self._node = None
            self._listener = None

    def spin_once(self) -> None:
        if self._node is None:
            return
        try:
            import rclpy

            rclpy.spin_once(self._node, timeout_sec=0.0)
        except Exception:
            pass

    def close(self) -> None:
        if self._node is None:
            return
        try:
            self._node.destroy_node()
        except Exception:
            pass


class _RosCameraContext:
    """Optional latest RGB + intrinsics for AprilTag cross-check."""

    def __init__(self, args):
        self.rgb = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self._node = None
        topic = getattr(args, "camera_rgb_topic", None)
        info_topic = getattr(args, "camera_info_topic", None)
        if not topic or not info_topic:
            return
        try:
            import rclpy
            from cv_bridge import CvBridge
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError:
            return
        try:
            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node("geodude_stringer_teleop_cam")
            self._bridge = CvBridge()
            self._lock = __import__("threading").Lock()
            self.rgb = None
            self.camera_matrix = None
            self.dist_coeffs = None

            def on_rgb(msg: Image) -> None:
                try:
                    rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                except Exception:
                    return
                with self._lock:
                    self.rgb = np.asarray(rgb)

            def on_info(msg: CameraInfo) -> None:
                k = msg.k
                camera_matrix = np.array(
                    [[float(k[0]), 0.0, float(k[2])], [0.0, float(k[4]), float(k[5])], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
                dist = np.array(msg.d, dtype=np.float64) if msg.d else np.zeros(5)
                with self._lock:
                    self.camera_matrix = camera_matrix
                    self.dist_coeffs = dist

            self._node.create_subscription(Image, str(topic), on_rgb, 10)
            self._node.create_subscription(CameraInfo, str(info_topic), on_info, 10)
        except Exception:
            self.rgb = None
            self.camera_matrix = None
            self.dist_coeffs = None
            self._node = None

    def snapshot(self):
        if self._node is None:
            return None, None, None
        try:
            import rclpy

            rclpy.spin_once(self._node, timeout_sec=0.0)
        except Exception:
            pass
        with self._lock:
            return self.rgb, self.camera_matrix, self.dist_coeffs

    def close(self) -> None:
        if self._node is None:
            return
        try:
            self._node.destroy_node()
        except Exception:
            pass


def resolve_T_tag0_base(args, *, buffer=None) -> tuple[np.ndarray, bool, float]:
    """Resolve ``T_tag0_base`` for this tick.

    Returns ``(T, tf_ok, tf_age_ms)``. ``T`` is always defined (never dangling).
    """
    global _live_tf_fallback_warned, _ros_tf_unavailable_warned
    from geodude_teleoperation.frames import load_T_tag0_base, lookup_T_tag0_base_tf

    live_tf_requested = bool(args.base_frame and args.tag0_frame)
    live_tf_failed = False
    tf_age_ms = 0.0

    if live_tf_requested:
        max_age_s = getattr(args, "tf_max_age_s", 0.5)
        T_live, age_ms = lookup_T_tag0_base_tf(
            buffer, args.base_frame, args.tag0_frame, max_age_s
        )
        if T_live is not None:
            return np.asarray(T_live, dtype=float), True, float(age_ms or 0.0)
        live_tf_failed = True
        tf_age_ms = float("nan")
        if buffer is None and not _ros_tf_unavailable_warned:
            print(
                "warning: live TF requested but ROS tf2 unavailable; "
                "tf_age_ms=nan",
                file=sys.stderr,
            )
            _ros_tf_unavailable_warned = True

    try:
        T = load_T_tag0_base(args.T_tag0_base)
    except (ValueError, OSError, KeyError) as exc:
        if not _live_tf_fallback_warned and live_tf_failed:
            print(
                "warning: live TF unavailable and fixed T_tag0_base failed "
                f"({exc}); refusing motion (speedJ(0))",
                file=sys.stderr,
            )
            _live_tf_fallback_warned = True
        return np.eye(4, dtype=float), False, float("nan")

    if live_tf_failed:
        if args.T_tag0_base is None:
            if not _live_tf_fallback_warned:
                print(
                    "warning: live TF unavailable and no fixed T_tag0_base; "
                    "refusing motion (speedJ(0))",
                    file=sys.stderr,
                )
                _live_tf_fallback_warned = True
            return T, False, tf_age_ms

        if not _live_tf_fallback_warned:
            print(
                "warning: live TF unavailable; using fixed T_tag0_base",
                file=sys.stderr,
            )
            _live_tf_fallback_warned = True

    return T, True, tf_age_ms if live_tf_requested else 0.0


def lookup_live_T_tag0_base(args, *, buffer=None) -> tuple[np.ndarray | None, float | None]:
    """Return live TF ``T_tag0_base`` when lookup succeeds (for cross-check)."""
    if not (args.base_frame and args.tag0_frame):
        return None, None
    from geodude_teleoperation.frames import lookup_T_tag0_base_tf

    max_age_s = getattr(args, "tf_max_age_s", 0.5)
    return lookup_T_tag0_base_tf(buffer, args.base_frame, args.tag0_frame, max_age_s)


def watchdog_is_stale(last_command_time: float, now: float, *, watchdog_s: float) -> bool:
    return (now - last_command_time) > watchdog_s


def apply_watchdog(
    qdot: np.ndarray,
    *,
    last_command_time: float,
    now: float,
    watchdog_s: float,
) -> np.ndarray:
    if watchdog_is_stale(last_command_time, now, watchdog_s=watchdog_s):
        return np.zeros_like(qdot)
    return qdot


class _GafropyKinematicsBridge:
    """Optional gafropy FK + Jacobian bridge for live RTDE speedJ."""

    def __init__(self, interface, task_space, joint_idx: np.ndarray):
        self._interface = interface
        self._task_space = task_space
        self._joint_idx = np.asarray(joint_idx, dtype=int)
        self._n_arm = len(self._joint_idx)

    @classmethod
    def try_create(cls, model_path: Path = GEODUDE_MODEL, task_space_name: str = TASK_SPACE_NAME):
        try:
            import gafropy as ga
            from gafropy import robot_interface as ri
        except ImportError:
            return None

        if not model_path.exists():
            return None

        config = ri.Configuration()
        config.model_file = str(model_path)
        config.type = "mujoco"
        interface = ri.RobotInterface(config)
        interface.initialize()
        task_space = interface.get_task_space(task_space_name)
        if task_space is None:
            interface.shutdown()
            return None
        joint_idx = np.asarray(task_space.get_joint_indices())
        return cls(interface, task_space, joint_idx)

    @property
    def task_space(self):
        return self._task_space

    @property
    def interface(self):
        return self._interface

    @property
    def joint_indices(self) -> np.ndarray:
        return self._joint_idx

    def shutdown(self) -> None:
        try:
            self._interface.shutdown()
        except Exception:
            pass

    def sync_joints(self, q_arm: Sequence[float]) -> None:
        q_arm = np.asarray(q_arm, dtype=float).reshape(self._n_arm)
        state = self._interface.get_state()
        position = np.asarray(state.get_position(), dtype=float).copy()
        position[self._joint_idx] = q_arm
        setter = getattr(state, "set_position", None)
        if not callable(setter):
            raise RuntimeError("gafropy state has no set_position; cannot sync RTDE joints")
        setter(position)

    def compute_ee_motor(self):
        position = self._interface.get_state().get_position()[self._joint_idx]
        return self._task_space.compute_ee_motor(position)

    def compute_qdot(self, controller, *, velocity: np.ndarray | None = None) -> np.ndarray:
        import gafropy as ga

        state = self._interface.get_state()
        if velocity is None:
            velocity = np.zeros_like(state.get_velocity())
        task_space_state = self._task_space.compute_task_space_state(
            state.get_position()[self._joint_idx],
            velocity[self._joint_idx] if len(velocity) > max(self._joint_idx) else np.zeros(self._n_arm),
        )
        return np.asarray(
            controller.compute_control_command(task_space_state, state),
            dtype=float,
        ).reshape(self._n_arm)


def run_rtde_loop(
    args,
    field_bundle,
    vf,
    controller,
    state,
    *,
    keyboard=None,
) -> int:
    """Host loop at ``args.control_hz`` for RTDE backend."""
    from geodude_teleoperation.frames import warn_tip_offset_xy
    from geodude_teleoperation.keyboard import KeyboardTeleop
    from geodude_teleoperation.teleop_state import format_status_line

    pcloud, wos = field_bundle  # noqa: F841 — kept for API parity with plan

    tip_warning = warn_tip_offset_xy(np.asarray(args.tip_offset, dtype=float))
    if tip_warning is not None:
        print(tip_warning)

    kin = None if args.dry_run else _GafropyKinematicsBridge.try_create()
    kin_ok = kin is not None
    kin_warned = False
    tf_warned = False
    apriltag_warned = False

    tf_ctx = _RosTfContext() if (args.base_frame and args.tag0_frame) else None
    cam_ctx = _RosCameraContext(args) if getattr(args, "calib_meta", None) else None

    if not args.dry_run and kin is not None:
        import gafropy as ga

        controller.set_task_space(kin.task_space)
        controller.set_interface(kin.interface)
        home_ee = kin.compute_ee_motor()
        initial_target = vf.project(home_ee)
        objective = ga.control.SingleArmMotorControlObjective(kin.task_space, initial_target)
        controller.set_objective(objective)
    elif not args.dry_run:
        print(
            "warning: gafropy kinematics unavailable; live speedJ will stay zero "
            "(VF status from RTDE TCP still works)",
            file=sys.stderr,
        )
        kin_warned = True

    session = open(
        args.robot_ip,
        dry_run=args.dry_run,
        accel=args.rtde_accel,
        speedj_time=1.0 / args.control_hz,
    )

    pending_twist = {"twist": None}

    def on_enable() -> None:
        state.enable()

    def on_button(index: int) -> None:
        if index == 0:
            state.teleop_active = not state.teleop_active
        elif index == 1:
            state.vf_enabled = not state.vf_enabled
            controller.set_virtual_fixture(vf if state.vf_enabled else None)

    def on_twist(twist) -> None:
        pending_twist["twist"] = twist

    owns_keyboard = keyboard is None
    if owns_keyboard:
        keyboard = KeyboardTeleop(step=args.keyboard_step)
        keyboard.start(
            on_twist,
            on_button=on_button,
            on_enable=on_enable,
            enable_key=args.enable_key,
        )

    dt = 1.0 / args.control_hz
    watchdog_s = args.watchdog_ms / 1000.0
    last_command_time = time.monotonic()

    try:
        while keyboard._running.is_set():
            t0 = time.perf_counter()
            now = time.monotonic()

            if tf_ctx is not None:
                tf_ctx.spin_once()

            rtde_state = session.read_state()
            fault = bool(rtde_state["fault"])
            q = rtde_state["q"]

            tf_buffer = tf_ctx.buffer if tf_ctx is not None else None
            T_tag0_base, tf_ok, tf_age_ms = resolve_T_tag0_base(args, buffer=tf_buffer)
            if not tf_ok and not tf_warned:
                print(
                    "warning: T_tag0_base unavailable; refusing motion (speedJ(0))",
                    file=sys.stderr,
                )
                tf_warned = True

            tag_vs_tf_mm = None
            if state.mode == "calibrate":
                from geodude_teleoperation.tag_crosscheck import resolve_tag_vs_tf_mm

                T_tag_tf, _ = lookup_live_T_tag0_base(args, buffer=tf_buffer)
                rgb, camera_matrix, dist_coeffs = (
                    cam_ctx.snapshot() if cam_ctx is not None else (None, None, None)
                )
                tag_vs_tf_mm = resolve_tag_vs_tf_mm(
                    args,
                    T_tag_tf=T_tag_tf,
                    tcp_pose=rtde_state["tcp_pose"],
                    rgb=rgb,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                )
                if (
                    getattr(args, "calib_meta", None)
                    and T_tag_tf is not None
                    and tag_vs_tf_mm is None
                    and not apriltag_warned
                ):
                    print(
                        "warning: AprilTag cross-check unavailable "
                        "(missing camera feed or tag detection)",
                        file=sys.stderr,
                    )
                    apriltag_warned = True

            if kin is not None:
                try:
                    kin.sync_joints(q)
                    ee = kin.compute_ee_motor()
                except Exception as exc:
                    if not kin_warned:
                        print(
                            f"warning: gafropy joint sync failed ({exc}); "
                            "refusing live motion",
                            file=sys.stderr,
                        )
                        kin_warned = True
                    kin_ok = False
                    ee = ee_motor_from_tcp_pose(rtde_state["tcp_pose"])
            else:
                ee = ee_motor_from_tcp_pose(rtde_state["tcp_pose"])

            vf.update_from_state(ee, T_tag0_base)

            if state.vf_enabled and state.motion_enabled and not args.dry_run and kin_ok:
                projected = vf.project(ee)
                controller.get_objective().set_target(projected)

            twist = pending_twist["twist"]
            fresh_command = False
            if twist is not None:
                pending_twist["twist"] = None
                if state.commands_allowed(
                    dry_run=args.dry_run, fault=fault, tf_ok=tf_ok
                ):
                    if kin_ok:
                        controller.update_command(twist)
                        fresh_command = True

            qdot = np.zeros(6, dtype=float)
            qdot_norm = 0.0
            allowed = state.commands_allowed(
                dry_run=args.dry_run, fault=fault, tf_ok=tf_ok
            )
            if allowed and not args.dry_run and kin_ok and session.control_open:
                try:
                    qdot = kin.compute_qdot(controller)
                    qdot = apply_watchdog(
                        qdot,
                        last_command_time=last_command_time,
                        now=now,
                        watchdog_s=watchdog_s,
                    )
                    speed = float(np.linalg.norm(qdot))
                    if speed > args.max_joint_velocity:
                        qdot = qdot * (args.max_joint_velocity / speed)
                    qdot_norm = float(np.linalg.norm(qdot))
                except Exception:
                    qdot = np.zeros(6, dtype=float)
                    qdot_norm = 0.0

            if session.control_open:
                try:
                    send_qdot = qdot if allowed and not args.dry_run else np.zeros(6)
                    send_qdot = apply_watchdog(
                        send_qdot,
                        last_command_time=last_command_time,
                        now=now,
                        watchdog_s=watchdog_s,
                    )
                    session.speedj(send_qdot, args.rtde_accel)
                    if fresh_command:
                        last_command_time = now
                except RuntimeError:
                    pass

            line = format_status_line(
                state,
                vf.status,
                qdot_norm=qdot_norm if not args.dry_run else 0.0,
                tf_age_ms=tf_age_ms,
                tag_vs_tf_mm=tag_vs_tf_mm,
            )
            print(line, flush=True)

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        if owns_keyboard:
            keyboard.stop()
        session.close()
        if kin is not None:
            kin.shutdown()
        if tf_ctx is not None:
            tf_ctx.close()
        if cam_ctx is not None:
            cam_ctx.close()

    return 0
