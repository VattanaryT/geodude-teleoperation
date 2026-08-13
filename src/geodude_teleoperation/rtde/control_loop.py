"""RTDE receive-only dry-run and live speedJ host loop for stringer teleop.

Kinematics:
- **Dry-run calibrate:** ``getActualTCPPose`` tracks the real arm (pendant OK);
  control interface stays closed.
- **Dry-run after Enter:** same free-space / DualLoop math as live; integrate
  ``q_sim`` and publish TCP to Polyscope (never ``speedJ``). Prefer numpy UR5e.
- **Live speedJ (preferred):** gafropy MuJoCo ``RobotInterface`` FK/Jacobian
  via ``_GafropyKinematicsBridge``.
- **Live speedJ (backup):** vendored UR5e FK/J + RTDE TCP via
  ``NumpyUrKinematicsBridge`` when gafropy is unavailable. Pendant TCP must
  match the tool; Jacobian is mapped to that TCP.
  If both fail, motion is refused (``speedJ(0)``) with a warning; TCP VF
  queries still run.

Live TF via ``lookup_T_tag0_base_tf`` when ROS tf2 is available; otherwise fall
back to an explicit ``--T-tag0-base`` file (warn once, ``tf_ok=True``). If live TF
is requested but unavailable and ``--T-tag0-base`` was not provided (implicit
identity), ``T`` is identity but ``tf_ok=False`` so motion is refused.
``tf_age_ms`` is ``nan`` when live lookup fails. Calibrate mode may show
``tag_vs_tf_mm`` when live TF and AprilTag chain inputs are both available
(enable stays manual).
"""

from __future__ import annotations

import atexit
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent.parent
GEODUDE_MODEL = ROOT / "assets" / "robots" / "geodude" / "geodude.yaml"
TASK_SPACE_NAME = "right_ee"

_shutdown = threading.Event()
_active_session: "RtdeTeleopSession | None" = None
_hooks_installed = False
_hooks_lock = threading.Lock()


def rtde_loop_should_continue(
    *,
    keyboard_running: bool,
    shutdown: bool,
    viz_open: bool | None,
) -> bool:
    """Host loop gate: keyboard quit, SIGINT/atexit, or Polyscope closed."""
    if not keyboard_running or shutdown:
        return False
    if viz_open is False:
        return False
    return True


def request_rtde_shutdown(session: "RtdeTeleopSession | None" = None) -> None:
    """Stop UR speedJ immediately (Ctrl+C / viz close / atexit)."""
    _shutdown.set()
    s = session if session is not None else _active_session
    if s is not None:
        try:
            s.stop_motion()
        except Exception:
            pass


def install_rtde_shutdown_hooks() -> None:
    """SIGINT/SIGTERM/atexit → speedStop so the arm does not keep the last q̇."""
    global _hooks_installed
    with _hooks_lock:
        if _hooks_installed:
            return
        _hooks_installed = True

        def _on_signal(signum, frame):
            request_rtde_shutdown()
            raise KeyboardInterrupt

        atexit.register(request_rtde_shutdown)
        try:
            # SIGINT → KeyboardInterrupt on the main thread (viz). SIGTERM for
            # kill. Do not override SIGINT here: pytest and Ctrl+C on the viz
            # thread use KeyboardInterrupt handlers that call request_rtde_shutdown.
            signal.signal(signal.SIGTERM, _on_signal)
        except ValueError:
            # Not the main thread — loop/viz still call request_rtde_shutdown.
            pass


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
        speedj_time: float = 1,
    ):
        self.ip = str(ip)
        self._receive = receive
        self._control = control
        self._dry_run = bool(dry_run)
        self._accel = float(accel)
        self._speedj_time = float(speedj_time)
        self._stopping = False

    @property
    def control_open(self) -> bool:
        return self._control is not None

    def read_state(self) -> dict[str, Any]:
        wrench = [0.0] * 6
        get_ft = getattr(self._receive, "getActualTCPForce", None)
        if callable(get_ft):
            try:
                wrench = [float(v) for v in get_ft()]
            except Exception:
                wrench = [0.0] * 6
        return {
            "q": [float(v) for v in self._receive.getActualQ()],
            "tcp_pose": [float(v) for v in self._receive.getActualTCPPose()],
            "tcp_force": wrench,
            "fault": _runtime_fault(self._receive),
        }

    def speedj(self, qdot: Sequence[float], accel: float) -> None:
        if self._dry_run or self._control is None:
            raise RuntimeError("speedj unavailable in dry-run (receive-only)")
        if self._stopping:
            return
        qdot_list = [float(v) for v in qdot]
        self._control.speedJ(qdot_list, float(accel), self._speedj_time)

    def stop_motion(self) -> None:
        """Leave continuous speedJ / DTC cleanly (call before disconnect).

        UR ``speedJ`` keeps the last q̇ until a new command. Host exit without
        ``speedStop`` leaves the arm moving.
        """
        self._stopping = True
        if self._dry_run or self._control is None:
            return
        accel = float(max(abs(self._accel), 1.0))
        for name, args in (
            ("speedStop", ()),
            ("speedStop", (accel,)),
            ("stopJ", ([accel] * 6,)),
            ("speedJ", ([0.0] * 6, accel, 0.5)),
        ):
            fn = getattr(self._control, name, None)
            if not callable(fn):
                continue
            try:
                fn(*args)
            except TypeError:
                continue
            except Exception:
                continue

    def close(self) -> None:
        global _active_session
        if _active_session is self:
            _active_session = None
        self.stop_motion()
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
    global _active_session
    RTDEControlInterface, RTDEReceiveInterface = _import_rtde()
    receive = RTDEReceiveInterface(str(ip))
    control = None if dry_run else RTDEControlInterface(str(ip))
    session = RtdeTeleopSession(
        ip,
        receive,
        control,
        dry_run=dry_run,
        accel=accel,
        speedj_time=speedj_time,
    )
    if not dry_run:
        _shutdown.clear()
        _active_session = session
        install_rtde_shutdown_hooks()
    return session


def ee_motor_from_tcp_pose(tcp_pose: Sequence[float]):
    """UR ``getActualTCPPose`` [x,y,z,rx,ry,rz] axis-angle → minimal EE motor."""
    from geodude_teleoperation.teleop.contact_wos_vf import _FallbackMotor

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


def _T_tag0_for_rtde_control(args, T_tag0_in_parent: np.ndarray) -> np.ndarray:
    """Map tag_0 pose into RTDE base when parent frame is vention (crop_meta / TF).

    Live TF with ``--base-frame right_base*`` is already in the controller frame.
    """
    parent = str(getattr(args, "base_frame", None) or "")
    if parent in (
        "right_base",
        "right_base_link",
        "right_base_link_inertia",
    ):
        return np.asarray(T_tag0_in_parent, dtype=float)
    if getattr(args, "no_rtde_base_vention", False):
        return np.asarray(T_tag0_in_parent, dtype=float)
    from geodude_teleoperation.frames import (
        compose_T_tag0_rtde,
        load_T_rtde_base_vention,
    )

    try:
        T_rv = load_T_rtde_base_vention(getattr(args, "T_rtde_base_vention", None))
    except FileNotFoundError:
        return np.asarray(T_tag0_in_parent, dtype=float)
    return compose_T_tag0_rtde(T_tag0_in_parent, T_rv)


def resolve_T_tag0_base(args, *, buffer=None) -> tuple[np.ndarray, bool, float]:
    """Resolve ``T_tag0`` for VF queries (RTDE / control base).

    Returns ``(T, tf_ok, tf_age_ms)``. ``T`` is always defined (never dangling).
    Fixed crop_meta / vention TF is composed with ``T_rtde_base_vention`` so
    RTDE TCP and the cloud share one frame.
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
            from geodude_teleoperation.teleop.object_offset import apply_args_object_offset

            T_live = apply_args_object_offset(np.asarray(T_live, dtype=float), args)
            T = _T_tag0_for_rtde_control(args, T_live)
            return T, True, float(age_ms or 0.0)
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
            return _T_tag0_for_rtde_control(args, T), False, tf_age_ms

        if not _live_tf_fallback_warned:
            print(
                "warning: live TF unavailable; using fixed T_tag0_base",
                file=sys.stderr,
            )
            _live_tf_fallback_warned = True

    from geodude_teleoperation.teleop.object_offset import apply_args_object_offset

    T = apply_args_object_offset(T, args)
    T = _T_tag0_for_rtde_control(args, T)
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


def should_refresh_watchdog(
    *,
    commands_allowed: bool,
    dry_run: bool,
    kin_ok: bool,
    control_open: bool,
    fresh_command: bool,
) -> bool:
    """Keep the host watchdog alive while actively servoing a pose target.

    Keyboard teleop is one-shot (sets ``T_ref``); the loop must keep tracking
    that target between keypresses. Refreshing only on keypress made each WASD
    tick re-open the gate and dump the same VF orientation snap toward
    ``R_field`` — looking like “every key rotates the same way.”

    Dry-run sim also refreshes: Polyscope integrates ``qdot`` between keys with
    no control interface open.
    """
    if fresh_command:
        return True
    if not (commands_allowed and kin_ok):
        return False
    if dry_run:
        return True
    return bool(control_open)


def should_run_dry_vf_query(now: float, last_t: float, period_s: float) -> bool:
    """Free-space dry-run: WoS only on a wall-clock period, never every WASD tick.

    Holding keys sets ``fresh_command`` each control tick. Querying on that flag
    made ``query_ms≈2s`` block the Polyscope tip every frame.
    """
    return (float(now) - float(last_t)) >= max(0.0, float(period_s))


def should_run_vf_query(
    *,
    dry_run: bool,
    sim_active: bool,
    now: float,
    last_query_t: float,
    dry_period_s: float,
    orient_locking: bool,
    has_lock_R_star: bool,
    awaiting_approve: bool = False,
    vf_frozen: bool = False,
) -> bool:
    """Whether to call ``vf.update_from_state`` this tick.

    Live Enter-lock: once ``R*`` is frozen, skip WoS/NN through lock, the
    post-settle hold, and VF freeze (``f``). A multi-100ms query blocks
    ``speedJ``, the watchdog zeros q̇, then catch-up looks like oscillation.
    Dry-run still uses ``dry_query_period_s``.
    """
    if has_lock_R_star and (orient_locking or awaiting_approve or vf_frozen):
        return False
    if dry_run and sim_active:
        return should_run_dry_vf_query(now, last_query_t, dry_period_s)
    return True


def flange_tcp_offset_matrix(q: Sequence[float], tcp_pose: Sequence[float]) -> np.ndarray:
    """Fixed flange→TCP SE(3) from a joint/TCP snapshot (for dry-run FK)."""
    from geodude_teleoperation.rtde.ur5e_kinematics import forward_kinematics
    from geodude_teleoperation.se3 import tcp_pose_to_matrix

    T_flange, _ = forward_kinematics(np.asarray(q, dtype=float).reshape(6))
    T_tcp = tcp_pose_to_matrix(tcp_pose)
    return np.linalg.inv(T_flange) @ T_tcp


def tcp_pose_from_q_and_offset(
    q: Sequence[float], T_flange_tcp: np.ndarray
) -> np.ndarray:
    """TCP pose [x,y,z,rx,ry,rz] from joints and a flange→TCP offset."""
    from geodude_teleoperation.rtde.ur5e_kinematics import forward_kinematics
    from geodude_teleoperation.se3 import matrix_to_tcp_pose

    T_flange, _ = forward_kinematics(np.asarray(q, dtype=float).reshape(6))
    return matrix_to_tcp_pose(T_flange @ np.asarray(T_flange_tcp, dtype=float))


def _bool_arg(args, name: str, default: bool) -> bool:
    val = getattr(args, name, default)
    return bool(val) if isinstance(val, bool) else bool(default)


def _float_arg(args, name: str, default: float) -> float:
    val = getattr(args, name, default)
    try:
        x = float(val)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return x


def vf_lock_rotation(vf) -> np.ndarray | None:
    """Orientation used for Enter lock / ``snap_T_ref_orientation_to_vf``."""
    if vf is None:
        return None
    if hasattr(vf, "motion_basis"):
        try:
            return np.asarray(vf.motion_basis(), dtype=float).reshape(3, 3)
        except Exception:
            pass
    st = getattr(vf, "status", None) or {}
    raw = st.get("R_field_raw_smoothed")
    if raw is None:
        raw = st.get("R_field_base")
    if raw is None:
        return None
    return np.asarray(raw, dtype=float).reshape(3, 3)


def snap_T_ref_orientation_to_vf(kin, vf) -> bool:
    """Replace ``kin._T_ref`` rotation with VF ``R_field`` / ``R_motion``.

    Dry-run: instant viz snap (no ``speedJ``). Live Enter-lock uses the same
    snap with startup vel/accel so the robot eases in instead of jerking.
    Pure linear WASD still does not snap on its own.
    """
    T = getattr(kin, "_T_ref", None)
    if T is None:
        return False
    R = vf_lock_rotation(vf)
    if R is None:
        return False
    T_new = np.asarray(T, dtype=float).reshape(4, 4).copy()
    T_new[:3, :3] = R
    kin._T_ref = T_new
    return True


def resolve_live_kinematics(line_tracking=None):
    """Prefer gafropy when it imports cleanly; else numpy UR5e backup.

    ``import gafropy`` SIGILLs on some CPUs, so we probe in a subprocess first
    (see ``gafropy_compat``). Phase 1.5 line tracking prefers numpy either way.
    """
    from geodude_teleoperation.gafropy_compat import gafropy_importable
    from geodude_teleoperation.rtde.numpy_kinematics_bridge import NumpyUrKinematicsBridge

    prefer_numpy = bool(line_tracking is not None and getattr(line_tracking, "enabled", False))
    ga_ok = gafropy_importable()

    def _try_numpy():
        kin = NumpyUrKinematicsBridge.try_create(line_tracking=line_tracking)
        if kin is not None:
            print(
                "kinematics: numpy UR5e backup (RTDE TCP + vendored FK/J)",
                flush=True,
            )
        return kin

    def _try_gafropy():
        if not ga_ok:
            return None
        kin = _GafropyKinematicsBridge.try_create()
        if kin is not None:
            print("kinematics: gafropy (MuJoCo FK/J)", flush=True)
        return kin

    if prefer_numpy or not ga_ok:
        kin = _try_numpy()
        if kin is not None:
            return kin
        return _try_gafropy()

    kin = _try_gafropy()
    if kin is not None:
        return kin
    return _try_numpy()


def _is_numpy_kinematics(kin) -> bool:
    return getattr(kin, "backend_name", None) == "numpy"


class _GafropyKinematicsBridge:
    """Optional gafropy FK + Jacobian bridge for live RTDE speedJ."""

    backend_name = "gafropy"

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
    live_viz=None,
) -> int:
    """Host loop at ``args.control_hz`` for RTDE backend."""
    from geodude_teleoperation.frames import warn_tip_offset_xy
    from geodude_teleoperation.teleop.motion_trace import MotionTraceLogger, save_and_maybe_plot
    from geodude_teleoperation.teleop.teleop_state import format_status_line, should_emit_status

    pcloud, wos = field_bundle  # noqa: F841 — kept for API parity with plan

    tip_warning = warn_tip_offset_xy(np.asarray(args.tip_offset, dtype=float))
    if tip_warning is not None:
        print(tip_warning)

    from geodude_teleoperation.teleop.direct_torque import (
        DirectTorqueBridge,
        print_control_rate_report,
        print_dtc_rate_verification,
        polyscope_supports_dtc,
        query_polyscope_version,
        verify_control_hz,
        verify_direct_torque_rate,
    )
    from geodude_teleoperation.teleop.line_tracking import config_from_args as line_cfg_from_args
    from geodude_teleoperation.teleop.line_tracking import vf_R_field
    from geodude_teleoperation.teleop.tactile_impedance import (
        TactileAgent,
        tactile_config_from_args,
    )
    from geodude_teleoperation.rtde.orient_lock import (
        EnableOrientLock,
        MotionLimits,
        apply_orient_lock_reference,
        blend_startup_motion_limits,
        orientation_error_deg,
        twist_linear_only,
    )
    from geodude_teleoperation.teleop.contact_teleop import (
        ContactTeleopBridge,
        config_from_args as contact_cfg_from_args,
        vf_R_field_or_none,
        vf_approach_R,
    )
    from geodude_teleoperation.teleop.keyboard import (
        contact_field_linear_from_twist_coeffs,
        is_contact_tangent_key,
    )
    from geodude_teleoperation.teleop.spacemouse import (
        contact_tangent_linear_from_spacemouse,
    )
    from geodude_teleoperation.se3 import map_jacobian_to_point, tcp_pose_to_matrix
    from geodude_teleoperation.rtde.ur5e_kinematics import compute_jacobian, forward_kinematics

    in_contact = bool(getattr(args, "in_contact", False))
    contact_bridge: ContactTeleopBridge | None = None
    if in_contact:
        contact_bridge = ContactTeleopBridge(contact_cfg_from_args(args))
        print(
            f"In-contact DualLoop ON (Fd={contact_bridge.cfg.desired_force_n:g} N, "
            f"force_sign={contact_bridge.cfg.force_sign:g}, "
            f"approach≤{contact_bridge.cfg.approach_v_max_m_s*1e3:.0f} mm/s, "
            f"align_first={int(contact_bridge.cfg.align_first)}, "
            "tangent keys w/s a/d, f=freeze n*)",
            flush=True,
        )

    line_cfg = line_cfg_from_args(args)
    tactile_cfg = tactile_config_from_args(args)
    if in_contact:
        # Contact path owns normal force; do not layer line-tracking/tactile FF.
        line_cfg.enabled = False
        tactile_cfg.enabled = False
    if tactile_cfg.enabled and not line_cfg.enabled:
        line_cfg.enabled = True
        print(
            "info: --tactile enables line tracking (free normal) automatically",
            flush=True,
        )
    if line_cfg.enabled:
        print(
            f"Phase 1.5 line tracking ON "
            f"(soft_standoff={int(line_cfg.standoff_enabled)} "
            f"d*={line_cfg.standoff_distance_m:.4f} m k={line_cfg.standoff_gain:g})",
            flush=True,
        )
    if tactile_cfg.enabled:
        print(
            f"Phase 3 tactile ON (F_n*={tactile_cfg.desired_normal_force_n:g} N, speedJ FF)",
            flush=True,
        )

    rate_report = verify_control_hz(float(args.control_hz))
    print_control_rate_report(rate_report, prefix="control-rate")
    verify_500 = getattr(args, "verify_500hz", False)
    poly_ver = getattr(args, "polyscope_version", None)
    poly_ok = None
    dtc_live_ok = True
    if verify_500 is True:
        rate_500 = verify_control_hz(500.0, target_hz=500.0, tol=1.0)
        print_control_rate_report(rate_500, prefix="verify-500hz")
        if not (isinstance(poly_ver, str) and poly_ver):
            poly_ver = query_polyscope_version(str(args.robot_ip))
            if poly_ver:
                print(f"[dtc] PolyScope from dashboard: {poly_ver}", flush=True)
        if isinstance(poly_ver, str) and poly_ver:
            poly_ok = polyscope_supports_dtc(poly_ver)
            print(
                f"[dtc] PolyScope {poly_ver}: "
                f"{'supports' if poly_ok else 'DOES NOT support'} Direct Torque "
                f"(need ≥5.23 or ≥10.10)",
                flush=True,
            )
        dtc_bridge = DirectTorqueBridge(enabled=False)
        print(dtc_bridge.urscript_snippet(), flush=True)

    tactile_agent = TactileAgent(tactile_cfg) if tactile_cfg.enabled else None

    def _flag(name: str, default: bool = False) -> bool:
        v = getattr(args, name, default)
        return v if isinstance(v, bool) else default

    diag_hold_zero = _flag("diag_hold_zero", False)
    # Standalone --log-motion path only when not recording MCAP (MCAP session
    # owns CSV/PNG next to each rollout .mcap).
    record_mcap = _flag("record_mcap", False)
    log_path = getattr(args, "log_motion", None)
    if record_mcap:
        log_path = None
    elif not isinstance(log_path, (str, type(None))) and not hasattr(
        log_path, "__fspath__"
    ):
        log_path = None
    plot_motion = _flag("plot_motion", False)
    motion_log = MotionTraceLogger() if log_path is not None else None

    # Dry-run sim needs the same FK/J path as live; prefer numpy (no MuJoCo).
    if args.dry_run:
        from geodude_teleoperation.rtde.numpy_kinematics_bridge import (
            NumpyUrKinematicsBridge,
        )

        kin = NumpyUrKinematicsBridge.try_create(line_tracking=line_cfg)
        if kin is not None:
            print(
                "kinematics: numpy UR5e backup (dry-run Polyscope sim)",
                flush=True,
            )
        else:
            print(
                "warning: numpy kinematics unavailable; dry-run sim tip will not move",
                file=sys.stderr,
            )
    else:
        kin = resolve_live_kinematics(line_tracking=line_cfg)
    kin_ok = kin is not None
    kin_warned = False
    tf_warned = False
    apriltag_warned = False
    use_numpy_kin = _is_numpy_kinematics(kin)
    if use_numpy_kin and kin is not None and hasattr(kin, "set_line_tracking"):
        kin.set_line_tracking(line_cfg)

    tf_ctx = _RosTfContext() if (args.base_frame and args.tag0_frame) else None
    cam_ctx = _RosCameraContext(args) if getattr(args, "calib_meta", None) else None

    if not args.dry_run and kin is not None and not use_numpy_kin:
        import gafropy as ga

        controller.set_task_space(kin.task_space)
        controller.set_interface(kin.interface)
        home_ee = kin.compute_ee_motor()
        # Hold current EE on startup — do not project onto live R_field here.
        # Idle orientation lock caused saturated q̇ / wobble after Enter with no keys;
        # twists still project inside TeleoperationController.update_command.
        initial_target = home_ee
        objective = ga.control.SingleArmMotorControlObjective(kin.task_space, initial_target)
        controller.set_objective(objective)
    elif not args.dry_run and kin is None:
        print(
            "warning: no kinematics bridge (gafropy and numpy backup failed); "
            "live speedJ will stay zero (VF status from RTDE TCP still works)",
            file=sys.stderr,
        )
        kin_warned = True

    session = open(
        args.robot_ip,
        dry_run=args.dry_run,
        accel=args.rtde_accel,
        speedj_time=1.0 / args.control_hz,
    )

    if verify_500 is True:
        if args.dry_run or not session.control_open:
            print(
                "[dtc-live] SKIP: need live RTDE control "
                "(omit --dry-run) to command directTorque",
                flush=True,
            )
            dtc_live_ok = False
        else:
            dur = float(getattr(args, "dtc_verify_seconds", 2.0))
            print(
                f"[dtc-live] commanding directTorque([0]*6) for {dur:g}s @ 500 Hz…",
                flush=True,
            )
            result = verify_direct_torque_rate(
                session._control,
                duration_s=dur,
                target_hz=500.0,
                friction_comp=True,
            )
            if isinstance(poly_ver, str) and poly_ver:
                result.polyscope_version = poly_ver
                result.polyscope_ok = poly_ok
            print_dtc_rate_verification(result)
            dtc_live_ok = bool(result.ok)
        if _flag("verify_500hz_exit", False):
            session.close()
            if kin is not None:
                kin.shutdown()
            if tf_ctx is not None:
                tf_ctx.close()
            if cam_ctx is not None:
                cam_ctx.close()
            return 0 if dtc_live_ok or args.dry_run else 1

    pending_twist = {"twist": None, "key": ""}
    pending_contact_arm = {"arm": False}
    sm_twist_t = {"t": None}
    mcap = None
    try:
        from geodude_teleoperation.teleop.teleop_mcap_recorder import recorder_from_args

        mcap = recorder_from_args(args)
    except Exception as exc:
        print(f"warning: record-mcap disabled ({exc})", file=sys.stderr)
        mcap = None
    prev_motion_enabled = False
    orient_lock = EnableOrientLock()
    prev_orient_locking = False
    prev_awaiting_approve = False
    vf_freeze_on = {"v": False}
    general_limits = MotionLimits(
        max_joint_velocity=float(args.max_joint_velocity),
        rtde_accel=float(args.rtde_accel),
        max_linear_speed=_float_arg(args, "max_linear_speed", 0.05),
        max_angular_speed=_float_arg(args, "max_angular_speed", 0.4),
    )
    startup_limits = MotionLimits(
        max_joint_velocity=_float_arg(
            args, "startup_max_joint_velocity", general_limits.max_joint_velocity
        ),
        rtde_accel=_float_arg(args, "startup_rtde_accel", general_limits.rtde_accel),
        max_linear_speed=general_limits.max_linear_speed,
        max_angular_speed=_float_arg(
            args, "startup_max_angular_speed", general_limits.max_angular_speed
        ),
    )
    startup_ramp_s = _float_arg(args, "startup_ramp_s", 0.0)
    orient_lock_tol_deg = _float_arg(args, "orient_lock_tol_deg", 2.0)
    orient_lock_on_enable = _bool_arg(args, "orient_lock_on_enable", False)

    def _apply_vf_frozen(frozen: bool) -> None:
        vf_freeze_on["v"] = bool(frozen)
        orient_lock.vf_frozen = bool(frozen)
        setter = getattr(vf, "set_frozen", None)
        if callable(setter):
            setter(bool(frozen))

    def on_enable() -> None:
        if (
            not args.dry_run
            and orient_lock_on_enable
            and contact_bridge is None
            and not diag_hold_zero
            and orient_lock.enabled_at is not None
        ):
            if not orient_lock.settled:
                print(
                    "info: orientation lock still running — wait to settle, "
                    "then Enter again to approve teleop",
                    flush=True,
                )
                return
            if not orient_lock.approved:
                orient_lock.approve()
                _apply_vf_frozen(True)
                print(
                    "info: teleop approved — VF frozen "
                    "(press f to unfreeze queries)",
                    flush=True,
                )
                return
            print("info: teleop already approved", flush=True)
            return
        state.enable()
        if (
            not args.dry_run
            and orient_lock_on_enable
            and contact_bridge is None
            and not diag_hold_zero
        ):
            orient_lock.on_enable(time.monotonic())
            _apply_vf_frozen(True)
            print(
                "info: orientation lock in-place → frozen R_field "
                f"(startup q̇≤{startup_limits.max_joint_velocity:g} rad/s, "
                f"accel≤{startup_limits.rtde_accel:g} rad/s², "
                f"ramp={startup_ramp_s:g}s, tol={orient_lock_tol_deg:g}deg); "
                "press Enter again after settle to approve teleop",
                flush=True,
            )
        if args.dry_run:
            print(
                "info: motion enabled (Enter) — teleop_active=1; "
                "dry_run=1 (Polyscope sim; robot not commanded); "
                f"vf={int(bool(state.vf_enabled))}",
                flush=True,
            )
        else:
            print(
                "info: motion enabled (Enter) — teleop_active=1; "
                f"dry_run=0; vf={int(bool(state.vf_enabled))}",
                flush=True,
            )
        if contact_bridge is not None:
            pending_contact_arm["arm"] = True

    def on_button(index: int) -> None:
        if index == 0:
            state.teleop_active = not state.teleop_active
        elif index == 1:
            if diag_hold_zero:
                return
            if contact_bridge is not None:
                return
            state.vf_enabled = not state.vf_enabled
            controller.set_virtual_fixture(vf if state.vf_enabled else None)
        elif index == 2:
            if contact_bridge is not None:
                frozen = contact_bridge.toggle_freeze()
                print(f"info: n* freeze={'ON' if frozen else 'OFF'}", flush=True)
                return
            if not state.motion_enabled:
                return
            if orient_lock.enabled_at is not None:
                frozen = orient_lock.toggle_vf_freeze()
            else:
                frozen = not vf_freeze_on["v"]
            _apply_vf_frozen(frozen)
            print(f"info: VF freeze={'ON' if frozen else 'OFF'}", flush=True)

    def on_twist(twist, *, source: str = "keyboard") -> None:
        if diag_hold_zero:
            return
        if contact_bridge is not None:
            if source == "spacemouse":
                coeffs = None
                if hasattr(twist, "to_array"):
                    coeffs = twist.to_array()
                elif hasattr(twist, "get_angular") and hasattr(twist, "get_linear"):
                    coeffs = np.concatenate(
                        [
                            np.asarray(twist.get_angular()),
                            np.asarray(twist.get_linear()),
                        ]
                    )
                if coeffs is None:
                    return
                lin = contact_tangent_linear_from_spacemouse(coeffs)
                if float(np.linalg.norm(lin[:2])) < 1e-12:
                    return
                pending_twist["twist"] = lin
                pending_twist["key"] = ""
                sm_twist_t["t"] = time.monotonic()
                return
            key = getattr(keyboard, "last_key", "") or ""
            if not is_contact_tangent_key(key):
                return
            coeffs = getattr(keyboard, "last_twist_coeffs", None)
            if coeffs is None and hasattr(twist, "to_array"):
                coeffs = twist.to_array()
            elif coeffs is None:
                coeffs = np.concatenate(
                    [np.asarray(twist.get_angular()), np.asarray(twist.get_linear())]
                )
            pending_twist["twist"] = contact_field_linear_from_twist_coeffs(coeffs)
            pending_twist["key"] = key
            return
        pending_twist["twist"] = twist
        pending_twist["key"] = (
            (getattr(keyboard, "last_key", "") or "") if source == "keyboard" else ""
        )
        if source == "spacemouse":
            sm_twist_t["t"] = time.monotonic()

    from geodude_teleoperation.teleop.inputs import (
        start_teleop_inputs,
        stop_teleop_inputs,
    )

    keyboard, spacemouse = start_teleop_inputs(
        args,
        on_twist=lambda tw: on_twist(tw, source="keyboard"),
        on_button=on_button,
        on_enable=on_enable,
        keyboard=keyboard,
        on_spacemouse_twist=lambda tw: on_twist(tw, source="spacemouse"),
    )

    dt = 1.0 / args.control_hz
    watchdog_s = args.watchdog_ms / 1000.0
    last_command_time = time.monotonic()
    last_status_t = float("-inf")
    status_hz = float(getattr(args, "status_hz", 5.0))
    # Dry-run Polyscope sim: after Enter, integrate q_sim (never speedJ).
    sim_active = False
    q_sim: np.ndarray | None = None
    tcp_sim: np.ndarray | None = None
    T_flange_tcp: np.ndarray | None = None
    last_dry_vf_query_t = float("-inf")
    dry_vf_period_s = float(getattr(args, "vf_dry_query_period_s", 1.0))

    try:
        while rtde_loop_should_continue(
            keyboard_running=keyboard._running.is_set(),
            shutdown=_shutdown.is_set(),
            viz_open=None if live_viz is None else live_viz.viz_open,
        ):
            t0 = time.perf_counter()
            now = time.monotonic()

            if live_viz is not None:
                for key in live_viz.drain_teleop_keys():
                    keyboard.handle_key(key)
                if not keyboard._running.is_set():
                    break

            if tf_ctx is not None:
                tf_ctx.spin_once()

            rtde_state = session.read_state()
            fault = bool(rtde_state["fault"])
            live_q = np.asarray(rtde_state["q"], dtype=float).reshape(6)
            live_tcp = np.asarray(rtde_state["tcp_pose"], dtype=float).reshape(6)

            if mcap is not None and state.motion_enabled and not prev_motion_enabled:
                mcap.begin_rollout(
                    {
                        "robot_ip": getattr(args, "robot_ip", None),
                        "backend": "rtde",
                        "diag_fixed_orient": bool(
                            getattr(args, "diag_fixed_orient", False)
                        ),
                    }
                )
            prev_motion_enabled = state.motion_enabled

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
                    tcp_pose=live_tcp,
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

            # Until Enter: track live RTDE. After Enter in dry-run: own q/tcp.
            if args.dry_run and state.motion_enabled:
                if not sim_active:
                    q_sim = live_q.copy()
                    tcp_sim = live_tcp.copy()
                    T_flange_tcp = flange_tcp_offset_matrix(q_sim, tcp_sim)
                    sim_active = True
                    print(
                        "info: dry-run sim TCP locked; WASD moves Polyscope tip only "
                        f"(vf={int(bool(state.vf_enabled))})",
                        flush=True,
                    )
                q = np.asarray(q_sim, dtype=float).reshape(6)
                tcp_pose = np.asarray(tcp_sim, dtype=float).reshape(6)
            else:
                q = live_q
                tcp_pose = live_tcp

            math_ok = state.control_math_allowed(fault=fault, tf_ok=tf_ok)
            send_ok = state.commands_allowed(
                dry_run=args.dry_run, fault=fault, tf_ok=tf_ok
            )

            if kin is not None and use_numpy_kin:
                try:
                    kin.sync_joints(q)
                    kin.set_tcp_pose(tcp_pose)
                    kin_ok = True
                except Exception as exc:
                    if not kin_warned:
                        print(
                            f"warning: numpy kinematics sync failed ({exc}); "
                            "refusing live motion",
                            file=sys.stderr,
                        )
                        kin_warned = True
                    kin_ok = False
                ee = ee_motor_from_tcp_pose(tcp_pose)
            elif kin is not None:
                try:
                    kin.sync_joints(q)
                    ee = kin.compute_ee_motor()
                    kin_ok = True
                except Exception as exc:
                    if not kin_warned:
                        print(
                            f"warning: gafropy joint sync failed ({exc}); "
                            "refusing live motion",
                            file=sys.stderr,
                        )
                        kin_warned = True
                    kin_ok = False
                    ee = ee_motor_from_tcp_pose(tcp_pose)
            else:
                ee = ee_motor_from_tcp_pose(tcp_pose)

            # Apply keyboard twists BEFORE the slow WoS/VF query so dry-run
            # Polyscope tip updates even when query_ms is multi-second.
            motion_limits = general_limits
            orient_locking = False
            hold_still = False
            if (
                not args.dry_run
                and orient_lock.enabled_at is not None
                and contact_bridge is None
                and not diag_hold_zero
                and state.vf_enabled
                and state.motion_enabled
            ):
                R_star = orient_lock.capture_R_star(vf_lock_rotation(vf))
                if R_star is not None:
                    ang_err = orientation_error_deg(
                        tcp_pose_to_matrix(tcp_pose)[:3, :3], R_star
                    )
                    orient_locking = orient_lock.locking(
                        ang_err, orient_lock_tol_deg
                    )
                else:
                    orient_locking = not orient_lock.settled
                hold_still = orient_lock.motion_hold()
                if orient_lock.awaiting_approve and not prev_awaiting_approve:
                    print(
                        "info: orientation lock settled — press Enter again "
                        "to approve teleop (VF stays frozen)",
                        flush=True,
                    )
                prev_awaiting_approve = orient_lock.awaiting_approve
                if prev_orient_locking and not orient_locking and orient_lock.approved:
                    print(
                        "info: orientation lock settled; using general UR limits "
                        f"(q̇≤{general_limits.max_joint_velocity:g} rad/s, "
                        f"accel≤{general_limits.rtde_accel:g} rad/s²)",
                        flush=True,
                    )
                prev_orient_locking = orient_locking
                motion_limits = blend_startup_motion_limits(
                    t_since_enable=orient_lock.t_since_enable(now),
                    ramp_s=startup_ramp_s,
                    startup=startup_limits,
                    general=general_limits,
                    locking=orient_locking or hold_still,
                )
                if orient_lock.R_star is not None:
                    _apply_vf_frozen(
                        hold_still or orient_locking or orient_lock.vf_frozen
                    )

            if (
                contact_bridge is None
                and not diag_hold_zero
                and state.motion_enabled
                and kin_ok
                and use_numpy_kin
            ):
                kin.ensure_reference_from_tcp()
                # Dry-run: align EE to live R_field immediately (safe — no speedJ).
                # Live Enter-lock: freeze R* and pin T_ref to current TCP (in-place).
                if args.dry_run and sim_active and state.vf_enabled:
                    snap_T_ref_orientation_to_vf(kin, vf)
                elif (
                    (orient_locking or hold_still)
                    and orient_lock.R_star is not None
                ):
                    apply_orient_lock_reference(kin, tcp_pose, orient_lock.R_star)

            twist = pending_twist["twist"]
            key_this_tick = ""
            fresh_command = False

            def _apply_twist(twist_obj, *, key: str = "") -> None:
                nonlocal fresh_command, key_this_tick
                if orient_locking or hold_still:
                    twist_obj = twist_linear_only(twist_obj)
                R_prev = vf_R_field_or_none(vf)
                if R_prev is None:
                    R_prev = vf_R_field(vf)
                if contact_bridge is not None and R_prev is not None:
                    n_app = None
                    if contact_bridge._R_star is not None:
                        n_app = contact_bridge._R_star[:, 2]
                    else:
                        n_app = np.asarray(R_prev, dtype=float).reshape(3, 3)[:, 2]
                    contact_bridge.push_tangent_field_linear(
                        twist_obj, R_prev, n_approach=n_app
                    )
                    fresh_command = True
                    if key:
                        key_this_tick = key
                elif kin_ok:
                    if use_numpy_kin:
                        kin.update_command(
                            twist_obj,
                            vf=vf if state.vf_enabled else None,
                            min_scale=getattr(
                                controller, "_min_twist_scaling_factor", 1.0
                            ),
                            max_scale=getattr(
                                controller, "_max_twist_scaling_factor", 2.0
                            ),
                            key=key or None,
                        )
                        if args.dry_run and state.vf_enabled:
                            snap_T_ref_orientation_to_vf(kin, vf)
                    else:
                        controller.update_command(twist_obj)
                    fresh_command = True
                    if key:
                        key_this_tick = key

            if twist is not None:
                pending_twist["twist"] = None
                key_this_tick = str(pending_twist.get("key") or "")
                pending_twist["key"] = ""
                if math_ok and not orient_locking and not hold_still:
                    _apply_twist(twist, key=key_this_tick)

            # Polyscope held WASD/X/Y → one combined step every control tick
            # (ImGui key-repeat alone was ~1–2 Hz → felt frozen with 2 mm steps).
            if (
                live_viz is not None
                and math_ok
                and not diag_hold_zero
                and not orient_locking
                and not hold_still
                and (kin_ok or contact_bridge is not None)
            ):
                held = live_viz.held_teleop_keys()
                if held:
                    from geodude_teleoperation.teleop.keyboard import (
                        dispatch_key,
                        make_twist,
                    )

                    coeffs = np.zeros(6, dtype=float)
                    for ch in held:
                        kind, payload = dispatch_key(
                            ch,
                            step=float(args.keyboard_step),
                            enable_key=getattr(args, "enable_key", "\n"),
                        )
                        if kind == "twist":
                            coeffs = coeffs + np.asarray(payload, dtype=float).reshape(6)
                    if float(np.linalg.norm(coeffs)) > 1e-12:
                        _apply_twist(
                            make_twist(coeffs),
                            key="+".join(sorted(held)),
                        )

            # Free-space dry-run: tip IS the pose reference (instant WASD feedback).
            # Joint servo to T_ref is for live speedJ only; with query_ms≫dt the
            # old qdot·dt integrate looked frozen.
            if (
                args.dry_run
                and sim_active
                and math_ok
                and contact_bridge is None
                and use_numpy_kin
                and kin is not None
                and getattr(kin, "_T_ref", None) is not None
            ):
                from geodude_teleoperation.se3 import matrix_to_tcp_pose

                tcp_sim = matrix_to_tcp_pose(kin._T_ref)
                tcp_pose = tcp_sim
                q = np.asarray(q_sim, dtype=float).reshape(6)
                try:
                    kin.set_tcp_pose(tcp_pose)
                except Exception:
                    pass
                ee = ee_motor_from_tcp_pose(tcp_pose)
                if live_viz is not None:
                    st0 = vf.status
                    live_viz.publish(
                        tcp_pose,
                        R_field_tcp=st0.get("R_field_raw_smoothed"),
                        signed_d=st0.get("signed_d"),
                        in_contact=st0.get("in_contact"),
                    )

            # WoS/VF query is expensive (~seconds on dense clouds). In free-space
            # dry-run, throttle it so WASD→tip stays near control_hz. Live
            # Enter-lock skips once R* is frozen (query hitch → speedJ gap →
            # oscillation).
            run_vf_query = should_run_vf_query(
                dry_run=bool(args.dry_run),
                sim_active=bool(sim_active),
                now=now,
                last_query_t=last_dry_vf_query_t,
                dry_period_s=dry_vf_period_s,
                orient_locking=orient_locking,
                has_lock_R_star=orient_lock.R_star is not None,
                awaiting_approve=orient_lock.awaiting_approve,
                vf_frozen=bool(vf_freeze_on["v"] or orient_lock.vf_frozen)
                and orient_lock.R_star is not None,
            )
            if run_vf_query:
                vf.update_from_state(ee, T_tag0_base)
                last_dry_vf_query_t = now
                if (
                    args.dry_run
                    and sim_active
                    and contact_bridge is None
                    and state.vf_enabled
                    and use_numpy_kin
                    and kin is not None
                ):
                    snap_T_ref_orientation_to_vf(kin, vf)
                    from geodude_teleoperation.se3 import matrix_to_tcp_pose

                    if getattr(kin, "_T_ref", None) is not None:
                        tcp_sim = matrix_to_tcp_pose(kin._T_ref)
                        tcp_pose = tcp_sim
                        try:
                            kin.set_tcp_pose(tcp_pose)
                        except Exception:
                            pass
                        ee = ee_motor_from_tcp_pose(tcp_pose)
                elif (
                    (orient_locking or hold_still)
                    and orient_lock.R_star is not None
                    and use_numpy_kin
                    and kin is not None
                    and contact_bridge is None
                ):
                    apply_orient_lock_reference(kin, tcp_pose, orient_lock.R_star)

            T_tcp = tcp_pose_to_matrix(tcp_pose)
            R_tcp = T_tcp[:3, :3]
            p_tcp = T_tcp[:3, 3]
            # DualLoop n* = surface approach; WASD = teleop local frame (R_motion).
            R_approach = vf_approach_R(vf)
            if R_approach is None:
                R_approach = vf_R_field_or_none(vf)
            if R_approach is None:
                R_approach = vf_R_field(vf)
            R_local = vf_R_field_or_none(vf)
            if R_local is None:
                R_local = R_approach
            R_field = R_approach  # DualLoop / arm target frame

            if (
                contact_bridge is not None
                and pending_contact_arm["arm"]
                and R_field is not None
            ):
                pending_contact_arm["arm"] = False
                bias_samples = None
                ft = rtde_state.get("tcp_force")
                if ft is not None:
                    bias_samples = [np.asarray(ft, dtype=float).reshape(-1)[:3]]
                contact_bridge.arm(
                    p_tcp,
                    R_tcp,
                    R_field,
                    force_bias_samples=bias_samples,
                )
                print(
                    f"info: in-contact armed (Fd={contact_bridge.cfg.desired_force_n:g} N, "
                    f"force_sign={contact_bridge.cfg.force_sign:g}, "
                    f"n_freeze={int(contact_bridge.n_frozen)}, "
                    f"phase={contact_bridge.phase})",
                    flush=True,
                )

            if (
                tactile_agent is not None
                and use_numpy_kin
                and kin is not None
                and hasattr(kin, "set_extra_normal_ff")
            ):
                R = vf_R_field(vf)
                if R is not None and math_ok:
                    out = tactile_agent.update(
                        wrench_meas=rtde_state.get("tcp_force", [0.0] * 6),
                        R_field=R,
                        jacobian=None,
                        dt=dt,
                        mode="speedj",
                    )
                    kin.set_extra_normal_ff(out["vn_world"])
                else:
                    kin.set_extra_normal_ff(None)

            qdot = np.zeros(6, dtype=float)
            qdot_norm = 0.0
            control_diag: dict = {}
            # Active teleop keeps the watchdog alive between one-shot keypresses
            # (pose servo to T_ref / VF orientation lock). Refresh before any
            # apply_watchdog so a long calibrate dwell doesn't zero the first
            # motion tick after Enter.
            if should_refresh_watchdog(
                commands_allowed=math_ok,
                dry_run=bool(args.dry_run),
                kin_ok=kin_ok or contact_bridge is not None,
                control_open=bool(session.control_open),
                fresh_command=fresh_command
                or diag_hold_zero
                or (
                    contact_bridge is not None
                    and contact_bridge.armed
                    and math_ok
                ),
            ):
                last_command_time = now
            run_control = (
                not diag_hold_zero
                and math_ok
                and (bool(session.control_open) or bool(args.dry_run))
            )
            # Free-space dry-run already snapped tip to T_ref; skip joint chase.
            dry_run_free_snap = bool(
                args.dry_run and sim_active and contact_bridge is None
            )
            if run_control and not dry_run_free_snap:
                try:
                    if contact_bridge is not None and contact_bridge.armed and R_field is not None:
                        if kin is not None and use_numpy_kin:
                            kin.sync_joints(q)
                            if hasattr(kin, "set_tcp_pose"):
                                kin.set_tcp_pose(tcp_pose)
                        T_flange, _ = forward_kinematics(q)
                        J_flange = compute_jacobian(q)
                        J_tcp = map_jacobian_to_point(
                            J_flange, T_flange[:3, 3], p_tcp
                        )
                        ft = rtde_state.get("tcp_force", [0.0] * 6)
                        qdot = contact_bridge.step(
                            p_tcp=p_tcp,
                            R_tcp=R_tcp,
                            R_field=R_field,
                            force_base=np.asarray(ft, dtype=float).reshape(-1)[:3],
                            jacobian=J_tcp,
                            dt=dt,
                        )
                        control_diag = dict(contact_bridge.last_diag or {})
                        if float(control_diag.get("phase_changed", 0.0)) > 0.5:
                            ph = (
                                "align"
                                if float(control_diag.get("phase_align", 0.0)) > 0.5
                                else "seek"
                            )
                            print(
                                f"info: in-contact phase → {ph} "
                                f"(ang_z={control_diag.get('ang_z_deg', float('nan')):.1f}deg)",
                                flush=True,
                            )
                        kin_ok = True
                    elif kin_ok and use_numpy_kin:
                        qdot = kin.compute_qdot(
                            gain=float(getattr(controller, "_gain", 4.0)),
                            damping=float(getattr(controller, "_damping", 0.05)),
                            max_joint_velocity=float(motion_limits.max_joint_velocity),
                            vf=vf if state.vf_enabled else None,
                            max_linear_speed=float(motion_limits.max_linear_speed),
                            max_angular_speed=float(motion_limits.max_angular_speed),
                        )
                        control_diag = dict(getattr(kin, "last_control_diag", {}) or {})
                    elif kin_ok:
                        qdot = kin.compute_qdot(controller)
                    else:
                        qdot = np.zeros(6)
                    qdot = apply_watchdog(
                        qdot,
                        last_command_time=last_command_time,
                        now=now,
                        watchdog_s=watchdog_s,
                    )
                    speed = float(np.linalg.norm(qdot))
                    if speed > motion_limits.max_joint_velocity:
                        # Outer clamp (qdot_from_error already clamps; flag if still over).
                        control_diag["qdot_clipped"] = True
                        control_diag.setdefault("qdot_unclipped_norm", speed)
                        qdot = qdot * (motion_limits.max_joint_velocity / speed)
                    qdot_norm = float(np.linalg.norm(qdot))
                except Exception:
                    qdot = np.zeros(6, dtype=float)
                    qdot_norm = 0.0

            # In-contact dry-run: integrate with wall-clock dt (loop may be slow).
            if (
                args.dry_run
                and sim_active
                and math_ok
                and contact_bridge is not None
                and T_flange_tcp is not None
                and q_sim is not None
            ):
                wall_dt = float(min(max(time.perf_counter() - t0, dt), 0.1))
                q_sim = np.asarray(q_sim, dtype=float).reshape(6) + np.asarray(
                    qdot, dtype=float
                ).reshape(6) * wall_dt
                tcp_sim = tcp_pose_from_q_and_offset(q_sim, T_flange_tcp)
                q = q_sim
                tcp_pose = tcp_sim
                if kin is not None and use_numpy_kin:
                    try:
                        kin.sync_joints(q)
                        kin.set_tcp_pose(tcp_pose)
                    except Exception:
                        pass
                ee = ee_motor_from_tcp_pose(tcp_pose)
                T_tcp = tcp_pose_to_matrix(tcp_pose)
                p_tcp = T_tcp[:3, 3]
                R_tcp = T_tcp[:3, :3]

            if live_viz is not None:
                st = vf.status
                live_viz.publish(
                    tcp_pose,
                    R_field_tcp=st.get("R_field_raw_smoothed"),
                    signed_d=st.get("signed_d"),
                    in_contact=st.get("in_contact"),
                )

            send_qdot = np.zeros(6, dtype=float)
            if session.control_open:
                try:
                    if diag_hold_zero:
                        send_qdot = np.zeros(6)
                    else:
                        send_qdot = qdot if send_ok else np.zeros(6)
                        send_qdot = apply_watchdog(
                            send_qdot,
                            last_command_time=last_command_time,
                            now=now,
                            watchdog_s=watchdog_s,
                        )
                    session.speedj(send_qdot, motion_limits.rtde_accel)
                    qdot_norm = float(np.linalg.norm(send_qdot))
                except RuntimeError:
                    pass

            status = vf.status
            if should_emit_status(last_status_t, now, status_hz=status_hz):
                last_status_t = now
                line = format_status_line(
                    state,
                    status,
                    qdot_norm=qdot_norm,
                    tf_age_ms=tf_age_ms,
                    tag_vs_tf_mm=tag_vs_tf_mm,
                )
                if contact_bridge is not None and contact_bridge.armed:
                    d = contact_bridge.last_diag or {}
                    phase = "align" if float(d.get("phase_align", 0.0)) > 0.5 else "seek"
                    line = (
                        f"{line}  contact=1  phase={phase}  "
                        f"ang_z={d.get('ang_z_deg', float('nan')):.1f}deg  "
                        f"Fd={d.get('Fd_N', float('nan')):.2f}  "
                        f"Fs={d.get('Fs_N', float('nan')):.2f}  "
                        f"Fe={d.get('Fe_N', float('nan')):.2f}  "
                        f"n_freeze={int(d.get('n_freeze', 0))}"
                    )
                    if float(d.get("force_wrong_sign", 0.0)) > 0.5:
                        line += "  WARN=force_sign?"
                    if float(d.get("force_fault", 0.0)) > 0.5:
                        line += "  OVERLOAD=retract"
                print(line, flush=True)

            elapsed = time.perf_counter() - t0
            if live_viz is not None:
                sm_age = None
                if sm_twist_t["t"] is not None:
                    sm_age = (time.monotonic() - sm_twist_t["t"]) * 1000.0
                relock_s = float(status.get("dt_since_relock_s", float("nan")))
                relock_ms = (
                    relock_s * 1000.0 if np.isfinite(relock_s) else float("nan")
                )
                live_viz.publish_latency(
                    loop_ms=elapsed * 1000.0,
                    query_ms=float(status.get("query_ms", float("nan"))),
                    relock_age_ms=relock_ms,
                    relock_reason=str(status.get("relock_reason", "") or ""),
                    sm_age_ms=sm_age,
                    target_dt_ms=1000.0 / max(float(args.control_hz), 1e-6),
                )
            active_motion_log = motion_log
            if mcap is not None and getattr(mcap, "motion_log", None) is not None:
                active_motion_log = mcap.motion_log
            if active_motion_log is not None:
                from geodude_teleoperation.teleop.contact_wos_vf import twist_field_alignment_rad
                from geodude_teleoperation.se3 import matrix_to_tcp_pose

                cmd_dir = None
                scale_factor = float("nan")
                distance_m = float("nan")
                twist_lin = None
                T_ref = None
                qdot_raw = control_diag.get("qdot_raw")
                if use_numpy_kin and kin is not None:
                    cmd_dir = getattr(kin, "last_cmd_dir_world", None)
                    scale_factor = float(getattr(kin, "last_scale_factor", float("nan")))
                    distance_m = float(getattr(kin, "last_distance_m", float("nan")))
                    twist_lin = getattr(kin, "last_twist_linear_field", None)
                    T_ref = getattr(kin, "_T_ref", None)
                align_field = twist_field_alignment_rad(
                    cmd_dir, status.get("R_field_raw_smoothed")
                )
                align_motion = twist_field_alignment_rad(
                    cmd_dir, status.get("R_motion_base")
                )
                ref_pose = None
                if T_ref is not None:
                    ref_pose = matrix_to_tcp_pose(T_ref)
                twist_lin_log = (
                    twist_lin if key_this_tick and twist_lin is not None else None
                )
                contact_kw = {}
                if contact_bridge is not None and contact_bridge.armed:
                    d = contact_bridge.last_diag or {}
                    contact_kw = {
                        "Fd_N": float(d.get("Fd_N", float("nan"))),
                        "Fs_N": float(d.get("Fs_N", float("nan"))),
                        "Fe_N": float(d.get("Fe_N", float("nan"))),
                        "n_freeze": float(d.get("n_freeze", float("nan"))),
                        "force_sign": float(d.get("force_sign", float("nan"))),
                        "k_ori": float(d.get("k_ori", float("nan"))),
                        "ang_z_deg": float(d.get("ang_z_deg", float("nan"))),
                        "phase_align": float(d.get("phase_align", float("nan"))),
                        "force_enable": float(d.get("force_enable", float("nan"))),
                    }
                    if key_this_tick and twist_lin_log is None:
                        twist_lin_log = np.asarray(
                            contact_bridge.last_tangent_field, dtype=float
                        ).reshape(3)
                active_motion_log.append(
                    t_mono=now,
                    qdot=send_qdot if not args.dry_run else qdot,
                    qdot_raw=qdot_raw,
                    R_motion=(
                        status.get("R_motion_base")
                        if status.get("R_motion_base") is not None
                        else status.get("R_field_base")
                    ),
                    relocked=bool(status.get("relocked", False)),
                    query_ms=float(status.get("query_ms", 0.0)),
                    loop_ms=elapsed * 1000.0,
                    qdot_unclipped_norm=float(
                        control_diag.get("qdot_unclipped_norm", float("nan"))
                    ),
                    qdot_clipped=bool(control_diag.get("qdot_clipped", False)),
                    tcp_pose=tcp_pose,
                    ref_pose=ref_pose,
                    err_pos_m=float(control_diag.get("err_pos_m", float("nan"))),
                    err_rot_rad=float(control_diag.get("err_rot_rad", float("nan"))),
                    err_norm=float(control_diag.get("err_norm", float("nan"))),
                    key=key_this_tick or None,
                    twist_linear=twist_lin_log,
                    scale_factor=scale_factor,
                    distance_m=distance_m
                    if np.isfinite(distance_m)
                    else float(status.get("signed_d", float("nan"))),
                    twist_align_rad=align_field,
                    twist_align_motion_rad=align_motion,
                    tip_delta_m=float(status.get("tip_delta_m", float("nan"))),
                    dt_since_relock_s=float(
                        status.get("dt_since_relock_s", float("nan"))
                    ),
                    relock_reason=str(status.get("relock_reason", "hold")),
                    **contact_kw,
                )

            if mcap is not None and mcap.should_sample(now):
                from geodude_teleoperation.teleop.motion_trace import key_wasdxy_pulses
                from geodude_teleoperation.se3 import matrix_to_tcp_pose

                ref_pose = None
                R_motion = status.get("R_motion_base")
                twist_lin = None
                if use_numpy_kin and kin is not None:
                    T_ref = getattr(kin, "_T_ref", None)
                    if T_ref is not None:
                        ref_pose = matrix_to_tcp_pose(T_ref)
                    if key_this_tick:
                        twist_lin = getattr(kin, "last_twist_linear_field", None)
                st_pack = dict(status)
                st_pack["tf_age_ms"] = float(tf_age_ms)
                st_pack["motion_enabled"] = bool(state.motion_enabled)
                st_pack["teleop_active"] = bool(state.teleop_active)
                st_pack["vf_enabled"] = bool(state.vf_enabled)
                mcap.record_tick(
                    now,
                    tcp_pose=tcp_pose,
                    ref_pose=ref_pose,
                    wrench=rtde_state.get("tcp_force"),
                    qdot=send_qdot if not args.dry_run else qdot,
                    qdot_raw=control_diag.get("qdot_raw"),
                    status=st_pack,
                    R_motion=R_motion,
                    key_pulses=key_wasdxy_pulses(key_this_tick or None),
                    twist_linear=twist_lin,
                )
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        print("info: interrupted — stopping RTDE motion", flush=True)
        try:
            session.stop_motion()
        except Exception:
            pass
    finally:
        stop_teleop_inputs(keyboard, spacemouse)
        try:
            session.stop_motion()
        except Exception:
            pass
        if mcap is not None:
            try:
                mcap.close()
            except Exception as exc:
                print(f"warning: record-mcap close failed ({exc})", file=sys.stderr)
        session.close()
        if kin is not None:
            kin.shutdown()
        if tf_ctx is not None:
            tf_ctx.close()
        if cam_ctx is not None:
            cam_ctx.close()
        if motion_log is not None and log_path is not None:
            try:
                style = "contact" if in_contact else "auto"
                csv, png = save_and_maybe_plot(
                    motion_log, log_path, plot=plot_motion, style=style
                )
                msg = f"motion log: wrote {csv} ({len(motion_log)} samples)"
                if png is not None:
                    msg += f"  plot={png}  style={style}"
                print(msg, flush=True)
            except Exception as exc:
                print(f"warning: failed to write motion log ({exc})", file=sys.stderr)

    return 0
