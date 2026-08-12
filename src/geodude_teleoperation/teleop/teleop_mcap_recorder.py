"""Standalone MCAP recorder for UR stringer teleop (Foxglove + diagnostics).

Session layout under ``output_dir``::

    YYYY-MM-DD/YYYYMMDD_HHMMSS/
        session_manifest.json
        cli_args.json
        YYYYMMDD_HHMMSS_ffffff.mcap
        YYYYMMDD_HHMMSS_ffffff_metadata.json

Rollout lifecycle: ``begin_rollout`` on Enter, ``end_rollout`` on quit / exit.
Does **not** import ``InferenceRunLogger``; uses ``mcap_ros2`` directly.
"""
from __future__ import annotations

import datetime
import json
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_CAMERA_TOPICS = (
    "/wrist_cam_0/color/image_rect_raw/compressed",
    "/wrist_cam_1/color/image_rect_raw/compressed",
    "/camera_0/color/image_raw/compressed",
)

DEFAULT_OUTPUT_DIR = "/tmp/ur_stringer_teleop_logs"


def _ns_to_stamp_dict(stamp_ns: int) -> dict[str, int]:
    return {"sec": int(stamp_ns // 1_000_000_000), "nanosec": int(stamp_ns % 1_000_000_000)}


def build_ros2_schema(root_type: str, ros_share: str = "/opt/ros/humble/share") -> str:
    """Build concatenated ROS2 msgdef text for ``mcap_ros2.register_msgdef``."""
    share = Path(ros_share)
    defs: OrderedDict[str, str] = OrderedDict()

    def resolve_type(ftype: str, context_pkg: str) -> str | None:
        base = ftype.rstrip("[]")
        if "/" in base:
            dep_pkg, dep_msg = base.split("/")
            return f"{dep_pkg}/msg/{dep_msg}"
        return f"{context_pkg}/msg/{base}"

    def collect(type_name: str) -> None:
        if type_name in defs:
            return
        pkg, _, msg = type_name.partition("/msg/")
        path = share / pkg / "msg" / f"{msg}.msg"
        if not path.is_file():
            raise FileNotFoundError(f"ROS message definition not found: {path}")
        body = path.read_text().strip()
        defs[type_name] = body
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            ftype = stripped.split()[0]
            base = ftype.rstrip("[]")
            if base in {
                "bool",
                "byte",
                "char",
                "float32",
                "float64",
                "int8",
                "uint8",
                "int16",
                "uint16",
                "int32",
                "uint32",
                "int64",
                "uint64",
                "string",
                "wstring",
            }:
                continue
            dep = resolve_type(ftype, pkg)
            if dep is not None:
                collect(dep)

    collect(root_type)
    sep = "=" * 80
    parts: list[str] = []
    for i, (type_name, body) in enumerate(defs.items()):
        if i > 0:
            parts.append(sep)
            parts.append(f"MSG: {type_name.replace('/msg/', '/')}")
        parts.append(body)
    return "\n".join(parts)


def _pose6_to_pose_dict(pose6: Sequence[float] | None) -> dict | None:
    if pose6 is None:
        return None
    p = np.asarray(pose6, dtype=float).reshape(6)
    if np.any(~np.isfinite(p)):
        return None
    quat = Rotation.from_rotvec(p[3:6]).as_quat()  # xyzw
    return {
        "position": p[:3].tolist(),
        "quat_xyzw": quat.tolist(),
    }


def _pose_dict_to_mcap_msg(pose: dict | None, frame_id: str, stamp_ns: int) -> dict:
    msg = {
        "header": {"stamp": _ns_to_stamp_dict(stamp_ns), "frame_id": frame_id},
        "pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    if pose is not None:
        p = pose["position"]
        q = pose["quat_xyzw"]
        msg["pose"]["position"] = {
            "x": float(p[0]),
            "y": float(p[1]),
            "z": float(p[2]),
        }
        msg["pose"]["orientation"] = {
            "x": float(q[0]),
            "y": float(q[1]),
            "z": float(q[2]),
            "w": float(q[3]),
        }
    return msg


def _wrench_to_mcap_msg(wrench: Sequence[float] | None, frame_id: str, stamp_ns: int) -> dict:
    msg = {
        "header": {"stamp": _ns_to_stamp_dict(stamp_ns), "frame_id": frame_id},
        "wrench": {
            "force": {"x": 0.0, "y": 0.0, "z": 0.0},
            "torque": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
    }
    if wrench is not None:
        w = np.asarray(wrench, dtype=float).reshape(6)
        msg["wrench"]["force"] = {
            "x": float(w[0]),
            "y": float(w[1]),
            "z": float(w[2]),
        }
        msg["wrench"]["torque"] = {
            "x": float(w[3]),
            "y": float(w[4]),
            "z": float(w[5]),
        }
    return msg


def _float64_multiarray_msg(data: Sequence[float]) -> dict:
    arr = [float(x) for x in np.asarray(data, dtype=float).reshape(-1)]
    return {
        "layout": {
            "dim": [{"label": "data", "size": len(arr), "stride": len(arr)}],
            "data_offset": 0,
        },
        "data": arr,
    }


def _is_compressed_image_topic(topic: str) -> bool:
    return topic.rstrip("/").endswith("/compressed")


def _ensure_ros_on_sys_path() -> list[str]:
    """Prepend common Humble Python paths when ``PYTHONPATH=…`` wiped ROS.

    ``source /opt/ros/humble/setup.bash`` sets ``PYTHONPATH``; many launch
    lines then replace it with ``PYTHONPATH=src:…`` and ``import rclpy`` fails.
    If ``LD_LIBRARY_PATH`` is still set from sourcing, restoring these paths
    is enough to import again.
    """
    candidates = [
        "/opt/ros/humble/lib/python3.10/site-packages",
        "/opt/ros/humble/local/lib/python3.10/dist-packages",
    ]
    # Also honor AMENT_PREFIX_PATH (set by setup.bash even when PYTHONPATH is replaced).
    ament = __import__("os").environ.get("AMENT_PREFIX_PATH", "")
    for prefix in ament.split(":"):
        if not prefix:
            continue
        candidates.append(f"{prefix}/lib/python3.10/site-packages")
        candidates.append(f"{prefix}/local/lib/python3.10/dist-packages")
    added: list[str] = []
    for path in candidates:
        if Path(path).is_dir() and path not in sys.path:
            sys.path.insert(0, path)
            added.append(path)
    return added


def _import_rclpy_sensor_msgs():
    """Import rclpy + sensor_msgs, recovering Humble paths if needed."""
    try:
        import rclpy
        from sensor_msgs.msg import CompressedImage, Image

        return rclpy, CompressedImage, Image
    except ImportError:
        added = _ensure_ros_on_sys_path()
        try:
            import rclpy
            from sensor_msgs.msg import CompressedImage, Image

            if added:
                print(
                    "record-mcap: restored ROS Python paths after PYTHONPATH wipe "
                    f"({len(added)} entries); prefer "
                    "PYTHONPATH=src:…:${PYTHONPATH}",
                    file=sys.stderr,
                    flush=True,
                )
            return rclpy, CompressedImage, Image
        except ImportError as exc:
            print(
                "warning: record-mcap images DISABLED — cannot import rclpy "
                f"({exc}). Cameras will not be recorded.\n"
                "  Fix: source /opt/ros/humble/setup.bash (and your ws), then\n"
                "       PYTHONPATH=src:${PYTHONPATH} \\\n"
                "         python3 -m geodude_teleoperation.teleop.ur_stringer_teleop …",
                file=sys.stderr,
                flush=True,
            )
            raise


def _new_session_dir(output_dir: Path) -> tuple[Path, str]:
    ts = datetime.datetime.now()
    session_id = ts.strftime("%Y%m%d_%H%M%S")
    rel = f"{ts.strftime('%Y-%m-%d')}/{session_id}"
    path = (output_dir / rel).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path, rel


class TeleopImageBuffer:
    """Best-effort ROS image subscriptions; warn once per topic if never seen."""

    def __init__(self, topics: Sequence[str], *, warn_after_s: float = 2.0):
        self.topics = [str(t) for t in topics]
        self.warn_after_s = float(warn_after_s)
        self._latest: dict[str, Any] = {}
        self._recv_counts: dict[str, int] = {t: 0 for t in self.topics}
        self._lock = threading.Lock()
        self._node = None
        self._t0 = time.monotonic()
        self._warned: set[str] = set()
        self._ros_ok = False
        self.disable_reason: str | None = None
        self._init_ros()

    def _init_ros(self) -> None:
        if not self.topics:
            return
        try:
            rclpy, CompressedImage, Image = _import_rclpy_sensor_msgs()
        except ImportError as exc:
            self.disable_reason = f"ROS import failed: {exc}"
            return
        try:
            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node("geodude_stringer_teleop_mcap_cam")
        except Exception as exc:
            print(
                f"warning: record-mcap images DISABLED (rclpy init failed: {exc})",
                file=sys.stderr,
                flush=True,
            )
            self.disable_reason = f"rclpy init failed: {exc}"
            self._node = None
            return

        self._ros_ok = True
        for topic in self.topics:
            if _is_compressed_image_topic(topic):
                self._node.create_subscription(
                    CompressedImage,
                    topic,
                    self._make_compressed_cb(topic),
                    10,
                )
            else:
                self._node.create_subscription(
                    Image,
                    topic,
                    self._make_raw_cb(topic),
                    10,
                )
        print(
            "record-mcap: subscribed cameras: " + ", ".join(self.topics),
            flush=True,
        )

    def _make_compressed_cb(self, topic: str):
        def _cb(msg) -> None:
            with self._lock:
                self._latest[topic] = {
                    "kind": "compressed",
                    "format": str(msg.format),
                    "data": bytes(msg.data),
                    "stamp_ns": time.time_ns(),
                }
                self._recv_counts[topic] = self._recv_counts.get(topic, 0) + 1

        return _cb

    def _make_raw_cb(self, topic: str):
        def _cb(msg) -> None:
            try:
                from cv_bridge import CvBridge

                bridge = getattr(self, "_bridge", None)
                if bridge is None:
                    self._bridge = CvBridge()
                    bridge = self._bridge
                rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            except Exception:
                return
            with self._lock:
                self._latest[topic] = {
                    "kind": "raw",
                    "rgb": np.asarray(rgb, dtype=np.uint8),
                    "stamp_ns": time.time_ns(),
                }
                self._recv_counts[topic] = self._recv_counts.get(topic, 0) + 1

        return _cb

    def spin_once(self) -> None:
        if self._node is None:
            return
        try:
            import rclpy

            # Drain a few ready callbacks (cameras >> control rate sometimes).
            for _ in range(8):
                rclpy.spin_once(self._node, timeout_sec=0.0)
        except Exception:
            pass

    def snapshot(self) -> dict[str, Any]:
        """Return latest frames; emit warn-once for topics still empty."""
        self.spin_once()
        with self._lock:
            out = dict(self._latest)
        if self._ros_ok and (time.monotonic() - self._t0) >= self.warn_after_s:
            for topic in self.topics:
                if topic not in out and topic not in self._warned:
                    self._warned.add(topic)
                    print(
                        f"warning: record-mcap camera topic never received: {topic}",
                        file=sys.stderr,
                        flush=True,
                    )
        return out

    def recv_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._recv_counts)

    def close(self) -> None:
        if self._node is None:
            return
        try:
            self._node.destroy_node()
        except Exception:
            pass
        self._node = None


class TeleopMcapRecorder:
    """Standalone MCAP writer for stringer teleop rollouts."""

    def __init__(
        self,
        output_dir: Path | str,
        *,
        sample_hz: float,
        save_images: bool = True,
        camera_topics: Sequence[str] | None = None,
        target_frame: str = "vention_frame",
        arm: str = "right",
        cli_args: Mapping[str, Any] | None = None,
        save_motion_trace: bool = True,
        plot_motion_trace: bool = True,
        plot_motion_style: str = "auto",
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.sample_hz = float(sample_hz)
        if self.sample_hz <= 0:
            raise ValueError("sample_hz must be > 0")
        self.save_images = bool(save_images)
        self.target_frame = str(target_frame)
        self.arm = str(arm)
        self.camera_topics = list(
            camera_topics if camera_topics is not None else DEFAULT_CAMERA_TOPICS
        )
        self.save_motion_trace = bool(save_motion_trace)
        self.plot_motion_trace = bool(plot_motion_trace)
        self.plot_motion_style = str(plot_motion_style or "auto")
        self.motion_log = None

        self.session_dir, self.session_name = _new_session_dir(self.output_dir)
        self._write_session_bootstrap(cli_args)

        self._rollout_lock = threading.Lock()
        self._rollout_active = False
        self._rollout_id = ""
        self._metadata: dict[str, Any] = {}
        self._t0 = 0.0
        self._last_sample_time = 0.0
        self._num_samples = 0
        self._mcap_writer = None
        self._mcap_schemas: dict[str, Any] = {}
        self._mcap_path: Path | None = None
        self._mcap_file = None
        self._prev_p: np.ndarray | None = None
        self._prev_t: float | None = None

        self._image_buf: TeleopImageBuffer | None = None
        if self.save_images:
            self._image_buf = TeleopImageBuffer(self.camera_topics)

    def _write_session_bootstrap(self, cli_args: Mapping[str, Any] | None) -> None:
        manifest = {
            "session_name": self.session_name,
            "session_id": self.session_dir.name,
            "layout": "day/session",
            "created_at": datetime.datetime.now().isoformat(),
            "kind": "ur_stringer_teleop",
            "rollouts": [],
            "rollout_count": 0,
        }
        (self.session_dir / "session_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        if cli_args is not None:
            serializable = {}
            for k, v in dict(cli_args).items():
                try:
                    json.dumps(v)
                    serializable[k] = v
                except TypeError:
                    serializable[k] = str(v)
            (self.session_dir / "cli_args.json").write_text(
                json.dumps(serializable, indent=2), encoding="utf-8"
            )

    def begin_rollout(self, metadata: Mapping[str, Any] | None = None) -> Path:
        with self._rollout_lock:
            if self._rollout_active:
                # Nested enable: close previous quietly then reopen.
                self._end_rollout_unlocked()
            ts = datetime.datetime.now()
            self._rollout_id = ts.strftime("%Y%m%d_%H%M%S_%f")
            self._t0 = time.time()
            self._last_sample_time = 0.0
            self._num_samples = 0
            self._prev_p = None
            self._prev_t = None
            self._metadata = {
                "rollout_id": self._rollout_id,
                "wall_time_start": ts.isoformat(),
                "record_format": "mcap",
                "session_name": self.session_name,
                "arm": self.arm,
                "sample_hz": self.sample_hz,
                "save_images": self.save_images,
            }
            if metadata:
                self._metadata.update(dict(metadata))

            from mcap_ros2.writer import Writer as McapWriter

            self._mcap_path = self.session_dir / f"{self._rollout_id}.mcap"
            self._mcap_file = open(self._mcap_path, "wb")
            self._mcap_writer = McapWriter(self._mcap_file)
            self._mcap_schemas = {
                "pose": self._mcap_writer.register_msgdef(
                    "geometry_msgs/msg/PoseStamped",
                    build_ros2_schema("geometry_msgs/msg/PoseStamped"),
                ),
                "wrench": self._mcap_writer.register_msgdef(
                    "geometry_msgs/msg/WrenchStamped",
                    build_ros2_schema("geometry_msgs/msg/WrenchStamped"),
                ),
                "float64_array": self._mcap_writer.register_msgdef(
                    "std_msgs/msg/Float64MultiArray",
                    build_ros2_schema("std_msgs/msg/Float64MultiArray"),
                ),
                "image": self._mcap_writer.register_msgdef(
                    "sensor_msgs/msg/Image",
                    build_ros2_schema("sensor_msgs/msg/Image"),
                ),
                "compressed_image": self._mcap_writer.register_msgdef(
                    "sensor_msgs/msg/CompressedImage",
                    build_ros2_schema("sensor_msgs/msg/CompressedImage"),
                ),
            }
            meta_path = self.session_dir / f"{self._rollout_id}_metadata.json"
            meta_path.write_text(json.dumps(self._metadata, indent=2), encoding="utf-8")
            if self.save_motion_trace:
                from geodude_teleoperation.teleop.motion_trace import MotionTraceLogger

                self.motion_log = MotionTraceLogger()
            else:
                self.motion_log = None
            self._rollout_active = True
            print(
                f"record-mcap: began rollout {self._mcap_path}",
                flush=True,
            )
            return self._mcap_path

    def should_sample(self, now: float) -> bool:
        if not self._rollout_active:
            return False
        if self._mcap_writer is None:
            return False
        if self._last_sample_time <= 0.0:
            return True
        return (now - self._last_sample_time) >= (1.0 / self.sample_hz) - 1e-9

    def record_tick(
        self,
        now: float,
        *,
        tcp_pose: Sequence[float] | None,
        ref_pose: Sequence[float] | None = None,
        wrench: Sequence[float] | None = None,
        qdot: Sequence[float] | None = None,
        qdot_raw: Sequence[float] | None = None,
        status: Mapping[str, Any] | None = None,
        R_motion: np.ndarray | None = None,
        key_pulses: Sequence[float] | None = None,
        twist_linear: Sequence[float] | None = None,
        v_field: Sequence[float] | None = None,
    ) -> None:
        with self._rollout_lock:
            if not self._rollout_active or self._mcap_writer is None:
                return
            if not self.should_sample(now):
                return
            self._last_sample_time = now
            self._num_samples += 1
            stamp_ns = time.time_ns()
            arm = self.arm

            actual = _pose6_to_pose_dict(tcp_pose)
            desired = _pose6_to_pose_dict(ref_pose) or actual
            self._mcap_writer.write_message(
                f"/{arm}/tcp_pose_broadcaster/pose",
                self._mcap_schemas["pose"],
                _pose_dict_to_mcap_msg(actual, self.target_frame, stamp_ns),
                log_time=stamp_ns,
            )
            self._mcap_writer.write_message(
                f"/{arm}/desired_pose",
                self._mcap_schemas["pose"],
                _pose_dict_to_mcap_msg(desired, self.target_frame, stamp_ns),
                log_time=stamp_ns,
            )
            self._mcap_writer.write_message(
                f"/{arm}/force_torque_sensor_broadcaster/wrench",
                self._mcap_schemas["wrench"],
                _wrench_to_mcap_msg(wrench, f"{arm}_tool0", stamp_ns),
                log_time=stamp_ns,
            )

            q = (
                np.asarray(qdot, dtype=float).reshape(6)
                if qdot is not None
                else np.full(6, np.nan)
            )
            q_raw = (
                np.asarray(qdot_raw, dtype=float).reshape(6)
                if qdot_raw is not None
                else np.full(6, np.nan)
            )
            self._write_f64("/teleop/qdot", q, stamp_ns)
            self._write_f64("/teleop/qdot_raw", q_raw, stamp_ns)

            st = status or {}
            status_vec = [
                float(st.get("signed_d", float("nan"))),
                1.0 if st.get("in_contact") else 0.0,
                1.0 if st.get("used_wos") else 0.0,
                float(st.get("tip_nn", float("nan"))),
                float(np.linalg.norm(q)) if np.all(np.isfinite(q)) else float("nan"),
                float(st.get("query_ms", float("nan"))),
                float(st.get("tf_age_ms", float("nan"))),
                1.0 if st.get("motion_enabled") else 0.0,
                1.0 if st.get("teleop_active") else 0.0,
                1.0 if st.get("vf_enabled", True) else 0.0,
            ]
            self._write_f64("/teleop/status", status_vec, stamp_ns)

            if R_motion is not None:
                R = np.asarray(R_motion, dtype=float).reshape(3, 3)
                rotvec = Rotation.from_matrix(R).as_rotvec()
            else:
                rotvec = np.full(3, np.nan)
            self._write_f64("/teleop/R_motion", rotvec, stamp_ns)

            keys = (
                np.asarray(key_pulses, dtype=float).reshape(6)
                if key_pulses is not None
                else np.zeros(6)
            )
            tw = (
                np.asarray(twist_linear, dtype=float).reshape(3)
                if twist_linear is not None
                else np.zeros(3)
            )
            self._write_f64("/teleop/key_twist", np.concatenate([keys, tw]), stamp_ns)

            # Field-frame tip velocity: explicit or TCP finite difference.
            vf = v_field
            if vf is None and tcp_pose is not None and R_motion is not None:
                p = np.asarray(tcp_pose, dtype=float).reshape(6)[:3]
                if self._prev_p is not None and self._prev_t is not None:
                    dt = float(now) - float(self._prev_t)
                    if dt > 1e-9:
                        from geodude_teleoperation.teleop.motion_trace import (
                            tip_velocity_field_frame,
                        )

                        v_world = (p - self._prev_p) / dt
                        vf = tip_velocity_field_frame(v_world, R_motion)
                self._prev_p = p.copy()
                self._prev_t = float(now)
            if vf is None:
                vf = (float("nan"), float("nan"), float("nan"))
            self._write_f64("/teleop/v_field", vf, stamp_ns)

            if self.save_images and self._image_buf is not None:
                self._write_images(self._image_buf.snapshot(), stamp_ns)

    def _write_f64(self, topic: str, data: Sequence[float], stamp_ns: int) -> None:
        assert self._mcap_writer is not None
        self._mcap_writer.write_message(
            topic,
            self._mcap_schemas["float64_array"],
            _float64_multiarray_msg(data),
            log_time=stamp_ns,
        )

    def _write_images(self, images: Mapping[str, Any], stamp_ns: int) -> None:
        assert self._mcap_writer is not None
        for topic, payload in images.items():
            kind = payload.get("kind")
            frame_id = topic.strip("/")
            if kind == "compressed":
                msg = {
                    "header": {
                        "stamp": _ns_to_stamp_dict(stamp_ns),
                        "frame_id": frame_id,
                    },
                    "format": payload.get("format", "jpeg"),
                    "data": payload["data"],
                }
                self._mcap_writer.write_message(
                    topic,
                    self._mcap_schemas["compressed_image"],
                    msg,
                    log_time=stamp_ns,
                )
            elif kind == "raw":
                arr = np.ascontiguousarray(payload["rgb"], dtype=np.uint8)
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                msg = {
                    "header": {
                        "stamp": _ns_to_stamp_dict(stamp_ns),
                        "frame_id": frame_id,
                    },
                    "height": int(arr.shape[0]),
                    "width": int(arr.shape[1]),
                    "encoding": "rgb8",
                    "is_bigendian": 0,
                    "step": int(arr.shape[1] * 3),
                    "data": arr.tobytes(),
                }
                self._mcap_writer.write_message(
                    topic,
                    self._mcap_schemas["image"],
                    msg,
                    log_time=stamp_ns,
                )

    def end_rollout(self) -> Path | None:
        with self._rollout_lock:
            return self._end_rollout_unlocked()

    def _end_rollout_unlocked(self) -> Path | None:
        if not self._rollout_active:
            return None
        self._rollout_active = False
        self._metadata["wall_time_end"] = datetime.datetime.now().isoformat()
        self._metadata["num_samples"] = self._num_samples
        cam_counts: dict[str, int] = {}
        if self._image_buf is not None:
            cam_counts = self._image_buf.recv_counts()
            self._metadata["camera_recv_counts"] = cam_counts
            if self._image_buf.disable_reason:
                self._metadata["camera_disable_reason"] = self._image_buf.disable_reason
            elif not self._image_buf._ros_ok and self.save_images:
                self._metadata["camera_disable_reason"] = "ros_unavailable"
        path = self._mcap_path
        if self._mcap_writer is not None:
            try:
                self._mcap_writer.finish()
            except Exception as exc:
                print(f"warning: mcap finish failed: {exc}", file=sys.stderr)
            self._mcap_writer = None
        if self._mcap_file is not None:
            try:
                self._mcap_file.close()
            except Exception:
                pass
            self._mcap_file = None
        if self._rollout_id:
            meta_path = self.session_dir / f"{self._rollout_id}_metadata.json"
            meta_path.write_text(json.dumps(self._metadata, indent=2), encoding="utf-8")
            self._append_manifest()
            self._flush_motion_trace()
        if path is not None:
            print(
                f"record-mcap: wrote {path} ({self._num_samples} samples)",
                flush=True,
            )
            if self.save_images:
                if cam_counts:
                    parts = [f"{t}={n}" for t, n in cam_counts.items()]
                    print(
                        "record-mcap: camera frames received: " + ", ".join(parts),
                        flush=True,
                    )
                    if sum(cam_counts.values()) == 0:
                        print(
                            "warning: record-mcap saved 0 camera frames "
                            "(rclpy missing or topics down — see earlier warnings)",
                            file=sys.stderr,
                            flush=True,
                        )
                elif self._image_buf is not None and self._image_buf.disable_reason:
                    print(
                        "warning: record-mcap cameras were disabled: "
                        f"{self._image_buf.disable_reason}",
                        file=sys.stderr,
                        flush=True,
                    )
        return path

    def _flush_motion_trace(self) -> None:
        log = self.motion_log
        self.motion_log = None
        if log is None or not self._rollout_id:
            return
        if len(log) == 0:
            return
        from geodude_teleoperation.teleop.motion_trace import save_and_maybe_plot

        csv_path = self.session_dir / f"{self._rollout_id}_motion.csv"
        try:
            csv, png = save_and_maybe_plot(
                log,
                csv_path,
                plot=self.plot_motion_trace,
                style=self.plot_motion_style,
            )
            msg = f"record-mcap: motion log {csv} ({len(log)} samples)"
            if png is not None:
                msg += f"  plot={png}  style={self.plot_motion_style}"
            print(msg, flush=True)
            self._metadata["motion_csv"] = csv.name
            if png is not None:
                self._metadata["motion_png"] = png.name
            meta_path = self.session_dir / f"{self._rollout_id}_metadata.json"
            meta_path.write_text(json.dumps(self._metadata, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"warning: motion log beside mcap failed ({exc})", file=sys.stderr)

    def _append_manifest(self) -> None:
        path = self.session_dir / "session_manifest.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {
                "session_name": self.session_name,
                "rollouts": [],
                "rollout_count": 0,
            }
        rollouts = manifest.setdefault("rollouts", [])
        rollouts.append(
            {
                "rollout_id": self._metadata.get("rollout_id"),
                "wall_time_start": self._metadata.get("wall_time_start"),
                "wall_time_end": self._metadata.get("wall_time_end"),
                "num_samples": self._metadata.get("num_samples"),
                "record_format": "mcap",
                "mcap": f"{self._rollout_id}.mcap",
            }
        )
        manifest["rollout_count"] = len(rollouts)
        manifest["updated_at"] = datetime.datetime.now().isoformat()
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def close(self) -> None:
        self.end_rollout()
        if self._image_buf is not None:
            self._image_buf.close()
            self._image_buf = None


def recorder_from_args(args) -> TeleopMcapRecorder | None:
    """Build recorder from argparse namespace, or None if disabled."""
    if not bool(getattr(args, "record_mcap", False)):
        return None
    control_hz = float(getattr(args, "control_hz", 50.0))
    sample_hz = float(getattr(args, "record_sample_hz", 0.0))
    if sample_hz <= 0:
        sample_hz = control_hz
    out = getattr(args, "record_output_dir", None) or DEFAULT_OUTPUT_DIR
    topics = getattr(args, "record_camera_topics", None)
    save_images = bool(getattr(args, "record_save_images", True))
    # Namespace → plain dict for JSON snapshot.
    cli = {
        k: getattr(args, k)
        for k in dir(args)
        if not k.startswith("_") and not callable(getattr(args, k))
    }
    return TeleopMcapRecorder(
        out,
        sample_hz=sample_hz,
        save_images=save_images,
        camera_topics=topics,
        target_frame=str(getattr(args, "record_target_frame", "vention_frame")),
        arm=str(getattr(args, "arm", "right")),
        cli_args=cli,
        save_motion_trace=True,
        plot_motion_trace=True,
        plot_motion_style=(
            "contact" if bool(getattr(args, "in_contact", False)) else "auto"
        ),
    )
