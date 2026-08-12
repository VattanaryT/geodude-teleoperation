"""Unit tests for TeleopMcapRecorder (no robot / optional ROS)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geodude_teleoperation.teleop.teleop_mcap_recorder import (
    DEFAULT_CAMERA_TOPICS,
    TeleopMcapRecorder,
    _ensure_ros_on_sys_path,
    _float64_multiarray_msg,
    _pose6_to_pose_dict,
    build_ros2_schema,
    recorder_from_args,
)


def test_ensure_ros_on_sys_path_is_idempotent():
    # Should not raise; may or may not add paths depending on the host.
    first = _ensure_ros_on_sys_path()
    second = _ensure_ros_on_sys_path()
    assert second == []
    assert isinstance(first, list)



def test_build_ros2_schema_pose():
    text = build_ros2_schema("geometry_msgs/msg/PoseStamped")
    assert "PoseStamped" in text or "Pose pose" in text
    assert "Header" in text or "std_msgs/Header" in text


def test_pose6_and_float_array_helpers():
    d = _pose6_to_pose_dict([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    assert d is not None
    assert abs(d["position"][0] - 0.1) < 1e-9
    assert abs(d["quat_xyzw"][3] - 1.0) < 1e-9
    msg = _float64_multiarray_msg([1.0, 2.0, 3.0])
    assert msg["data"] == [1.0, 2.0, 3.0]


def test_recorder_writes_mcap(tmp_path: Path):
    pytest.importorskip("mcap_ros2")
    rec = TeleopMcapRecorder(
        tmp_path,
        sample_hz=50.0,
        save_images=False,
        cli_args={"backend": "rtde", "record_mcap": True},
    )
    assert (rec.session_dir / "session_manifest.json").is_file()
    assert (rec.session_dir / "cli_args.json").is_file()

    path = rec.begin_rollout({"note": "unit"})
    assert path.suffix == ".mcap"
    t0 = 1000.0
    for i in range(5):
        now = t0 + 0.02 * i
        assert rec.should_sample(now)
        rec.record_tick(
            now,
            tcp_pose=[0.01 * i, 0.0, 0.5, 0.0, 0.0, 0.0],
            ref_pose=[0.012 * i, 0.0, 0.5, 0.0, 0.0, 0.0],
            wrench=[0.0, 0.0, -5.0, 0.0, 0.0, 0.0],
            qdot=[0.1, 0, 0, 0, 0, 0],
            qdot_raw=[0.5, 0, 0, 0, 0, 0],
            status={
                "signed_d": 0.05,
                "in_contact": False,
                "used_wos": True,
                "tip_nn": 0.04,
                "query_ms": 0.5,
                "motion_enabled": True,
                "teleop_active": True,
                "vf_enabled": True,
            },
            R_motion=np.eye(3),
            key_pulses=[1, 0, 0, 0, 0, 0] if i == 2 else [0] * 6,
            twist_linear=[0.01, 0, 0] if i == 2 else None,
        )
    out = rec.end_rollout()
    assert out is not None and out.is_file()
    assert out.stat().st_size > 0
    meta = rec.session_dir / f"{rec._rollout_id}_metadata.json"
    assert meta.is_file()
    manifest = (rec.session_dir / "session_manifest.json").read_text()
    assert "rollout_id" in manifest
    rec.close()


def test_second_begin_rollout_ends_previous(tmp_path: Path):
    pytest.importorskip("mcap_ros2")
    rec = TeleopMcapRecorder(tmp_path, sample_hz=10.0, save_images=False)
    p1 = rec.begin_rollout()
    rec.record_tick(1.0, tcp_pose=[0, 0, 0, 0, 0, 0], qdot=np.zeros(6))
    p2 = rec.begin_rollout()
    assert p1 != p2
    assert p1.is_file()
    rec.record_tick(2.0, tcp_pose=[0, 0, 0.1, 0, 0, 0], qdot=np.zeros(6))
    rec.close()
    assert p2.is_file()


def test_recorder_writes_motion_csv_and_plot(tmp_path: Path):
    pytest.importorskip("mcap_ros2")
    rec = TeleopMcapRecorder(
        tmp_path,
        sample_hz=50.0,
        save_images=False,
        save_motion_trace=True,
        plot_motion_trace=True,
    )
    rec.begin_rollout()
    assert rec.motion_log is not None
    for i in range(3):
        rec.motion_log.append(
            t_mono=0.02 * i,
            qdot=[0.1, 0, 0, 0, 0, 0],
            R_motion=np.eye(3),
            tcp_pose=[0.01 * i, 0, 0.4, 0, 0, 0],
            ref_pose=[0.01 * i, 0, 0.4, 0, 0, 0],
        )
        rec.record_tick(
            1000.0 + 0.02 * i,
            tcp_pose=[0.01 * i, 0, 0.4, 0, 0, 0],
            qdot=[0.1, 0, 0, 0, 0, 0],
            R_motion=np.eye(3),
        )
    rid = rec._rollout_id
    rec.end_rollout()
    csv = rec.session_dir / f"{rid}_motion.csv"
    png = rec.session_dir / f"{rid}_motion.png"
    assert csv.is_file()
    assert png.is_file()
    assert rec.motion_log is None
    rec.close()


def test_recorder_from_args_enabled(tmp_path: Path):
    pytest.importorskip("mcap_ros2")

    class A:
        record_mcap = True
        record_output_dir = tmp_path
        record_sample_hz = 0.0
        control_hz = 25.0
        record_save_images = False
        record_camera_topics = None
        record_target_frame = "vention_frame"
        arm = "right"

    rec = recorder_from_args(A())
    assert rec is not None
    assert abs(rec.sample_hz - 25.0) < 1e-9
    assert rec.camera_topics == list(DEFAULT_CAMERA_TOPICS)
    rec.close()
