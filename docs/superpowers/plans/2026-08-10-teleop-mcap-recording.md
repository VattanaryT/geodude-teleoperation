# Teleop MCAP recording — Implementation Plan

> Implemented after design approval (2026-08-10).

**Goal:** Standalone MCAP recorder for stringer teleop (core Foxglove topics + teleop diagnostics), one file per Enter→quit session.

**Architecture:** `teleop_mcap_recorder.py` owns session dirs + `mcap_ros2` writing; RTDE loop starts/stops rollouts and feeds ticks; optional ROS image buffer with warn-once.

**Tech stack:** Python 3.10, `mcap_ros2`, optional `rclpy` / `sensor_msgs` for cameras.

## Tasks (done)

- [x] Design spec `docs/superpowers/specs/2026-08-10-teleop-mcap-recording-design.md`
- [x] `TeleopMcapRecorder` + `TeleopImageBuffer` + `recorder_from_args`
- [x] CLI flags on `ur_stringer_teleop`
- [x] Wire RTDE loop (rising-edge Enter → begin; quit/finally → close; per-tick sample)
- [x] Unit tests `tests/test_teleop_mcap_recorder.py` + CLI coverage
