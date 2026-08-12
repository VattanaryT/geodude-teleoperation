# Teleop MCAP recording (standalone)

**Date:** 2026-08-10  
**Status:** approved → implementing  
**Package:** `geodude-teleoperation` (ur-stringer-teleop worktree)

## Goal

Optional MCAP recording for UR stringer keyboard teleop, similar to
`diffusion_rl` `recording:` (format mcap, output_dir, sample_hz, save_images),
but implemented as a **standalone** recorder (no `InferenceRunLogger` import).

## Decisions

| Choice | Decision |
|--------|----------|
| Payload | Core inference-compatible topics **plus** teleop diagnostics |
| Lifecycle | One `.mcap` per **Enter → quit** enable session (re-Enter = new file) |
| Images | Try default camera topics when ROS is up; **warn once** and skip if missing |
| Architecture | Standalone `TeleopMcapRecorder` using `mcap_ros2` |

## CLI

- `--record-mcap` — enable
- `--record-output-dir PATH` — session root (default `/tmp/ur_stringer_teleop_logs`)
- `--record-sample-hz FLOAT` — `0` → `control_hz`
- `--record-save-images` / `--no-record-save-images` (default on)
- `--record-camera-topics TOPIC [TOPIC ...]` — override defaults

Default camera topics (match diffusion layup):

- `/wrist_cam_0/color/image_rect_raw/compressed`
- `/wrist_cam_1/color/image_rect_raw/compressed`
- `/camera_0/color/image_raw/compressed`

Independent of `--log-motion`.

## Session layout

```
output_dir/YYYY-MM-DD/YYYYMMDD_HHMMSS/
  session_manifest.json
  cli_args.json
  YYYYMMDD_HHMMSS_ffffff.mcap
  YYYYMMDD_HHMMSS_ffffff_metadata.json
  YYYYMMDD_HHMMSS_ffffff_motion.csv   # auto with --record-mcap
  YYYYMMDD_HHMMSS_ffffff_motion.png   # auto plot beside CSV
```

`--log-motion PATH` is only for standalone CSV (no MCAP). With `--record-mcap`,
CSV/PNG live in the session folder next to each rollout `.mcap` (no separate path).

- Process start: create session dir + manifest + CLI snapshot
- Enter: `begin_rollout()`
- Each control tick (throttled to `sample_hz`): `record_tick(...)`
- `q` / Ctrl-C / process exit: `end_rollout()`; re-Enter opens a new rollout

## Topics

**Core (right arm, Foxglove-friendly):**

- `/right/tcp_pose_broadcaster/pose` — actual TCP (`PoseStamped`)
- `/right/desired_pose` — `T_ref` if available else actual
- `/right/force_torque_sensor_broadcaster/wrench` — RTDE TCP force when present
- Camera topics as `CompressedImage` / `Image`

**Teleop diagnostics (`Float64MultiArray` unless noted):**

- `/teleop/qdot` — sent joint rates (6)
- `/teleop/qdot_raw` — pre-clamp (6; NaN if unknown)
- `/teleop/status` — `d, in_contact, wos, tip_nn, qdot_norm, query_ms, tf_age_ms, motion_enabled, teleop_active, vf_enabled`
- `/teleop/R_motion` — rotvec (3) of motion basis
- `/teleop/key_twist` — `key_w..y (6) + twist_lx/ly/lz (3)`
- `/teleop/v_field` — `v_t1, v_t2, v_n` (from TCP Δp/Δt × R_motion)

## Module

- `geodude_teleoperation/teleop_mcap_recorder.py` — session + mcap writer + helpers
- Optional thin ROS image buffer in same module or loop (warn-once per topic)
- Wire in `rtde_control_loop.py`; mujoco path optional / same helper if easy

## Non-goals (v1)

- PKL format
- Left arm
- Action-chunk sidecars
- Coupling to `robot_learning.InferenceRunLogger`
