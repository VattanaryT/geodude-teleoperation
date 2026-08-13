# SSoT_dof hyperparameter notes

Companion to [`SSoT_dof.yaml`](SSoT_dof.yaml). Operator flow stays in the [repo README](../README.md).

| File | Role |
| --- | --- |
| [`SSoT_dof.yaml`](SSoT_dof.yaml) | Default scratch (`--config` omitted) |
| [`SSoT_dof_keyboard_dryrun.yaml`](SSoT_dof_keyboard_dryrun.yaml) | Saved keyboard dry-run + Polyscope tuning |
| [`SSoT_dof_spacemouse_dryrun.yaml`](SSoT_dof_spacemouse_dryrun.yaml) | Saved SpaceMouse dry-run + device feel (scales / deadzone / hz / signs) |
| [`SSoT_dof_spacemouse_rtde.yaml`](SSoT_dof_spacemouse_rtde.yaml) | Saved SpaceMouse **live** RTDE (`dry_run: false`); Enter orientation lock + UR speed/accel |

Same schema in every file; switch with `--config`.

**Where it applies:** live = RTDE speedJ; dry = `--dry-run` Polyscope sim; viz = Polyscope display / nudge.

---

## session / robot / paths / pointcloud

| Key | Notes | Applies |
| --- | --- | --- |
| `session.backend` | `mujoco` or `rtde`. CLI `--backend` overrides. | both |
| `session.dry_run` | `true` = receive-only / Polyscope sim. Live motion: `false` or `--no-dry-run`. | both |
| `robot.arm` | v1: `right` only. | both |
| `robot.ip` | Required for RTDE. | live + dry receive |
| `paths.ply` | Object cloud in **tag_0**. `null` → package default under `data/processed/`. | field + viz |
| `paths.T_tag0_base` | tag_0 pose (crop_meta). `null` → default file if present. | VF frames |
| `paths.T_rtde_base_vention` | Right UR base in vention. `null` → `assets/frames/…`. | VF + viz |
| `paths.T_left_base_vention` | Left UR base in vention (laser). | laser viz |
| `paths.calib_meta` | JSON with `T_cam_EE` (AprilTag / camera frustum). | viz + cross-check |
| `paths.object_offset` | `t_vention_m` sidecar. `null` → next to `T_tag0_base`. | VF + viz |
| `paths.laser_ply` / `laser_offset` | Laser overlay/replace cloud + offset JSON. | viz |
| `paths.log_motion` / `record_output_dir` | Motion CSV / MCAP root. | record |
| `pointcloud.voxel_size` | Downsample (m). Smaller → denser cloud, slower WoS. | field |
| `pointcloud.normal_orientation` | PCA normal sign `+1` / `-1`. | field |

---

## field / wos / query

Shared by free-space and in-contact (same orientation field).

| Key | Notes | Applies |
| --- | --- | --- |
| `field.mode` | `end_lines` or `boundary` BCs. | startup field |
| `field.flow` | `length` → end-cap bands; `width` → long-edge bands. | startup field |
| `field.end_line_fraction` | Band thickness as fraction of extent `(0, 0.5)`. | startup field |
| `field.swap_end_lines` | Swap source/sink. | startup field |
| `field.diffusion_scalar` | τ/h²; `dt = scalar × mean_edge²`. Larger → smoother field. | startup field |
| `wos.batch_size` | Walk-on-Spheres samples per query. Lower → cheaper `query_ms`, noisier R. | live + dry VF |
| `wos.convergence_threshold_edge_mult` | Stop radius = `mean_edge × this`. Density-aware. | live + dry VF |
| `query.contact_threshold_m` | `|d|` band for `in_contact` flag. | VF status |
| `query.query_radius_m` | Neighborhood for local surface fit (m). | VF query |
| `query.fit_mode` | `sphere` or `plane`. | VF query |

If `query_ms` is ~seconds, lower `wos.batch_size` (and/or raise voxel size) before raising `control.hz`.

---

## vf (free-space pose-ref; still queried in-contact for n*)

These are **different jobs** — smoothing ≠ relock ≠ query rate.

| Key | Notes | Applies |
| --- | --- | --- |
| `r_field_smoothing` | EMA on **R_field**: keep this fraction of previous (`0`=snap, `~0.75`=sticky). | live + dry + viz preview |
| `relock_dist_m` | After this much **tip travel**, relock **R_motion** toward R_field. | live (+ dry when a query ran) |
| `relock_period_s` | **Min time between queries** (hard floor) and time-based relock if still. | live (dry-run is gated by `dry_query_period_s` first) |
| `motion_track` | Slerp amount of R_motion toward R_field on relock. Clipped to **`[0, 1]`** (`0`=keep old, `1`=snap). | live + dry |
| `dry_query_period_s` | Dry-run only: how often to run WoS. Held WASD must **not** query every tick. Live still queries every tick (subject to `relock_period_s`). Raise if Polyscope tip hitches; `query_ms` ≫ this → hitch once per period. | **dry + viz only** |

---

## control

| Key | Notes | Applies |
| --- | --- | --- |
| `hz` | Host loop rate. Period ≈ `1000/hz` ms. Dry-run WASD + viz TCP publish run at this rate **unless** a WoS query blocks the tick. | live + dry |
| `status_hz` | Status-line print rate. `≤0` → every tick. Does not move the robot/tip. | console |
| `max_joint_velocity` | General `speedJ` `‖q̇‖` clamp (rad/s) after Enter lock settles. | **live** |
| `rtde_accel` | General `speedJ` acceleration (rad/s²) after lock settles. | **live** |
| `pose_gain` | Pose servo: `q̇ ≈ −gain J⁺ e`. Too high saturates on tiny steps. | live (numpy bridge) |
| `max_linear_speed` | Cap tip linear from pose error (m/s). Always this (Enter does not translate). | live |
| `max_angular_speed` | Cap tip angular from pose error (rad/s) after lock settles. | live |
| `orient_lock_on_enable` | Live Enter freezes `R*` = `R_field` and rotates **in place** (no translation servo; **WoS/VF queries paused**). After settle, arm **holds** until a **second Enter** approves teleop. VF stays frozen until `f`. Dry-run still snaps instantly. `false` → hold current EE. | **live** |
| `startup_max_joint_velocity` | `‖q̇‖` cap (rad/s) at Enter; lerps to `max_joint_velocity` over `startup_ramp_s`. | **live** |
| `startup_rtde_accel` | `speedJ` accel (rad/s²) at Enter; lerps to `rtde_accel`. Keep low so lock does not snap. | **live** |
| `startup_max_angular_speed` | Tip angular cap (rad/s) at Enter; lerps to `max_angular_speed`. | **live** |
| `startup_ramp_s` | Seconds to lerp startup → general limits. `0` → general immediately. | **live** |
| `orient_lock_tol_deg` | Lock done when `∠(R_tcp, R_field) ≤` this; then wait for second Enter before teleop. | **live** |
| `keyboard_step` | Meters of tip travel **per keypress / held tick** (before 0.25× distance scale). Free-space pose-ref increment; in-contact tangent slide size. Also ghost arrows / `--viz-teleop-keys`. | live + dry + viz |
| `spacemouse` | Opt-in analog 6-DOF. Keyboard bindings unchanged. CLI `--spacemouse` / `--no-spacemouse`. Open failure → warn, keyboard-only. Snapshot: `SSoT_dof_spacemouse_dryrun.yaml`. | live + dry |
| `spacemouse_linear_scale` | Meters of tip increment at **full** linear deflection, **per poll**, before 0.25× VF scale. Raise if sluggish; lower if jumpy. | live + dry |
| `spacemouse_angular_scale` | Same for angular (rad per full deflection per poll). | live + dry |
| `spacemouse_deadzone` | Ignore |axis| below this (0–1 device units). Raise if idle drift. | live + dry |
| `spacemouse_hz` | Device poll rate. `null` / `≤0` → `control.hz`. Independent of WASD key-repeat. | live + dry |
| `spacemouse_axis_sign` | Length-6 `[x, y, z, roll, pitch, yaw]`; use `-1` to invert an axis. | live + dry |
| `spacemouse_path` | HID path (`/dev/hidrawN`) or `null` = auto. `pyspacemouse --list-spacemouse`. | live + dry |
| `max_twist_scale` | Cap on adaptive distance scale (far → larger steps; min still 1.0). | live + dry |
| `watchdog_ms` | No active control tick this long → `speedJ(0)`. Not latency. | **live** |
| `enable_key` | Enable after calibrate. `"enter"` or one char (not `t`/`v`/`f`). | both |
| `tip_offset` | Tip in TCP frame `[tx, ty, tz]` (m). | VF + viz |
| `base_frame` / `tag0_frame` | Live TF names (rtde). `null` = unused. | live TF |
| `no_object_offset` / `no_rtde_base_vention` / `no_laser_offset` | Ignore sidecars / mounts. | frames |
| `object_offset_step_m` | Polyscope object-cloud nudge per tap (`j/l i/k u/o`). Not robot motion. | **viz** |

---

## viz / laser / tag_crosscheck / record

| Key | Notes | Applies |
| --- | --- | --- |
| `viz.frame` | `tag0` / `vention` / `rtde` / `base` (alias of vention). | viz |
| `viz.frame_stride` | Draw every Nth orientation frame. | viz |
| `viz.hide_object` / `show_object` | Hide / keep object cloud (`--viz-teleop-keys` implies hide unless show). | viz |
| `viz.teleop_keys` | Offline WASD preview (no RTDE loop). Prefer live `--viz` for dry-run sim. | viz |
| `viz.laser` | `off` / `overlay` / `replace`. CLI `--viz-laser` overrides. | viz |
| `viz.latency_table` | Live Polyscope **Latency** window: `loop` / WoS `query` / relock age / SpaceMouse age / viz age (ms). Warns `!` if loop or query > `1000/hz`. CLI `--viz-latency` / `--no-viz-latency`. Default on in spacemouse dry-run snapshot. | **viz** |
| `laser.voxel` | Voxel (m) when loading laser PLY. | viz |
| `tag_crosscheck.*` | Wrist RGB / CameraInfo / tag size & family. | calibrate + viz camera |
| `record.mcap` | Record MCAP per Enter→quit session. | record |
| `record.sample_hz` | `0` → `control.hz`. | record |
| `record.save_images` / `camera_topics` / `target_frame` | Image topics + pose `frame_id`. | record |
| `record.plot_motion` | Write motion PNG (needs `paths.log_motion` or MCAP session). | record |

---

## line_tracking / tactile (free-space only)

Ignored / forced off when in-contact is on. Do not combine `--tactile` with `--in-contact`.

| Key | Notes |
| --- | --- |
| `line_tracking.enabled` | Free tip along field normal (kill normal pose error). RTDE + numpy kin. |
| `line_tracking.soft_standoff` | `v_n = −k(d − d*)` along outward normal. |
| `standoff_distance` / `standoff_gain` / `standoff_max_speed` | d* (m), k (1/s), `|v_n|` clamp (m/s). |
| `no_keyboard_normal_ff` | Ignore field-z keys when line-tracking. |
| `tactile.enabled` | Phase-3a: `ActualTCPForce` → normal FF on speedJ (not DualLoop). |
| `tactile.desired_normal_force` | Desired contact force (N). **Also `Fd` for in-contact DualLoop.** |
| `wrench_kp` / `ki` / `kd` | Force PID gains (free-space tactile path). |
| `wrench_to_vn` | m/s per N → normal speed. |
| `max_vn` | Clamp tactile normal speed (m/s). |
| `ft_bias` | Optional 6-vector subtracted from `actual_TCP_force`; `null` = none. |

---

## in_contact (DualLoop, RTDE only)

| Key | Notes |
| --- | --- |
| `enabled` | Same as CLI `--in-contact`. Align + Force PID; tangent keys only. |
| `force_sign` | `Fs = sign · (F − bias) · n*`. Use `-1` if Enter seeks away from surface. |
| `align_first` | Seek force only after tool-Z ≈ n*. `--no-align-first` skips. |
| `align_enter_tol_deg` / `align_exit_tol_deg` | Enter Force / drop back to Align (deg). |
| `approach_v_max` | Soft approach along n* before light contact (m/s). |
| `force_max_v` | Force PID along-normal speed cap (m/s). |

---

## diag

| Key | Notes |
| --- | --- |
| `hold_zero` | After enable, `speedJ(0)` every tick (no field / ignore keys). Wobble → RTDE/watchdog. |
| `fixed_orient` | Lock `R_motion` to EE at first update. Smooth here → wobble is field-tracking. Mutually exclusive with `hold_zero`. |
| `verify_500hz` / `dtc_verify_seconds` / `verify_500hz_exit` | Direct Torque rate check. |
| `polyscope_version` | Optional string for DTC check; else DashboardClient. |

---

## Quick lag / hitch checklist (dry-run + viz)

Enable `--viz --viz-latency` (or `viz.latency_table: true`) and watch the Polyscope **Latency** table.

1. `query (WoS)` ≫ `target dt` → drop `wos.batch_size`, then raise `pointcloud.voxel_size`.
2. Dry-run hitch: raise `vf.dry_query_period_s` (try `0.2–1`) so WoS does not run every tick.
3. Live: `vf.relock_period_s` ≥ `query` ms. Don’t relock faster than WoS finishes.
4. Only then raise `control.hz` / `spacemouse_hz`. SpaceMouse poll does not unblock WoS.
5. `watchdog_ms` and `status_hz` do not change tip latency.
