# UR stringer keyboard teleop (geodude-teleoperation)

**Date:** 2026-08-09  
**Status:** Ready for review  

**Package:** `geodude-teleoperation`  
**Approach:** Modular package + one entry script (Approach 1)

## Goal

Keyboard teleoperate a real UR5e (Geodude right arm) against the stringer layup orientation field, reusing:

- Field build / BC from the stringer demo (`compute_orientation_field`, `--field-flow length|width`, `boundary`)
- Contact / free-space query from `offline_demo_to_local_actions.py` (`query_basis_surface` / `query_basis_wos`)
- Control from geodude (`TeleoperationController.update_command` + `compute_control_command`)
- Input from `KeyboardTeleop`

**Safety-first mode machine:** calibrate (motion off, pendant OK) → operator checks prints → explicit enable → teleop. Limit velocity (and RTDE accel). Support MuJoCo dry backend and RTDE real backend with the same controller math and mode machine.

## Non-goals (v1)

- Left arm (same pattern later)
- Mode-dependent control-law table (gains / kill normal / soft stand-off) — **v1.5+**
- Pose streaming (`speedL` / servo pose) — v1 hardcodes `speedJ`
- Contact hysteresis (note for later)
- Replacing pendant TCP calibration; tip offset is a hook (default zeros)
- Baking PLY into the MuJoCo world as a second frame path (v1 always uses PLY in tag_0 + `--T-tag0-base`)

## Architecture

### Modules

| Module | Role |
|--------|------|
| `stringer_field.py` | Load stringer PLY once; voxel / `normal_orientation` from `pointclouds.yaml` (`stringer_layup`); build BC field; startup field validation prints |
| `field_query.py` | Pure query API ported from offline demo: neighborhood plane/sphere → `n`, signed `d`, surface basis; `query_basis_wos` off-contact. **No** `correct_distance_smooth` |
| `contact_wos_vf.py` | Frame-aware VF: `project` / `get_distance` / `update_from_state`; status getters for signed `d` / `in_contact` |
| `frames.py` | Tip offset in TCP; SE(3) helpers; live TF `base↔tag_0` (rtde); fixed `T_tag0_base` (mujoco); AprilTag chain for calibrate cross-check |
| `ur_stringer_teleop.py` | CLI + mode machine + backend loops |
| Existing | `keyboard.py`, `controller.py` |

### Runtime modes

1. **Startup** — load PLY, build field once, print field validation; abort on empty PLY / BC failure.
2. **Calibrate** (default, `motion_enabled=False`) — pendant OK; continuous status prints from per-tick VF refresh. Twists do not move the robot.
3. **Teleop** — after dedicated enable key (`motion_enabled=True`, `teleop_active=True` by default); keyboard → `update_command`; control loop applies `compute_control_command` and refreshes VF every tick.

### Mode machine (identical for mujoco and rtde)

```
Startup → Calibrate (motion_enabled=False, teleop_active unused for motion)
              │
              │ enable key (Enter / --enable-key; NOT t/v)
              ▼
           Teleop (motion_enabled=True, teleop_active=True by default)
              │
              │ t toggles teleop_active; v toggles VF
              │ commands only if motion_enabled AND teleop_active
              │ disable / quit / fault / stale TF → q̇=0
```

### Command gating precedence

Apply motion commands only when **all** of:

1. `motion_enabled` (calibrate gate passed via `--enable-key`)
2. `teleop_active` (keyboard `t` toggle; **defaults to True at enable** so operator does not need Enter then `t`)
3. Not dry-run / not fault / TF fresh (rtde)

VF on/off (`v`) is independent: when VF off, `update_command` / tick refresh skip projection but gating above still applies.

### Per-tick VF / query refresh (not only on keypress)

`TeleoperationController.update_command` runs only when a twist arrives. That is insufficient for:

- Orientation staying current between keypresses
- Calibrate status prints while the operator moves with the pendant

**v1 rule:** every control tick (or a fixed query cadence ≤ control rate), refresh VF from the current EE/TCP state **before** computing `q̇` (and always during calibrate for prints):

- Preferred API: `vf.update_from_state(ee_motor, T_tag0_base)` which runs the same query path as `project`, caches signed `d` / `R_field` / tip_nn, and updates the objective orientation target when VF is enabled and teleop is active.
- Host (rtde) and mujoco side-loop both call this. Do **not** rely on keypress-only `project` inside `update_command` for distance/status freshness.
- `update_command` may still call `project` when composing a twist increment; it must reuse the VF’s **cached** last query (or call `update_from_state` once) so tip/TCP points stay consistent — **no double query with inconsistent points** in one tick.

## Frames and SE(3) naming

Store **`T_tag0_base`** as crop_meta does: pose of tag_0 expressed in base  
`p_base = R @ p_tag0 + t` with `R,t = T_tag0_base[:3,:3], T_tag0_base[:3,3]`.

Then:

- `R_base_from_tag0 = T_tag0_base[:3, :3]`
- `p_tip_tag0 = R_base_from_tag0.T @ (p_tip_base - t)`
- `R_field_base = R_base_from_tag0 @ R_field_tag0`

Docstrings must state this explicitly. Do **not** name an inverted matrix `T_tag0_base`. Prefer helpers `base_to_tag0_point` / `tag0_rot_to_base` over ambiguous aliases in call sites.

PLY vertices stay in **tag_0**. Query in tag_0; apply orientation correction in **base/control**. The VF must never assume colocated object/robot frames.

### Tip vs TCP

- Query at tip: `p_tip = p_tcp + R_tcp @ tip_offset` (default `tip_offset = [0,0,0]` when pendant TCP = contact tip).
- gafropy task space remains TCP/EE. `project` corrects **TCP orientation** from `R_field_base` (+ approach convention). Position target stays “where TCP is”; tip follows via fixed tool offset.
- No second kinematic tip frame inside gafropy.
- Non-zero tip-offset TX/TY → startup warning (same as offline).

### Backend frame sources

| Backend | `T_tag0_base` |
|---------|----------------|
| `mujoco` | **Always** PLY in tag_0 + `--T-tag0-base` (file or identity default). No live TF. No separate “bake PLY into world” path in v1. |
| `rtde` | Live TF (`--base-frame`, `--tag0-frame`). AprilTag wrist-cam chain in calibrate validates placement vs TF. |

## Field build and query API

### Startup field (once)

- Object: stringer layup PLY in tag_0.
- Defaults: `voxel_size`, `normal_orientation` from diffused_fields `pointclouds.yaml` entry `stringer_layup`.
- BC: same as stringer demo — `--field-flow length|width` (end-cap / side lines), `--field-mode end_lines|boundary`.
- Build via the same scalar diffusion + `get_local_bases` path as `compute_orientation_field` / offline `compute_field`.

**Startup validation prints:**

- `#verts`
- `#source` / `#sink` (or boundary sink count)
- Sample `local_bases[i]`: orthonormal / det≈+1
- Mean edge length and WoS `convergence_threshold` (typically `mean_edge * 2`)

### Pure query API (`field_query.py`)

Mirror offline:

1. Neighborhood ball (+ k-NN fallback, max neighbors) → plane or sphere fit → normal `n`, **signed** distance `d`, surface basis from neighborhood `local_bases` re-orthonormalized onto `n`.
2. `in_contact = abs(d) <= contact_threshold` (band; hysteresis later).
3. If not in contact → `query_basis_wos`; else keep surface basis.
4. Do **not** use `correct_distance_smooth`.

Calibrate tip check also prints Euclidean tip↔nearest-vertex distance for operator validation (separate from signed `d` / contact).

## Virtual fixture (v1)

### Contract

`ContactWosOrientationField`:

- **`update_from_state(ee_motor, T_tag0_base)`** — single query per tick (or cadence): tip from TCP+offset → tag_0 → surface/WoS → cache status + `R_field_base`. Used in calibrate and teleop.
- **`project(motor)`** — frame-aware orientation replace using the **same tip query semantics**; may reuse cache if called in the same tick after `update_from_state` for that motor’s TCP. Steps when computing:
  1. Extract TCP pose in base from motor.
  2. Tip in base → tip in tag_0.
  3. Surface query → signed `d`, surface `R_field_tag0`.
  4. If not contact → WoS for `R_field_tag0`.
  5. Double-cover continuity vs last `R_field` (quat dot `< 0` → flip).
  6. `R_field_base = R_base_from_tag0 @ R_field_tag0`.
  7. Apply approach-axis convention; replace motor orientation; keep translator (TCP position).
- **`get_distance(position)`** — returns cached **`abs(d)`** from the last `update_from_state` / `project` tip query. The `position` argument matches today’s geodude call site (EE/TCP translation in base) but **must not trigger a second inconsistent query**; if cache is cold, compute once at tip derived from current EE (not raw NN at TCP when tip_offset ≠ 0). Adaptive scale always uses `abs(d)` — never signed `d` in `_adaptive_linear_scale`.
- **Status / getters** — signed `d`, `in_contact`, `tip_nn`, fit diagnostics, `used_wos`, `query_ms`, for status line (do not overload `get_distance`).

### Approach-axis convention (mujoco = rtde)

Offline / stringer field columns: `[t1, t2, n]` with **z = normal**.  
Apply existing geodude `_orient_approach_into_surface` (flip y and z) so tool approach matches `WalkOnSpheresVirtualFixture` / `OrientationField`. Same path on both backends.

### v1 behavior: only swap how `R_field` is computed

| Regime | `R_field` source |
|--------|------------------|
| Free | WoS(x) |
| Contact | Surface neighborhood fit |

`project()` / tick refresh always replace orientation with that `R_field` (+ approach / continuity). Gains, twist reframe, linear scale: **unchanged** from current `TeleoperationController`.

**v1.5+ (out of scope):** use `in_contact` to change gains, kill normal component, soft stand-off, yaw-about-n preference, twist reframe into surface frame.

### Continuity

Each new `R_field`, compare to previous (quat dot). If `< 0`, flip so successive updates do not 180° snap — same stitching as offline demo relabeling.

### Runtime query failure

WoS / query failure mid-teleop → fall back to last `R_field` or nearest surface basis; log + print; **do not abort**.

## Backends

### MuJoCo (`--backend mujoco`, default)

- `gafropy` `RobotInterface` with `config.type = "mujoco"`.
- `interface.control(controller, ...)` owns the servo loop; a companion tick (or controller hook) still calls `vf.update_from_state` every cycle for status + orientation refresh.
- Honor `max_joint_velocity`, motion disabled until enable, same status line / mode machine.
- Frame: PLY in tag_0 + `--T-tag0-base` (identity allowed as default).

### RTDE (`--backend rtde`)

- **`interface.control` is mujoco-only.** Host loop owns a fixed-rate cycle (`--control-hz`):
  1. Read robot state / runtime state via **RTDEReceive**.
  2. Refresh TF (and AprilTag in calibrate).
  3. `vf.update_from_state(...)` (query cadence).
  4. If protective stop / fault → do not command; `speedJ(0)` when control interface is open.
  5. If not (`motion_enabled` and `teleop_active`) or TF stale → `speedJ(0)`.
  6. Else reuse the **same** `TeleoperationController` object — **do not reimplement P-velocity**. Obtain `q̇` by calling `compute_control_command` with constructed task-space / joint state compatible with gafropy, **or** (if those state objects cannot be fed from RTDE) extract a thin pure helper from the controller’s math (`qdot_from_error(...)`) used by both paths. Feeding compatible state vs extracting the helper is the main integration risk; resolve in implementation with a spike before hardening the rtde loop.
  7. Clamp `‖q̇‖` and per-joint caps; send **`speedJ(q̇)`** (v1 hardcoded; no `--servo` yet).
- **Watchdog:** if no fresh command for `N` ms → `speedJ(0)` (not only on disconnect), when control is open.
- Disconnect / quit → `speedJ(0)` only if control was opened; then exit.
- Abort before calibrate if `--backend rtde` without `--robot-ip`.

### Dry-run / shadow

`--dry-run`: query + prints + mode machine; **never send motion commands**.

For `--backend rtde --dry-run`:

- May open **RTDEReceive only** (require `--robot-ip` to read live state).
- **Never** open `RTDEControl` / never call `speedJ` (including no “send 0 on exit”).
- Rule: **dry-run ⇒ no control interface.**

For `--backend mujoco --dry-run`: do not apply actuator commands (or skip starting control); still refresh VF / prints.

## Keyboard

- Movement keys → `on_twist` only when `motion_enabled and teleop_active` (and not dry-run for real motion).
- **`t` / `v`** → toggle `teleop_active` / VF only. **Do not** overload `t` as calibrate enable.
- **Enable gate:** `--enable-key` (default Enter) sets `motion_enabled=True` and **`teleop_active=True`** so one key enters usable teleop; `t` can later pause twists without leaving the teleop mode / re-running calibrate.

## CLI

| Flag | Notes |
|------|--------|
| `--backend {mujoco,rtde}` | Default `mujoco` |
| `--robot-ip` | Required for rtde (including dry-run receive-only) |
| `--arm` | v1: `right` |
| `--ply` | Object cloud in tag_0 |
| `--voxel-size`, `--normal-orientation` | Override config |
| `--field-flow {length,width}` | Default `length` |
| `--field-mode {end_lines,boundary}` | Demo BC |
| `--end-line-fraction`, `--swap-end-lines` | BC knobs |
| `--tip-offset TX TY TZ` | Default zeros; TX/TY warn |
| `--contact-threshold`, `--query-radius`, `--fit-mode` | Query |
| `--wos-batch-size` | WoS walkers |
| `--control-hz` | Host / status / query cadence |
| `--max-joint-velocity`, `--keyboard-step` | Safety / feel |
| `--rtde-accel` | RTDE accel limit for `speedJ` (control path only) |
| `--watchdog-ms` | No fresh command → `speedJ(0)` when control open |
| `--enable-key` | Calibrate→teleop gate (default Enter); documented separately from `t`/`v` |
| `--T-tag0-base` | Fixed SE(3) file; identity default for mujoco |
| `--base-frame`, `--tag0-frame` | Live TF (rtde) |
| AprilTag params | Calibrate cross-check only |
| `--dry-run` | No control interface / no motion commands |

## Safety

- Motion starts **disabled** (`motion_enabled=False`).
- Clamp `‖q̇‖` and per-joint caps; optional Cartesian cap only if `speedL` is added later.
- RTDE (control open): watchdog `N` ms; protective-stop / runtime-state check; fault → stop commanding.
- MuJoCo: same enable gate + max-vel so modes match.
- Conservative defaults for real hardware.

## Errors

| Condition | Behavior |
|-----------|----------|
| Empty PLY / field BC failure | Abort at startup |
| `rtde` without `--robot-ip` | Abort before calibrate (required for receive-only dry-run too) |
| Missing / stale TF | Warn; hold `q̇=0` / keep motion gated |
| Missing AprilTag in calibrate | Warn; continue prints (TF still used for query when valid); enable remains **manual** (operator presses `--enable-key` after checking prints — no auto-enable) |
| WoS/query failure at runtime | Fallback last / nearest basis; log; continue |
| tip_offset TX/TY ≠ 0 | Startup warning |
| UR fault / protective stop | `speedJ(0)` if control open; do not keep commanding |
| RTDE disconnect | `speedJ(0)` only if control was open; exit |

## Status print contract

One recurring status line (calibrate and teleop), e.g.:

```
mode=calibrate|teleop  motion_enabled=0|1  teleop_active=0|1  d=±x.xxxx  in_contact=0|1  wos=0|1  tip_nn=x.xxxx  tag_vs_tf_mm=…  query_ms=…  qdot_norm=…  tf_age_ms=…
```

Fields:

- `mode` — calibrate / teleop
- `motion_enabled` — calibrate gate
- `teleop_active` — `t` toggle (default 1 after enable)
- `d` — signed distance from geometric fit
- `in_contact` — `abs(d) <= threshold`
- `wos` — whether free-space WoS supplied `R_field`
- `tip_nn` — tip↔nearest-vertex (operator validation)
- `tag_vs_tf` — AprilTag vs TF disagreement in calibrate (when available)
- `query_ms` — last query cost
- `qdot_norm` — commanded joint speed norm (0 when disabled / dry-run)
- `tf_age_ms` — TF freshness (rtde); N/A or 0 for fixed mujoco `T_tag0_base`

## Testing / verification

1. **Startup field prints** look sane (`#verts`, source/sink, orthonormal sample, edge / WoS threshold).
2. **`--dry-run` + mujoco:** sim pose → tip_nn / `d` / frames print via per-tick refresh; no motion until enable; enable still no commands if dry-run.
3. **MuJoCo teleop:** enable alone (no extra `t`) → keyboard moves with VF orientation lock; orientation continues updating between keys; continuity no 180° snaps.
4. **Calibrate AprilTag:** disagree with TF → printed; enable remains manual after operator review.
5. **RTDE shadow:** `--dry-run --backend rtde --robot-ip …` opens receive only; never opens control; never `speedJ`.
6. **RTDE live:** low max-vel; watchdog stops; fault stops; same `TeleoperationController` math for `q̇`.

## Dependencies

- Existing: `gafropy`, `diffused_fields`, `numpy`, keyboard/controller stack
- Optional: `ur-rtde` (`[ur]` extra) for rtde backend
- ROS / TF / AprilTag for live frames and calibrate cross-check (rtde path)

## Open follow-ups (not blocking v1)

- Contact hysteresis
- v1.5 mode-dependent gains / stand-off
- Left arm
- `speedL` / servo pose path
- Extract shared query module upstream into `diffused_fields` later (v1 copies/ports into geodude)
- Spike: RTDE state → `compute_control_command` vs pure `qdot_from_error` helper
)
