# UR stringer keyboard teleop — test guide

Keyboard teleop the Geodude **right** UR5e against the stringer layup orientation field
(`ContactWosOrientationField` + `TeleoperationController`).

**Safety mode machine:** calibrate (motion off) → check prints → **Enter** to enable → teleop.  
`t` / `v` only toggle teleop-active / VF — they do **not** leave calibrate.

Design: `docs/superpowers/specs/2026-08-09-ur-stringer-keyboard-teleop-design.md`  
Short checklist: `docs/superpowers/plans/2026-08-09-ur-stringer-teleop-manual-checklist.md`

---

## Install

From this package (feature branch / worktree):

```bash
cd /path/to/geodude-teleoperation   # or .worktrees/ur-stringer-teleop
pip install -e '.[ur]'              # ur-rtde for --backend rtde
# also need: diffused_fields, gafropy (for MuJoCo / live kinematics), open3d, scipy
```

Entry point:

```bash
geodude-ur-stringer-teleop --help
# or (after sourcing ROS — keep ${PYTHONPATH} so rclpy stays importable):
source /opt/ros/humble/setup.bash
PYTHONPATH=src:${PYTHONPATH} python -m geodude_teleoperation.ur_stringer_teleop --help
```

If `diffused_fields` is a local checkout:

```bash
export PYTHONPATH=src:/path/to/diffused_fields/src:${PYTHONPATH}
```

**MCAP cameras:** `--record-mcap` needs `rclpy`. Do **not** set `PYTHONPATH=src:…` alone after sourcing ROS — that wipes Humble’s Python path and silently drops cameras. Always append `${PYTHONPATH}`.

---

## What you need


| Asset                   | Purpose                                               |
| ----------------------- | ----------------------------------------------------- |
| Object PLY in **tag_0** | e.g. `object_tag0_crop_voxel0.0020.ply`               |
| `T_tag0_base`           | `crop_meta.json` / `.npy` / `.npz`, or live TF (rtde) |
| Right-arm robot IP      | Only for `--backend rtde`                             |
| gafropy + geodude model | MuJoCo FK/control and live RTDE `q̇` bridge           |


Defaults: `--field-flow length`, `--field-mode end_lines`, `--arm right`, tip offset `0 0 0`.

---

## Keyboard


| Key           | Action                                                  |
| ------------- | ------------------------------------------------------- |
| **Enter**     | Enable motion gate (`motion_enabled` + `teleop_active`) |
| `t`           | Toggle teleop-active (pause/resume twists)              |
| `v`           | Toggle virtual fixture                                  |
| `f`           | Freeze/unfreeze VF `n*` (**`--in-contact` only**)       |
| `w/s a/d y/x` | Linear twist (contact mode: **`w/s a/d` tangent only**) |
| `n/m j/k u/i` | Angular twist (ignored in `--in-contact`)               |
| `q`           | Quit keyboard thread / exit loop                        |

### In-contact mode

```bash
PYTHONPATH="/home/boeing/ros2_ws/src/geodude-teleoperation/.worktrees/ur-stringer-teleop/src:/home/boeing/ros2_ws/src/diffused_fields/src${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m geodude_teleoperation.ur_stringer_teleop \
  --backend rtde --robot-ip "$RIGHT_UR_IP" \
  --ply /home/boeing/ros2_ws/src/diffused_fields/data/processed/object_tag0_crop_voxel0.0020.ply \
  --T-tag0-base /home/boeing/ros2_ws/src/diffused_fields/data/processed/crop_meta.json \
  --field-flow width --max-joint-velocity 1 \
  --in-contact --desired-normal-force 5.0 \
  --viz \
  --record-mcap \
  --record-output-dir "/media/boeing/Vat_uWAMIT/layup data archive/dof_logs"
```

`--viz` (with `--in-contact`) shows the object cloud, tip→NN, magenta **tangent** (±field-x), cyan **approach** `n*`, and field axes. With `--backend rtde`, tip/TCP keep tracking live RTDE (pendant + teleop) until you close Polyscope; terminal `q` ends the loop.

**Object align (session):** with live `--viz`, focus Polyscope and use `j/l` `i/k` `u/o` to translate the cloud (±X/Y/Z in the viz frame), `[`/`]` for step size, `r` reset, `p` save to `<crop_meta>.object_offset.json` (auto-loaded next run; `--no-object-offset` skips).

`--record-mcap` / `--plot-motion` write an **in-contact** PNG: Fd vs Fs/Fe, cumulative `a_t1/a_t2/a_n`, tip velocity, `R_motion`, keyboard, `q̇`.

Pendant-jog near the surface → **Enter** → Force PID auto-seeks/holds `Fd` along VF `n*` → `w/s a/d` slides tangent. If the arm slams into the part instead of loading, try `--force-sign 1` (default is `-1` for UR `ActualTCPForce`).

Design: `docs/superpowers/specs/2026-08-10-in-contact-keyboard-teleop-design.md`.


---

## 1) Unit tests (no robot)

```bash
cd /path/to/geodude-teleoperation   # feature worktree recommended
PYTHONPATH=src:/path/to/diffused_fields/src pytest \
  tests/test_frames.py \
  tests/test_field_query.py \
  tests/test_stringer_field.py \
  tests/test_contact_wos_vf.py \
  tests/test_keyboard_buttons.py \
  tests/test_teleop_state.py \
  tests/test_controller_qdot.py \
  tests/test_ur_stringer_teleop_cli.py \
  tests/test_rtde_control_loop.py \
  tests/test_tag_crosscheck.py -v
```

Expect all green before any hardware run.

---

## 2) MuJoCo dry-run (recommended first)

Builds the field, prints validation, refreshes tip/contact status. **Never** starts `interface.control`.

```bash
geodude-ur-stringer-teleop \
  --ply /path/to/object_tag0.ply \
  --dry-run \
  --field-flow length \
  --T-tag0-base /path/to/crop_meta.json
```

Identity object↔base (cloud already aligned to world):

```bash
geodude-ur-stringer-teleop --ply /path/to/object_tag0.ply --dry-run
# omits --T-tag0-base → identity
```

**Check:**

- [ ] Startup: `#verts`, `#source/#sink`, sample `local_bases` orthonormal, mean edge / WoS threshold
- [ ] Status line ticks (`mode=calibrate`, `d=…`, `tip_nn=…`, `motion_enabled=0`)
- [ ] **Enter** → `motion_enabled=1` `teleop_active=1` (still no motion in dry-run)
- [ ] `t` / `v` do not enable
- [ ] Tip-offset TX/TY warning if you pass non-zero XY

---

## 3) MuJoCo teleop (sim motion)

Same as above **without** `--dry-run`. Motion starts only after **Enter**.

```bash
geodude-ur-stringer-teleop \
  --ply /path/to/object_tag0.ply \
  --field-flow length \
  --T-tag0-base /path/to/crop_meta.json \
  --max-joint-velocity 0.3 \
  --keyboard-step 0.0025
```

**Check:**

- [ ] Before Enter: sim does not servo on the teleop objective
- [ ] After Enter: WASD… moves EE; VF locks orientation near surface
- [ ] Between keypresses, orientation still updates (per-tick VF refresh)
- [ ] `v` off → free orientation; status `d` / `tip_nn` still update
- [ ] `t` pauses twists without re-entering calibrate

Needs **gafropy** installed.

---

## 4) RTDE shadow (receive-only)

Opens **RTDEReceive** only — never `RTDEControl` / never `speedJ`. Use pendant to move; watch tip vs cloud.

```bash
export RIGHT_UR_IP=192.168.x.x   # your right arm

geodude-ur-stringer-teleop \
  --backend rtde \
  --robot-ip "$RIGHT_UR_IP" \
  --dry-run \
  --ply /path/to/object_tag0.ply \
  --field-flow length \
  --T-tag0-base /path/to/crop_meta.json \
  --control-hz 50 \
  --contact-threshold 0.001
```

**Check:**

- [ ] Connects receive; no control interface
- [ ] Pendant motion updates status (`tip_nn`, signed `d`, `in_contact`)
- [ ] Tip↔nearest-vertex looks sane before you ever enable live motion
- [ ] Enter only flips flags; arm still only moves from pendant

---

## 5) Live RTDE (real motion — last)

Use only after steps 2–4 look good. Keep velocity low.

```bash
geodude-ur-stringer-teleop \
  --backend rtde \
  --robot-ip "$RIGHT_UR_IP" \
  --ply /path/to/object_tag0.ply \
  --field-flow length \
  --T-tag0-base /path/to/crop_meta.json \
  --max-joint-velocity 0.2 \
  --rtde-accel 0.5 \
  --watchdog-ms 500 \
  --keyboard-step 0.002
```

**Frame rules:**

- Prefer an explicit `--T-tag0-base` file for first live runs.
- If you pass `--base-frame` / `--tag0-frame` without working TF and **without** `--T-tag0-base`, motion stays gated (`tf_ok=0`).
- Live TF + AprilTag cross-check (calibrate prints): see checklist; enable stays **manual**.

**Check:**

- [ ] Before Enter: `speedJ(0)` / no tracking of teleop twists
- [ ] After Enter + `t` active: small keyboard steps move the arm
- [ ] e-stop / protective stop → commanding stops
- [ ] Quit / disconnect → safe stop when control was open

Needs **gafropy** for the kinematics bridge; if it fails to sync, live `q̇` stays zero by design.

---

## Useful flags


| Flag                                | Notes                               |
| ----------------------------------- | ----------------------------------- |
| `--dry-run`                         | No control / no motion commands     |
| `--field-flow {length,width}`       | Demo BC (default `length`)          |
| `--field-mode {end_lines,boundary}` |                                     |
| `--tip-offset TX TY TZ`             | Default zeros; avoid non-zero TX/TY |
| `--contact-threshold`               | Default `0.001` m                   |
| `--wos-batch-size`                  | Default `256`                       |
| `--control-hz`                      | Default `50`                        |
| `--max-joint-velocity`              | Clamp `‖q̇‖`                        |
| `--watchdog-ms`                     | Live RTDE stale-command stop        |
| `--enable-key`                      | Default Enter; must not be `t`/`v`  |
| `--line-tracking`                   | Phase 1.5: free along `R_field` normal (kill normal pose error) |
| `--soft-standoff`                   | Phase 1.5: `v_n=-k(d-d*)` (implies line tracking) |
| `--standoff-distance` / `--standoff-gain` | Soft stand-off `d*` (m) and `k` (1/s) |
| `--tactile`                         | Phase 3a: `actual_TCP_force` → normal FF (RTDE; enables line tracking) |
| `--desired-normal-force`            | Target contact force (N) for tactile |
| `--verify-500hz`                    | Print 500 Hz readiness; on live RTDE also command `directTorque([0]*6)` and measure Hz |
| `--dtc-verify-seconds`              | Duration of live DTC rate test (default 2) |
| `--verify-500hz-exit`               | Exit after verify (no teleop) |
| `--polyscope-version`               | e.g. `5.23.0`; else queried via dashboard on `--verify-500hz` |


---

## Phase 1.5 / 2 / 3 (contact-ready path)

```bash
# Geometric free-normal + soft stand-off (still speedJ)
geodude-ur-stringer-teleop --backend rtde --robot-ip "$RIGHT_UR_IP" \
  --ply … --T-tag0-base … --line-tracking --soft-standoff \
  --standoff-distance 0.002 --standoff-gain 1.0

# Phase 2: live 500 Hz directTorque(0) rate check (omit --dry-run; remote + External Control on)
geodude-ur-stringer-teleop --backend rtde --robot-ip "$RIGHT_UR_IP" \
  --ply … --T-tag0-base … \
  --verify-500hz --verify-500hz-exit --dtc-verify-seconds 2

# Phase 3a tactile on speedJ (F/T → normal velocity; needs line freedom)
geodude-ur-stringer-teleop --backend rtde --robot-ip "$RIGHT_UR_IP" \
  --ply … --tactile --desired-normal-force 5.0 --line-tracking
```

**Notes:** Field build is one-shot (Laplacian + surface bases); WoS stays in the realtime loop. `--diffusion-scalar` is τ/h². Live `--verify-500hz` uses ur_rtde `directTorque` every ~2 ms (zero torque = gravity-compensated hold); PolyScope ≥ 5.23 / 10.10 required.

---

## Status line

Example fields:

```text
mode=calibrate|teleop  motion_enabled=0|1  teleop_active=0|1
d=±…  in_contact=0|1  wos=0|1  tip_nn=…  tag_vs_tf_mm=…
query_ms=…  qdot_norm=…  tf_age_ms=…
```

---

## Troubleshooting


| Symptom                                   | Likely cause                                         |
| ----------------------------------------- | ---------------------------------------------------- |
| Import errors for `diffused_fields`       | Add its `src` to `PYTHONPATH` or install the package |
| MuJoCo / live RTDE won't move after Enter | Missing gafropy, or kin sync failed (check warnings) |
| `tf_ok=0` with frame flags                | Live TF down and no explicit `--T-tag0-base`         |
| Tip far from cloud                        | Wrong `T_tag0_base`, tip offset, or PLY frame        |
| Sticky / huge steps                       | Lower `--keyboard-step` and `--max-joint-velocity`   |
| )                                         |                                                      |


