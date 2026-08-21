# UR stringer keyboard / SpaceMouse teleop

Right UR5e keyboard teleop against the stringer layup field. Optional SpaceMouse analog 6-DOF (`--spacemouse` / `control.spacemouse`) runs **in addition** to keyboard — key bindings do not change.

**Config:** edit [`config/SSoT_dof.yaml`](config/SSoT_dof.yaml) for robot IP, paths, diffusion / WoS / VF, and control defaults. Per-key notes: [`config/SSoT_dof.md`](config/SSoT_dof.md). CLI is mainly for session toggles (`--backend`, `--dry-run`, `--viz`, `--in-contact`, …).

Local data defaults live under `data/processed/`. Install `diffused_fields` as a package (no sibling checkout on `PYTHONPATH` for config/viz).

## Modes

Two **mutually exclusive** control paths. Pick one per session.

| | **Free-space** (default) | **In-contact** (`--in-contact` / `in_contact.enabled`) |
| --- | --- | --- |
| Backend | MuJoCo or RTDE | **RTDE only** (needs `ActualTCPForce`) |
| Controller | Pose-reference teleop + orientation VF | DualLoop: Align + Force PID |
| Normal / depth | Keyboard `y/x` (field ±z); optional `line_tracking` / `tactile` FF | Force PID holds `Fd` along `n*` (auto-seek on Enter) |
| Tangential | `w/s a/d` in field frame | Same keys, tangent plane only |
| Angular keys | `n/m j/k u/i` | Ignored |
| Extra | — | `f` freeze/unfreeze VF `n*` |
| SSoT knobs | `control.*`, `vf.*`, `query.*`, optional `line_tracking.*` / `tactile.*` | `in_contact.*` + `tactile.desired_normal_force` (`Fd`) |

Shared for both: `paths.*`, `pointcloud.*`, `field.*`, `wos.*`, `query.*`, `viz.*`.

Do **not** combine `--in-contact` with `--tactile`.

## Install

Needs **Python ≥3.12**. Do **not** use system Python 3.10, and do **not** use conda env `pyrite` (that’s the UMI-FT / diffusion-policy / ROS research stack).

Dedicated env: **`geodude-teleop`** (Python 3.12, already created).

```bash
conda activate geodude-teleop
# Important: clear ROS/UMI overlays so they don't shadow this env
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH

cd ~/ros2_ws/src/geodude-teleoperation
pip install -e '.[ur]'
# Prefer local diffused_fields (complete package) over the GitHub dep alone:
pip install -e ~/ros2_ws/src/diffused_fields
# polyscope comes with local diffused_fields; needed for --viz
```

Confirm:

```bash
which geodude-ur-stringer-teleop   # …/miniforge3/envs/geodude-teleop/bin/…
python -c "import diffused_fields; print(diffused_fields.__file__)"
# should NOT be under ros2_ws/install/.../python3.10/...
geodude-ur-stringer-teleop --help
```

**Recreate the env** (if needed):

```bash
conda create -y -n geodude-teleop python=3.12 pip
conda activate geodude-teleop
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
cd ~/ros2_ws/src/geodude-teleoperation
pip install -e '.[ur]'
pip install -e ~/ros2_ws/src/diffused_fields
```

Avoid a sourced ROS overlay that puts an old `diffused_fields` (Python 3.10) ahead of the env on `PYTHONPATH`.

**Note:** On this machine `import gafropy` SIGILLs (bad CPU opcode in the wheel). RTDE teleop auto-probes gafropy in a subprocess and uses `numpy_kinematics_bridge` instead. MuJoCo backend still needs a working gafropy.

## Config (`SSoT_dof.yaml`)

| File | Role |
| --- | --- |
| [`config/SSoT_dof.yaml`](config/SSoT_dof.yaml) | Default scratch (omit `--config`) |
| [`config/SSoT_dof_keyboard_dryrun.yaml`](config/SSoT_dof_keyboard_dryrun.yaml) | Snapshot of keyboard dry-run + Polyscope tuning |
| [`config/SSoT_dof_spacemouse_dryrun.yaml`](config/SSoT_dof_spacemouse_dryrun.yaml) | Snapshot of SpaceMouse dry-run (device scales / deadzone / hz / signs) |

```bash
# replay saved keyboard dry-run
geodude-ur-stringer-teleop --config config/SSoT_dof_keyboard_dryrun.yaml \
  --backend rtde --dry-run --viz

# replay saved SpaceMouse dry-run (keyboard still works)
geodude-ur-stringer-teleop --config config/SSoT_dof_spacemouse_dryrun.yaml \
  --backend rtde --dry-run --viz
```

| Section | Typical knobs | Mode |
| --- | --- | --- |
| `robot` | `ip` (required for RTDE) | both |
| `paths` | `ply`, `T_tag0_base`, mounts, offsets (`null` → package defaults) | both |
| `pointcloud` / `field` / `wos` | voxel, diffusion, WoS stop radius | both |
| `query` / `vf` | contact band, fit; R_field smoothing & relock | both (VF); free-space pose-ref uses relock heavily |
| `control` | rates, `keyboard_step`, `spacemouse` (+ `linear_scale` / `angular_scale` / `deadzone` / `hz` / `axis_sign` / `path`), general UR limits (`max_joint_velocity`, `rtde_accel`, `max_linear_speed`, `max_angular_speed`), Enter lock (`orient_lock_on_enable`, `startup_*`, `orient_lock_tol_deg`), `pose_gain`, `max_twist_scale`, `tip_offset` | both |
| `line_tracking` / `tactile` | soft standoff / normal FF | **free-space only** (forced off in-contact) |
| `in_contact` | DualLoop Force PID + Align + tangent → **speedJ**, **or** Eq.56/`pygafro` → **directTorque** (`actuation: speedj\|dtc\|dtc_pure`; yellow banner) | **in-contact only** |
| `tactile.desired_normal_force` | free-space tactile `Fd`; contact prefers `in_contact.desired_force` | both |

Knob-by-knob notes: [`config/SSoT_dof.md`](config/SSoT_dof.md).

## Safety

Calibrate (motion off) → check prints → **Enter** to enable → after lock settles **Enter again** to approve teleop.  
`t` / `v` toggle teleop-active / VF only. `f` holds pose in place (free-space) or freezes `n*` (in-contact). `q` quits. **Ctrl+C** must `speedStop` the UR (last `speedJ` otherwise keeps running).

| Key | Free-space | In-contact |
| --- | --- | --- |
| Enter | Enable + lock TCP orientation to `R_field` (live: startup ramp). **Enter again** after settle to approve teleop | Enable + arm DualLoop (bias F/T, seek `Fd` along `n*`) |
| `t` / `v` | Pause teleop / toggle VF | Pause teleop / toggle VF |
| `w/s a/d` | Field ±x / ±y | Tangent slide ⊥ `n*` |
| `y/x` | Field ±z (normal) | Ignored (Force owns normal) |
| `f` | Hold pose (`speedStop` + recapture, VF frozen). Again: resume + unlock VF | Freeze / unfreeze `n*` |
| `q` | Quit | Quit |

With `--viz`, use these keys in the **Polyscope window** (not only the terminal).

`--viz --viz-latency` (or `viz.latency_table: true`) opens a **Latency** table: loop ms vs `1000/hz`, WoS `query`, relock age, SpaceMouse age, viz publish→draw. `!` means loop or query is slower than the target tick.

**SpaceMouse** (`--spacemouse` or `control.spacemouse: true`): analog 6-DOF on the same twist path. Button 0 = Enter (enable), button 1 = `t` (pause). `v` / `f` / `q` stay on the keyboard. In-contact: continuous tangent xy only (z/angular ignored; tip holds during Align). If the device fails to open, teleop stays keyboard-only. Tune feel in [`config/SSoT_dof_spacemouse_dryrun.yaml`](config/SSoT_dof_spacemouse_dryrun.yaml) (`spacemouse_linear_scale`, `_angular_scale`, `_deadzone`, `_hz`, `_axis_sign`, `_path`). Live free-space: [`config/SSoT_dof_spacemouse_rtde.yaml`](config/SSoT_dof_spacemouse_rtde.yaml). Live in-contact: [`config/SSoT_dof_spacemouse_rtde_in_contact.yaml`](config/SSoT_dof_spacemouse_rtde_in_contact.yaml).

## Free-space

### Dry-run (Polyscope sim, no motion)

Set `robot.ip` in `config/SSoT_dof.yaml`, leave `in_contact.enabled: false`, then:

```bash
geodude-ur-stringer-teleop --backend rtde --dry-run --viz
# SpaceMouse dry-run snapshot (keyboard still works; Latency table on):
geodude-ur-stringer-teleop --config config/SSoT_dof_spacemouse_dryrun.yaml \
  --backend rtde --dry-run --viz
```

1. Calibrate: tip tracks the **real** RTDE TCP (pendant OK).
2. Focus Polyscope → **Enter**: sim locks a TCP snapshot and snaps EE axes to `R_field` (VF on). **Hold** WASD / `x`/`y` — tip slides continuously along the field (~control_hz × keyboard_step). Robot stays receive-only.
3. `v` toggles VF (default on; when on, EE stays field-aligned in viz). `t` pauses sim. Confirm axes vs the cloud before live.

`--viz-teleop-keys` remains a separate offline preview (no RTDE control loop).

### Shadow receive (pendant only)

Same receive-only RTDE; without Enter, tip only follows the pendant. After Enter with `--dry-run`, tip follows the sim (above).

### Live free-space

`session.dry_run` defaults to `true` in SSoT — clear it for motion:

```bash
# either set session.dry_run: false in SSoT_dof.yaml, or:
geodude-ur-stringer-teleop --backend rtde --no-dry-run --viz
# SpaceMouse live snapshot (UR limits + Enter orientation lock):
geodude-ur-stringer-teleop --config config/SSoT_dof_spacemouse_rtde.yaml --viz
```

Calibrate → focus the **Polyscope window** → **Enter** (enable + rotate TCP **in place** to frozen `R_field`) → wait for settle → **Enter again** to approve teleop → `w/s a/d x/y` move tip along the **field** frame (±x/±y/±z). VF stays frozen after approve (no WoS relock). **`f`** holds the live TCP (`speedStop` + recapture, VF frozen). **`f` again** resumes teleop and unlocks VF queries. Live Enter uses `startup_*` vel/accel (linear speed 0 during lock/hold) then general limits after approve. One keypress ≈ `keyboard_step` × adaptive scale (capped by `max_twist_scale`).

Optional free-space layers (SSoT or CLI): `--line-tracking`, `--tactile` (not with `--in-contact`).

## In-contact

Tune `in_contact.*`, `tactile.desired_normal_force` (`Fd`), `field.*`, and keep `control.max_joint_velocity` low, then:

```bash
geodude-ur-stringer-teleop --backend rtde --no-dry-run --in-contact --viz
# SpaceMouse in-contact snapshot:
geodude-ur-stringer-teleop --config config/SSoT_dof_spacemouse_rtde_in_contact.yaml --viz
```

(Or set `in_contact.enabled: true` and `session.dry_run: false` in SSoT.)

Pendant near surface → focus Polyscope → **Enter** → Force PID seeks/holds along VF normal → `w/s a/d` slide tangent → `f` to freeze `n*` if the field jitters.

Design: [`docs/superpowers/specs/2026-08-10-in-contact-keyboard-teleop-design.md`](docs/superpowers/specs/2026-08-10-in-contact-keyboard-teleop-design.md).

## Field / VF tuning with viz

1. Edit `field.diffusion_scalar`, `wos.*`, `query.*`, or `vf.*` in `SSoT_dof.yaml`.
2. Re-run with `--viz` (optionally `--viz-teleop-keys` for offline WASD tip preview).
3. Iterate until the orientation field and contact frame look right.

## Layout

- `config/SSoT_dof.yaml` — default scratch SSoT
- `config/SSoT_dof_keyboard_dryrun.yaml` — saved keyboard dry-run tuning
- `config/SSoT_dof_spacemouse_dryrun.yaml` — saved SpaceMouse dry-run tuning
- `config/SSoT_dof_spacemouse_rtde.yaml` — saved SpaceMouse live RTDE tuning
- `config/SSoT_dof.md` — hyperparameter notes
- `src/geodude_teleoperation/{field,viz,teleop,rtde}/` — library code
- `data/processed/` — PLY / crop_meta / laser
- `scripts/` — thin wrappers
- `docs/superpowers/` — design history (not required for ops)

Design: [`docs/superpowers/specs/2026-08-12-ssot-dof-config-design.md`](docs/superpowers/specs/2026-08-12-ssot-dof-config-design.md)
