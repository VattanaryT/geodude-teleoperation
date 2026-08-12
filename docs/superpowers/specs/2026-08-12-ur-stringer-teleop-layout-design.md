# Design: UR stringer teleop layout (local-first, modular)

## Goal

Reorganize the `ur-stringer-teleop` worktree so an operator can install and run dry-run → shadow → live without a sibling `diffused_fields` checkout on `PYTHONPATH`, with clearer package boundaries and one minimal operator README.

## Decisions (approved)

- Prefer local over other repo: **B** — local `config/` + polyscope helpers; keep `diffused_fields` as an installed library for Pointcloud / WoS / scalar diffusion only.
- Layout scope: **moderate** — subpackages `field/`, `viz/`, `teleop/`, `rtde/` with compatibility shims at old import paths.
- Operator README: **dry-run/shadow + live in-contact** (not full phase/MCAP encyclopedia).

## Target layout

```text
ur-stringer-teleop/
  README.md
  config/pointclouds.yaml
  data/processed/…
  scripts/                  # thin wrappers (optional; console scripts remain)
  src/geodude_teleoperation/
    paths.py                # repo/data/config resolution
    field/                  # stringer_field, field_query
    viz/                    # field viz + local polyscope helpers
    teleop/                 # CLI, keyboard, contact, offsets, recording helpers
    rtde/                   # RTDE loop + numpy kin bridge pieces
    calibration/            # unchanged
    in_contact_control/     # unchanged
    frames.py, se3.py, …    # shared math / IO stay top-level
  tests/
  docs/superpowers/         # design history (not operator-facing)
```

## Compatibility shims

Keep importable modules at previous paths (`geodude_teleoperation.field_viz`, `.stringer_field`, `.ur_stringer_teleop`, `.rtde_control_loop`, …) that re-export from the new packages so existing tests and scripts keep working.

## Local wrappers

1. `config/pointclouds.yaml` — `stringer_layup` voxel/normal defaults; `config_defaults()` reads this first (hardcoded fallback if missing).
2. `viz/polyscope_helpers.py` — local `set_camera_and_plane` / `plot_orientation_field` (no `diffused_fields.visualization` import).
3. CLI defaults for `--ply` / `--T-tag0-base` resolve under this worktree’s `data/processed/` when paths are omitted or relative.

## Modular splits (minimum)

- `field_viz` → `viz/` with helpers + session logic separated from polyscope adapters.
- Teleop CLI stays the single operator entry (`geodude-ur-stringer-teleop` / `python -m geodude_teleoperation.ur_stringer_teleop`).
- RTDE loop lives under `rtde/`; numpy kinematics bridge co-located or re-exported.

## Operator README

Replace long `README_ur_stringer_teleop.md` with one `README.md`: install, local data paths, dry-run, RTDE shadow, live in-contact + viz, safety keys. Move deep design links to `docs/superpowers/` only.

## Non-goals

- Vendoring WoS / Pointcloud out of `diffused_fields`.
- Aggressive rename of calibration / examples / assets.
- Expanding operator README to cover every teleop phase flag.

## Success criteria

- `pip install -e '.[ur]'` + installed `diffused_fields` (no sibling `PYTHONPATH`) runs CLI help and field build against local `data/processed/`.
- `--viz` works without importing `diffused_fields.visualization`.
- Existing tests pass via shims (or updated imports).
- One short operator `README.md` at repo root.
