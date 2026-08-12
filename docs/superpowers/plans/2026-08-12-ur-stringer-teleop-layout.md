# UR stringer teleop layout — Implementation Plan

> **For agentic workers:** Execute inline in this session (user requested execute).

**Goal:** Local-first modular layout with shims, operator README, no sibling `diffused_fields` on PYTHONPATH for viz/config/data.

**Architecture:** Subpackages `field/`, `viz/`, `teleop/`, `rtde/` + shims; local `config/` and polyscope helpers; `paths.py` for data defaults.

**Tech Stack:** Python 3.10+, existing pytest suite, polyscope, diffused_fields (library only).

## Global Constraints

- Prefer local config/data/viz helpers over sibling repo checkout.
- Keep old import paths working via shims.
- One minimal operator `README.md` (dry-run/shadow → live).
- Do not vendor WoS/Pointcloud.

---

### Task 1: paths + local config

- Create `src/geodude_teleoperation/paths.py`
- Create `config/pointclouds.yaml`
- Update `stringer_field.config_defaults` to prefer local config

### Task 2: packages + moves + shims

- Create `field/`, `viz/`, `teleop/`, `rtde/`
- Move modules; leave shim re-exports at old paths
- Entry point still `geodude_teleoperation.ur_stringer_teleop:main`

### Task 3: local polyscope + viz split

- `viz/polyscope_helpers.py` (no DF visualization import)
- Extract `LiveVizState` to `viz/live_state.py`
- Point `_ensure_polyscope` at local helpers

### Task 4: CLI local data defaults + scripts + README

- Default `--ply` / `--T-tag0-base` under `data/processed/`
- `scripts/ur_stringer_teleop.sh` thin wrapper
- Replace root operator doc with `README.md`; retire long guide from operator path

### Task 5: Verify tests

- Run pytest with `PYTHONPATH=src` and installed/local DF library only as needed for Pointcloud
