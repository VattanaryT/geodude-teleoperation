# SSoT_dof Config Implementation Plan

> **For agentic workers:** Implement task-by-task. Steps use checkbox syntax.

**Goal:** Make `config/SSoT_dof.yaml` the source of truth for UR stringer teleop / field / viz config; thin CLI for session toggles only.

**Architecture:** Nested YAML → flatten to existing argparse Namespace attrs; ephemeral CLI overrides after load; fold `pointclouds.yaml` into SSoT; expose WoS edge multiplier on `build_wos`.

**Tech Stack:** PyYAML, argparse, existing `geodude_teleoperation` package.

## Global Constraints

- No CLI for field/VF/path/IP tuning after cutover.
- WoS threshold = `mean_edge_length * convergence_threshold_edge_mult`.
- Preserve Namespace attr names used by teleop/viz/RTDE.

---

### Task 1: SSoT YAML + loader

**Files:**
- Create: `config/SSoT_dof.yaml`
- Create: `src/geodude_teleoperation/ssot_dof.py`
- Modify: `src/geodude_teleoperation/paths.py`

- [ ] Write commented nested YAML with current defaults
- [ ] Implement `load_ssot_dof`, `flatten_ssot`, `ssot_dof_config_path`
- [ ] Unit-test flatten + load

### Task 2: `build_wos` + `config_defaults`

**Files:**
- Modify: `field/stringer_field.py`
- Modify: `tests/test_stringer_field.py`, `tests/test_paths_local.py`

- [ ] Add `convergence_threshold_edge_mult` to `build_wos`
- [ ] Point `config_defaults` at SSoT `pointcloud`
- [ ] Deprecate/remove live use of `pointclouds.yaml`

### Task 3: Slim CLI

**Files:**
- Modify: `teleop/ur_stringer_teleop.py`
- Modify: CLI / laser / frames tests

- [ ] Replace hyperparam/path/IP argparse with `--config` + ephemeral flags
- [ ] Wire `parse_args` to load/flatten SSoT then apply CLI
- [ ] Pass edge mult into `build_wos` from args
- [ ] Update tests

### Task 4: Verify

- [ ] Run targeted pytest modules; fix fallout
