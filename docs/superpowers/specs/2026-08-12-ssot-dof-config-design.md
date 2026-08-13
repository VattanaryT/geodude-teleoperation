# Design: SSoT_dof.yaml as teleop/field config source of truth

## Goal

Replace scattered CLI hyperparameter / path / robot defaults with one commented
`config/SSoT_dof.yaml`. Operators tune diffusion / WoS / VF (and paths / IP) in
YAML; `field_viz` and teleop consume the same values. CLI stays thin for session
toggles only.

## Decisions (approved)

- Cutover: **full** — robot IP, file paths, field/VF/control/phase defaults live in YAML.
- CLI: `--config` (default `config/SSoT_dof.yaml`) + ephemeral overrides:
  `--backend`, `--dry-run`, `--viz` and related viz session flags, plus common
  mode toggles (`--in-contact`, `--line-tracking`, `--tactile`, `--record-mcap`,
  diags, etc.) that flip without editing YAML.
- WoS threshold: **mean-edge multiplier** (`wos.convergence_threshold_edge_mult`,
  default `2.0`); YAML comments explain `threshold = mean_edge * mult`.
- `pointclouds.yaml`: **fold** voxel/normal into `SSoT_dof.yaml`; teleop/field
  stop reading `pointclouds.yaml`.
- Loader approach: **nested YAML → flatten to existing Namespace attrs** so
  teleop / viz / RTDE keep using `args.diffusion_scalar`, `args.robot_ip`, etc.

## Architecture

```text
config/SSoT_dof.yaml
        │
        ▼
  load_ssot_dof(path)  →  nested dict
        │
        ▼
  flatten_ssot(dict)   →  flat keys matching today’s argparse dests
        │
        ▼
  parse_args: set_defaults(flat) → parse ephemeral CLI → resolve null paths
        │
        ▼
  validate_args / _load_field_bundle / field_viz / RTDE (unchanged consumers)
```

- `build_wos(..., convergence_threshold_edge_mult=...)` uses the YAML multiplier.
- `config_defaults()` reads `pointcloud` from SSoT (hardcoded fallback if missing).
- Default `--config` resolves via `paths.ssot_dof_config_path()`.

## YAML sections

`robot`, `paths`, `pointcloud`, `field`, `wos`, `query`, `vf`, `control`,
`viz`, `tag_crosscheck`, `laser`, `record`, `line_tracking`, `tactile`,
`in_contact`, `diag` — each commented for operators.

## Non-goals

- Typed pydantic schema.
- CLI overrides for tuning knobs (`--diffusion-scalar`, `--wos-batch-size`, …).
- Keeping `pointclouds.yaml` as a live teleop dependency.

## Success criteria

- Editing `diffusion_scalar` / `wos.batch_size` / `convergence_threshold_edge_mult`
  in YAML changes field build + viz without CLI flags.
- `ur_stringer_teleop --viz --dry-run` works with defaults from SSoT.
- Existing tests updated to use SSoT / ephemeral flags; suite passes.
