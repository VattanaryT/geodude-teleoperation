# Design: per-session object translation offset (viz + VF)

## Goal

Before each teleop session, nudge the stringer point cloud in live `--viz` so tip and object align, then save a sidecar config that auto-loads next run. Offset applies to **`T_tag0` translation** (viz + VF), not tip-offset.

## Model

- Sidecar: `<stem>.object_offset.json` next to `--T-tag0-base` (e.g. `crop_meta.object_offset.json`).
- Contents: `{ "t_vention_m": [x,y,z], "step_m": 0.001 }`.
- Apply: `T_tag0_vention[:3,3] += t_vention` after load / live TF parent, before RTDE compose.
- Nudge keys move in current `--viz-frame` axes; delta mapped into vention for storage.

## UX (Polyscope focused)

| Key | Action |
|-----|--------|
| `j`/`l` | ±X |
| `i`/`k` | ±Y |
| `u`/`o` | ±Z |
| `[`/`]` | finer / coarser step |
| `r` | reset offset to zero (memory) |
| `p` | save sidecar |

## Runtime

- Shared mutable `args.object_offset_t` updated by Polyscope; RTDE loop applies via `resolve_T_tag0_base`.
- Live viz recomputes object/BC/field point positions from tag0 verts + effective `T_tag0`.

## Non-goals

Rotation, mutating `crop_meta.json`, tip-offset editing.
