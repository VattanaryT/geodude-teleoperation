# Design: `--viz-teleop-keys` preview

**Date:** 2026-08-09  
**Status:** implemented (interactive)

## Goal

Optional Polyscope overlay that **interactively** previews live keyboard teleop: hold WASD/XYZ in the Polyscope window to move an EE tip + `R_field` axes; ghost tips show one scaled keypress.

## CLI

- `--viz-teleop-keys` — requires `--viz`
- Uses `--keyboard-step` and VF query params (`--contact-threshold`, etc.)

## Behavior

1. Peek TCP (same as existing `--viz` markers).
2. Query tip → tag_0 → surface/WoS → `_orient_approach_into_surface` (live VF path).
3. Polyscope user callback (focus the window):
   - Hold `w/s/a/d/x/y` → tip slides along `R_field` (±x/±y/±z)
   - Re-query `R_field` while moving; update EE axes + one-step ghosts
4. If TCP missing: warn and skip overlay.

## Field parity vs `stringer_layup_ee_teleop.py`

| Item | Match? |
|------|--------|
| `end_lines` / `boundary` BC | Yes (same PCA band logic) |
| `field-flow` length→ends, width→sides | Yes |
| `diffusion_scalar` default | Yes (`5000`; was `2000`) |
| Orientation query in viz | Hybrid contact/WoS like offline+geodude VF (teleop demo defaults to **always WoS**) |
| Motion UX | Field-frame WASD (geodude live); demo uses **camera-tangent** arrows |

Offline `offline_demo_to_local_actions` still defaults `diffusion_scalar=2000` — override with `--diffusion-scalar` if comparing NPZs.
