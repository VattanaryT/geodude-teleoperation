# Tag-based extrinsics validation (multi-view + tip-contact)

**Date:** 2026-08-06  
**Status:** Approved  
**Package:** `geodude-teleoperation`

## Goal

Validate wrist-camera / AprilTag / EE (pendant TCP) extrinsics using
`ee_object_calib_recorder` session marks:

1. **Multi-view:** tag pose is consistently known in base (and thus wrt EE) across free-space views.
2. **Tip-contact:** when the EE/TCP tip touches the tag, tip ↔ tag and cam ↔ tag match expectations.

## Assumptions

- Tip = EE frame origin in each mark (`T_EE_base`, typically `*_tool0` with pendant TCP in that TF).
- No extra demo `--tip-offset`; pendant TCP *is* the contact tip.
- Data from `marked/<id>/<cam>/meta.json` + `tags.json` (same formulas as the recorder).

## Check 1 — Multi-view `T_tag_base`

`T_tag_base = T_EE_base @ T_cam_EE @ T_tag_cam`

- Reference: mean translation + SVD mean rotation over views that see `--tag-id`.
- Per view: Δt (m), ΔR (deg = angle of `R_ref.T @ R_i`).
- Aggregate: RMS / max Δt and ΔR.
- Pass if `max_trans <= --max-trans` and `max_rot_deg <= --max-rot`.

## Check 2 — Tip-contact

On contact marks (default: all selected marks, or `--contact-marks`):

- `δ_tip_tag = ‖p_tag_base − p_EE_base‖` (expect ≈ 0)
- `t_pred = inv(T_cam_EE)[:3, 3]` (EE origin in camera)
- `t_meas = T_tag_cam[:3, 3]`
- `δ_cam = ‖t_meas − t_pred‖`
- Also report `‖t_meas‖` and `‖t_pred‖` (cam↔tag vs cam↔tip)
- Pass if `max δ_tip_tag <= --max-tip-tag` and `max δ_cam <= --max-cam`

## CLI

```bash
python -m geodude_teleoperation.calibration tag-extrinsics \
  --session /path/to/ee_object_calib/<ts> \
  --cam wrist_cam_0 --tag-id 0 --arm right \
  [--mode both|multiview|tip-contact] \
  [--contact-marks 0001,0002] \
  [--max-trans 0.005] [--max-rot 2.0] \
  [--max-tip-tag 0.005] [--max-cam 0.01]
```

Exit 0 on pass, 2 on fail / bad input, 1 on I/O errors.

## Persistence

Merge into `calibration/<arm>.yaml` under `tag_extrinsics` (summary metrics, timestamps, session path, pass/fail). Preserve existing `tcp` / `payload` / `ft_sensor` keys.

## Layout

- `tag_extrinsics.py` — load marks, geometry, scoring
- `cli.py` — `tag-extrinsics` subcommand
- Tests with synthetic SE3 marks (no robot)
