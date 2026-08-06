# Calibration tools (pendant TCP / payload)

TCP and payload are calibrated on the **UR teach pendant**. This package provides:

1. **FT sensor** zero / bias recording  
2. **Tag extrinsics** validation from `ee_object_calib_recorder` marks (multi-view + tip-contact)

## Setup

```bash
pip install 'geodude-teleoperation[ur]'   # only needed for live FT
PYTHONPATH=src python -m geodude_teleoperation.calibration --help
```

---

## FT sensor

```bash
PYTHONPATH=src python -m geodude_teleoperation.calibration ft \
  --arm right --ip 192.168.1.2 --samples 100 --apply --yes \
  --note "still; tool mounted"
```

| Command | Purpose |
|---------|---------|
| `ft` | Sample `ActualTCPForce`, write `ft_sensor`; `--apply` → `zeroFtSensor()` |
| `show` | Print `calibration/<arm>.yaml` |

---

## Tag extrinsics validation

Uses marks from `bimanual` `ee_object_calib_recorder` (`session/marked/<id>/<cam>/`).

**Tip** = EE frame origin in the mark (`ee_frame`, usually `*_tool0` with pendant TCP in that TF). No extra tip offset.

### What it checks

| Mode | Question answered |
|------|-------------------|
| `multiview` | Across free-space views, is `T_tag_base` consistent? (tag known wrt EE/base) |
| `tip-contact` | With tip on the tag, is tip ≈ tag and cam↔tag ≈ cam↔tip? |
| `both` (default) | Both of the above |

`T_tag_base = T_EE_base @ T_cam_EE @ T_tag_cam` (same as the recorder).

### Operator procedure

1. Record several **free-space** marks where the wrist camera sees the tag (different EE poses).
2. Optionally record **tip-contact** mark(s): touch TCP tip to tag center, mark.
3. Run:

```bash
PYTHONPATH=src python -m geodude_teleoperation.calibration tag-extrinsics \
  --arm right \
  --session /path/to/ee_object_calib/<timestamp> \
  --cam <wrist_cam_name> \
  --tag-id 0 \
  --mode both \
  --contact-marks 0003
```

Omit `--contact-marks` to score tip-contact on **all** loaded marks (only do that if every mark is a tip-touch).

Thresholds (defaults): `--max-trans 0.005`, `--max-rot 2.0` (deg), `--max-tip-tag 0.005`, `--max-cam 0.01`.

Exit **0** = pass, **2** = fail / bad input. Summary is merged into `calibration/<arm>.yaml` under `tag_extrinsics` (use `--no-save` to skip).

### Interpreting failures

| Failure | Likely cause |
|---------|----------------|
| Multi-view Δt / ΔR large | Wrist-cam calib (`T_cam_EE`) or TF / EE frame mismatch |
| Tip↔tag large at contact | Pendant TCP not aligned with mark `ee_frame`, or tip not on tag center |
| Cam err large while tip↔tag small | Inconsistent `T_cam_EE` vs tag detection |

---

## YAML

`ft` / `tag-extrinsics` merge only their sections. Existing `tcp` / `payload` keys are preserved.

## Module layout

| File | Role |
|------|------|
| `cli.py` | `ft` / `show` / `tag-extrinsics` |
| `tag_extrinsics.py` | Multi-view + tip-contact scoring |
| `io.py` | YAML load/save/merge |
| `rtde_session.py` / `ft_sensor.py` | FT hardware helpers |

## Tests

```bash
PYTHONPATH=src python -m pytest tests/test_calibration_*.py -v
```
