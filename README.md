# UR stringer keyboard teleop

Right UR5e keyboard teleop against the stringer layup field. Local data defaults live under `data/processed/`. Install `diffused_fields` as a package (no sibling checkout on `PYTHONPATH` needed for config/viz).

## Install

```bash
cd /path/to/ur-stringer-teleop
pip install -e '.[ur]'
# also: polyscope (for --viz), gafropy (MuJoCo / live kinematics)
```

```bash
geodude-ur-stringer-teleop --help
# or
PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH} python -m geodude_teleoperation.teleop.ur_stringer_teleop --help
```

## Safety

Calibrate (motion off) → check prints → **Enter** to enable → teleop.  
`t` / `v` toggle teleop-active / VF only. `q` quits.

| Key | Action |
|-----|--------|
| Enter | Enable motion |
| `t` / `v` | Pause teleop / toggle VF |
| `w/s a/d` | Linear (contact: tangent) |
| `q` | Quit |

## Dry-run (no motion)

```bash
export RIGHT_UR_IP=192.168.x.x
geodude-ur-stringer-teleop \
  --backend rtde --robot-ip "$RIGHT_UR_IP" \
  --dry-run --viz
```

Uses default `--ply` / `--T-tag0-base` under `data/processed/` when present.

## Shadow receive (pendant only)

Same as dry-run: receive-only RTDE, jog with pendant, confirm tip vs cloud before live motion.

## Live in-contact

```bash
geodude-ur-stringer-teleop \
  --backend rtde --robot-ip "$RIGHT_UR_IP" \
  --field-flow width --max-joint-velocity 0.2 \
  --in-contact --desired-normal-force 5.0 \
  --viz
```

Keep velocity low. Pendant near surface → **Enter** → force PID holds along VF normal → `w/s a/d` slide tangent.

## Layout

- `src/geodude_teleoperation/{field,viz,teleop,rtde}/` — library code  
- `config/pointclouds.yaml` — local field defaults  
- `data/processed/` — PLY / crop_meta / laser  
- `scripts/` — thin wrappers  
- `docs/superpowers/` — design history (not required for ops)

Design notes: `docs/superpowers/specs/2026-08-12-ur-stringer-teleop-layout-design.md`
