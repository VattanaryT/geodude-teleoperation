# Force/torque calibration (pendant for TCP / payload)

TCP and payload are calibrated on the **UR teach pendant**. This package only
helps with the **force/torque sensor**: sample bias, optionally `zeroFtSensor()`,
and record results under `calibration/<arm>.yaml`.

Pendant TCP/payload values may still be stored in that YAML by hand (as on
`right.yaml`) for documentation — this tool does not set them on the robot.

## Setup

```bash
pip install 'geodude-teleoperation[ur]'   # ur-rtde for live FT
```

```bash
PYTHONPATH=src python -m geodude_teleoperation.calibration --help
# or: geodude-calibrate --help
```

## Workflow

1. Set TCP + payload on the pendant (already done for right).
2. Hold the arm still in free space (tool mounted).
3. Sample and zero FT; confirm residual wrench is near zero and responds to a light push.

```bash
cd /path/to/geodude-teleoperation

PYTHONPATH=src python -m geodude_teleoperation.calibration ft \
  --arm right --ip 192.168.1.2 --samples 100 --apply --yes \
  --note "still; tool mounted"

PYTHONPATH=src python -m geodude_teleoperation.calibration show --arm right
```

## Commands

| Command | Purpose |
|---------|---------|
| `ft` | Average `ActualTCPForce`, write `ft_sensor` to YAML; `--apply` → `zeroFtSensor()` |
| `show` | Print `calibration/<arm>.yaml` |

Shared flags: `--arm {left,right}`, `--calib-dir`, `--ip`.

`ft` extras: `--samples` (default 100), `--note`, `--bias Fx,Fy,Fz,Tx,Ty,Tz` (skip live sample), `--apply`, `--yes`.

## Sanity checks after zero

| Check | Expect |
|--------|--------|
| Still in free space | Force ≈ 0 ± ~1–2 N |
| Reorient in air | Stays near zero if pendant payload/CoG is good |
| Light push on tool | Clear force spike; release → near zero |

## YAML

`ft` merges only the `ft_sensor` block (and optional `robot_ip`). Existing
`tcp` / `payload` keys are left alone.

```yaml
ft_sensor:
  zeroed_at: "..."
  bias_wrench: [Fx, Fy, Fz, Tx, Ty, Tz]
  sample_count: 100
  note: "still; tool mounted"
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 2 | Validation error |
| 1 | RTDE / import / connection failure |

## Module layout

| File | Role |
|------|------|
| `cli.py` | `ft` / `show` entrypoint |
| `io.py` | YAML load/save/merge |
| `rtde_session.py` | `zeroFtSensor` / wrench average |
| `ft_sensor.py` | section builder + helpers |

## Tests

```bash
PYTHONPATH=src python -m pytest tests/test_calibration_*.py -v
```
