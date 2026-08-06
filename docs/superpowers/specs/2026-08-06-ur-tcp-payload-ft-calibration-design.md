# UR TCP / payload / FT calibration tools (geodude-teleoperation)

**Date:** 2026-08-06  
**Status:** Approved for implementation planning  
**Package:** `geodude-teleoperation`

## Goal

Add calibration tools for Universal Robots (Geodude dual UR5e) **TCP pose**, **payload** (mass + CoG), and **force/torque sensor** zeroing inside `geodude-teleoperation`. Tools both **persist results to YAML** and optionally **apply them to the robot** via RTDE.

## Non-goals (v1)

- Multi-pose TCP or payload CoG solvers (operator-supplied values only for TCP/payload)
- ROS 2 nodes or coupling to `bimanual` / `SSoT.yaml`
- Motion commands during calibration
- Lite6 / non-UR robots
- Changing MuJoCo teleop control laws

## Context

`bimanual` / `bimanual_moveit` expose camera and Mello/Gello calibration, and broadcast FT wrench topics, but have no TCP / payload / FT calibration utilities. `geodude-teleoperation` is a pure-Python teleop package (gafropy + MuJoCo) with a Geodude (dual UR5e) model; this work adds a hardware-facing calibration submodule without requiring ROS.

## Approach

Thin RTDE wrappers + YAML IO under `geodude_teleoperation.calibration`, with a single CLI module and subcommands. Optional dependency on `ur_rtde` for live apply / FT sampling.

## Package layout

```
geodude-teleoperation/
  calibration/                    # persisted results (YAML per arm)
    left.yaml                     # created on first save (may be empty stubs later)
    right.yaml
  src/geodude_teleoperation/
    calibration/
      __init__.py
      __main__.py                 # python -m geodude_teleoperation.calibration
      cli.py                      # argparse: tcp | payload | ft | apply | show
      io.py                       # load/save YAML, merge fields
      rtde_session.py             # connect; setTcp / setPayload / zeroFtSensor; read wrench
      tcp.py
      payload.py
      ft_sensor.py
```

- Console script in `pyproject.toml`: `geodude-calibrate = geodude_teleoperation.calibration.cli:main`
- Optional dependency group (e.g. `hardware` or `ur`) for `ur_rtde`

## YAML schema

Per-arm file, e.g. `calibration/right.yaml`:

```yaml
arm: right
robot_ip: 192.168.1.20
updated_at: "2026-08-06T13:39:00-07:00"

tcp:
  pose: [0.0, 0.0, 0.15, 0.0, 0.0, 0.0]   # [x,y,z,rx,ry,rz] m / axis-angle, flange→TCP
  frame: tool0
  source: manual                          # manual | measured (future)

payload:
  mass_kg: 0.98
  cog: [0.017, -0.007, 0.058]              # m, relative to tool flange
  source: manual

ft_sensor:
  zeroed_at: "2026-08-06T13:39:00-07:00"
  bias_wrench: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Fx,Fy,Fz,Tx,Ty,Tz
  sample_count: 100
  note: "arm still; tool attached"
```

**Merge semantics:** each subcommand updates only its section and top-level `updated_at` / `arm` / optional `robot_ip`. Other sections are preserved.

**Partial files:** missing sections are allowed. `apply` pushes only requested sections that exist (or errors if a requested section is absent).

## CLI & RTDE behavior

### Invocation

```bash
python -m geodude_teleoperation.calibration tcp --arm right --pose 0,0,0.15,0,0,0 [--apply] [--ip ...]
python -m geodude_teleoperation.calibration payload --arm right --mass 0.98 --cog 0.017,-0.007,0.058 [--apply]
python -m geodude_teleoperation.calibration ft --arm right [--samples 100] [--apply]
python -m geodude_teleoperation.calibration apply --arm right [--what tcp,payload,ft]
python -m geodude_teleoperation.calibration show --arm right
```

### Defaults

- `--arm` required: `left` | `right`
- Robot IP: `--ip` if given, else YAML `robot_ip`, else error when RTDE is needed
- Calibration directory: package-root `calibration/` (overridable via `--calib-dir`)

### RTDE usage

Only when `--apply` is set or when `ft` needs a live wrench average:

| Subcommand | Write to controller | Read |
|---|---|---|
| `tcp` | `setTcp(pose)` | optional echo of active TCP if available |
| `payload` | `setPayload(mass, cog)` | — |
| `ft` | `zeroFtSensor()` when `--apply` | average `ActualTCPForce` over N samples → `bias_wrench` |
| `apply` | push selected sections from YAML | — |

### Interactive fallback

If `--pose` or `--mass`/`--cog` are omitted, prompt on stdin with the same validation as flags.

### Safety

- No motion commands; operator must hold the arm still for FT sampling
- `--apply` prints a one-line confirmation of values; `--yes` skips the prompt
- Apply failures after a successful YAML write leave the file intact and exit non-zero

## Errors

| Condition | Exit | Behavior |
|---|---|---|
| Invalid pose/CoG length or non-numeric / mass ≤ 0 | 2 | No YAML write |
| `--apply` / live FT without `ur_rtde` or unreachable IP | 1 | YAML kept if already saved; clear message |
| `apply` missing a requested section | 2 | List present sections |
| FT sample connection drop mid-average | 1 | Do not update `ft_sensor` block |

## Tests

No robot required:

- `io.py`: merge sections, YAML round-trip, unrelated keys preserved
- Validators: pose length 6, mass > 0, CoG length 3
- CLI smoke: `tcp` / `payload` / `show` with a temp calib dir; mock RTDE for `--apply` path

## Dependencies

- Existing: `numpy`, PyYAML (add if not already a direct dependency)
- Optional: `ur_rtde` for hardware apply / FT sampling

## Success criteria

1. Operator can save TCP, payload, and FT bias/zero metadata to `calibration/<arm>.yaml` without a robot when not applying.
2. With `ur_rtde` and a reachable UR5e, `--apply` (or `apply` subcommand) sets TCP, payload, and zeros the FT sensor on the controller.
3. Re-running one subcommand does not erase other calibrated sections.
4. Unit tests for IO and validators pass offline.
