# In-contact keyboard teleop (dual-loop)

**Date:** 2026-08-10  
**Status:** Approved for implementation  
**Approach:** Dedicated `--in-contact` RTDE path using `in_contact_control` DualLoop + keyboard tangent FF

## Goal

Enable UR keyboard teleop **in contact** with constant desired normal force **5 N**, using the existing Align + Force dual-loop, without free-space pose-reference teleop in this mode.

## Behavior

### Startup / approach

1. Launch with `--in-contact` (motion gated off / calibrate).
2. Operator pendant-jogs near the surface.
3. **Enter:** bias F/T, seed tip-tracked target, start Force PID with `Fd = 5 N` → **auto-seek along `n*`** (soft-approach caps in `force_pid`).
4. On load: Force holds ~5 N; Align aims tool-Z to `n*`; `w/s a/d` slide in the tangent plane.
5. `f` freezes / unfreezes live VF `n*`. `t` pauses. `q` quits.

### Axis ownership

| Source | Owns |
|--------|------|
| Force PID | Motion along `n*` only (`Fd` constant) |
| Align | Tool-Z → `n*`; lateral `p*` tip-tracked each tick (error ≈ 0) |
| Keyboard | Tangent linear only (`w/s a/d` in field frame ⊥ `n*`) |
| `f` | Freeze / unfreeze `n*` and `R*` |

Ignored in this mode: `y/x` (normal), angular keys.

### Architecture

- Fix `in_contact_control` imports to `geodude_teleoperation.in_contact_control.*`.
- New bridge `contact_teleop.py`: tip-track `p*`, live/frozen `n*`, keyboard tangent buffer, DualLoop step, sum twists, `q̇ = J⁺[v;ω]`.
- `rtde_control_loop`: when `--in-contact`, bypass free-space pose-ref path; still use RTDE + Jacobian + `speedJ`.
- Free-space path unchanged when flag off.

### CLI

- `--in-contact`
- `--desired-normal-force` default `5.0` → `ForcePIDConfig.Fd_N`
- Reuse `--keyboard-step`, Enter / `t` / `q` gates

### Out of scope

Free-space changes, DTC, Spacemouse, auto mode switch, keyboard `Fd` tuning, lateral pose-hold capture.

## Tests

- Package imports resolve (no `sandbox.line_admittance`).
- Tip-track + freeze `n*` unit tests on bridge.
- Keyboard tangent maps ⊥ `n*`; normal/angular ignored.
- CLI accepts `--in-contact`.
- DualLoop force seek direction smoke (Fe → v along n*).
