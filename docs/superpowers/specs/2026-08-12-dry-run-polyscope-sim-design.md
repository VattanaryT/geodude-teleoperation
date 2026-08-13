# Design: Dry-run Polyscope teleop simulation

**Date:** 2026-08-12  
**Status:** implemented  
**Command:** `geodude-ur-stringer-teleop --backend rtde --dry-run --viz`

## Goal

Validate free-space and in-contact keyboard teleop in Polyscope **without** commanding the robot. Same controller path as live; tip/TCP motion is integrated and published to viz only.

## Operator behavior

| Phase | Polyscope tip | Robot |
| --- | --- | --- |
| Calibrate (before Enter) | Tracks live RTDE TCP (pendant OK) | Receive-only |
| After Enter (`dry_run=1`) | Follows **simulated** TCP from controller | Never `speedJ` / control closed |
| `t` pause | Sim freezes (no integrate) | unchanged |
| `v` | VF on/off — **same as live** (default on) | n/a |
| `q` | Quit | disconnect receive |

In-contact (`--in-contact`): DualLoop Align + Force + tangent WASD run on the sim state. Force uses **real** F/T (typically ~0 off-surface) → approach seeks along `n*`; no synthetic contact model in v1.

## Architecture

1. **Kinematics in dry-run** — load numpy UR5e bridge (not `kin=None`). Prefer numpy so gafropy/MuJoCo is not required for sim.
2. **Gating**
   - `control_math_allowed(fault, tf_ok)` — `motion_enabled ∧ teleop_active ∧ ¬fault ∧ tf_ok` (ignores `dry_run`).
   - `commands_allowed(...)` — unchanged: math allowed **and** `¬dry_run` (gates `speedJ` only).
3. **Sim handoff** — on first tick after Enter in dry-run: snapshot `q`/`tcp_pose`. Subsequent ticks use sim state for VF query and viz.
4. **Free-space tip** — apply keyboard twists **before** the slow WoS/VF query; snap Polyscope TCP to `T_ref` immediately (pose-ref is the tip). Do **not** chase `T_ref` with `qdot·dt` in dry-run free-space (that looked frozen when `query_ms ≫ dt`).
5. **In-contact** — DualLoop `qdot` integrated with wall-clock `dt` (capped); real F/T.
6. **Watchdog** — refresh while dry-run sim is servoing.
7. **Status** — print `info: dry-run sim key=…` on each applied WASD; enable log notes `dry_run=1` + sim.
8. **Polyscope** — WASD/X/Y use ImGui key-repeat; Enter/t/v/f/q stay one-shot.

## Non-goals (v1)

- Synthetic contact force / Fd hold without real contact
- Ghost overlay of live TCP after Enter
- Changing `--viz-teleop-keys` offline preview (remains separate)

## Tests

- `control_math_allowed` vs `commands_allowed` dry-run gating
- Dry-run loop: Enter + twist → sim TCP moves / viz publish changes; FakeControl never constructed / no `speedJ`
- VF still applied when enabled; `v` still clears fixture for math path

## README

Dry-run section: Enter → practice WASD in Polyscope; tip moves in viz only; then `--no-dry-run` for live.
