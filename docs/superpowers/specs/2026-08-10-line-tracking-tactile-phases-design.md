# Phase 1.5–3: line tracking, DTC, tactile agent

**Date:** 2026-08-10  
**Status:** Phase 1.5 + 3a implemented on `speedJ`; Phase 2/3b architected

## Phase 1.5 — Line tracking + soft stand-off

- **Line tracking** (`--line-tracking`): zero the normal component of pose error along `R_field` (outward = −approach column). Tangent position + orientation still track; depth is free so contact force does not fight a stiff pose objective.
- **Soft stand-off** (`--soft-standoff`): feedforward `v = -k(d - d*) n_out` from VF signed distance. Implies line tracking.
- Keyboard field-z becomes normal feedforward (not a stiff depth target) unless `--no-keyboard-normal-ff`.
- RTDE prefers **numpy** kinematics when line tracking is on (explicit error + FF path).

## Phase 2 — Direct Torque Control (architecture)

- UR DTC requires PolyScope ≥ **5.23** / PolyScope X ≥ **10.10** and a **500 Hz** URScript loop calling `direct_torque(tau)` every timestep.
- Host `--control-hz` (default 50) is for Phase 1/1.5 `speedJ` teleop. Use `--verify-500hz` (+ optional `--polyscope-version`) to print readiness and a URScript skeleton (`direct_torque.py`).

## Phase 3 — Tactile agent

- **3a (implemented):** `--tactile` reads `actual_TCP_force`, runs a wrench PID vs `W_d` along the approach normal, maps force error → normal velocity on the free axis (`speedJ` + line tracking).
- **3b (next):** same `W_c` with `τ = -Jᵀ W` on the DTC bridge.

## Preprocessing vs runtime / τ

- One-time: PLY load, Laplacian heat diffusion (`diffusion_scalar` = τ/h²), surface `local_bases`.
- Realtime: tip → surface/WoS query every control tick.
- `dt = diffusion_scalar * mean_edge_length²` keeps smoothness density-independent.
