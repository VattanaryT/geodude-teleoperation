# In-contact Eq. 56 DTC (pygafro) — design

**Date:** 2026-08-13  
**Status:** Implemented (host `directTorque` bring-up; URScript 500 Hz consumer next)

## Decision summary

| Choice | Value |
|--------|--------|
| Algebra | `pygafro` (PyPI `gafropy` SIGILLs on this CPU) |
| Actuation dual-path | `in_contact.actuation: speedj \| dtc` |
| Terminal clarity | Yellow `\033[93m` banner on start / Enter |
| Tangent in DTC | Extra wrench \(W_\text{tan}\) in Eq. 56 |
| 500 Hz | Host `directTorque` now; URScript consumer later |

## Control law (Eq. 56)

\[
\tau = -J_{ee,G}^\top \big( K_L W_L - D_V W_V + W_C + W_\text{tan} \big)
\]

- \(W_L\): line-to-line motor \(\log(M_{L_{ee}}^{L_{a,\perp}})\) via pygafro (`Line.getMotor`)
- \(W_C\): wrench PID on EE-frame \(W_d - W_m'\) (`actual_TCP_force`)
- \(W_V\): \(J\dot q\) damping
- \(W_\text{tan}\): SpaceMouse/WASD tangent → force FF
- \(J\): calibrated numpy UR5e geometric Jacobian at TCP (hardware fidelity)

## Modules

- `geodude_teleoperation/cga/` — primitives, line_tracker, wrenches, tactile_law
- `contact_teleop.py` — `actuation`, `step` / `step_tau`, yellow banner
- `rtde/control_loop.py` — `session.direct_torque`, DTC send path
- SSoT: `in_contact.actuation`, gains `K_line` / `D_twist`

## Safety

- DualLoop `speedj` remains flip-to-safe
- \(\|\tau\|\) clamp (`max_tau_nm`)
- Watchdog → \(\tau=0\)
- Gravity not included (UR DTC compensates)
