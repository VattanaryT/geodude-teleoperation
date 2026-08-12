# RTDE numpy kinematics backup (gafropy optional)

## Problem

Live RTDE `speedJ` teleop currently requires `gafropy` (`RobotInterface` MuJoCo FK/Jacobian). Published `gafropy` wheels use AVX-512 and SIGILL on CPUs without it (e.g. i9-13900KF). MuJoCo sim remains gafropy-only.

## Goals

- Prefer existing `_GafropyKinematicsBridge` when import/init succeeds.
- Automatic fallback to vendored UR5e FK/Jacobian + `qdot_from_error` + `speedJ`.
- VF / status EE pose from RTDE `getActualTCPPose` (pendant TCP, tool already set).
- Jacobian expressed at that same TCP frame.
- MuJoCo backend unchanged (still requires gafropy).

## Non-goals

- OSC torque control, Pinocchio, rewriting VF field math, gafropy-free MuJoCo.

## Architecture

```text
RTDE live (not dry-run)
  try _GafropyKinematicsBridge
  else _NumpyUrKinematicsBridge

Loop:
  ee (VF/status) = RTDE TCP duck motor
  kin.sync(q) / update twist target / qdot = kin.compute_qdot(...) → speedJ
```

## Components

| Piece | Role |
|-------|------|
| `ur5e_kinematics.py` | Vendored slim calibrated FK/J; UR controller frame (`apply_base_rotation=False`) |
| `se3.py` | Numpy twist / body-frame pose increment / pose error |
| `numpy_kinematics_bridge.py` | Backup bridge: flange J → TCP J via FK vs RTDE TCP; hold `T_ref`; `qdot_from_error` |
| `keyboard.py` | `gafropy.Twist` when available; else numpy twist duck |
| `rtde_control_loop.py` | Selection + numpy twist/qdot path; dry-run still `kin=None` |

## Control (numpy path)

1. `T_flange = FK(q)`, `T_tcp` from RTDE.
2. `J_tcp` from spatial map: \(v_{tcp}=v_f+\omega\times(p_{tcp}-p_f)\), \(\omega_{tcp}=\omega_f\).
3. Keyboard updates `T_ref`: with VF, twist is in **field / `R_field` frame** (parity with
   `TeleoperationController` reframe), composed on the projected EE, then `project` again;
   without VF, body-frame compose on TCP.
4. `error = [p-p_ref; rotvec(R R_ref^T)]` (current − reference); `q̇ = qdot_from_error(...)`.

Warn once if `‖t_flange→tcp‖ > 0.5` m.

## Errors

Both bridges fail → warn; VF from TCP; no motion (`speedJ(0)`). Fault/TF/dry-run gates unchanged.

## Testing

Unit FK/J map; numpy twist→qdot; loop selects numpy when gafropy `try_create` is None and can issue `speedJ`.

## Docs

README: RTDE live auto-fallback; pendant TCP must be set; MuJoCo still needs gafropy.
