# Sandbox dual-loop control structure

This package runs **two independent control loops** that share one geometric
target and produce **one** robot twist:

\[
\mathbf{v}_{\mathrm{cmd}} = \mathbf{v}_{\mathrm{align}} + \mathbf{v}_{\mathrm{force}},
\qquad
\boldsymbol{\omega}_{\mathrm{cmd}} = \boldsymbol{\omega}_{\mathrm{align}} + \boldsymbol{\omega}_{\mathrm{force}}
\]

| Loop | Type | Owns | File |
|------|------|------|------|
| **Align** | Geometric spring–damper | Lateral pose + tool-\(Z\) direction | `alignment_admittance.py` |
| **Force** | **PID** on force error | Motion **along** \(n^*\) only | `force_pid.py` |
| **Orchestrator** | Sum + clip (no third law) | Calls both, merges twists | `dual_loop.py` |

**Force is not admittance.** The name “admittance” appears only in older align /
pose-hold helpers. Contact force is handled by **PID** in `force_pid.py`.

---

## Control structure (one diagram)

```text
  CAPTURE (once, or mesh waypoint)
  ─────────────────────────────────
  p*, R*  →  n* = R*[:,2]     (tool +Z of the target)
  L* = p* + s n*              (target line through the point)


  EVERY CONTROL TICK
  ────────────────────────────────────────────────────────────────

     live TCP (p, R)                    live wrench F (F/T)
            │                                    │
            │                                    │
            ▼                                    ▼
   ┌────────────────────┐              ┌────────────────────┐
   │  ALIGN             │              │  FORCE (PID)       │
   │  alignment_        │              │  force_pid.py      │
   │  admittance.py     │              │                    │
   │                    │              │  Fs = (F−bias)·n*  │
   │  e⊥  (lateral)     │              │  Fe = Fd − Fs      │
   │  e_R (tool Z→n*)   │              │  v∥ = PID(Fe)      │
   │  spring–damper     │              │  v_force = v∥ n*   │
   │  → v_align, ω      │              │  ω_force = 0       │
   │  then strip        │              │                    │
   │    v · n*  = 0     │              │  owns along n*     │
   │  owns ⟂ n*         │              │                    │
   └─────────┬──────────┘              └─────────┬──────────┘
             │                                   │
             └────────────────┬──────────────────┘
                              ▼
                    dual_loop.py
                    v  = v_align  + v_force
                    ω  = ω_align  + ω_force
                    clip ‖v‖, ‖ω‖
                              │
                              ▼
                       TwistCommand  →  Kinova JOG
```

### Axis ownership (why they do not fight)

```text
                 FORCE only (along n*)
                        │
                        n*  ▲
                            │
              ──────────────●──────────────  plane ⟂ n*
                           p*                 ALIGN only
                                              (lateral + aim tool Z)
```

- **Align** never commands along \(n^*\) (velocity along the normal is zeroed).
- **Force** only commands along \(n^*\) (no lateral, no \(\omega\)).
- **dual_loop** only adds the two twists and clips.

---

## What each loop does (math)

### Shared target

| Symbol | Meaning |
|--------|---------|
| \(p^*, R^*\) | Target pose (capture TCP or mesh waypoint) |
| \(n^* = R^*[:,2]\) | Target tool \(+Z\) in robot base |
| \(p, R\) | Live TCP each tick |
| \(F_d, F_s\) | Desired / measured force along \(n^*\) [N] |
| \(F_e = F_d - F_s\) | Force error (PID drives this → 0) |

### Align (`AlignmentAdmittance`)

| Step | |
|------|--|
| Lateral error | \(e_\perp = (I - n^*{n^*}^\top)(p^* - p)\) |
| Orientation | rotate live tool \(Z\) toward \(n^*\) (or full \(R^*\)) |
| Dynamics | spring–damper → \(v, \omega\) (`spring_damper.py`) |
| Free normal | \(v \leftarrow v - (v\cdot n^*)\,n^*\) |

### Force (`ForcePIDController`)

| Step | |
|------|--|
| Sense | \(F_s =\) (biased, optional LPF) force projected on \(n^*\) |
| Error | \(F_e = F_d - F_s\) |
| PID | \(v_\parallel = K_p F_e + K_i \int F_e + K_d \dot F_e\) |
| Command | \(\mathbf{v}_{\mathrm{force}} = v_\parallel\, n^*\), \(\boldsymbol{\omega}=0\) |

Optional shaping (still in `force_pid.py`): soft approach, spike hold, overshoot freeze, oscillation gain cut — **not** a separate admittance law.

---

## How the loops are used

```text
  window.py  (GUI / JOG thread)
       │
       │  capture() or waypoints → set_target()
       │  each tick: pose + wrench
       ▼
  dual_loop.py
       ├── force_pid.py          ← Force PID
       └── alignment_admittance.py ← Align
              └── spring_damper.py, so3.py
       │
       ▼
  robot_bridge.py  →  send_twist / go-to
```

| Layer | Files | Responsibility |
|-------|--------|----------------|
| **Laws** | `force_pid.py`, `alignment_admittance.py` | Compute each loop’s twist |
| **Combine** | `dual_loop.py` | One target, call both, sum, clip, diagnostics |
| **I/O** | `robot_bridge.py`, `window.py` | Sensors, JOG, UI |
| **Targets** | `waypoints.py` (optional) | Mesh points → \(p^*, R^*\) |

Minimal code use (no GUI):

```python
from geodude_teleoperation.in_contact_control.dual_loop import DualLoopController
from geodude_teleoperation.in_contact_control.types import PoseSO3

ctl = DualLoopController()
ctl.capture(PoseSO3(p=p_star, R=R_star))   # freeze target line L* / n*

cmd, diag = ctl.step(measured_pose, force_base=(fx, fy, fz), dt=0.02)
# cmd.linear_*  [m/s], cmd.angular_*  [deg/s]  → JOG
# force gains: ctl.config.force  (ForcePIDConfig: Fd_N, kp, ki, kd, …)
# align gains: ctl.config.align  (AlignmentAdmittanceConfig: k_perp, k_ori, …)
```

---

## Files to read (control only)

**Start here:**

1. **`dual_loop.py`** — structure and sum  
2. **`force_pid.py`** — force PID  
3. **`alignment_admittance.py`** — align geometry + spring–damper  

**Then if needed:**

4. `spring_damper.py` — Align dynamics only  
5. `so3.py` — orientation helpers for Align  
6. `types.py` — `PoseSO3`, `TwistCommand`  

**Usage / targets (not the laws themselves):**

7. `window.py` — how the GUI runs the loop  
8. `robot_bridge.py` — robot I/O  
9. `waypoints.py` — mesh scan points as targets  

```text
dual_loop.py
├── force_pid.py                 # FORCE = PID
│   └── types.py
└── alignment_admittance.py      # ALIGN = spring–damper
    ├── types.py
    ├── so3.py
    └── spring_damper.py
```

---

## Tests

```bash
pytest tests/test_force_pid.py tests/test_alignment_line.py -v
```

---

## Out of scope for this README

- Production outer-loop admittance: `surface_tracing/admittance_controller.py` (separate stack)  
- Older demos: `pose_hold.py`, `line_admittance.py`, `line_target.py`  
- Path / drive loop between many points (not implemented as a third summed law yet)
