# Deep CGA Teleop (gafro) Implementation Plan

> **For agentic workers:** Implement task-by-task. Keep Euclidean Phase 1.5 (`--line-tracking`) working until CGA parity tests pass. Prefer dual-path (`--control-algebra se3|cga`) rather than a hard cutover.

**Goal:** Replace the thin gafropy Motor/Twist veneer + SE(3) “kill-normal” line tracking with a research-faithful CGA control path: algebraic **Line** objectives, **bivector** twists/wrenches, **multivector Geometric Jacobian**, and Phase 3 torque law  
\(\tau = -J_{ee,G}^\top \cdot (K_{L_{a,\perp}} W_{L_{a,\perp}} - D_V W_V + W_C)\)  
on UR Direct Torque Control at 500 Hz.

**Architecture:** Keep current `speedJ` + numpy SE(3) as **Backend A** (stable). Add **Backend B** (`cga_control/`) that speaks gafro/gafropy multivectors end-to-end, then a **DTC host** that only streams \(\tau\) / gains into a URScript 500 Hz loop.

**Tech Stack:** Python 3.10+, gafropy (or gafro C++ bindings), numpy (interop / tests only), ur-rtde, URScript DTC (PolyScope ≥ 5.23), pytest. Optional: existing `diffused_fields` for \(R_\text{field}\) / signed \(d\) only (field stays outside CGA until a later port).

**Spec companion:** `docs/superpowers/specs/2026-08-10-line-tracking-tactile-phases-design.md` (Phase 1.5 Euclidean). This plan is the **deepening** path.

---

## Current vs target (do not blur)

| Concept | Today (worktree) | Target (this plan) |
|--------|------------------|--------------------|
| Line tracking | Kill normal component of 6D pose error along \(R_\text{field}[:,2]\) | \(L = P_1 \wedge P_2 \wedge e_\infty\); motor \(M_{L_{ee}}^{L_{a,\perp}}\) via log map |
| Pose | `Motor` or 4×4 / rotvec | Motor + conformal points; sandwich \(X' = M X \tilde{M}\) |
| Twist / wrench | 6-vectors / `Twist` duck | Bivectors \(V\), \(W\); \(W' = \tilde{M} W M\) |
| Force PID | numpy \(W_d - W_\text{meas}\) → \(v_n\) | Same loop in bivector space; \(I_c\) duality where needed |
| Jacobian | \(6\times 6\) geometric matrix | Multivector \(J_G\); \(J_{ee,G} = \tilde{M} J_G M\) (per element) |
| Actuation | `speedJ(q̇)` @ ~50 Hz | `direct_torque(τ)` @ 500 Hz (URScript) |
| Algebra flag | (none) | `--control-algebra se3\|cga` |

**Hard rule:** Phase 1.5 Euclidean remains the default until Backend B matches kinematics/line error on recorded logs (± tolerance).

---

## Global constraints

- Do **not** delete `line_tracking.py` / numpy bridge until CGA path is proven on hardware dry-run.
- Field query (`ContactWosOrientationField`) may still emit \(R_\text{field}\), signed \(d\) in numpy; **adapters** build \(L_{a,\perp}\) and \(W_C\) from those until a CGA field exists.
- DTC: control loop **in URScript**; Python publishes \(\tau\) or \((W_C, gains)\) via RTDE registers / socket. Never claim Python alone is 500 Hz.
- Gravity: UR DTC compensates gravity — commanded \(\tau\) must **not** include \(g\).
- Safety: calibrate → Enter enable; fault / watchdog → \(\tau=0\) / leave DTC; tip speed & \(\|W\|\) clamps.
- gafropy AVX / missing wheel: CGA path **requires** gafropy; SE(3) path stays as fallback (no silent mix of algebras in one tick).
- Arm: right UR5e only (same as v1).
- Cite coverage Eq. 56 in code comments where the torque sum is assembled.

---

## File structure (new / touched)

| Path | Responsibility |
|------|----------------|
| `src/geodude_teleoperation/cga/__init__.py` | Lazy exports |
| `src/geodude_teleoperation/cga/primitives.py` | Conformal point, Line, Plane helpers; \(L_{a,\perp}\) from tip + \(R_\text{field}\) |
| `src/geodude_teleoperation/cga/motors.py` | Motor ↔ SE(3) round-trip; `log(M)` → Twist bivector; sandwich |
| `src/geodude_teleoperation/cga/wrenches.py` | 6-vector ↔ wrench bivector; frame change via sandwich; \(I_c\) dual helpers |
| `src/geodude_teleoperation/cga/jacobian_g.py` | Build \(J_G\) / \(J_{ee,G}\) from gafropy robot model or UR5e screws |
| `src/geodude_teleoperation/cga/line_tracker.py` | \(M_{L_{ee}}^{L_a}\), line cost / twist from `log(M)` |
| `src/geodude_teleoperation/cga/tactile_law.py` | Assemble Eq. 56 terms → \(\tau\) |
| `src/geodude_teleoperation/cga/adapters.py` | Bridge VF status + `actual_TCP_force` → CGA types |
| `src/geodude_teleoperation/dtc_host.py` | RTDE publish \(\tau\); rate/version checks (extend `direct_torque.py`) |
| `assets/urscript/direct_torque_loop.script` | 500 Hz `direct_torque` consumer |
| `tests/test_cga_*.py` | Algebra parity + motor/line/wrench/J tests |
| `ur_stringer_teleop.py` / `rtde_control_loop.py` | `--control-algebra`, wire Backend B |
| `docs/superpowers/specs/2026-08-10-deep-cga-teleop-design.md` | Short approved design (write in Task 0) |

---

## Task 0: Design lock + API spike

**Files:**
- Create: `docs/superpowers/specs/2026-08-10-deep-cga-teleop-design.md`
- Spike notes in plan or `docs/superpowers/plans/2026-08-10-gafropy-cga-api-spike.md`

- [ ] **Step 1: Inventory gafropy symbols** (Point, Line, Motor, Twist, Wrench/Motorized, Jacobian, `log`, reverse, dual). Document what exists vs what must be polyfilled in pure GA ops.
- [ ] **Step 2: Write design** — dataflow diagram: VF → \(L_{a,\perp}\) → line motor → \(V\); FT → \(W\); \(J_{ee,G}\) → \(\tau\); DTC.
- [ ] **Step 3: Decide interop convention** — when to drop to numpy (logging only) vs stay in multivectors.
- [ ] **Step 4: User review of design** before coding Tasks 1–7.

---

## Task 1: Conformal primitives + SE(3) round-trip

**Files:**
- Create: `cga/primitives.py`, `cga/motors.py`
- Create: `tests/test_cga_primitives.py`, `tests/test_cga_motors.py`

**Interfaces:**
- `euclidean_to_conformal_point(x: np.ndarray) -> Point`
- `line_from_point_direction(p, d) -> Line`  # \(P \wedge (P+d) \wedge e_\infty\) or library equivalent
- `motor_from_SE3(T: (4,4)) -> Motor`, `SE3_from_motor(M) -> (4,4)`
- `sandwich(M, X) -> X'`
- `motor_log(M) -> Twist`  # bivector / gafropy Twist

- [ ] **Step 1: Failing tests** — random rigid motions: sandwich point equals \(R x + t\); motor log ∥ finite screw; line normalized.
- [ ] **Step 2: Implement primitives/motors** with gafropy; skip/`pytest.importorskip("gafropy")` if missing.
- [ ] **Step 3: Parity table** — 100 random SE(3): max point error \(< 1\text{e-9}\) m.

---

## Task 2: Algebraic line objective (replace kill-normal)

**Files:**
- Create: `cga/line_tracker.py`, `cga/adapters.py`
- Create: `tests/test_cga_line_tracker.py`

**Logic:**
1. From TCP/tip + approach axis \(e_3\) of \(R_\text{field}\), build **end-effector line** \(L_{ee}\) (tool approach line through tip).
2. From desired contact geometry, build **target line** \(L_{a,\perp}\) (same point or projected tip; direction = surface approach / outward policy — lock in design).
3. Compute motor \(M\) taking \(L_{ee} \mapsto L_{a,\perp}\) (library line-to-line motor or decompose: rotate direction + translate in common perpendicular).
4. Twist command \(V = \log(M)\) (scale by gain \(K_L\)); **do not** manually zero Euclidean error axes.

**Adapters:**
- `target_line_from_vf(vf_status, tip_base) -> Line`
- `ee_line_from_tcp(tcp_pose, tip_offset) -> Line`

- [ ] **Step 1: Unit test** — EE already on target line ⇒ \(\|\log(M)\| \approx 0\).
- [ ] **Step 2: Unit test** — pure lateral offset ⇒ twist has translation in normal plane; free along line (no restoring component along \(L\) within tol).
- [ ] **Step 3: Compare to Euclidean** — same tip/\(R_\text{field}\) snapshot: CGA free-axis residual vs `kill_normal_linear_error` (document expected differences; assert free-axis stiffness \(\approx 0\)).
- [ ] **Step 4: Optional MuJoCo dry path** — `--control-algebra cga --dry-run` prints \(\|V\|\) without motion.

---

## Task 3: Bivector wrenches + frame change

**Files:**
- Create: `cga/wrenches.py`
- Create: `tests/test_cga_wrenches.py`
- Touch: `tactile_impedance.py` (thin wrapper or deprecate numpy-only path behind flag)

**Interfaces:**
- `wrench_from_tcp_force(f6: array) -> W`  # bivector
- `transform_wrench(M, W) -> W'`  # \(\tilde{M} W M\)
- `twist_wrench_pairing / Ic_dual` as needed for PID → \(W_C\)
- Keep `joint_torques_from_wrench` for SE(3) parity tests only

- [ ] **Step 1: Tests** — sandwich wrench vs dual-adjoint matrix formula on random \(M\) (parity).
- [ ] **Step 2: Port tactile PID** to operate on \(W_d, W_\text{meas}\) as bivectors; output \(W_C\).
- [ ] **Step 3: Duality smoke test** — twist from line tracker → candidate wrench via \(I_c\) (even if gains zero initially).

---

## Task 4: Geometric Jacobian \(J_G\) / \(J_{ee,G}\)

**Files:**
- Create: `cga/jacobian_g.py`
- Create: `tests/test_cga_jacobian.py`

**Approach:**
1. Prefer gafropy robot / task-space Jacobian if it already returns motorized screws; else build UR5e joint screw bivectors from known DH/calibrated chain (`ur5e_kinematics.py` as ground truth for axes).
2. \(J_{ee,G,i} = \tilde{M} J_{G,i} M\) for each column.
3. Map to matrix form only at the \(\tau = -J^\top W\) product boundary if gafropy lacks multivector matmul — document the conversion.

- [ ] **Step 1: Finite-difference check** — perturb \(q_i\); compare screw action on tip to \(J_G\) column (numeric).
- [ ] **Step 2: Frame change test** — \(J_{ee}\) vs body Jacobian from numpy `map_jacobian_to_point` within tolerance.
- [ ] **Step 3: Wire into tactile law** as \(J_{ee,G}\).

---

## Task 5: Eq. 56 torque assembly (Phase 3 law)

**Files:**
- Create: `cga/tactile_law.py`
- Create: `tests/test_cga_tactile_law.py`

**Control law (coverage Eq. 56):**
\[
\tau = -J_{ee,G}^{\top}\cdot\big(K_{L_{a,\perp}} W_{L_{a,\perp}} - D_V W_V + W_C\big)
\]

Where:
- \(W_{L_{a,\perp}}\) — wrench/cost from line tracking error (from \(\log(M)\) dualized or library line metric)
- \(W_V\) — damping on measured / commanded twist
- \(W_C\) — contact wrench PID from F/T
- \(K_{L}, D_V\) — gains (CLI)t

- [ ] **Step 1: Pure function tests** — zero all terms ⇒ \(\tau=0\); only \(W_C\) ⇒ matches \(-J^\top W_C\) numpy reference.
- [ ] **Step 2: Line-only term** — lateral error ⇒ restoring \(\tau\); slide along line ⇒ near-zero line cost.
- [ ] **Step 3: CLI gains** — `--cga-k-line`, `--cga-d-twist`, reuse `--wrench-kp` family for \(W_C\).

---

## Task 6: Dual algebra in the teleop loop

**Files:**
- Touch: `ur_stringer_teleop.py`, `rtde_control_loop.py`, `numpy_kinematics_bridge.py` (leave as SE(3))
- Create: `cga/control_loop.py` (tick function)

**CLI:**
- `--control-algebra {se3,cga}` default `se3`
- `cga` implies gafropy required; refuse numpy-only DTC fantasy
- While on `cga` + still `speedJ`: map \(V=\log(M)\) → \(q̇\) via \(J\) **or** temporary SE(3) projection of \(V\) (document as Phase 2a); prefer full Eq. 56 only when `--dtc` enabled

- [ ] **Step 1: `cga` dry-run** — prints line error, \(\|W_C\|\), \(\|\tau\|\) estimate; no motion.
- [ ] **Step 2: `cga` + `speedJ` (optional bridge)** — convert \(V\) → \(q̇\) for bring-up only; assert no kill-normal code path runs.
- [ ] **Step 3: Status line** — `algebra=cga`, `line_err=…`, `tau_norm=…`.

---

## Task 7: Phase 2 DTC host + URScript (500 Hz)

**Files:**
- Extend: `direct_torque.py` → `dtc_host.py`
- Create: `assets/urscript/direct_torque_loop.script`
- Create: `tests/test_dtc_host.py` (fake RTDE registers)

**URScript responsibilities:**
- Every timestep: read \(\tau[6]\) (or wrench+gains if computing on robot — prefer \(\tau\) from host for Python CGA), call `direct_torque(tau)`, `sync()`.
- Watchdog: if host stale → `direct_torque(0)` then exit to `speedj(0)` / `stopj`.

**Host responsibilities:**
- Verify PolyScope ≥ 5.23 / 10.10 (`--polyscope-version` / teach pendant).
- Publish \(\tau\) at best-effort rate (125–250 Hz acceptable if URScript holds last \(\tau\) for 1–2 cycles — **specify hold policy**).
- Log timestamp skew; fail if age > N ms.

- [ ] **Step 1: Script reviewed + loaded on pendant in manual mode.**
- [ ] **Step 2: Host dry publish zeros; confirm robot stays still.**
- [ ] **Step 3: Tiny \(\tau\) pulse test (joint 0) behind enable + e-stop ready.**
- [ ] **Step 4: Wire Eq. 56 output to publisher when `--dtc --control-algebra cga`.**

---

## Task 8: Hardware validation matrix

| # | Mode | Pass criteria |
|---|------|----------------|
| H1 | `se3` + `--line-tracking` (regression) | Unchanged slide behavior |
| H2 | `cga` dry-run | Finite line/wrench logs; no NaNs |
| H3 | `cga` + `speedJ` low gain | Approaches surface; free along approach under soft contact |
| H4 | `cga` + DTC zeros | No drift / no protective stop |
| H5 | `cga` + DTC + \(W_C\) only | Tracks `--desired-normal-force` within band |
| H6 | Full Eq. 56 | Tangent keyboard + force along normal without stutter |

- [ ] **Step 1: Fill checklist in README** (link this plan).
- [ ] **Step 2: Record one NPZ/CSV of \(q, \tau, F, d\) for H5–H6.**

---

## Task 9: Cleanup / docs

- [ ] Mark Euclidean kill-normal as **legacy** when `cga` is default for tactile (keep flag).
- [ ] Update `README_ur_stringer_teleop.md` algebra section + DTC warnings.
- [ ] Cross-link Eq. 56 and this plan from Phase 1.5 spec.
- [ ] Decide fate of `AdmittanceTeleoperationController` (pose-MSD): document “not CGA tactile”; optional later rewrite in bivectors.

---

## Suggested schedule (engineering weeks)

| Week | Focus |
|------|--------|
| 0 | Task 0 spike + design approval |
| 1 | Tasks 1–2 (primitives + algebraic line) |
| 2 | Tasks 3–4 (wrenches + \(J_G\)) |
| 3 | Task 5–6 (Eq. 56 + loop flag); MuJoCo/`speedJ` bring-up |
| 4 | Task 7 DTC zeros → \(W_C\) |
| 5 | Task 8 full law + docs |

---

## Risk register

| Risk | Mitigation |
|------|------------|
| gafropy missing Line/Wrench APIs | Polyfill with documented GA identities; keep tests against SE(3) dual adjoint |
| AVX/SIGILL on gafropy wheels | Pin build; CGA path optional; SE(3) remains |
| 500 Hz host latency | Compute \(\tau\) in Python @ 125 Hz; URScript holds; or move PID to URScript later |
| Algebra mix bugs | Single `--control-algebra` per process; assert in tick |
| Line definition ambiguity (which point on \(L_{a,\perp}\)) | Freeze in Task 0: tip projection onto target line through nearest surface point |

---

## Out of scope (explicit)

- Porting diffused_fields itself to CGA
- Left arm / bimanual CGA
- Replacing WoS with GA sphere primitives (future)
- Ergodic coverage planner (uses this stack as low-level only)

---

## Definition of done

1. `--control-algebra cga --dry-run` exercises Line + wrench + \(J_G\) without motion.
2. Parity tests: sandwich ≡ SE(3); wrench sandwich ≡ dual adjoint; \(J_G\) FD check green.
3. Eq. 56 implemented as one pure function with unit tests.
4. DTC URScript + host zeros proven; full tactile H5/H6 signed off.
5. Euclidean Phase 1.5 still available behind `--control-algebra se3`.
