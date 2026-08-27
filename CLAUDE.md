# geodude-teleoperation

Bimanual UR teleoperation with in-contact control over a laser-scanned stringer.
SpaceMouse/keyboard drives a gafropy speedJ plant; contact mode adds a force loop
along the surface normal of a diffused orientation field.

**Read `docs/HANDOFF.md` first** — it says what was in flight when the last
session ended, and what was already ruled out.

---

## Facts that cost real time to rediscover

These are established, not guesses. Trusting the wrong one here moves hardware.

### Force feedback

- **`interface.get_state().get_wrench()` is always zero on hardware.** gafropy's
  `URRobot::sampleState` copies position/velocity/torque only — it never calls
  `SystemState::setWrench`, even though the RTDE recipe includes
  `actual_TCP_force`. Any force loop that trusts it presses open-loop. That is
  what caused C153A1. Read force through `tcp_force.py` instead.
- **UR `getActualTCPForce()`** is `[Fx, Fy, Fz, Rx, Ry, Rz]` in the **tool**
  frame, so Fz is index 2. It reports the **reaction** (environment on TCP), so
  a press along +Z reads `Fz < 0`. The repo-wide convention is `Fs > 0` in
  compression, hence `Fs = -press_sign * fz`.
- **Only one RTDE client** is allowed on most controllers, and gafropy's urcl
  already holds it. `connect_tcp_force_reader()` can therefore fail on a
  perfectly healthy robot — treat that as "no force feedback" and refuse to
  press, never as a reason to fall back to open loop.
- **The UR5e has a real 6-axis wrist F/T sensor and `getActualTCPForce()` reads
  it** on the e-series — it is not a motor-current estimate. The measured
  tick-to-tick spread backs this up: 0.19 N median in contact, ±0.3 N in free
  space, far better than any current-based estimate. Consequence: **large
  in-contact excursions are physically real, not measurement artifact.** The
  operator has observed the friction grabs directly. Do not "fix" them by
  filtering — that is lying to yourself about a real contact condition.
- **The free-space offset is sensor zeroing, not payload error.** The TCP and
  payload are calibrated, so the ~10 N seen with nothing touching is F/T bias,
  which `zero_ftsensor()` would clear — and the speedj backend cannot issue it
  (single RTDE client; only the `force_mode` URScript path can). The per-session
  base-frame tare is the stand-in, and it is strictly better than a one-shot
  zero because it stays valid as the wrist turns.
- **`getActualTCPForce()` is not gravity compensated** unless the UR payload is
  configured, and the **speedj backend never issues `zero_ftsensor()`** (only
  the `force_mode` URScript does). A perfectly live sensor therefore reads
  several N of tool weight with nothing touching. `ContactTeleoperationController`
  now tares once per session in `start()` (`tare_force()`, `--no-force-tare` to
  disable) and both seek laws subtract it; the dead-feedback abort deliberately
  still reads `fz_raw`, or a tared-to-zero sensor would look dead. Untared, the
  dtc loop trips `wrong_sign` at `Fs ≤ min(−2, −0.5·Fd)` and commands
  `−unload_v_m_s` forever — button 0 retracted instead of approaching. The tare
  is stored in **base** and rotated back into the tool frame each tick, because
  the weight it removes is fixed in base — a tool-frame constant is only right
  at the orientation it was captured at, which is why two runs minutes apart
  tared at `fz=+3.74 N` and `fz=+10.42 N` on the same hardware.
- **Twist blade order** is `[e12, e13, e23, e1i, e2i, e3i]` = `[angular,
  linear]`, so linear Z is **index 5**. That is the slot `set_desired_wrench`
  and `AdmittanceConfig` use (`spacemouse.py:279`).
- **UNVERIFIED — do not build on it:** `controller.py` claims
  `Wrench.to_array()` is `[fx, fy, tz, fz, ty, tx]` (fz at index 3). That came
  from a comment written in commit `7f303ae`, not from gafropy docs. Settle it
  with `ga.Wrench(tx=0,ty=0,tz=0,fx=0,fy=0,fz=1.0).to_array()` and see which
  slot holds the `1.0`, then fix or delete the comment. It is moot while RTDE is
  the force source. `examples/visualize_robot.py:67` shows the index-free idiom
  (build a `ga.Wrench(...)` and compare arrays) — prefer that over any literal
  index.

### Contact control invariants

- **The outer PID owns `Fd`, and nothing else does.** In the `speedj` backend
  the contact controller is never handed to `interface.control(...)` (see
  `enter_speedj_contact`) and `_run` skips `set_desired_wrench` whenever
  `_on_vz` is set, so the gafropy VelocityAdmittance plant beneath it is never
  stepped. `contact_config` and `z_damping` are dead weight on that path. Two
  loops authoritative on the same axis is the fight that produces drift.
- **Never keep inching into a rigid stop (C153).** At or over target, press
  velocity is 0 or retract. speedJ cannot yield.
- **`unload_v_m_s` is a *floor* on the escape speed, never a ceiling.** The
  overload guards used to assign it, throwing away the PID's own answer exactly
  when the load was worst: at `|Fs| = 40 N` with `Fd = 10 N` the PID wants the
  full `max_v_along` and the guard forced half of it, so shedding 30 N through a
  stiff contact took ~2 s — the pendant protective-stopped well inside that.
  Use `min(v_along, -unload)`, not `= -unload`.
- **A UR protective stop must stop the press, not queue one up.** While stopped
  the controller discards the speedJ stream, but the force loop went on
  computing against a robot that was not moving, so the error stayed large and
  the held command would have been applied the instant someone cleared the
  pendant. `safety_mode_fn` (from the same RTDE receive socket as the force
  reader — there is only one client allowed) zeroes the overlay and the slew
  while safety mode is outside NORMAL/REDUCED. A *failed* read is not a stop:
  refusing to press on a missing status signal is the worse failure.
- **The dtc guards must shape the seek, not own it.** Measured on this rig
  (`contact_20260826_145209.csv`, 9152 ticks): `SPIKE` on 75.4% of ticks, `v_z`
  exactly 0 on 68.6%, `gain_scale` at 1.0 on 4.3% — about **0.9 mm/s of mean
  normal authority against a 24 mm/s lateral slide**, so any slope past ~2°
  outran the loop and the tool left the surface. Three constants, all inherited
  from the in-contact-dtc rig's F/T sensor, were doing it:
  `spike_dF_dt_N_s` 40 → **250** (ordinary tick-to-tick spread here is already
  ~123 N/s; a real impact hit 1260 N/s); the spike gain cut is now
  **rising-edge only** (re-cutting every tick outpaced `osc_recover_per_s`);
  and a spike **under** target now clamps to `spike_v_cap_m_s` instead of
  zeroing, because overshoot/mag_overload/fault are *level* checks on `fs_raw`
  and far more trustworthy than any rate on this signal.
- **`force_rate_damp_max_frac` bounds the rate damping at 0.5·`max_v_along`.**
  Uncapped, `0.00015 m/s per N/s` subtracts 5.6 mm/s at this rig's median rate
  and 18.5 mm/s at p95, against an 8 mm/s ceiling. Past 1.0× it does not damp
  the seek, it inverts it — a rising force commands full retract.
- **`spike_dF_dt_N_s` is a rate over `spike_rate_window_s` (20 ms), not over one
  tick, and only counts while `|Fs| ≥ contact_threshold_N`.** At 200 Hz a
  single-tick difference turns UR's own ±0.3 N of noise into ±120 N/s, well
  past the 40 N/s threshold, so the guard latched forever: every log line read
  `SPIKE gain=0.50` with `Fs≈0` and `v_z=+0.0000`, and contact mode never
  approached at all. An impact is a *sustained* rise and still clears 40 N/s
  against a 20 ms baseline.
- **The contact loop integrates on the *measured* tick period, not on
  `update_period`.** `_run` sleeps `update_period` after the work, and the work
  includes a blocking RTDE read, so the real period is always longer than
  nominal. Nominal dt rescales the integral and the LPF pole and inflates the
  spike guard's rate by the same ratio — dividing a real ~32 ms change by a
  nominal 20 ms baseline manufactures trips. `_tick_dt()` measures it and the
  `dt` column in the contact CSV shows what it actually is. **Measured: 5.18 ms
  median against 5.00 nominal** — worth ~4%, not the 2× that was suspected, so
  this is correct but was never the cause of anything.
- **Both seek laws answer a wrong-sign reading the same way.** `Fs` at or below
  `min(−2, −0.5·Fd)` is a bounce or a flipped reading, not more force owed.
  Without that guard the legacy PID reads `fe = Fd − Fs` growing without bound
  and answers `Fs = −11.6 N` with full approach speed — that is what drove the
  tool into the part right before a C153 protective stop.
- **`v_slew_mps2` is 0.4, not 0.05.** At 0.05 a reversal from +8 to −8 mm/s
  takes 0.32 s, so the loop keeps pressing ~1.3 mm after deciding to back off
  (hardware printed `v_z=+0.0023 m/s` at `Fs=21.7 N`, `Fd=10 N` — the command
  was already negative, the slew was still descending). The joint ramp is
  separately bounded by `max_qddot`, so this only ever had to be gentle.
- **`push_stop_frac_of_fd` is 1.0, not 0.5.** The ported guard zeroed every
  approach command from `0.5·Fd` upwards, which is right for DTC's *approach*
  phase handing off to the force loop but wrong for the loop that has to hold
  `Fd`: `--target-force 5` could only ever settle near 2.5 N, which reads as
  under-press and then as lift-off over curvature.
- **Free along the normal — align never owns the unbound axis, force does.**
  `stiffness[5,5] = 0` frees only the *body Z of the pose error*, which is not
  the surface normal: field-follow slerps tool Z behind the field normal on
  purpose, so mid-slide the two differ and the stiffness on the remaining axes
  gets a component along the true normal. `exp/in-contact-dtc` encodes this as
  `free_along_normal` in `alignment_admittance.py`.
- **`robot_state.get_velocity()` inside `compute_control_command` is the
  control-group slice** (len 7 for one arm + vention), *not* the full bimanual
  q. `get_joint_indices()` are system-global and must not index it — that
  crashed with index 15 on size 7 (`controller.py:293`).

### Joint space — the path the force loop cannot see

- **C153A1 "position deviates from path" (shoulder) is kinematic, not force.**
  It fired with the contact loop perfectly healthy (`Fs=+9.67 N` vs `Fd=10`,
  `v_z=+0.0005`, no guards). Do not look for it in the force diagnostics.
- **The DLS regularizer was `1e-6` (λ=1e-3), giving the jacobian inverse a peak
  gain of `1/(2λ)` = 500.** Inherited from ufactory_demos via the root commit,
  never validated here, while the sibling `TeleoperationController` has always
  used `damping=0.05` for the same math — 2500× more. At σ_min = 0.005 (an
  ordinary near-singular UR5e pose) the logged 2.6 mm pose error commanded
  **0.8 rad/s, pinned at the clamp**, delivering almost no TCP motion — the arm
  windmills one joint until the UR protective-stops. Now `dls_damping=0.05`
  (`--dls-damping 0.001` restores the old behavior). Reduction to the working
  case: at σ = 0.2 the gain goes 5.00 → 4.71, a 6% change.
- **The velocity clamp was `norm(qdot)` only, so one joint could carry the whole
  budget** — exactly what a near-singular pose produces. Now also per-joint
  (`--no-per-joint-velocity` reverts).
- **`sigma_min` must come from `J @ Jᵀ` (6×6), not the `Jᵀ J` (7×7) the inverse
  is built from.** With 7 joints and 6 task DOF the latter is rank-deficient by
  construction, so its smallest eigenvalue is always ~0 and every pose reads as
  singular. That extra dimension is the redundancy nullspace, not
  ill-conditioning.
- **The vz press overlay is added *before* the shared clamp** (`controller.py`,
  `twist[5] = vz` then `norm(qdot)`), so a saturating pose spring silently
  scales the press down by the same factor while the contact log keeps printing
  the *pre-clamp* `v_z`. A healthy-looking force log is not evidence the press
  was healthy. `clamp_scale` in the contact CSV is the one that tells you.
- **The constants that run on hardware are in `_make_approach_controller`, not
  the class defaults.** It passes `stiffness=2000, max_joint_velocity=0.8`; the
  class defaults are 3000/1.0 and `hardware_free_teleop.py` uses those. A stale
  comment claimed the two matched — they never did.
- **Lateral stiffness is the cheapest contact compliance we have.** The pose
  spring delivers `v_task = dt·K/M · e`, so at K=2000 and the 4 mm `max_step_m`
  cap the lateral command tops out at 8 mm/s and shoves with 8 N into a tool
  held by friction — which is what the UR reports as "could not follow path".
  `--xy-stiffness` softens only blade indices 3 and 4 (tool X/Y), leaving
  angular and the force-owned Z alone. Max lateral speed scales with it: 8 mm/s
  at 2000, 2 mm/s at 500.
- **`exp/in-contact-dtc` called `speedJ` with an explicit `accel=0.5 rad/s²`.**
  On this branch that argument lives inside gafropy's urcl backend and nothing
  here controls it. `max_qddot` bounds *our commanded qdot* rate; it does not
  change the acceleration the UR plans its ramp with. Unverified, still a
  candidate.

### Geometry and fields

- `OrientationField.approach_basis_at(p)[:, 2]` is the **into-surface** approach
  Z near the object, blending to table-flat (EE +Z = world −Z) far away.
  `display_basis_at` / `field_z_arrows(outward=True)` flip it for viser, so the
  arrows you see point the opposite way from what the EE snaps to.
- `field_follow` runs `vf.project()` in a helper process precisely so the 500 Hz
  speedJ thread never blocks on it. Do not query `vf` from a control thread.

---

## Branch map

Six branches, two mutually incompatible stacks. Check which one you are on
before trusting any module name.

| branch | stack |
|---|---|
| `exp/hw-field-follow` | **active.** gafropy speedJ + viser + field-follow + SpaceMouse + contact |
| `exp/in-contact-dtc` | RTDE-direct `ur_stringer_teleop`: `in_contact_control/`, `cga/`, `calibration/`, `rtde/`, `teleop/`, `viz/`. Origin of the DualLoop force loop |
| `exp/sim-replay` | speedJ live plots, bag-TCP and sim replay |
| `base/stringer-teleop`, `base/viser-force` | older bases |
| `main` | contact mode here is **non-functional** — it PIDs on the dead `get_wrench()` path |

---

## Running things

```bash
pip install -e .        # once per environment
pytest -q               # thereafter, from anywhere in the repo
```

`tests/conftest.py` also puts `src/` on the path when the package is not
installed, so plain `pytest -q` works without the editable install too. Most
tests need `gafropy` and `scipy`; `tests/test_force_pid.py` needs only numpy.

Library data (`frames/`, laser `crop_meta`/offsets, `chamfer_ee.stl`) lives in
`src/geodude_teleoperation/data/` and ships in the wheel, because module-scope
defaults in `laser_frames`/`ee_viz` read it. The bulk meshes and scans stay in
`assets/`, which belongs to `examples/` and is not packaged.

`--contact-log` (on by default) streams one CSV row per force tick — `dt`,
`Fs`, `Fd`, `v_z`, the tare, and the dtc flags — next to `speedj_logs/`. It is
flushed as it runs, so a protective-stopped or Ctrl+C'd session still leaves
the record that explains it; the speedj `.npz` only lands on a clean exit.
`python examples/analyze_contact_log.py` (newest log by default) summarizes
loop period, force noise per tick, time inside the target band and which guard
held `v_along` — the 0.5 s console lines alias all of that away.

Hardware entry point is `examples/hardware_object_contact_teleop.py`. Contact
lift-off fixes are behind three independent flags so a session can bisect them:
`--press-loop {dtc,legacy}`, `--no-free-along-normal`,
`--no-snap-follow-to-surface`. All three off reproduces the pre-fix behavior.

**Nothing in `src/` may import from `examples/`.**

---

## Working style here

- Subagents in `.claude/agents/` exist to keep expensive reading out of this
  conversation. Prefer `geo-history` over running `git show` yourself — raw
  diffs across six branches are the worst context sink in this repo. `geo-control`,
  `geo-field` and `geo-hw` are research-only; the main agent makes the edits.
- Sign and frame conventions are the failure mode that reaches hardware. When
  changing one, show the reduction to the previously-working case rather than
  arguing from first principles.
- Before `/clear`, rewrite `docs/HANDOFF.md`.
