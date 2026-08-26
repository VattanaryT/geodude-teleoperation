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
  `dt` column in the contact CSV shows what it actually is.
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
