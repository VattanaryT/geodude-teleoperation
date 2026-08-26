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
