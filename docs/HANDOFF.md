# Handoff

Rewritten by the main agent before a `/clear`. Read it at the start of every
session. Keep it short — this is the state of play, not a changelog. Anything
that is permanently true belongs in `CLAUDE.md` instead.

**Last updated:** 2026-08-26 · branch `exp/hw-field-follow`

---

## In flight

Contact-mode lift-off fix. **Implemented, uncommitted, and not yet run on
hardware or against the full test suite.**

Three changes, each behind its own flag so they can be bisected on the robot:

| flag | default | what it does |
|---|---|---|
| `--press-loop {dtc,legacy}` | `dtc` | `dtc` delegates the press seek to `ForcePIDController`, ported from `exp/in-contact-dtc`. `legacy` is the previous bare PID on fz alone. |
| `--no-free-along-normal` | on | projects the pose spring's task velocity off the surface normal while in contact |
| `--no-snap-follow-to-surface` | on | takes the along-normal component of the commanded point from `vf.project()` |

All three off reproduces the pre-fix behavior exactly.

Touched: `src/geodude_teleoperation/in_contact_control/` (new),
`controller.py`, `field_follow.py`, `tcp_force.py`,
`examples/hardware_object_contact_teleop.py`, four test files.

## Next

1. `pytest -q` on the hardware PC. This machine has neither `gafropy` nor
   `scipy` (and its `pytest` is itself broken), so only
   `tests/test_force_pid.py` (11 passing, run via `unittest`) has actually been
   exercised. Everything else is compile-checked only.
2. Hardware run: `--contact-backend speedj`, button 0 to latch contact. Success
   is `Fs` holding near `Fd` **while sliding laterally across curvature** — that
   is where it currently lifts off. The 0.5 s log line now reports the loop and
   the along-normal force: `contact force[rtde/dtc]: … Fs_n=+14.8N gain=1.00`.
3. Split into three commits (one per change) if the hardware run is good.

## Ruled out — do not re-run these

- **The regression is not force-output vs velocity-output.** All three contact
  loops with hardware hours behind them are velocity output, including
  `in_contact_control/force_pid.py` on `exp/in-contact-dtc`, which is the most
  mature. The only force-output paths are `7f303ae`'s `set_desired_wrench` (dead
  gafropy path) and UR native `force_mode`.
- **`hardware_free_teleop.py` never had a contact mode.** Comments elsewhere
  saying "same plant as hardware_free_teleop" refer to reusing its pose-spring
  math, not a force loop.
- **`541028e` ("Fs holds Fd well and moves tangentially") was already velocity
  output.** What `8f6d3c5` changed was the *plant*: it deleted the `_z_compliant`
  Cartesian branch (`qdot = J# @ merged_contact_twist(vz)`, pose spring off) and
  put the K=400 pose spring back underneath the press. `merged_contact_twist`,
  `contact_slide_twist` and `slew_slide` are now vestigial.

## Open questions

- **`Wrench.to_array()` blade order is unverified.** The
  `[fx, fy, tz, fz, ty, tx]` comment in `controller.py` came from commit
  `7f303ae`, not from gafropy. Settle with
  `ga.Wrench(tx=0,ty=0,tz=0,fx=0,fy=0,fz=1.0).to_array()`, then fix or delete
  the comment. Moot while RTDE is the force source.
- The overlay downstream of the press loop is still a scalar tool-Z speed, so a
  press axis tilted off tool Z is delivered scaled by `cos²θ`. That is a bounded
  positive gain inside a closed force loop — it costs response, not steady
  state — with a fallback if the axis approaches perpendicular. Revisit only if
  the hardware run shows a slow approach.
