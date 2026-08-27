# Handoff

Rewritten by the main agent before a `/clear`. Read it at the start of every
session. Keep it short — this is the state of play, not a changelog. Anything
that is permanently true belongs in `CLAUDE.md` instead.

**Last updated:** 2026-08-26 · branch `exp/hw-field-follow` · head `2fb363f`

---

## Where we are

**The contact force loop works.** Best run so far (`contact_20260826_165503.csv`,
49 s): `Fs` median 7.94 N against `Fd` 10, guards nearly silent (spike 0.5%,
wrong_sign 0.0%, fault 0.5%), `gain_scale` at 1.0 for 94.4% of ticks, mean
normal authority 2.80 mm/s. Peak fault force 12.3 N, down from 81.6 N earlier
in the day. Stationary holds sit at 9–10 N with `v_z` hunting ±3 mm/s.

**What still fails: the UR protective-stops.** Pendant text is
`C153A1 — "robot could not follow path; either there was a collision, or a
setting was incorrect"`, and it named the **shoulder joint**. Suggested checks:
objects in path, payload, centre of gravity, acceleration settings. The operator
reports the arm pose looks fine at the trips, and that **one lateral direction
trips it far more than the other**.

Critically: at the instant of one stop the force loop was perfectly healthy
(`Fs=+9.67 N`, `v_z=+0.0005`, no guards). **The fault is kinematic/lateral, not
force.** Stop looking for it in the force diagnostics.

## Leading hypothesis

A friction grab *is* the "collision" the pendant names. The lateral axes are
pure position control: the pose spring delivers `v_task = dt·K/M · e`, so at
K=2000 with the 4 mm `max_step_m` cap it shoves ~8 N into a tool held by
friction, the arm cannot follow, and the UR calls that path deviation. Nothing
in the loop knows the tool is stuck — `fx`/`fy` are read from the wrist F/T
sensor every tick and thrown away.

## Ranked, with the next action for each

1. **Lateral compliance.** `--xy-stiffness` (new, `2fb363f`) softens only blade
   indices 3/4. Try 1000, then 500. Cost is proportional lateral speed: 8 mm/s
   at 2000, 2 mm/s at 500. If this fixes it, the durable answer is real lateral
   admittance using the `fx`/`fy` we already measure (~30 lines in the pose
   spring), or extending `--contact-backend force_mode`, which is Z-compliant
   only today.
2. **Payload / CoG precision.** Named by the pendant. TCP is calibrated but the
   operator is unsure how precisely. Re-run the UR payload wizard with the
   chamfer EE mounted. No code.
3. **speedJ acceleration.** Named by the pendant. `exp/in-contact-dtc` set it
   explicitly to `accel=0.5 rad/s²`; on this branch it lives inside gafropy's
   urcl backend and nothing here controls it. `max_qddot` bounds *our commanded
   qdot* rate, not the acceleration the UR plans its ramp with. Unverified.
4. **Near-singular amplification.** Already fixed in `1222d57` (DLS λ
   0.001 → 0.05, per-joint clamp). "Pose looks fine" makes this less likely to
   have been the cause, but it was a real 500× amplifier. `sigma_min` in the
   contact CSV now settles it objectively.

## Ruled out — do not re-investigate

- **Loop timing.** `dt` median 5.19 ms against 5.00 nominal. The fix in `6a28a5e`
  was worth ~4%, not the 2× suspected. Never the cause of anything.
- **Sensor noise as the source of the big excursions.** Tick-to-tick `|dFs|` is
  0.19–0.23 N median. The ±10 N swings seen in the console were 0.5 s aliasing.
  The UR5e has a real 6-axis wrist F/T sensor and the grabs are physically real —
  the operator has observed them. Do not filter them.
- **The force loop as the cause of the protective stops.** See above.

## Tools built for this (use them, do not re-derive)

- `python examples/analyze_contact_log.py` — newest `speedj_logs/contact_*.csv`
  by default. Loop period, force spread and the rate it implies against
  `spike_dF_dt_N_s`, tracking, guard duty cycles, safety-halt windows,
  `sigma_min`/`clamp_scale`/`|qdot|`, per-axis lateral error, dominant
  oscillation frequency of `Fs` vs `v_z`, and worst events with timestamps.
  **~25 lines instead of pasting a terminal.**
- `python examples/plot_contact_spectrum.py [--column v_z]` — the same data as
  four plots, including what the Hann window is doing.
- The contact CSV streams to disk and survives a Ctrl+C or a protective stop;
  the speedJ `.npz` only lands on a clean exit.

## Next run

```bash
git pull
python examples/hardware_object_contact_teleop.py --contact-backend speedj --target-force 10
python examples/analyze_contact_log.py
```

Paste the analyzer output, not the terminal. Two things wanted from the
console: the `jacobian: 6xN sigma_min=... dls_lambda=...` line at startup (the
operator believes the vention rail is *not* in the jacobian — this confirms it),
and any `near-singular pose:` warnings.
