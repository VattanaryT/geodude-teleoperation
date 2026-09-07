# geodude-teleoperation — runbook

Read `CLAUDE.md` for why any of these numbers are what they are, and
`docs/HANDOFF.md` for the state of play. Both are gitignored; this file is not,
so the commands survive a fresh clone.

Branch: `exp/hw-field-follow`. Everything below is `--contact-backend dtc
--dtc-law wrench`, which is the default on this branch.

## Tuned gains (2026-09-05)

Found by a five-run campaign on the width traverse, one variable at a time.
**The two blades ring at different frequencies and take opposite fixes** — the
linear/shoulder burst is at 30–36 Hz = `f_90`, where the damper is a *source*
(so less D); the angular/wrist ring is at 15.2 Hz, where it still works (more
D). See CLAUDE.md.

```
--dtc-hz 500 --dtc-fn-lin 3.0 --dtc-damping-ratio 0.25 --dtc-damping-max 16
--dtc-fn-ang 3.0 --dtc-damping-ratio-ang 0.25
--follow-hz 50 --loop-hz 200 --follow-linear-approach
--max-lead-m 0.03 --dtc-tau-max 140 140 140 30 30 30
--dtc-joint-hold-joints 3 4 5 --dtc-stiction-assist-nm 2.0 --dtc-stiction-max-nm 4.0
```

Ceilings, measured — do not exceed without a fresh log:
`--dtc-fn-lin 3.5` bursts (11 windows, 71 N·m on j1) at both D 11.0 and 13.99.
`--dtc-damping-ratio 0.40` gave 11 burst windows where 0.25 gives 0.

## Push test — deflect the arm by hand, it springs back

No align, no device, no field-follow: the target is frozen at the handoff pose
and the pose spring is the only thing holding the arm.

```bash
python examples/hardware_object_contact_teleop.py --dtc-push-test \
    --dtc-hz 500 --dtc-fn-lin 3.0 --dtc-damping-ratio 0.25 --dtc-damping-max 16 \
    --dtc-fn-ang 3.0 --dtc-damping-ratio-ang 0.25 \
    --dtc-tau-max 140 140 140 30 30 30 \
    --dtc-joint-hold-joints 3 4 5 --dtc-stiction-assist-nm 2.0 --dtc-stiction-max-nm 4.0
```

`--dtc-push-test` is what makes the arm hold: it implies
`--dtc-compliance-hold`, sets `hold_at_tick=False` (so the target does **not**
chase your hand) and `startup_align=False`. **Without it the idle hold re-bases
the target onto the live pose every tick and the arm has no spring at all — it
stays wherever you push it.** That is the usual reason "it didn't deflect back".

For a paired push (push `+u`, release, settle; then `−u`), hold still 5 s after
each release, then:

```bash
python examples/analyze_paired_push.py     # bias vs friction, per joint
python examples/analyze_dtc_log.py         # bursts, ringing, stall, leverage
```

## Free-space teleop

```bash
python examples/hardware_object_contact_teleop.py --dtc-compliance-hold \
    --dtc-hz 500 --dtc-fn-lin 3.0 --dtc-damping-ratio 0.25 --dtc-damping-max 16 \
    --dtc-fn-ang 3.0 --dtc-damping-ratio-ang 0.25 \
    --follow-hz 50 --loop-hz 200 --follow-linear-approach \
    --max-lead-m 0.03 --dtc-tau-max 140 140 140 30 30 30 \
    --dtc-joint-hold-joints 3 4 5 --dtc-stiction-assist-nm 2.0 --dtc-stiction-max-nm 4.0
```

## Contact

Add to the free-space command:

```
--target-force 10 --press-engage-frac 1.0 --press-auto-approach
```

`--press-auto-approach` has the press close the `Fd − Fs` gap itself from the
moment button 0 is pressed (the pose spring cannot press on this rig), and it
breakaway-sizes `--press-kp` to 1.80, so the free-space command is **28 N**.
`--press-engage-frac 1.0` keeps the surface normal with the pose spring — and
the stick able to steer along it — until `Fs` actually reaches `Fd`.

Press button 0 with the tool **clear of the part**: the force tare happens on
that first press, and pressing while touching bakes the contact force into the
bias. Check the sign first — push the tool by hand and confirm `Fs` goes
*positive* in compression, else flip `--force-sign`.

```bash
python examples/analyze_contact_log.py
```

Watch `F_emit` (what reached the arm) against `F_press` (what the law asked
for), `engaged` going 0→1, and `n_deg` staying near 2–3°.

## Other

- The live matplotlib plot is **off** by default; `--speedj-plot` opens it.
- `pip install -e .` once, then `pytest -q`. 21 tests fail on a clean tree
  (speedj_plot and lateral_compliance) — that is the pre-existing baseline.
