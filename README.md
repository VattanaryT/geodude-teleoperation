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

## Deterministic path (`--path-mode`) — every knob

No input device. Align, click **one point** on the object in viser (the end),
`p` previews with the orange ghost, Enter executes: descend where the tool is →
trace to the click → retract. `a` aborts, `r` clears, `f` flips the tool 180°.

### Geometry — where the plan goes

| flag | default | what it does |
|---|---|---|
| `--path-standoff` | 0.03 m | height above the surface the approach and retract sit at |
| `--path-depth` | 0.002 m | how far **below** the scanned surface the plunge and traverse are commanded. The knob for "it does not go far enough in" **before** the hand-off; after the latch engages the spring loses the normal and `--target-force` owns it alone |
| `--path-ds` | 0.005 m | traverse waypoint spacing. Lower it on tight curvature |
| `--path-two-point` | off | go back to picking start *and* end |
| `--no-path-axial-approach` | — | one diagonal lead-in instead of normal-then-tangential |

### Speed — and slower is NOT gentler

| flag | default | what it does |
|---|---|---|
| `--path-speed` | 0.02 m/s | free-space carrot speed |
| `--path-contact-speed` | 0.005 m/s | plunge and traverse speed |
| `--path-dither-mm` | 0 | transverse wobble on the carrot; **the measured difference between the stick and the plan** |
| `--path-dither-hz` | 3.0 | its frequency (under the 15 Hz mode and the 31 Hz `f_90`) |

A slow command is *stickier*, not gentler: `--stick-linear-scale` 0.0015 →
0.0008 took the stall fraction 5.6% → 37.5%. The stick runs at ~75 mm/s. Try
`--path-speed 0.05`.

### Contact — when the press takes over

| flag | default | what it does |
|---|---|---|
| `--target-force` | 15 N | the press, feed-forward, once latched |
| `--path-engage-gap` | 0.002 m | latch ON at this distance to the surface. Smaller = later = deeper |
| `--path-release-gap` | 0.005 m | latch OFF above this (the gap is hysteresis) |
| `--path-touch-n` | 0 | also require this much measured `Fs`. **The gate that catches a scan offset** |
| `--path-max-seek` | 0.01 m | how far past the planned depth the touch seek may go — millimetres the *arm* followed, not seconds |
| `--path-seek-lag` | 0.003 m | the seek deepens only while the arm is this close to the carrot |
| `--path-seek-timeout-s` | 20 s | give up on a seek the arm never follows |
| `--path-contact-at` | plunge | `execute` switches to contact mode at the start instead |
| `--press-overload-n` | 2×Fd | \|Fs\| above this cuts the press |

### Pacing and safety

| flag | default | what it does |
|---|---|---|
| `--path-hold-frac` | 0.8 | fraction of `--max-lead-m` at which the plan waits. **A force cap**: 0.6 delivered 13 N where 0.8 delivered 30 |
| `--path-stall-abort-s` | 2.0 s | window a stall is judged over |
| `--path-stall-mm` | 1.0 mm | TCP travel below this over that window, while behind, counts as stalled |
| `--path-plot` / `--no-path-plot` | on | render the PNG after every run and on Ctrl+C |
| `--no-path-ghost-arm` | — | hide the second, commanded-joints arm in viser |

### The gains that matter most here

`--dtc-fn-lin` (loop gain is `leverage × K`; leverage is the pose),
`--dtc-k-ang` under `--no-dtc-lambda` (the angular blade competes with the
descent for the same shoulder torque), and `--max-lead-m` (force ceiling).
`--no-dtc-lambda --dtc-k-lin 3000` makes `W = -K e` antiparallel to the error
by construction — it removed a 50° misalignment and all the burst windows.

## Other

- The live matplotlib plot is **off** by default; `--speedj-plot` opens it.
- `pip install -e .` once, then `pytest -q`. 21 tests fail on a clean tree
  (speedj_plot and lateral_compliance) — that is the pre-existing baseline.
