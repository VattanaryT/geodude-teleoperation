# UR stringer teleop — manual dry-run checklist

Operator checklist before enabling motion on hardware.

## MuJoCo dry-run (field + prints only)

```bash
geodude-ur-stringer-teleop --ply /path/to/object_tag0.ply --dry-run \
  --field-flow length --T-tag0-base /path/to/crop_meta.json
```

- [ ] Startup field validation lines print (BC, flow, contact threshold)
- [ ] Status line updates while moving the pendant (or sim EE)
- [ ] Enter sets enable flags; `t` / `v` do **not** enable
- [ ] No robot motion in dry-run (`interface.control` never called)

## RTDE shadow (receive only)

```bash
geodude-ur-stringer-teleop --backend rtde --robot-ip $IP --dry-run --ply /path/to/object_tag0.ply \
  --field-flow length --T-tag0-base /path/to/crop_meta.json
```

- [ ] RTDE receive connects; no `speedJ` commands sent
- [ ] Status line shows live tip / contact / VF fields
- [ ] Enter enables flags only; pendant still required for motion on real arm
- [ ] Watchdog stays quiet while gated; no fault on dry-run startup

## Optional: live TF + tag cross-check

With ROS TF and camera running, omit `--T-tag0-base` if TF is available:

```bash
geodude-ur-stringer-teleop --backend rtde --robot-ip $IP --dry-run --ply ...
```

- [ ] `T_tag0_base` resolves from TF (or falls back to JSON path)
- [ ] Tag vs TF cross-check prints when detection is available; teleop continues if absent
