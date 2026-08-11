# In-contact keyboard teleop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `--in-contact` so RTDE keyboard teleop runs DualLoop (Align + Force @ 5 N) with tangent-only keys and VF `n*` freeze.

**Architecture:** Dedicated contact path in `rtde_control_loop`; bridge owns tip-track / freeze / tangent FF; free-space pose-ref unused when flag set.

**Tech Stack:** Python, numpy, ur-rtde `speedJ`, existing `in_contact_control`, VF field basis.

## Global Constraints

- `Fd` default 5 N; auto-seek on Enter; tip-track `p*`; freeze key `f`; ignore normal/angular keys in contact mode.
- DualLoop angular is deg/s → convert to rad/s before Jacobian.
- Do not break free-space path when `--in-contact` is off.

---

### Task 1: Fix `in_contact_control` package imports

**Files:**
- Modify: `src/geodude_teleoperation/in_contact_control/*.py`
- Modify: `src/geodude_teleoperation/in_contact_control/README.md` (import paths)
- Test: `tests/test_in_contact_imports.py`

- [ ] Write import smoke test
- [ ] Replace `sandbox.line_admittance` with relative / package imports
- [ ] Run test; commit

### Task 2: `ContactTeleopBridge`

**Files:**
- Create: `src/geodude_teleoperation/contact_teleop.py`
- Test: `tests/test_contact_teleop.py`

- [ ] Failing tests: tip-track, freeze n*, tangent key → v ⊥ n*, DualLoop sum
- [ ] Implement bridge
- [ ] Pass tests; commit

### Task 3: Keyboard contact helpers

**Files:**
- Modify: `src/geodude_teleoperation/keyboard.py`
- Test: `tests/test_keyboard_buttons.py` (extend)

- [ ] Dispatch `f` as freeze button; document contact help
- [ ] Helper: field-tangent coeffs from key / twist
- [ ] Tests; commit

### Task 4: CLI + RTDE wiring

**Files:**
- Modify: `src/geodude_teleoperation/ur_stringer_teleop.py`
- Modify: `src/geodude_teleoperation/rtde_control_loop.py`
- Test: `tests/test_ur_stringer_teleop_cli.py`, extend RTDE loop tests if needed

- [ ] Add `--in-contact`, wire `--desired-normal-force`
- [ ] Contact branch: bridge.step → speedJ; Enter arms seek
- [ ] Tests; commit
