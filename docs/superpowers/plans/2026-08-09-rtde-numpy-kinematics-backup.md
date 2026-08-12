# RTDE numpy kinematics backup — Implementation Plan

> **For agentic workers:** TDD each task; worktree `.worktrees/ur-stringer-teleop`.

**Goal:** Auto-fallback live RTDE kinematics when gafropy unavailable.

**Tech:** numpy, scipy, existing `qdot_from_error`, fake ur_rtde tests.

## Task 1: Vendored FK/J + SE3 helpers

- Add `ur5e_kinematics.py` (slim copy; default no R_180Z).
- Add `se3.py` (twist duck, exp compose, pose error, Jacobian TCP map).
- Tests: FK shape; J map identity when tcp=flange; pose error sign for qdot.

## Task 2: Numpy kinematics bridge

- Add `numpy_kinematics_bridge.py` with `try_create`, twist update, `compute_qdot`.
- Unit tests with fake tcp/q.

## Task 3: Wire RTDE loop + keyboard

- Selection: gafropy then numpy.
- Keyboard numpy twist fallback.
- Integrate twist/qdot for numpy kin without gafropy objective.
- Extend `test_rtde_control_loop.py`.
- README note.

## Task 4: Verify

- `pytest` on touched tests.
