"""Tests that encode live teleop contracts — not CLI/viz/calibration smoke.

Keep a test when it would have caught a lab failure:

- Idle/enable must not retract, snap, or chase R_field (startup ramp, 2nd Enter).
- Ctrl+C / watchdog must speedStop and refuse further speedJ.
- `f` freeze must stop WoS requery; WASD/SpaceMouse stay in field frame, not tool.
- DualLoop `speedj` vs Eq.56 `dtc`: far = q̇, contact = τ; overload retracts, not push.
- Hardware Jacobian is numpy UR5e at TCP; SSoT flatten is the config source of truth.
"""
