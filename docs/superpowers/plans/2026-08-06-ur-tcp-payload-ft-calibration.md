# UR TCP / Payload / FT Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `geodude_teleoperation.calibration` CLI that saves TCP/payload/FT calibration to YAML and optionally applies them to a UR5e via RTDE.

**Architecture:** Pure-Python package under `src/geodude_teleoperation/calibration/` with validators + YAML merge IO, thin `RtdeSession` wrapper (optional `ur_rtde`), and argparse subcommands. No ROS.

**Tech Stack:** Python 3.10+, PyYAML, numpy (existing), optional `ur_rtde`, pytest.

## Global Constraints

- Arms: `left` | `right` only
- TCP pose: length-6 `[x,y,z,rx,ry,rz]` meters / axis-angle, flange→TCP
- Payload CoG: length-3 meters relative to tool flange; `mass_kg > 0`
- No motion commands; FT requires still arm
- Merge YAML sections; never wipe unrelated keys
- Exit 2 = validation / missing section; Exit 1 = RTDE / apply failure
- Default calib dir: repo `calibration/`; override `--calib-dir`
- Optional dep group name: `ur` → `ur_rtde`

---

### Task 1: Validators + YAML IO

**Files:**
- Create: `src/geodude_teleoperation/calibration/__init__.py`
- Create: `src/geodude_teleoperation/calibration/io.py`
- Create: `tests/test_calibration_io.py`
- Modify: `pyproject.toml` (add `pyyaml`, pytest optional if needed)

**Interfaces:**
- Produces:
  - `parse_float_list(text: str, expected_len: int) -> list[float]`
  - `validate_tcp_pose(pose: list[float]) -> list[float]`
  - `validate_payload(mass_kg: float, cog: list[float]) -> tuple[float, list[float]]`
  - `default_calib_dir() -> Path`  # repo-root `calibration/`
  - `arm_yaml_path(calib_dir: Path, arm: str) -> Path`
  - `load_arm_calib(path: Path) -> dict`
  - `save_arm_calib(path: Path, data: dict) -> None`
  - `merge_section(data: dict, arm: str, section: str, section_body: dict, robot_ip: str | None = None) -> dict`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_calibration_io.py
from pathlib import Path
import pytest
from geodude_teleoperation.calibration.io import (
    parse_float_list, validate_tcp_pose, validate_payload,
    load_arm_calib, save_arm_calib, merge_section,
)

def test_parse_float_list_ok():
    assert parse_float_list("1,2,3", 3) == [1.0, 2.0, 3.0]

def test_parse_float_list_bad_len():
    with pytest.raises(ValueError):
        parse_float_list("1,2", 3)

def test_validate_tcp_pose_ok():
    assert validate_tcp_pose([0, 0, 0.1, 0, 0, 0]) == [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]

def test_validate_payload_rejects_nonpositive_mass():
    with pytest.raises(ValueError):
        validate_payload(0.0, [0, 0, 0])

def test_merge_preserves_other_sections(tmp_path: Path):
    path = tmp_path / "right.yaml"
    data = {"arm": "right", "tcp": {"pose": [0]*6, "frame": "tool0", "source": "manual"}}
    save_arm_calib(path, data)
    loaded = load_arm_calib(path)
    merged = merge_section(loaded, "right", "payload", {
        "mass_kg": 0.98, "cog": [0.01, 0.0, 0.05], "source": "manual",
    })
    save_arm_calib(path, merged)
    again = load_arm_calib(path)
    assert again["tcp"]["pose"] == [0]*6
    assert again["payload"]["mass_kg"] == 0.98
    assert "updated_at" in again
```

- [ ] **Step 2: Run tests — expect FAIL (import error)**

Run: `cd /home/boeing/ros2_ws/src/geodude-teleoperation && PYTHONPATH=src pytest tests/test_calibration_io.py -v`

- [ ] **Step 3: Implement `io.py` + empty `__init__.py`; add `pyyaml` to dependencies**

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/geodude_teleoperation/calibration tests/test_calibration_io.py
git commit -m "feat(calibration): add YAML IO and validators"
```

---

### Task 2: RTDE session wrapper

**Files:**
- Create: `src/geodude_teleoperation/calibration/rtde_session.py`
- Create: `tests/test_calibration_rtde_session.py`
- Modify: `pyproject.toml` (optional-dep `ur = ["ur-rtde"]`)

**Interfaces:**
- Produces:
  - `class RtdeSession` with context manager `__enter__/__exit__`
  - `RtdeSession.connect(ip: str) -> RtdeSession` (class method or ctor)
  - `set_tcp(self, pose: list[float]) -> None`
  - `set_payload(self, mass_kg: float, cog: list[float]) -> None`
  - `zero_ft_sensor(self) -> None`
  - `average_tcp_force(self, samples: int, settle_s: float = 0.01) -> list[float]`
  - ImportError with clear message if `ur_rtde` missing when connecting

- [ ] **Step 1: Write failing tests with a fake ur_rtde module injected**

```python
# tests/test_calibration_rtde_session.py
import sys, types
import numpy as np

class FakeC:
    def __init__(self, ip): self.ip = ip; self.tcp=None; self.payload=None; self.zeroed=False
    def setTcp(self, pose): self.tcp = list(pose)
    def setPayload(self, mass, cog): self.payload = (mass, list(cog))
    def zeroFtSensor(self): self.zeroed = True
    def disconnect(self): pass

class FakeR:
    def __init__(self, ip): self.ip = ip; self.n = 0
    def getActualTCPForce(self):
        self.n += 1
        return [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    def disconnect(self): pass

def test_average_and_apply(monkeypatch):
    mod = types.ModuleType("rtde_control"); mod.RTDEControlInterface = FakeC
    mod_r = types.ModuleType("rtde_receive"); mod_r.RTDEReceiveInterface = FakeR
    # package ur_rtde often exposes these as top-level; support both import styles used in rtde_session
    ...
```

Implement `rtde_session` to import `rtde_control.RTDEControlInterface` and `rtde_receive.RTDEReceiveInterface` (standard `ur_rtde` install layout).

- [ ] **Step 2–4: Fail → implement → pass**
- [ ] **Step 5: Commit** `feat(calibration): add RTDE session wrapper`

---

### Task 3: tcp / payload / ft_sensor domain helpers

**Files:**
- Create: `src/geodude_teleoperation/calibration/tcp.py`
- Create: `src/geodude_teleoperation/calibration/payload.py`
- Create: `src/geodude_teleoperation/calibration/ft_sensor.py`
- Create: `tests/test_calibration_commands.py`

**Interfaces:**
- Produces:
  - `build_tcp_section(pose: list[float]) -> dict`
  - `build_payload_section(mass_kg: float, cog: list[float]) -> dict`
  - `build_ft_section(bias_wrench: list[float], sample_count: int, note: str = "") -> dict`
  - `apply_tcp(session, pose)`, `apply_payload(session, mass, cog)`, `apply_ft_zero(session)`
  - `sample_ft_bias(session, samples: int) -> list[float]`

- [ ] **Step 1: Tests for section builders + apply delegates to session**
- [ ] **Step 2–4: TDD cycle**
- [ ] **Step 5: Commit** `feat(calibration): add tcp/payload/ft command helpers`

---

### Task 4: CLI + `__main__` + entry point

**Files:**
- Create: `src/geodude_teleoperation/calibration/cli.py`
- Create: `src/geodude_teleoperation/calibration/__main__.py`
- Create: `tests/test_calibration_cli.py`
- Modify: `pyproject.toml` (`[project.scripts]` `geodude-calibrate`)
- Create: `calibration/.gitkeep` (dir exists; YAML created on save)

**Interfaces:**
- Produces: `main(argv: list[str] | None = None) -> int`
- Subcommands: `tcp`, `payload`, `ft`, `apply`, `show`
- Shared flags: `--arm`, `--calib-dir`, `--ip`, `--apply`, `--yes`

- [ ] **Step 1: CLI tests with tmp calib dir, no real RTDE**

```python
def test_cli_tcp_writes_yaml(tmp_path):
    rc = main(["tcp", "--arm", "right", "--pose", "0,0,0.15,0,0,0", "--calib-dir", str(tmp_path)])
    assert rc == 0
    data = load_arm_calib(tmp_path / "right.yaml")
    assert data["tcp"]["pose"][2] == 0.15

def test_cli_apply_missing_section(tmp_path):
    save_arm_calib(tmp_path / "left.yaml", {"arm": "left"})
    rc = main(["apply", "--arm", "left", "--what", "tcp", "--calib-dir", str(tmp_path), "--yes", "--ip", "127.0.0.1"])
    assert rc == 2
```

- [ ] **Step 2–4: TDD cycle**
- [ ] **Step 5: Commit** `feat(calibration): add CLI subcommands and entrypoint`

---

### Task 5: Verification

- [ ] Run full suite: `PYTHONPATH=src pytest tests/test_calibration_*.py -v`
- [ ] Manual dry-run: `PYTHONPATH=src python -m geodude_teleoperation.calibration show --arm right --calib-dir /tmp/...`
- [ ] Commit any leftover fixes

---

## Spec coverage

| Spec item | Task |
|---|---|
| Layout + entrypoint | 1, 4 |
| YAML schema + merge | 1 |
| tcp/payload/ft/apply/show | 3, 4 |
| RTDE setTcp/setPayload/zeroFt/sample | 2, 3 |
| Errors exit codes | 4 |
| Offline tests | 1–4 |
| Optional ur_rtde | 2 |

## Execution

User requested immediate implementation → **inline execution** in this session (executing-plans style, batched).
