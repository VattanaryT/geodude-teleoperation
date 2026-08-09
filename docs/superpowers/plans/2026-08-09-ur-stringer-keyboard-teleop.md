# UR Stringer Keyboard Teleop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add modular stringer-field keyboard teleop for Geodude right UR5e: calibrate-then-enable mode machine, frame-aware contact/WoS VF, MuJoCo + RTDE backends, dry-run receive-only shadow.

**Architecture:** Pure helpers (`frames`, `field_query`, `stringer_field`) + `ContactWosOrientationField` VF + thin `ur_stringer_teleop` CLI/loops. Reuse `TeleoperationController` / `KeyboardTeleop`. Query in tag_0; correct orientation in base. Per-tick `vf.update_from_state` (not keypress-only).

**Tech Stack:** Python 3.10+, numpy, scipy, open3d, PyYAML, diffused_fields, gafropy, optional ur-rtde + ROS TF/AprilTag, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-09-ur-stringer-keyboard-teleop-design.md`
- v1 arm: `right` only; task space `right_ee` (geodude.yaml)
- `T_tag0_base` = pose of tag_0 in base (`p_base = R @ p_tag0 + t`); never invert the name
- Query at tip; gafropy TCP/EE; no second tip frame in gafropy
- `in_contact = abs(d) <= threshold`; `get_distance` returns cached `abs(d)`
- dry-run ⇒ no RTDEControl / no speedJ
- Commands only if `motion_enabled and teleop_active`; after enable, `teleop_active=True`
- Enable key ≠ `t`/`v`; default Enter
- MuJoCo: PLY in tag_0 + `--T-tag0-base` (identity default); no bake-into-world path
- MuJoCo motion gate (**A**): do **not** call `interface.control` until enable, and never in dry-run (same idea as RTDE `speedJ(0)` while gated)
- When `vf_enabled=False`: still `update_from_state` every tick (status/d/tip_nn); skip applying projection to the objective
- RTDE v1: `speedJ` only; reuse `TeleoperationController` math
- RTDE before Task 10: use live TF if available, else `--T-tag0-base`; if neither → refuse motion (`speedJ(0)` / no enable path for commands). Never leave `T` undefined.
- No `correct_distance_smooth`
- Commit per task when user has asked for commits during execution; otherwise leave working tree for review

## File structure

| Path | Responsibility |
|------|----------------|
| `src/geodude_teleoperation/frames.py` | SE(3) helpers, tip offset, load `T_tag0_base` |
| `src/geodude_teleoperation/field_query.py` | Surface/WoS query API (offline port) |
| `src/geodude_teleoperation/stringer_field.py` | Load PLY, BC field, startup validation |
| `src/geodude_teleoperation/contact_wos_vf.py` | Frame-aware VF + cache + status |
| `src/geodude_teleoperation/teleop_state.py` | Mode flags, gating, status line |
| `src/geodude_teleoperation/rtde_control_loop.py` | Receive/control RTDE host loop |
| `src/geodude_teleoperation/ur_stringer_teleop.py` | CLI + mujoco/rtde entry |
| `src/geodude_teleoperation/keyboard.py` | Fix `t`/`v`/`q` + enable-key callback |
| `src/geodude_teleoperation/controller.py` | Expose `qdot_from_error` helper used by both paths |
| `src/geodude_teleoperation/__init__.py` | Lazy exports for new symbols |
| `pyproject.toml` | deps + console script |
| `tests/test_frames.py` | |
| `tests/test_field_query.py` | |
| `tests/test_stringer_field.py` | |
| `tests/test_contact_wos_vf.py` | |
| `tests/test_teleop_state.py` | |
| `tests/test_keyboard_buttons.py` | |
| `tests/test_ur_stringer_teleop_cli.py` | |
| `tests/test_rtde_control_loop.py` | |

---

### Task 1: Frame helpers

**Files:**
- Create: `src/geodude_teleoperation/frames.py`
- Create: `tests/test_frames.py`

**Interfaces:**
- Produces:
  - `load_T_tag0_base(path: Path | None) -> np.ndarray`  # (4,4); identity if path is None
  - `base_to_tag0_point(p_base: np.ndarray, T_tag0_base: np.ndarray) -> np.ndarray`
  - `tag0_rot_to_base(R_tag0: np.ndarray, T_tag0_base: np.ndarray) -> np.ndarray`
  - `tip_from_tcp(p_tcp: np.ndarray, R_tcp: np.ndarray, tip_offset: np.ndarray) -> np.ndarray`
  - `warn_tip_offset_xy(tip_offset: np.ndarray) -> str | None`  # warning message or None

- [ ] **Step 1: Write failing tests**

```python
# tests/test_frames.py
import numpy as np
from geodude_teleoperation.frames import (
    base_to_tag0_point,
    load_T_tag0_base,
    tag0_rot_to_base,
    tip_from_tcp,
    warn_tip_offset_xy,
)

def test_identity_roundtrip():
    T = np.eye(4)
    p = np.array([0.1, 0.2, 0.3])
    assert np.allclose(base_to_tag0_point(p, T), p)

def test_crop_meta_convention():
    # T_tag0_base = pose of tag_0 in base: origin of tag_0 at t, axes R
    T = np.eye(4)
    T[:3, 3] = [1.0, 0.0, 0.0]
    p_base = np.array([1.1, 0.0, 0.0])
    assert np.allclose(base_to_tag0_point(p_base, T), [0.1, 0.0, 0.0])

def test_tag0_rot_to_base():
    T = np.eye(4)
    R = np.eye(3)
    assert np.allclose(tag0_rot_to_base(R, T), R)

def test_tip_from_tcp():
    p = np.zeros(3)
    R = np.eye(3)
    tip = tip_from_tcp(p, R, np.array([0.0, 0.0, 0.1]))
    assert np.allclose(tip, [0.0, 0.0, 0.1])

def test_warn_tip_offset_xy():
    assert warn_tip_offset_xy(np.array([0.01, 0.0, 0.1])) is not None
    assert warn_tip_offset_xy(np.array([0.0, 0.0, 0.1])) is None

def test_load_none_is_identity():
    assert np.allclose(load_T_tag0_base(None), np.eye(4))
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `cd /home/boeing/ros2_ws/src/geodude-teleoperation && PYTHONPATH=src pytest tests/test_frames.py -v`

Expected: import error / missing module

- [ ] **Step 3: Implement `frames.py`**

```python
# src/geodude_teleoperation/frames.py
"""SE(3) helpers. T_tag0_base = pose of tag_0 expressed in base (crop_meta convention)."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np

def load_T_tag0_base(path: Path | None) -> np.ndarray:
    if path is None:
        return np.eye(4, dtype=float)
    path = Path(path)
    if path.suffix == ".json":
        with path.open() as f:
            meta = json.load(f)
        T = np.asarray(meta["T_tag0_base"] if "T_tag0_base" in meta else meta, dtype=float)
    else:
        T = np.load(path)
        if isinstance(T, np.lib.npyio.NpzFile):
            T = T["T_tag0_base"] if "T_tag0_base" in T.files else T[T.files[0]]
        T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T_tag0_base must be 4x4, got {T.shape}")
    return T

def base_to_tag0_point(p_base: np.ndarray, T_tag0_base: np.ndarray) -> np.ndarray:
    R = T_tag0_base[:3, :3]
    t = T_tag0_base[:3, 3]
    return R.T @ (np.asarray(p_base, dtype=float).reshape(3) - t)

def tag0_rot_to_base(R_tag0: np.ndarray, T_tag0_base: np.ndarray) -> np.ndarray:
    return T_tag0_base[:3, :3] @ np.asarray(R_tag0, dtype=float).reshape(3, 3)

def tip_from_tcp(p_tcp, R_tcp, tip_offset) -> np.ndarray:
    return np.asarray(p_tcp, dtype=float).reshape(3) + (
        np.asarray(R_tcp, dtype=float).reshape(3, 3)
        @ np.asarray(tip_offset, dtype=float).reshape(3)
    )

def warn_tip_offset_xy(tip_offset: np.ndarray) -> str | None:
    tip = np.asarray(tip_offset, dtype=float).reshape(3)
    if abs(tip[0]) > 1e-9 or abs(tip[1]) > 1e-9:
        return (
            "WARNING: tip-offset TX/TY non-zero; prefer TX=TY=0 and only TZ "
            f"(got {tip.tolist()})"
        )
    return None
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit** (if committing during execution)

```bash
git add src/geodude_teleoperation/frames.py tests/test_frames.py
git commit -m "feat(teleop): add tag_0/base frame helpers"
```

---

### Task 2: Field query API (offline port)

**Files:**
- Create: `src/geodude_teleoperation/field_query.py`
- Create: `tests/test_field_query.py`
- Modify: `pyproject.toml` — ensure `scipy`, `open3d` available (add if missing; open3d may already come via diffused_fields — pin explicit if import fails)

**Interfaces:**
- Produces (signatures match offline semantics):
  - `query_basis_surface(pcloud, x, *, query_radius=0.025, fit_mode="sphere", min_neighbors=12, k_fallback=24, max_neighbors=48) -> tuple[np.ndarray, int, float, dict]`
  - `query_basis_wos(wos, x, ref) -> np.ndarray`
  - `query_field_basis(pcloud, wos, x, *, contact_threshold, **surface_kwargs) -> tuple[np.ndarray, float, bool, dict]`  
    # returns (R, signed_d, in_contact, diag); uses WoS when not in contact
- Consumes: `diffused_fields` Pointcloud with `local_bases`, `normals`, kd_tree

- [ ] **Step 1: Write failing tests with a synthetic planar cloud**

```python
# tests/test_field_query.py
import numpy as np
from diffused_fields.manifold import Pointcloud
from geodude_teleoperation.field_query import query_basis_surface, query_field_basis

def _planar_cloud():
    xs, ys = np.meshgrid(np.linspace(-0.1, 0.1, 21), np.linspace(-0.1, 0.1, 21))
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    pc = Pointcloud(vertices=verts, voxel_size=0.01, normal_orientation=1)
    pc.get_normals()
    # identity-ish local bases: z = normal
    n = pc.num_vertices
    bases = np.tile(np.eye(3), (n, 1, 1))
    pc.local_bases = bases
    pc.get_kd_tree()
    return pc

def test_surface_signed_distance_above_plane():
    pc = _planar_cloud()
    x = np.array([0.0, 0.0, 0.02])
    R, idx, d, diag = query_basis_surface(pc, x, fit_mode="plane", query_radius=0.05)
    assert d > 0
    assert abs(abs(d) - 0.02) < 0.005
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)

def test_in_contact_band():
    pc = _planar_cloud()
    class FakeWos:
        def get_batch_from_point(self, x): return x
        def diffuse_rotations(self, batch):
            return np.eye(3), None, np.array(["converged"])
    R, d, contact, diag = query_field_basis(
        pc, FakeWos(), np.array([0.0, 0.0, 0.0005]),
        contact_threshold=0.001, fit_mode="plane", query_radius=0.05,
    )
    assert contact is True
    assert diag.get("used_wos") is False
```

- [ ] **Step 2: Run — expect FAIL**

Run: `PYTHONPATH=src pytest tests/test_field_query.py -v`

- [ ] **Step 3: Port helpers from `diffused_fields_robotics/scripts/offline_demo_to_local_actions.py` into `field_query.py`**

Copy/adapt: `_mean_quaternion`, `_orient_normal`, `fit_local_tangent_plane`, `fit_local_tangent_sphere`, `sphere_signed_distance`, `query_neighborhood`, `orthonormalize_basis_with_normal`, `query_basis_surface`, `query_basis_wos`, plus:

```python
def query_field_basis(pcloud, wos, x, *, contact_threshold, **kwargs):
    basis, nearest, sd, diag = query_basis_surface(pcloud, x, **kwargs)
    in_contact = abs(sd) <= contact_threshold
    used_wos = False
    if not in_contact:
        basis = query_basis_wos(wos, x, basis)
        used_wos = True
    diag = {**diag, "nearest_idx": nearest, "used_wos": used_wos}
    return basis, float(sd), bool(in_contact), diag
```

Do **not** import or call `correct_distance_smooth`.

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/geodude_teleoperation/field_query.py tests/test_field_query.py pyproject.toml
git commit -m "feat(teleop): port contact/WoS field query API"
```

---

### Task 3: Stringer field load + BC + startup validation

**Files:**
- Create: `src/geodude_teleoperation/stringer_field.py`
- Create: `tests/test_stringer_field.py`

**Interfaces:**
- Produces:
  - `config_defaults(object_name: str = "stringer_layup") -> dict`  # voxel_size, normal_orientation
  - `load_stringer_pointcloud(ply: Path, voxel_size: float, normal_orientation: int) -> Pointcloud`
  - `side_source_sink_lines(pcloud, fraction=0.06, swap=False, side="ends"|"sides") -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  - `compute_orientation_field(pcloud, *, field_mode, field_flow="length", end_line_fraction=0.06, swap_end_lines=False, diffusion_scalar=2000.0) -> PointcloudScalarDiffusion`
  - `validate_field_prints(pcloud, sd, wos_threshold: float) -> str`  # multi-line report
  - `build_wos(pcloud, batch_size: int) -> WalkOnSpheresDiffusion`

- [ ] **Step 1: Failing test — orthonormal sample + source/sink counts on tiny cloud**

```python
# tests/test_stringer_field.py
import numpy as np
from pathlib import Path
import open3d as o3d
from geodude_teleoperation.stringer_field import (
    compute_orientation_field,
    load_stringer_pointcloud,
    validate_field_prints,
    build_wos,
)

def test_field_build_end_lines(tmp_path: Path):
    xs, ys = np.meshgrid(np.linspace(0, 0.2, 40), np.linspace(0, 0.05, 10))
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    ply = tmp_path / "tiny.ply"
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(verts))
    o3d.io.write_point_cloud(str(ply), pcd)
    pc = load_stringer_pointcloud(ply, voxel_size=0.01, normal_orientation=1)
    pc.get_normals(); pc.get_kd_tree()
    sd = compute_orientation_field(
        pc, field_mode="end_lines", field_flow="length", end_line_fraction=0.1,
    )
    assert pc.local_bases is not None
    R0 = pc.local_bases[0]
    assert np.allclose(R0.T @ R0, np.eye(3), atol=1e-5)
    thr = pc.get_mean_edge_length() * 2
    report = validate_field_prints(pc, sd, thr)
    assert "#verts" in report or "verts" in report
    wos = build_wos(pc, batch_size=8)
    assert wos is not None
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `stringer_field.py`**

Port BC logic from `stringer_layup_ee_teleop.compute_orientation_field` / offline `side_source_sink_lines` + `compute_field`. Map `field_flow length→ends`, `width→sides`. Load config from `diffused_fields` package `config/pointclouds.yaml` via `importlib.resources` or path relative to installed package; fall back to `{voxel_size: 0.004, normal_orientation: 1}`.

`validate_field_prints` must include: `#verts`, `#source/#sink`, sample basis orthonormal + det, mean edge length, WoS threshold.

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/geodude_teleoperation/stringer_field.py tests/test_stringer_field.py
git commit -m "feat(teleop): stringer field load and BC validation"
```

---

### Task 4: `ContactWosOrientationField`

**Files:**
- Create: `src/geodude_teleoperation/contact_wos_vf.py`
- Create: `tests/test_contact_wos_vf.py`
- Modify: `src/geodude_teleoperation/__init__.py` (lazy export later in Task 10 OK)

**Interfaces:**
- Consumes: `query_field_basis`, `base_to_tag0_point`, `tag0_rot_to_base`, `tip_from_tcp`, `_orient_approach_into_surface` from `virtual_fixture.py`
- Produces:
  - `class ContactWosOrientationField`
  - `__init__(self, pcloud, wos, *, tip_offset, contact_threshold, T_tag0_base, query_kwargs...)`
  - `set_T_tag0_base(self, T: np.ndarray) -> None`
  - `update_from_state(self, ee_motor, T_tag0_base=None) -> None`
  - `project(self, proposed_reference) -> motor`
  - `get_distance(self, position) -> float`  # cached abs(d)
  - `status` property → dataclass/dict with signed_d, in_contact, tip_nn, used_wos, query_ms, R_field_base

- [ ] **Step 1: Failing tests (mock pcloud/wos or tiny planar cloud + identity T)**

```python
# tests/test_contact_wos_vf.py
import numpy as np
# build tiny planar pc as in Task 2/3; identity T; tip_offset=0
# motor: use gafropy.Motor / Rotor.from_matrix if available in env;
# if gafropy missing in CI, skip with pytest.importorskip("gafropy")

def test_get_distance_is_abs_cached():
    ...
    vf.update_from_state(ee_above)
    d = vf.get_distance(np.array([0, 0, 0.02]))
    assert d >= 0
    assert abs(d - abs(vf.status["signed_d"])) < 1e-9

def test_continuity_flip():
    # two updates with opposite-hemisphere bases should not snap det
    ...

def test_project_keeps_translation():
    ...
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement VF**

Core of `update_from_state`:

1. TCP `p,R` from `ee_motor` translator/rotor  
2. `p_tip_base = tip_from_tcp(...)`  
3. `p_tip_tag0 = base_to_tag0_point(...)`  
4. `R_tag0, sd, contact, diag = query_field_basis(...)` with try/except → last basis  
5. Continuity quat-dot flip vs `_last_R_tag0`  
6. `R_base = tag0_rot_to_base(R_tag0, T)` then `_orient_approach_into_surface`  
7. Cache status; `tip_nn = ‖p_tip_tag0 - verts[nearest]‖`  
8. `get_distance` returns `abs(cached signed_d)` without re-query  

`project`: if cache warm for this tick, apply cached `R_field_base` to motor translator; else call `update_from_state` once then apply.

**VF off vs query:** callers (Task 8/9) always invoke `update_from_state` each tick when a VF object exists — including `vf_enabled=False` and calibrate — so status / `d` / `tip_nn` keep updating. Skipping applies only to **writing** the projected orientation into the controller objective (and to `controller.vf` projection during twists when VF is toggled off — see Task 8).

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/geodude_teleoperation/contact_wos_vf.py tests/test_contact_wos_vf.py
git commit -m "feat(teleop): frame-aware ContactWosOrientationField"
```

---

### Task 5: Keyboard enable + `t`/`v`/`q` buttons

**Files:**
- Modify: `src/geodude_teleoperation/keyboard.py`
- Create: `tests/test_keyboard_buttons.py`

**Interfaces:**
- Produces (behavior change):
  - `KeyboardTeleop.start(on_twist, on_button=None, on_enable=None, enable_key="\n")`
  - `t` → `on_button(0)`; `v` → `on_button(1)`; `q` → stop / optional quit callback
  - `enable_key` (default `"\n"`) → `on_enable()` once per press; never treated as twist

- [ ] **Step 1: Failing unit test of key dispatch helper**

Refactor a pure function to avoid termios in tests:

```python
# in keyboard.py
def dispatch_key(key: str, *, step, enable_key="\n") -> tuple[str, object]:
    """Return (kind, payload) where kind in twist|button|enable|quit|ignore."""
```

```python
# tests/test_keyboard_buttons.py
from geodude_teleoperation.keyboard import dispatch_key

def test_enable_enter():
    assert dispatch_key("\n", step=0.01)[0] == "enable"

def test_t_v_buttons():
    assert dispatch_key("t", step=0.01) == ("button", 0)
    assert dispatch_key("v", step=0.01) == ("button", 1)

def test_twist_w():
    kind, twist = dispatch_key("w", step=0.01)
    assert kind == "twist"
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `dispatch_key` and wire `_run` to call `on_enable` / `on_button` / `on_twist`**

Update `KEYBOARD_HELP` to document Enter enable vs `t`/`v`.

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/geodude_teleoperation/keyboard.py tests/test_keyboard_buttons.py
git commit -m "fix(keyboard): enable key and t/v button dispatch"
```

---

### Task 6: Teleop state + status line

**Files:**
- Create: `src/geodude_teleoperation/teleop_state.py`
- Create: `tests/test_teleop_state.py`

**Interfaces:**
- Produces:
  - `class TeleopState` with `mode`, `motion_enabled`, `teleop_active`, `vf_enabled`
  - `enable(self) -> None`  # motion_enabled=True, teleop_active=True, mode=teleop
  - `commands_allowed(self, *, dry_run: bool, fault: bool, tf_ok: bool) -> bool`
  - `format_status_line(state, vf_status, *, qdot_norm, tf_age_ms, tag_vs_tf_mm=None) -> str`

- [ ] **Step 1: Failing tests**

```python
from geodude_teleoperation.teleop_state import TeleopState, format_status_line

def test_enable_defaults_teleop_active():
    s = TeleopState()
    assert s.motion_enabled is False
    s.enable()
    assert s.motion_enabled and s.teleop_active and s.mode == "teleop"

def test_gating():
    s = TeleopState(); s.enable()
    assert s.commands_allowed(dry_run=False, fault=False, tf_ok=True)
    assert not s.commands_allowed(dry_run=True, fault=False, tf_ok=True)
    s.teleop_active = False
    assert not s.commands_allowed(dry_run=False, fault=False, tf_ok=True)

def test_status_line_has_flags():
    s = TeleopState(); s.enable()
    line = format_status_line(s, {"signed_d": 0.01, "in_contact": False, "used_wos": True,
                                  "tip_nn": 0.02, "query_ms": 1.5}, qdot_norm=0.0, tf_age_ms=0.0)
    assert "motion_enabled=1" in line
    assert "teleop_active=1" in line
```

- [ ] **Step 2–4: Implement + PASS**

- [ ] **Step 5: Commit**

```bash
git add src/geodude_teleoperation/teleop_state.py tests/test_teleop_state.py
git commit -m "feat(teleop): mode gating and status line"
```

---

### Task 7: Extract `qdot_from_error` from controller

**Files:**
- Modify: `src/geodude_teleoperation/controller.py`
- Create: `tests/test_controller_qdot.py`

**Interfaces:**
- Produces:
  - `qdot_from_error(error, jacobian, *, gain, damping, max_joint_velocity) -> np.ndarray`
  - `TeleoperationController.compute_control_command` calls this helper (behavior unchanged)

**Rationale:** RTDE host reuses the same P-velocity math without reimplementing it. Full gafropy `task_space_state` construction is Task 9; this task only extracts the pure linear algebra.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from geodude_teleoperation.controller import qdot_from_error

def test_qdot_scales_and_clamps():
    err = np.ones(6)
    J = np.eye(6)
    q = qdot_from_error(err, J, gain=1.0, damping=0.0, max_joint_velocity=0.5)
    assert np.linalg.norm(q) <= 0.5 + 1e-9
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement helper; refactor `TeleoperationController.compute_control_command` to use it**

```python
def qdot_from_error(error, jacobian, *, gain, damping, max_joint_velocity):
    error = np.asarray(error)
    jacobian = np.asarray(jacobian)
    j_inverse = np.linalg.inv(
        jacobian.T @ jacobian + damping**2 * np.eye(jacobian.shape[1]))
    qdot = -gain * j_inverse @ jacobian.T @ error
    speed = np.linalg.norm(qdot)
    if speed > max_joint_velocity:
        qdot = qdot * (max_joint_velocity / speed)
    return qdot
```

- [ ] **Step 4: PASS**

- [ ] **Step 5: Commit**

```bash
git add src/geodude_teleoperation/controller.py tests/test_controller_qdot.py
git commit -m "refactor(controller): extract qdot_from_error helper"
```

---

### Task 8: CLI + MuJoCo / dry-run loop (no RTDE motion)

**Files:**
- Create: `src/geodude_teleoperation/ur_stringer_teleop.py`
- Create: `tests/test_ur_stringer_teleop_cli.py`
- Modify: `pyproject.toml` — script `geodude-ur-stringer-teleop = geodude_teleoperation.ur_stringer_teleop:main`

**Interfaces:**
- Produces: `parse_args(argv=None) -> Namespace`, `main(argv=None) -> int`
- CLI flags per spec table (backend, robot-ip, ply, field-*, tip-offset, contact-threshold, wos-batch-size, control-hz, max-joint-velocity, keyboard-step, rtde-accel, watchdog-ms, enable-key, T-tag0-base, base-frame, tag0-frame, dry-run, arm)

- [ ] **Step 1: CLI tests (no robot)**

```python
# tests/test_ur_stringer_teleop_cli.py
import pytest
from geodude_teleoperation.ur_stringer_teleop import parse_args

def test_default_backend_mujoco():
    args = parse_args(["--ply", "/tmp/x.ply"])
    assert args.backend == "mujoco"
    assert args.dry_run is False

def test_rtde_requires_ip_validation():
    args = parse_args(["--ply", "/tmp/x.ply", "--backend", "rtde", "--dry-run"])
    # main() should abort; unit-test validate_args helper
    from geodude_teleoperation.ur_stringer_teleop import validate_args
    with pytest.raises(SystemExit):
        validate_args(args)  # missing robot-ip
```

- [ ] **Step 2: FAIL then implement argparse + `validate_args`**

- [ ] **Step 3: Implement `run_mujoco(args)` skeleton**

**MuJoCo motion-disabled (option A — required):** Do **not** call `interface.control(...)` until `state.enable()` has run, and **never** call it in `--dry-run`. Until then the sim must not servo toward a teleop objective (calibrate = FK/read + VF query/prints only). After enable (and not dry-run), start `interface.control(controller, blocking=False)` once. Same safety idea as RTDE holding `speedJ(0)` while gated.

Order:
1. Load field (`stringer_field`); print `validate_field_prints`
2. `warn_tip_offset_xy`
3. Load `T_tag0_base` (identity default)
4. Build WoS + `ContactWosOrientationField`
5. Create `TeleoperationController`; **`controller.set_virtual_fixture(vf)`** and `set_interface` / task space / objective as in `examples/teleoperation.py`
6. Create `TeleopState`; start `KeyboardTeleop` with gating callbacks (`v` toggles `vf_enabled` and whether twists use VF projection)
7. Init `RobotInterface` mujoco for FK/state always; **do not** start `interface.control` yet (and never if dry-run)
8. Loop at `--control-hz`:
   - read EE motor
   - **always** `vf.update_from_state(ee, T)` (even when `vf_enabled=False` / calibrate) so status/`d`/`tip_nn` update
   - if `vf_enabled` and commands will use VF: optionally refresh objective orientation from cached `project(ee)` **only after enable + control started**; if not `vf_enabled`, skip applying projection to the objective
   - print status line
   - on enable edge (not dry-run): start `interface.control` once
   - if `commands_allowed` and twist pending: `controller.update_command(twist)`  
     **Twist/VF cache:** leave `update_command` as-is (`project` / `get_distance`); the tick’s prior `update_from_state` warms the VF cache so `project`/`get_distance` reuse tip-query results — no second inconsistent query in the same tick
9. On quit: stop keyboard; stop control if it was started

For dry-run mujoco: FK/read only; never `interface.control`; never apply commands.

- [ ] **Step 4: Smoke import**

Run: `PYTHONPATH=src python -c "from geodude_teleoperation.ur_stringer_teleop import parse_args; print(parse_args(['--ply','x']).backend)"`

- [ ] **Step 5: Commit**

```bash
git add src/geodude_teleoperation/ur_stringer_teleop.py tests/test_ur_stringer_teleop_cli.py pyproject.toml
git commit -m "feat(teleop): ur_stringer_teleop CLI and mujoco/dry-run loop"
```

---

### Task 9: RTDE host loop (receive-only dry-run + live speedJ)

**Files:**
- Create: `src/geodude_teleoperation/rtde_control_loop.py`
- Create: `tests/test_rtde_control_loop.py`
- Modify: `src/geodude_teleoperation/ur_stringer_teleop.py` (wire `--backend rtde`)

**Interfaces:**
- Produces:
  - `class RtdeTeleopSession`
  - `open(ip, *, dry_run: bool) -> RtdeTeleopSession`  
    - dry_run: `RTDEReceive` only  
    - not dry_run: receive + `RTDEControl`
  - `read_state(self) -> dict`  # q, tcp pose, runtime fault flags
  - `speedj(self, qdot, accel) -> None`  # no-op / raise if dry_run
  - `close(self) -> None`  # speedJ(0) **only if control was opened**
  - `run_rtde_loop(args, field_bundle, vf, controller_or_kin, state: TeleopState) -> int`

**Kinematics bridge (integration spike, resolve here):**  
Prefer: keep a gafropy `RobotInterface` (mujoco model) as **kinematics-only** — write RTDE joint positions into its state each tick, compute EE motor + task-space error/jacobian via existing gafropy APIs, then `qdot = controller.compute_control_command(...)` or `qdot_from_error(...)`. Do **not** reimplement P-velocity. If writing joint state into RobotInterface is impossible, document the chosen adapter in a short comment at the top of `rtde_control_loop.py`.

- [ ] **Step 1: Fake RTDE modules test (mirror calibration tests)**

```python
# tests/test_rtde_control_loop.py
def test_dry_run_opens_receive_only(monkeypatch):
    # inject fake rtde_receive / rtde_control
    # open(..., dry_run=True) must not instantiate Control
    ...

def test_watchdog_zeros(monkeypatch):
    # after watchdog_ms without send, speedj called with zeros when control open
    ...

def test_close_dry_run_no_speedj(monkeypatch):
    ...
```

- [ ] **Step 2: FAIL → implement session + loop**

**`T_tag0_base` before Task 10:** Each tick resolve `T` as: live TF if Task 10 helper available and fresh → else fixed `--T-tag0-base` (CLI/file/identity). If RTDE live needs a frame and **neither** TF nor `--T-tag0-base` is usable → treat as missing frame: warn, `tf_ok=False`, `speedJ(0)`, refuse motion. Never leave `T` undefined (do not query with a dangling transform).

Loop body: resolve `T` (TF or `--T-tag0-base` fallback) → fault check → **always** `vf.update_from_state` (even if `vf_enabled=False`) → gate → if allowed and `vf_enabled`, apply cached projection to objective before `q̇` → compute `q̇` via same controller → clamp → `speedJ` → status print. If not allowed → `speedJ(0)`. Watchdog: track `last_command_time`; if stale → `speedJ(0)` when control open.

- [ ] **Step 3: Wire `main()` backend switch; abort without `--robot-ip`**

- [ ] **Step 4: PASS unit tests**

- [ ] **Step 5: Commit**

```bash
git add src/geodude_teleoperation/rtde_control_loop.py tests/test_rtde_control_loop.py src/geodude_teleoperation/ur_stringer_teleop.py
git commit -m "feat(teleop): RTDE receive-only dry-run and speedJ loop"
```

---

### Task 10: Live TF + AprilTag calibrate cross-check

**Files:**
- Modify: `src/geodude_teleoperation/frames.py` (TF lookup helpers)
- Create: `src/geodude_teleoperation/tag_crosscheck.py`
- Create: `tests/test_tag_crosscheck.py`
- Modify: `rtde_control_loop.py` / `ur_stringer_teleop.py` to pass `tag_vs_tf_mm` into status line

**Interfaces:**
- Produces:
  - `lookup_T_tag0_base_tf(buffer, base_frame, tag0_frame, max_age_s) -> tuple[np.ndarray | None, float | None]`  # (T, age_ms); None if missing/stale
  - `apriltag_T_tag0_base(T_EE_base, T_cam_EE, T_tag_cam) -> np.ndarray`  # wrap `compose_T_tag_base` formula
  - `tag_vs_tf_translation_mm(T_tag_tf, T_tag_april) -> float`

- [ ] **Step 1: Pure math tests (no ROS)**

```python
def test_disagree_mm():
    T1 = np.eye(4); T2 = np.eye(4); T2[0, 3] = 0.01
    assert abs(tag_vs_tf_translation_mm(T1, T2) - 10.0) < 1e-6
```

- [ ] **Step 2–4: Implement; TF lookup try/import rclpy+tf2; if unavailable, status shows `tf_age_ms=nan` and warn once**

Calibrate: print disagreement when both TF and AprilTag available; **never auto-enable**.

- [ ] **Step 5: Commit**

```bash
git add src/geodude_teleoperation/tag_crosscheck.py src/geodude_teleoperation/frames.py tests/test_tag_crosscheck.py
git commit -m "feat(teleop): TF lookup and AprilTag vs TF cross-check"
```

---

### Task 11: Package exports + end-to-end dry checklist

**Files:**
- Modify: `src/geodude_teleoperation/__init__.py` — lazy attrs for new public types
- Modify: `docs/superpowers/specs/2026-08-09-ur-stringer-keyboard-teleop-design.md` — Status: Approved / Implemented (optional note)

- [ ] **Step 1: Add lazy exports** for `ContactWosOrientationField`, `query_field_basis`, `TeleopState`, etc.

- [ ] **Step 2: Run full unit suite**

Run: `cd /home/boeing/ros2_ws/src/geodude-teleoperation && PYTHONPATH=src pytest tests/test_frames.py tests/test_field_query.py tests/test_stringer_field.py tests/test_contact_wos_vf.py tests/test_keyboard_buttons.py tests/test_teleop_state.py tests/test_controller_qdot.py tests/test_ur_stringer_teleop_cli.py tests/test_rtde_control_loop.py tests/test_tag_crosscheck.py -v`

Expected: all PASS

- [ ] **Step 3: Manual dry-run checklist (operator)**

```bash
# field + prints only (mujoco dry-run)
geodude-ur-stringer-teleop --ply /path/to/object_tag0.ply --dry-run \
  --field-flow length --T-tag0-base /path/to/crop_meta.json

# RTDE shadow (receive only)
geodude-ur-stringer-teleop --backend rtde --robot-ip $IP --dry-run --ply ...
```

Verify startup field validation lines; status line updates while pendant-moving; Enter enables flags; no motion when dry-run; `t`/`v` do not enable.

- [ ] **Step 4: Commit**

```bash
git add src/geodude_teleoperation/__init__.py
git commit -m "feat(teleop): export stringer teleop public API"
```

---

## Spec coverage checklist (self-review)

| Spec item | Task |
|-----------|------|
| frames / SE(3) naming / tip offset | 1 |
| field_query no correct_distance_smooth | 2 |
| stringer field BC + startup prints | 3 |
| ContactWos VF, abs(d), continuity, update_from_state | 4 |
| Keyboard enable ≠ t/v | 5 |
| Gating + status line fields | 6 |
| Reuse controller math / qdot helper | 7 |
| MuJoCo + dry-run + CLI | 8 |
| RTDE receive-only / speedJ / watchdog / fault | 9 |
| Live TF + AprilTag cross-check manual enable | 10 |
| Exports + verification | 11 |
| Per-tick VF refresh (also when VF off) | 4, 8, 9 |
| MuJoCo: no `interface.control` until enable / never dry-run | 8 |
| `set_virtual_fixture` + tick cache before `update_command` | 8 |
| RTDE `T` = TF or `--T-tag0-base`; never undefined | 9, 10 |
| v1.5 control-law table | explicitly out of scope |

## Placeholder / risk notes

- Main risk: Task 9 kinematics bridge (RobotInterface state sync vs adapter). Spike at start of Task 9 before hardening watchdog.
- AprilTag live detection may reuse `bimanual` geom helpers via import; if ROS stack absent, cross-check is optional and must not crash teleop.
- Task 9 may ship with `--T-tag0-base` only; Task 10 adds live TF without changing the “never undefined T” rule.
)
