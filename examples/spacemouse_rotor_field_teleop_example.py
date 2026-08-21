"""Interactive SpaceMouse exploration of geodude_teleoperation's own ambient OrientationField
(see virtual_fixture.py -- gafropy.RotorFieldDiffusion under the hood). Moves one point through
R^3 and plots its orientation *as read directly from the field* at wherever it currently is, as
a single RGB-for-XYZ frame (red=local x, green=local y, blue=local z) -- no other geometry drawn.
Self-contained within this repository: builds its own OrientationField from a mesh/point-cloud
asset via geodude_teleoperation.load_object_mesh, rather than depending on gafropy's own example
scripts.

**There is no independently-controlled orientation here.** The field assigns exactly one
orientation to every point in R^3. Only the point's R^3 position is free and SpaceMouse-
controlled; orientation is always OrientationField.project()'s readout at wherever the point
currently is. Rotational SpaceMouse input (roll/pitch/yaw) is read but intentionally never used.

**"Twist in the current local frame" means**: the SpaceMouse's *linear* reading (remapped, see
--forward-axis/--left-axis/--up-axis) is transformed by the field's own current rotation at the
point's current position before being added to the position. "Push forward" moves along whatever
direction the field's own tangent currently points, not a fixed world axis.

**Up-axis gating**: --up-threshold-fraction/--up-scale keep ordinary in-plane movement from
drifting between level-sets due to incidental z-axis noise/cross-talk (a well-known SpaceMouse
ergonomics issue) -- see apply_up_gate's docstring.

Run with (mesh/point-cloud optional; needs a physically-connected 3Dconnexion SpaceMouse)::

    python examples/spacemouse_rotor_field_teleop_example.py
    python examples/spacemouse_rotor_field_teleop_example.py --mesh assets/objects/bunny_real.pcd
    python examples/spacemouse_rotor_field_teleop_example.py --forward-axis -y --left-axis x

Pass --self-check to exercise the axis remap, up-gating, accumulate/drain, and field-readout
logic with synthetic readings instead of a real device.
"""

from __future__ import annotations

import argparse
import pathlib
import threading
import time

import numpy as np

import gafropy as g
from gafropy.viz import Visualizer

from geodude_teleoperation import OrientationField, load_object_mesh

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OBJECT = str(ROOT / "assets" / "objects" / "banana_half.ply")

# Standard RGB-for-XYZ convention: local x = red, y = green, z = blue.
AXIS_COLORS = ((220, 40, 40), (40, 190, 40), (40, 90, 220))

AXIS_NAMES = ("x", "y", "z")
_RAW_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def parse_signed_axis(spec: str) -> tuple[int, float]:
    """Parses a signed raw-axis spec like 'x', '-y', '+z' into (index, sign)."""
    spec = spec.strip()
    sign = -1.0 if spec.startswith("-") else 1.0
    name = spec.lstrip("+-")
    if name not in _RAW_AXIS_INDEX:
        raise ValueError(f"invalid axis spec {spec!r}: expected one of x, -x, y, -y, z, -z")
    return _RAW_AXIS_INDEX[name], sign


def remap_raw_linear(raw_xyz: np.ndarray, forward_spec: str, left_spec: str, up_spec: str) -> np.ndarray:
    """Remaps the SpaceMouse's raw (x, y, z) translation reading into this script's own
    (+x=forward, +y=left, +z=up) output convention -- see the module docstring for the default
    guess and why it needs your confirmation on real hardware."""
    idx_f, sign_f = parse_signed_axis(forward_spec)
    idx_l, sign_l = parse_signed_axis(left_spec)
    idx_u, sign_u = parse_signed_axis(up_spec)
    return np.array([sign_f * raw_xyz[idx_f], sign_l * raw_xyz[idx_l], sign_u * raw_xyz[idx_u]])


def apply_up_gate(remapped: np.ndarray, linear_scale: float, up_threshold_fraction: float, up_scale: float) -> np.ndarray:
    """Gates the remapped vector's up-component (index 2) separately and more strictly than the
    general SpaceMouseTeleop deadzone: zeroed entirely unless it exceeds
    `up_threshold_fraction * linear_scale` (a fraction of full deflection, not raw device units --
    `twist.get_linear()` is already scaled by linear_scale), then further scaled down by
    `up_scale`. Forward/left are untouched. This is specifically to keep ordinary in-plane
    movement (forward/left) from drifting between level-sets due to incidental z-axis noise or
    cross-talk -- a well-known SpaceMouse ergonomics issue (pushing forward/sideways often reports
    some spurious z deflection too) -- while still allowing a deliberate, larger up push through."""
    result = remapped.copy()
    threshold = up_threshold_fraction * linear_scale
    if abs(result[2]) < threshold:
        result[2] = 0.0
    else:
        result[2] *= up_scale
    return result


class LinearAccumulator:
    """Thread-safe accumulator for remapped linear readings between redraw ticks. Written (added
    to) by the SpaceMouse's ~50Hz background poll thread; drained (read-and-reset to zero) once
    per redraw tick by the main thread -- so no motion between ticks is lost even though the
    field query and position update only happen once per tick, not once per device reading."""

    def __init__(self):
        self._lock = threading.Lock()
        self._accumulated = np.zeros(3)

    def add(self, linear: np.ndarray) -> None:
        with self._lock:
            self._accumulated += linear

    def drain(self) -> np.ndarray:
        with self._lock:
            value, self._accumulated = self._accumulated, np.zeros(3)
            return value


class ProbeState:
    """Thread-safe holder for the probe's current R^3 position only -- no orientation state."""

    def __init__(self, position: np.ndarray):
        self._lock = threading.Lock()
        self._position = np.asarray(position, dtype=np.float64).copy()

    def move(self, local_frame: np.ndarray, local_linear: np.ndarray) -> None:
        """Adds `local_linear` to the position, expressed in `local_frame` (a 3x3 rotation
        matrix -- the field's own current orientation at the position *before* this move)."""
        with self._lock:
            self._position = self._position + local_frame @ np.asarray(local_linear, dtype=np.float64)

    def get(self) -> np.ndarray:
        with self._lock:
            return self._position.copy()


def add_frame_arrows(scene, name: str, origin: np.ndarray, rotation: np.ndarray, scale: float) -> None:
    """Draws all 3 local axes (rotation's columns) as RGB-for-XYZ arrows at `origin`."""
    points = np.stack([np.stack([origin, origin + scale * rotation[:, axis]]) for axis in range(3)])
    colors = np.array(AXIS_COLORS, dtype=np.uint8)
    scene.add_arrows(name, points=points, colors=colors, shaft_radius=0.08 * scale, head_radius=0.2 * scale, head_length=0.35 * scale)


def build_field(args) -> tuple:
    """Builds the mesh + OrientationField (geodude_teleoperation's own gafropy-based ambient
    orientation field, see virtual_fixture.py) -- self-contained within this repository, unlike
    the gafropy-examples version this file was copied from, which manually redid stages 1/1b/2
    inline. Returns (object_mesh, mean_edge, source_index, field, key_direction)."""
    key_direction = np.asarray(args.key_direction, dtype=np.float64)
    key_direction = key_direction / np.linalg.norm(key_direction)

    object_mesh = load_object_mesh(args.mesh)
    object_vertices = np.array(object_mesh.vertices)
    mean_edge = float(object_mesh.edges_unique_length.mean())
    source_index = args.source if args.source is not None else int(np.argmax(object_vertices[:, 2]))

    print(f"object: {len(object_vertices)} vertices, {len(object_mesh.faces)} faces, mean edge length = {mean_edge:.4f}")

    field = OrientationField(
        object_mesh, key_direction=key_direction, source_vertex=source_index,
        t_surface=args.t_surface, t_spatial=args.t_spatial, num_boundary_samples=args.num_boundary_samples,
        max_iterations=args.max_iterations, smoothing=args.smoothing,
    )
    print("OrientationField built.\n")

    return object_mesh, mean_edge, source_index, field, key_direction


def field_rotation_matrix(field: "OrientationField", position: np.ndarray) -> np.ndarray:
    """The field's own orientation at `position`, via OrientationField's public .project()
    (position pass-through, orientation replaced by the field readout, NLERP-smoothed against
    the field's own last query -- see virtual_fixture.py). A throwaway identity-rotor motor is
    used as the input since project() ignores the proposed orientation entirely."""
    proposed = g.Motor(g.Translator.exp(*position), g.Rotor())
    result = field.project(proposed)
    return np.asarray(result.get_rotor().to_rotation_matrix())


def run_self_check(field_rotation_matrix, start_position: np.ndarray, args) -> None:
    """Exercises the axis remap, the accumulate/drain logic, and the position-update/field-readout
    logic with synthetic readings, printed for direct inspection -- the substitute for hardware
    verification described in the module docstring."""
    print("--- self-check: raw-axis remap ---")
    for raw, label in [(np.array([1.0, 0.0, 0.0]), "raw +x"), (np.array([0.0, 1.0, 0.0]), "raw +y"),
                       (np.array([0.0, 0.0, 1.0]), "raw +z")]:
        remapped = remap_raw_linear(raw, args.forward_axis, args.left_axis, args.up_axis)
        print(f"  {label} -> (forward, left, up) = {remapped}")
    print()

    print("--- self-check: up-axis gating (threshold + reduced sensitivity) ---")
    small_up = np.array([0.0, 0.0, 0.1 * args.linear_scale])   # below default 0.3 threshold
    large_up = np.array([0.0, 0.0, 0.8 * args.linear_scale])   # above it
    print(f"  small up push (10% of full deflection): gated -> {apply_up_gate(small_up, args.linear_scale, args.up_threshold_fraction, args.up_scale)} (must be exactly zero)")
    print(f"  large up push (80% of full deflection): gated -> {apply_up_gate(large_up, args.linear_scale, args.up_threshold_fraction, args.up_scale)} (nonzero, scaled down by --up-scale)")
    forward = np.array([0.03, 0.0, 0.0])
    print(f"  forward push, untouched by up-gate: {apply_up_gate(forward, args.linear_scale, args.up_threshold_fraction, args.up_scale)}\n")

    print("--- self-check: accumulate-then-drain loses no motion between redraw ticks ---")
    accumulator = LinearAccumulator()
    readings = [np.array([0.01, 0.0, 0.0])] * 3 + [np.array([0.0, 0.02, 0.0])] * 2
    for r in readings:
        accumulator.add(r)
    drained = accumulator.drain()
    expected = sum(readings)
    print(f"  {len(readings)} readings summed directly: {expected}")
    print(f"  accumulator.drain():                 {drained}  (residual={np.linalg.norm(drained - expected):.2e})")
    print(f"  accumulator after drain (must be zero): {accumulator.drain()}\n")

    print("--- self-check: position moves in the field's own current local frame, orientation "
          "is always a fresh field readout (no independent orientation state) ---")
    probe = ProbeState(start_position)

    synthetic_linear_readings = [
        np.array([0.05, 0.0, 0.0]),   # local +x (forward)
        np.array([0.0, 0.05, 0.0]),   # local +y (left)
        np.array([0.0, 0.0, 0.0]),    # zero input -- position must not move at all
        np.array([0.02, -0.01, 0.03]),  # mixed
    ]

    for i, linear in enumerate(synthetic_linear_readings):
        position_before = probe.get()
        local_frame = field_rotation_matrix(position_before)
        probe.move(local_frame, linear)
        position_after = probe.get()

        expected_pos = position_before + local_frame @ linear
        residual = np.linalg.norm(position_after - expected_pos)

        det = np.linalg.det(local_frame)
        orthogonality = np.max(np.abs(local_frame.T @ local_frame - np.eye(3)))
        print(f"  step {i}: linear_input={linear}  position={np.round(position_after, 4)}  "
              f"local-frame det={det:.6f}  max|R^T R-I|={orthogonality:.2e}  "
              f"move-matches-expected residual={residual:.2e}")

        if np.allclose(linear, 0.0):
            moved = np.linalg.norm(position_after - position_before)
            print(f"    zero-input check: moved by {moved:.2e} (must be ~0)")

    print("--- self-check done. ---\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", type=str, default=DEFAULT_OBJECT,
                        help="mesh or point-cloud file (any format trimesh understands, or .pcd -- reconstructed via pymeshlab if it has no faces, see load_object_mesh)")
    parser.add_argument("--key-direction", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    parser.add_argument("--source", type=int, default=None)
    parser.add_argument("--t-surface", type=float, default=None)
    parser.add_argument("--t-spatial", type=float, default=None, help="RotorFieldDiffusion spatial kernel bandwidth (default: mean-edge^2)")
    parser.add_argument("--num-boundary-samples", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--smoothing", type=float, default=0.75, help="OrientationField NLERP smoothing factor (0 = no smoothing, closer to 1 = smoother/laggier)")
    parser.add_argument("--start-offset", type=float, default=None, help="starting distance above the source vertex, along key-direction (default: 10x mean edge length)")
    parser.add_argument("--linear-scale", type=float, default=0.05, help="SpaceMouse linear step size, passed straight to SpaceMouseTeleop")
    parser.add_argument("--deadzone", type=float, default=0.05)
    parser.add_argument("--frequency", type=float, default=50.0, help="SpaceMouse poll rate -- on_twist is now O(1) (see module docstring), so this can stay high without causing lag")
    parser.add_argument("--redraw-rate", type=float, default=15.0,
                        help="rate (Hz) at which accumulated motion is applied and the scene redrawn -- the only place a field query happens on the interactive path")
    parser.add_argument("--forward-axis", type=str, default="y", help="which raw SpaceMouse axis (e.g. 'y', '-y', 'x') maps to +forward -- best guess, see module docstring")
    parser.add_argument("--left-axis", type=str, default="x", help="which raw SpaceMouse axis maps to +left")
    parser.add_argument("--up-axis", type=str, default="-z",
                        help="which raw SpaceMouse axis maps to +up -- confirmed on real hardware to need the flipped sign (raw +z reads negative when physically pushing up on this device)")
    parser.add_argument("--up-threshold-fraction", type=float, default=0.3,
                        help="fraction of full up-axis deflection required before any up motion registers at all "
                             "(stricter than --deadzone, and up-specific) -- raise this if in-plane movement still "
                             "drifts off the level-set")
    parser.add_argument("--up-scale", type=float, default=0.3,
                        help="multiplicative sensitivity reduction applied to up motion once past --up-threshold-fraction")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--self-check", action="store_true", help="run a synthetic reading sequence instead of connecting to a real SpaceMouse (no hardware needed)")
    args = parser.parse_args()

    object_mesh, mean_edge, source_index, field, key_direction = build_field(args)

    start_offset = args.start_offset if args.start_offset is not None else 10.0 * mean_edge

    def query_rotation(position: np.ndarray) -> np.ndarray:
        return field_rotation_matrix(field, position)

    object_vertices = np.array(object_mesh.vertices)
    start_position = object_vertices[source_index] + start_offset * key_direction

    if args.self_check:
        run_self_check(query_rotation, start_position, args)
        return

    from geodude_teleoperation import SpaceMouseTeleop

    probe = ProbeState(start_position)
    accumulator = LinearAccumulator()

    # --- Visualization. ---
    viz = Visualizer(port=args.port)
    scene = viz.scene

    object_mesh.visual.vertex_colors = np.full((len(object_vertices), 4), [200, 200, 210, 90], dtype=np.uint8)
    viz.scene.add_mesh_trimesh("/object", object_mesh)
    bbox_diag = float(np.linalg.norm(object_mesh.bounding_box.extents))
    scene.add_frame("/axes", axes_length=0.15 * bbox_diag, axes_radius=0.004 * bbox_diag, origin_radius=0.0)

    arrow_scale = 3.0 * mean_edge

    viz.gui.add_markdown(
        "SpaceMouse rotor-field exploration: move the point with your SpaceMouse -- only "
        "translation matters, always applied along the field's own *current* local axes at the "
        "point's position (rotational SpaceMouse input has no effect: orientation is entirely "
        "determined by the field). The frame shown is the field's own rotor at the point's "
        "current position: red=local x, green=local y, blue=local z.\n\n"
        "Ctrl+C to exit."
    )

    def redraw_and_move(force: bool = False) -> bool:
        # IMPORTANT: returns immediately, sending NO scene messages at all, if nothing
        # accumulated since the last tick (unless force=True, used only for the one-time initial
        # draw below). The previous version always re-sent all 7 scene objects every tick,
        # forever, even at complete rest -- a continuous stream of redundant messages to viser's
        # websocket regardless of whether you were touching the device. If that pipeline has
        # *any* backlog (very plausible with an object mesh also in the scene), constant
        # unnecessary traffic never lets it settle, so after you let go, the browser keeps working
        # through a backlog of stale-but-queued updates -- which looks exactly like continued
        # movement, without any actual bug in the position/accumulator logic itself (verified
        # separately, see --self-check). Fixed: skip entirely when idle.
        pending = accumulator.drain()
        if not force and np.allclose(pending, 0.0):
            return False

        position = probe.get()
        local_frame = query_rotation(position)
        if not np.allclose(pending, 0.0):
            probe.move(local_frame, pending)
            position = probe.get()
            local_frame = query_rotation(position)

        add_frame_arrows(scene, "/probe", position, local_frame, arrow_scale)
        return True

    redraw_and_move(force=True)

    # --- IMPORTANT: on_twist does no field query and no viser call -- just a remap and an
    # accumulator add, both O(1). This is the only thing that runs on SpaceMouseTeleop's ~50Hz
    # poll thread, so that thread can never stall regardless of how many Riemannian-mean
    # iterations a field query happens to need. ---
    def on_twist(twist: "g.Twist") -> None:
        raw_linear = np.asarray(twist.get_linear())  # rotational (roll/pitch/yaw) part intentionally unused
        remapped = remap_raw_linear(raw_linear, args.forward_axis, args.left_axis, args.up_axis)
        gated = apply_up_gate(remapped, args.linear_scale, args.up_threshold_fraction, args.up_scale)
        accumulator.add(gated)

    def on_button(index: int) -> None:
        pass

    device = SpaceMouseTeleop(linear_scale=args.linear_scale, angular_scale=0.0,
                              deadzone=args.deadzone, frequency=args.frequency)
    device.start(on_twist, on_button)

    print("Open the viser URL printed above in a browser to view the scene.")
    print("Move the SpaceMouse to move the point through the field. Ctrl+C to exit.")
    redraw_period = 1.0 / args.redraw_rate
    try:
        while True:
            tick_start = time.time()
            redraw_and_move()
            elapsed = time.time() - tick_start
            time.sleep(max(0.0, redraw_period - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        device.stop()


if __name__ == "__main__":
    main()
