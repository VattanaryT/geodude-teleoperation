"""Check 5: wrist-cam + flange vs field axes (visual kinematics).

Scrub contact frames and confirm the chamfer is on the stringer. Does not
replace checks 1–4.

  PYTHONPATH=src python examples/local_frame_checks.py 5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import viser
from scipy.spatial.transform import Rotation

from geodude_teleoperation.local_frames_npz import (
    integrate_desired_xyz,
    load_cga_episode,
)
from geodude_teleoperation.mesh_fixture import load_points
from geodude_teleoperation.mode_segments import sandwich_local_actions
from geodude_teleoperation.wrist_cam import (
    WRIST_CAM_TOPIC,
    annotate_wrist_frame,
    decode_jpeg,
    load_wrist_cam,
    nearest_indices,
    overlay_caption,
)

_DATA = Path("/mnt/data/boeing/local_frames_data/08_16_half_stringer_local_frames")
_DEFAULT_NPZ = _DATA / "bags_08_16_half_stringer_bimanual_0_cga_actual.npz"
_DEFAULT_TRAIN = _DATA / "training" / "bags_08_16_half_stringer_bimanual_0.npz"
_DEFAULT_PLY = _DATA / "merged_rightbase.ply"
_DEFAULT_BAG = Path("/mnt/data/boeing/data/bags_08_16_half_stringer/bimanual_0")
_DEFAULT_CAM = _DATA / "bags_08_16_half_stringer_bimanual_0_wrist_cam.npz"

_AXIS_RGB = np.array([[255, 40, 40], [40, 200, 40], [40, 80, 255]], dtype=np.uint8)


def _polyline_segments(xyz: np.ndarray) -> np.ndarray:
    return np.stack([xyz[:-1], xyz[1:]], axis=1)


def _wxyz_from_R(R: np.ndarray) -> np.ndarray:
    q_xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def _contact_segment_colors(in_contact: np.ndarray, n_seg: int) -> np.ndarray:
    flags = np.asarray(in_contact[:n_seg], dtype=bool)
    colors = np.zeros((n_seg, 2, 3), dtype=np.uint8)
    colors[:, :, :] = (80, 180, 90)
    colors[flags] = (220, 60, 50)
    return colors


def _axis_arrows(origin: np.ndarray, R: np.ndarray, length: float) -> np.ndarray:
    o = np.asarray(origin, dtype=float).reshape(3)
    R = np.asarray(R, dtype=float).reshape(3, 3)
    return np.stack([np.stack([o, o + length * R[:, ax]]) for ax in range(3)])


def resolve_wrist_source(cam: Path | None, bag: Path) -> Path:
    if cam is not None:
        return cam
    if _DEFAULT_CAM.is_file():
        return _DEFAULT_CAM
    return bag


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, default=_DEFAULT_NPZ)
    p.add_argument("--training", type=Path, default=_DEFAULT_TRAIN)
    p.add_argument("--ply", type=Path, default=_DEFAULT_PLY)
    p.add_argument("--bag", type=Path, default=_DEFAULT_BAG)
    p.add_argument("--cam", type=Path, default=None,
                   help="wrist-cam NPZ cache; default is sidecar or --bag")
    p.add_argument("--lost-s", type=float, default=1.0)
    p.add_argument("--port", type=int, default=8080)
    return p


def main() -> None:
    args = build_parser().parse_args()
    train = args.training if args.training.is_file() else None
    ep = load_cga_episode(args.npz, training_path=train)
    n = len(ep.poses)
    t = np.asarray(ep.timestamps, dtype=float)
    t_rel = t - t[0]
    actual = np.asarray(ep.poses[:, :3], dtype=float)
    R_flange = Rotation.from_quat(np.asarray(ep.poses[:, 3:7], dtype=float)).as_matrix()
    contact, _, _, _, _ = sandwich_local_actions(
        R_field=ep.R_field,
        R_ee=R_flange,
        a_local=ep.a_local,
        in_contact=ep.in_contact,
        timestamps=t,
        f_n=ep.f_n,
        lost_s=args.lost_s,
    )
    # sticky flags are action-length; last pose keeps the last mode
    mode_flags = np.zeros(n, dtype=bool)
    mode_flags[: len(contact)] = contact
    if len(contact) < n:
        mode_flags[-1] = contact[-1]
    replay_xyz = integrate_desired_xyz(ep.action_xyz[0], ep.R_field, ep.a_local)
    f_n = None if ep.f_n is None else np.asarray(ep.f_n, dtype=float)

    src = resolve_wrist_source(args.cam, args.bag)
    print(f"loading wrist cam from {src} ({WRIST_CAM_TOPIC})")
    cam_t, jpegs = load_wrist_cam(src, WRIST_CAM_TOPIC)
    idx = nearest_indices(t, cam_t)
    lag = np.abs(t - cam_t[idx])
    print(
        f"episode {ep.meta.get('episode')}  N={n}  {ep.hz} Hz  "
        f"cam={len(jpegs)}  max |t_npz-t_cam|={float(lag.max()) * 1e3:.0f} ms"
    )
    print("Open viser. Slider is synced wrist JPEG + flange (bag) vs field axes.")
    print("Contact should show the chamfer on the stringer. Flange Z vs field n.")

    cloud = load_points(args.ply) if args.ply.is_file() else None
    server = viser.ViserServer(port=args.port)
    if cloud is not None:
        server.scene.add_point_cloud(
            "/object", points=cloud, colors=(180, 180, 185), point_size=0.002)
    server.scene.add_line_segments(
        "/replay", points=_polyline_segments(replay_xyz),
        colors=_contact_segment_colors(mode_flags, n - 1),
        thickness=0.0025,
    )
    server.scene.add_line_segments(
        "/actual", points=_polyline_segments(actual),
        colors=(255, 170, 40), thickness=0.0015,
    )
    server.scene.add_frame("/axes", axes_length=0.08, axes_radius=0.003)

    first = annotate_wrist_frame(
        decode_jpeg(jpegs[int(idx[0])]),
        mode="contact" if mode_flags[0] else "approach",
        t_s=float(t_rel[0]),
        f_n=None if f_n is None else float(f_n[0]),
    )
    img = server.gui.add_image(first, label="wrist_cam_0", format="jpeg")
    caption = server.gui.add_markdown(overlay_caption(
        "contact" if mode_flags[0] else "approach", float(t_rel[0]),
        None if f_n is None else float(f_n[0])))

    def draw_step(i: int) -> None:
        i = int(np.clip(i, 0, n - 1))
        p = actual[i]
        R_f = ep.R_field[min(i, len(ep.R_field) - 1)]
        R_b = R_flange[i]
        mode = "contact" if mode_flags[i] else "approach"
        fn = None if f_n is None else float(f_n[min(i, len(f_n) - 1)])
        rgb = annotate_wrist_frame(
            decode_jpeg(jpegs[int(idx[i])]), mode=mode, t_s=float(t_rel[i]), f_n=fn)
        img.image = rgb
        caption.content = overlay_caption(mode, float(t_rel[i]), fn)
        server.scene.add_frame(
            "/flange",
            wxyz=_wxyz_from_R(R_b),
            position=p,
            axes_length=0.05,
            axes_radius=0.0025,
        )
        server.scene.add_arrows(
            "/R_field", points=_axis_arrows(p, R_f, 0.06), colors=_AXIS_RGB,
            shaft_radius=0.0015, head_radius=0.0035, head_length=0.008,
        )
        server.scene.add_icosphere(
            "/tcp", radius=0.006, color=(255, 170, 40), position=p)

    slider = server.gui.add_slider("step", min=0, max=n - 1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("play", initial_value=False)
    draw_step(0)
    slider.on_update(lambda _: draw_step(int(slider.value)))

    print(f"viser http://localhost:{args.port}")
    last = time.monotonic()
    while True:
        now = time.monotonic()
        if playing.value and (now - last) >= 1.0 / max(ep.hz, 1):
            nxt = int(slider.value) + 1
            if nxt >= n:
                nxt = 0
            slider.value = nxt
            draw_step(nxt)
            last = now
        time.sleep(0.01)


if __name__ == "__main__":
    main()
