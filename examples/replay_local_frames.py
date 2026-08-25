"""Check 2: offline 3D + plot replay of one CGA local-frame episode (right_base).

Not sim and not the robot. Catalog of rerunnable checks:

  PYTHONPATH=src python examples/local_frame_checks.py

  PYTHONPATH=src python examples/replay_local_frames.py \\
    --npz /mnt/data/boeing/local_frames_data/08_16_half_stringer_local_frames/bags_08_16_half_stringer_bimanual_0_cga_actual.npz
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
    replay_arrays,
)
from geodude_teleoperation.mesh_fixture import load_points
from geodude_teleoperation.viser_view import apply_stringer_view

_DATA = Path("/mnt/data/boeing/local_frames_data/08_16_half_stringer_local_frames")
_DEFAULT_NPZ = _DATA / "bags_08_16_half_stringer_bimanual_0_cga.npz"
_DEFAULT_TRAIN = _DATA / "training" / "bags_08_16_half_stringer_bimanual_0.npz"
_DEFAULT_PLY = _DATA / "merged_rightbase.ply"

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, default=_DEFAULT_NPZ)
    p.add_argument("--training", type=Path, default=_DEFAULT_TRAIN)
    p.add_argument("--ply", type=Path, default=_DEFAULT_PLY)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--no-plots", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    train = args.training if args.training.is_file() else None
    ep = load_cga_episode(args.npz, training_path=train)
    replay_xyz = integrate_desired_xyz(ep.action_xyz[0], ep.R_field, ep.a_local)
    actual = np.asarray(ep.poses[:, :3], dtype=float)
    series = replay_arrays(ep)
    n = len(replay_xyz)
    cloud = load_points(args.ply) if args.ply.is_file() else None

    print(
        f"episode {ep.meta.get('episode')}  source={ep.action_source}  "
        f"frame={ep.pose_frame}  N={n}  {ep.hz} Hz  contact={float(ep.in_contact.mean()):.0%}"
    )
    print("Open the viser URL. Slider/play steps the field frame + plot playhead.")

    server = viser.ViserServer(port=args.port)
    apply_stringer_view(server, cloud if cloud is not None else actual)
    if cloud is not None:
        server.scene.add_point_cloud(
            "/object", points=cloud, colors=(180, 180, 185), point_size=0.002)
    segs = _polyline_segments(replay_xyz)
    server.scene.add_line_segments(
        "/replay", points=segs,
        colors=_contact_segment_colors(series["in_contact"], len(segs)),
        thickness=0.0025,
    )
    server.scene.add_line_segments(
        "/actual", points=_polyline_segments(actual),
        colors=(255, 170, 40), thickness=0.0015,
    )
    if ep.action_source == "actual":
        server.scene.add_line_segments(
            "/command", points=_polyline_segments(ep.poses_desired_xyz),
            colors=(140, 140, 220), thickness=0.0012,
        )
    server.scene.add_frame("/axes", axes_length=0.08, axes_radius=0.003)

    def draw_step(i: int) -> None:
        i = int(np.clip(i, 0, n - 1))
        R = ep.R_field[i]
        p = replay_xyz[i]
        server.scene.add_frame(
            "/tcp_desired",
            wxyz=_wxyz_from_R(R),
            position=p,
            axes_length=0.04,
            axes_radius=0.002,
        )
        origin = p
        arrows = np.stack([np.stack([origin, origin + 0.05 * R[:, ax]]) for ax in range(3)])
        server.scene.add_arrows(
            "/R_field", points=arrows, colors=_AXIS_RGB,
            shaft_radius=0.0015, head_radius=0.0035, head_length=0.008,
        )
        server.scene.add_icosphere(
            "/actual_tcp", radius=0.006, color=(255, 170, 40), position=actual[i])

    slider = server.gui.add_slider("step", min=0, max=n - 1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("play", initial_value=False)
    draw_step(0)

    fig = axes = vlines = None
    if not args.no_plots:
        import matplotlib.pyplot as plt

        plt.ion()
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 8))
        t_a = series["t_action_s"]
        axes[0].plot(t_a, series["a_t1"] * 1e3, label="a_t1")
        axes[1].plot(t_a, series["a_t2"] * 1e3, label="a_t2")
        axes[2].plot(t_a, series["a_n"] * 1e3, label="a_n")
        axes[0].set_ylabel("a_t1 mm")
        axes[1].set_ylabel("a_t2 mm")
        axes[2].set_ylabel("a_n mm")
        if "f_n" in series:
            axes[3].plot(t_a, series["f_n"], color="k", label="f_n")
            axes[3].set_ylabel("f_n N")
        else:
            axes[3].plot(series["t_s"], series["signed_distance"] * 1e3)
            axes[3].set_ylabel("signed mm")
        axes[3].set_xlabel("t s")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        vlines = [ax.axvline(0.0, color="0.2", lw=1.0) for ax in axes]
        fig.suptitle(
            f"{ep.meta.get('episode')}  {ep.action_source}  {ep.pose_frame}  "
            "(red=contact on replay path)"
        )
        fig.tight_layout()
        plt.show(block=False)

    def on_step(i: int) -> None:
        draw_step(i)
        if vlines is not None:
            t = float(series["t_s"][int(i)])
            for line in vlines:
                line.set_xdata([t, t])
            fig.canvas.draw_idle()

    slider.on_update(lambda _: on_step(int(slider.value)))

    print(f"viser http://localhost:{args.port}")
    last = time.monotonic()
    while True:
        now = time.monotonic()
        if playing.value and (now - last) >= 1.0 / max(ep.hz, 1):
            nxt = int(slider.value) + 1
            if nxt >= n:
                nxt = 0
            slider.value = nxt
            on_step(nxt)
            last = now
        if fig is not None:
            fig.canvas.flush_events()
        time.sleep(0.01)


if __name__ == "__main__":
    main()
