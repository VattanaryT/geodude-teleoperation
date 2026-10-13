"""UMI-FT zarr episode -> fake CGA local-frame NPZ for viser replay.

Confirms the schema gap (fixture pose7 wxyz @ ~200 Hz, no a_local / R_field)
and builds the minimal NPZ that ``replay_local_frames.py`` /
``replay_wrist_cam.py`` can load.

Example (stringer_scanner_v2 ep0):

  PYTHONPATH=src python examples/umift_episode_to_cga_npz.py \\
    --zarr /mnt/data/umidata/ft/datasets/stringer_scanner_v2/acp_replay_buffer.zarr \\
    --episode 0 \\
    --out-dir /mnt/data/boeing/local_frames_data/umift_stringer_scanner_v2_ep0

  PYTHONPATH=src python examples/replay_wrist_cam.py \\
    --npz       .../episode_0_cga_actual.npz \\
    --training  .../episode_0_training.npz \\
    --cam       .../episode_0_wrist_cam.npz \\
    --ply       .../merged_rightbase.ply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
import zarr
from scipy.spatial.transform import Rotation

from geodude_teleoperation.local_frames_npz import local_actions_from_xyz

_DEFAULT_ZARR = Path(
    "/mnt/data/umidata/ft/datasets/stringer_scanner_v2/acp_replay_buffer.zarr"
)
_DEFAULT_EXTRINSICS = Path(
    "/home/boeing/umi_ft_ws/UMI-FT/PyriteConfig/tasks/umift/layup_workcell_extrinsics.yaml"
)
_DEFAULT_SCAN_PLY = Path("/mnt/data/boeing/scans/scan_20261008_170729/merged.ply")
_FORCE_CONTACT_N = 5.0
_SIGNED_CONTACT_M = 0.005
_SIGNED_FREE_M = 0.05


def _load_T(path: Path, key: str) -> np.ndarray:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    return np.asarray(cfg[key], dtype=float)


def _pose7_wxyz_to_T(pose7: np.ndarray) -> np.ndarray:
    """UMI pose7 = xyz + quat_wxyz -> 4x4."""
    T = np.eye(4)
    T[:3, 3] = pose7[:3]
    wxyz = pose7[3:7]
    T[:3, :3] = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
    return T


def _T_to_pose7_xyzw(T: np.ndarray) -> np.ndarray:
    q_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.concatenate([T[:3, 3], q_xyzw])


def _downsample_indices(t: np.ndarray, hz: float) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(-1)
    if t.size == 0:
        return np.zeros(0, dtype=int)
    step = 1.0 / float(hz)
    targets = np.arange(t[0], t[-1] + 0.5 * step, step)
    idx = np.searchsorted(t, targets)
    idx = np.clip(idx, 0, len(t) - 1)
    # drop duplicates from plateaus
    keep = np.concatenate([[True], np.diff(idx) > 0])
    return idx[keep]


def _nearest(query: np.ndarray, source: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=float).reshape(-1)
    s = np.asarray(source, dtype=float).reshape(-1)
    i = np.clip(np.searchsorted(s, q), 0, len(s) - 1)
    prev = np.clip(i - 1, 0, len(s) - 1)
    pick_prev = np.abs(q - s[prev]) <= np.abs(q - s[i])
    return np.where(pick_prev, prev, i)


def _frames_from_normals(normals: np.ndarray) -> np.ndarray:
    """R columns = [t1, t2, n]; n from mesh, t1/t2 orthonormal."""
    n = np.asarray(normals, dtype=float)
    n = n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-9, None)
    # pick a stable helper axis not parallel to n
    helper = np.zeros_like(n)
    helper[:, 0] = 1.0
    swap = np.abs(n[:, 0]) > 0.9
    helper[swap] = (0.0, 1.0, 0.0)
    t1 = np.cross(helper, n)
    t1 = t1 / np.clip(np.linalg.norm(t1, axis=1, keepdims=True), 1e-9, None)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2, n], axis=-1)


def _as_point_cloud(ply: Path):
    """Scan PLYs are often vertex-only clouds, not triangle meshes."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(ply))
    if len(pcd.points) == 0:
        mesh = o3d.io.read_triangle_mesh(str(ply))
        pcd = o3d.geometry.PointCloud(mesh.vertices)
    if len(pcd.points) == 0:
        raise ValueError(f"empty cloud: {ply}")
    return pcd


def _estimate_normals_fast(pcd, *, voxel: float = 0.003):
    """Voxel-downsample then PCA normals; flip toward +Y (outward for this cell)."""
    import open3d as o3d

    cloud = pcd.voxel_down_sample(voxel) if voxel > 0 else pcd
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=40)
    )
    n = np.asarray(cloud.normals)
    if float(np.mean(n[:, 1])) < 0:
        cloud.normals = o3d.utility.Vector3dVector(-n)
    return cloud


def _query_mesh_normals(xyz: np.ndarray, ply: Path) -> np.ndarray:
    import open3d as o3d

    pcd = _estimate_normals_fast(_as_point_cloud(ply))
    nrm = np.asarray(pcd.normals, dtype=float)
    kdt = o3d.geometry.KDTreeFlann(pcd)
    out = np.zeros((len(xyz), 3), dtype=float)
    for i, p in enumerate(xyz):
        _, idx, _ = kdt.search_knn_vector_3d(p, 1)
        out[i] = nrm[idx[0]]
    return out


def _transform_ply(src: Path, dst: Path, T: np.ndarray) -> Path:
    import open3d as o3d

    pcd = _as_point_cloud(src)
    pcd.transform(T)
    pcd = _estimate_normals_fast(pcd, voxel=0.003)
    dst.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(dst), pcd)
    return dst


def _rgb_to_wrist_npz(
    rgb: np.ndarray, stamps: np.ndarray, dst: Path
) -> Path:
    import cv2

    jpegs: list[bytes] = []
    for frame in rgb:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        jpegs.append(buf.tobytes())
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        stamps=np.asarray(stamps, dtype=float),
        jpeg=np.array(jpegs, dtype=object),
        topic=np.array("/wrist_cam_0/color/image_rect_raw/compressed"),
        bag=np.array("umift_rgb_0"),
    )
    return dst


def convert_episode(
    zarr_path: Path,
    episode: int,
    out_dir: Path,
    *,
    extrinsics: Path,
    scan_ply: Path,
    hz: float,
    force_n: float,
) -> dict[str, Path]:
    root = zarr.open(str(zarr_path), "r")
    ep = root[f"data/episode_{episode}"]
    t_robot = np.asarray(ep["robot_time_stamps_0"], dtype=float).reshape(-1)
    pose_fb = np.asarray(ep["ts_pose_fb_0"], dtype=float)
    pose_cmd = np.asarray(ep["ts_pose_command_0"], dtype=float)
    t_w = np.asarray(ep["wrench_time_stamps_0"], dtype=float).reshape(-1)
    wrench = np.asarray(ep["wrench_left_0"], dtype=float)
    t_rgb = np.asarray(ep["rgb_time_stamps_0"], dtype=float).reshape(-1)
    rgb = np.asarray(ep["rgb_0"])

    idx = _downsample_indices(t_robot, hz)
    t = t_robot[idx]
    fb = pose_fb[idx]
    cmd = pose_cmd[idx]
    w_idx = _nearest(t, t_w)
    force = np.linalg.norm(wrench[w_idx, :3], axis=1)
    in_contact = force >= float(force_n)

    T_rb_f = _load_T(extrinsics, "T_right_base_from_fixture")
    T_rb_s = _load_T(extrinsics, "T_right_base_from_scanner_base")

    poses = np.stack([_T_to_pose7_xyzw(T_rb_f @ _pose7_wxyz_to_T(p)) for p in fb])
    desired = np.stack([(T_rb_f @ _pose7_wxyz_to_T(p))[:3, 3] for p in cmd])
    xyz = poses[:, :3]

    out_dir.mkdir(parents=True, exist_ok=True)
    ply_out = out_dir / "merged_rightbase.ply"
    if scan_ply.is_file():
        _transform_ply(scan_ply, ply_out, T_rb_s)
        normals = _query_mesh_normals(xyz, ply_out)
        R_field = _frames_from_normals(normals)
        field_note = "mesh-NN normals on transformed scan PLY"
    else:
        R_field = np.tile(np.eye(3), (len(xyz), 1, 1))
        field_note = "identity (no scan PLY)"

    a_local = local_actions_from_xyz(R_field, xyz)
    signed = np.where(in_contact, _SIGNED_CONTACT_M, _SIGNED_FREE_M).astype(float)

    meta = {
        "episode": f"umift_episode_{episode}",
        "pose_frame": "right_base",
        "action_source": "actual",
        "source_zarr": str(zarr_path),
        "source_episode": int(episode),
        "hz_target": float(hz),
        "contact_threshold": _SIGNED_CONTACT_M,
        "contact_proxy": f"|F|>= {force_n} N from wrench_left_0 (signed_cga is a stand-in)",
        "field": field_note,
        "quat": "poses are xyz+xyzw (scipy); zarr was xyz+wxyz in fixture",
        "convention": "a_local = R_field^T (x_actual[t+1]-x_actual[t]); columns [t1,t2,n]",
        "mismatch_note": (
            "UMI zarr has no CGA SDF / true surface-frame actions; this NPZ is a "
            "viz/training-bridge: fixture->right_base, force-proxy contact, mesh-NN field."
        ),
    }
    cga_path = out_dir / f"episode_{episode}_cga_actual.npz"
    np.savez_compressed(
        cga_path,
        timestamps=t,
        poses=poses,
        poses_desired=desired,
        a_local_cga=a_local,
        R_field_cga=R_field,
        signed_cga=signed,
        meta=json.dumps(meta),
    )

    # sibling training NPZ so replay_wrist_cam can color by |fn|>=5
    n_act = len(a_local)
    f_n = force[: n_act + 1]
    obs = np.zeros((n_act, 12), dtype=float)
    obs[:, 9] = f_n[:n_act]
    obs[:, 10] = in_contact[:n_act].astype(float)
    action = np.zeros((n_act, 6), dtype=float)
    action[:, :3] = a_local
    train_path = out_dir / f"episode_{episode}_training.npz"
    np.savez_compressed(
        train_path,
        obs=obs,
        action=action,
        in_contact=in_contact[:n_act],
    )

    cam_path = out_dir / f"episode_{episode}_wrist_cam.npz"
    _rgb_to_wrist_npz(rgb, t_rgb, cam_path)

    print(
        f"episode {episode}: N={len(t)} @~{hz}Hz  contact={float(in_contact.mean()):.0%}  "
        f"force_max={float(force.max()):.1f}N  field={field_note}"
    )
    print(f"  cga:      {cga_path}")
    print(f"  training: {train_path}")
    print(f"  cam:      {cam_path}")
    print(f"  ply:      {ply_out if ply_out.is_file() else '(missing scan ply)'}")
    return {
        "cga": cga_path,
        "training": train_path,
        "cam": cam_path,
        "ply": ply_out,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr", type=Path, default=_DEFAULT_ZARR)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/data/boeing/local_frames_data/umift_stringer_scanner_v2_ep0"),
    )
    p.add_argument("--extrinsics", type=Path, default=_DEFAULT_EXTRINSICS)
    p.add_argument("--scan-ply", type=Path, default=_DEFAULT_SCAN_PLY)
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--force-n", type=float, default=_FORCE_CONTACT_N)
    return p


def main() -> None:
    args = build_parser().parse_args()
    convert_episode(
        args.zarr,
        args.episode,
        args.out_dir,
        extrinsics=args.extrinsics,
        scan_ply=args.scan_ply,
        hz=args.hz,
        force_n=args.force_n,
    )


if __name__ == "__main__":
    main()
