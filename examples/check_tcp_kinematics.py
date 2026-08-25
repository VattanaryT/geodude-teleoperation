"""Check 3: bag TCP vs gafro chamfer / flange / yaml EE (not sim).

Rerun this anytime; it does not replace checks 1–2. Catalog:

  PYTHONPATH=src python examples/local_frame_checks.py

  PYTHONPATH=src python examples/check_tcp_kinematics.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from geodude_teleoperation.tcp_kinematics import (
    MODEL,
    apply_right_ur6,
    poses_in_right_base,
    rotation_error_deg,
)

_REPO = Path(__file__).resolve().parent.parent
_FIXTURE = _REPO / "tests" / "fixtures" / "bimanual_0_tcp_fk_samples.npz"
_DEFAULT_BAG = Path("/mnt/data/boeing/data/bags_08_16_half_stringer/bimanual_0")


def _load_samples(bag: Path | None):
    if bag is not None and bag.exists():
        from geodude_teleoperation.tcp_kinematics import UR6_ORDER
        try:
            from rosbag2_py import (
                ConverterOptions, SequentialReader, StorageFilter, StorageOptions,
            )
            from rclpy.serialization import deserialize_message
            from rosidl_runtime_py.utilities import get_message
        except ImportError:
            bag = None
        else:
            reader = SequentialReader()
            reader.open(
                StorageOptions(uri=str(bag), storage_id="sqlite3"),
                ConverterOptions(input_serialization_format="cdr",
                                 output_serialization_format="cdr"),
            )
            topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
            reader.set_filter(StorageFilter(
                topics=["/right/joint_states", "/right/tcp_pose_broadcaster/pose"]))
            js, tcp = [], []
            while reader.has_next():
                topic, data, t = reader.read_next()
                msg = deserialize_message(data, get_message(topics[topic]))
                if topic.endswith("joint_states"):
                    js.append((t, dict(zip(msg.name, msg.position))))
                else:
                    p, o = msg.pose.position, msg.pose.orientation
                    tcp.append((t, np.array(
                        [p.x, p.y, p.z, o.x, o.y, o.z, o.w], dtype=float)))
            js_t = np.array([x[0] for x in js], dtype=np.int64)
            idx = np.linspace(0, len(tcp) - 1, 8, dtype=int)
            q6, poses = [], []
            for i in idx:
                t, pose = tcp[i]
                j = int(np.argmin(np.abs(js_t - t)))
                q6.append([js[j][1][n] for n in UR6_ORDER])
                poses.append(pose)
            return np.asarray(q6), np.asarray(poses), f"bag {bag}"
    data = np.load(_FIXTURE)
    return (np.asarray(data["q_ur6"]), np.asarray(data["tcp_pose_xyzw"]),
            f"fixture {_FIXTURE.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bag", type=Path, default=_DEFAULT_BAG)
    p.add_argument("--fixture-only", action="store_true")
    args = p.parse_args()
    import gafropy as ga

    system = ga.SystemSerialization.load(str(MODEL))
    q0 = np.asarray(system.get_default_configuration(), dtype=float)
    q6, poses, src = _load_samples(None if args.fixture_only else args.bag)
    print(f"check 3  source={src}  n={len(q6)}")
    print(f"{'i':>3} {'chamfer_mm':>11} {'flange_deg':>11} {'yamlEE_mm':>11}")
    ch, fl, ee = [], [], []
    for i, (qi, pose) in enumerate(zip(q6, poses)):
        frames = poses_in_right_base(system, apply_right_ur6(system, q0, qi))
        R_bag = Rotation.from_quat(pose[3:]).as_matrix()
        d_ch = float(np.linalg.norm(frames["chamfer"][:3, 3] - pose[:3]) * 1e3)
        a_fl = rotation_error_deg(frames["flange"][:3, :3], R_bag)
        d_ee = float(np.linalg.norm(frames["yaml_ee"][:3, 3] - pose[:3]) * 1e3)
        ch.append(d_ch); fl.append(a_fl); ee.append(d_ee)
        print(f"{i:3d} {d_ch:11.2f} {a_fl:11.2f} {d_ee:11.2f}")
    print(
        f"mean chamfer {np.mean(ch):.2f} mm  flange {np.mean(fl):.2f} deg  "
        f"yaml EE {np.mean(ee):.1f} mm"
    )
    print(
        "Use chamfer position + flange orientation in right_base "
        "(gafro base * Rz 180). Do not drive yaml right_ee for this bag."
    )


if __name__ == "__main__":
    main()
