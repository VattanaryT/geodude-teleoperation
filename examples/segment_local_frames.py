"""Check 4: split approach vs contact and sandwich local actions to EE-body twist.

Prints a numeric table (not plots). Rerun anytime; does not replace checks 1–3.

  PYTHONPATH=src python examples/local_frame_checks.py 4

  PYTHONPATH=src python examples/segment_local_frames.py \\
    --npz /mnt/data/boeing/local_frames_data/08_16_half_stringer_local_frames/bags_08_16_half_stringer_bimanual_0_cga_actual.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from geodude_teleoperation.local_frames_npz import load_cga_episode
from geodude_teleoperation.mode_segments import (
    format_segment_table,
    sandwich_local_actions,
)

_DATA = Path("/mnt/data/boeing/local_frames_data/08_16_half_stringer_local_frames")
_DEFAULT_NPZ = _DATA / "bags_08_16_half_stringer_bimanual_0_cga_actual.npz"
_DEFAULT_TRAIN = _DATA / "training" / "bags_08_16_half_stringer_bimanual_0.npz"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, default=_DEFAULT_NPZ)
    p.add_argument("--training", type=Path, default=_DEFAULT_TRAIN)
    p.add_argument("--lost-s", type=float, default=1.0,
                   help="DualLoop-style free-time before leaving contact")
    return p


def main() -> None:
    args = build_parser().parse_args()
    train = args.training if args.training.is_file() else None
    ep = load_cga_episode(args.npz, training_path=train)
    n = len(ep.a_local)
    t = np.asarray(ep.timestamps, dtype=float)
    t = t - t[0]
    R_ee = Rotation.from_quat(np.asarray(ep.poses[: n + 1, 3:7], dtype=float)).as_matrix()
    contact, v_body, twist, segs, stats = sandwich_local_actions(
        R_field=ep.R_field,
        R_ee=R_ee,
        a_local=ep.a_local,
        in_contact=ep.in_contact,
        timestamps=t,
        f_n=ep.f_n,
        lost_s=args.lost_s,
    )
    raw = np.asarray(ep.in_contact[:n], dtype=bool)
    print(
        f"episode {ep.meta.get('episode')}  source={ep.action_source}  "
        f"frame={ep.pose_frame}  N_act={n}  {ep.hz} Hz  "
        f"R_ee=flange quat in right_base"
    )
    print(
        f"raw in_contact={float(raw.mean()):.0%}  "
        f"sticky={float(contact.mean()):.0%}  lost_s={args.lost_s:g}  "
        f"segments={len(segs)}"
    )
    print("f_n is the force label (Fd), not plunge. Contact zeros a_n and body vz.")
    print()
    print(format_segment_table(stats))
    print()
    vz_c = float(np.abs(v_body[contact, 2]).max()) if contact.any() else 0.0
    vz_a = float(np.abs(v_body[~contact, 2]).mean()) if (~contact).any() else 0.0
    print(
        f"max |vz| contact={vz_c * 1e3:.3f} mm/s  "
        f"mean |vz| approach={vz_a * 1e3:.2f} mm/s  "
        f"twist={twist.shape} gafro [w, v]"
    )


if __name__ == "__main__":
    main()
