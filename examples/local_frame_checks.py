"""Numbered local-frame replay checks. Each stays rerunnable; later checks
do not replace earlier ones.

  PYTHONPATH=src python examples/local_frame_checks.py        # list
  PYTHONPATH=src python examples/local_frame_checks.py 2      # run check 2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_DATA = Path("/mnt/data/boeing/local_frames_data/08_16_half_stringer_local_frames")
_ACTUAL = _DATA / "bags_08_16_half_stringer_bimanual_0_cga_actual.npz"

CHECKS = (
    (
        1,
        "reconstruct actual-TCP local actions (unit tests)",
        ["python", "-m", "unittest", "tests.test_local_frames_npz", "-v"],
    ),
    (
        2,
        "viser 3D + plots of actual-TCP replay (not sim)",
        [
            "python", "examples/replay_local_frames.py",
            "--npz", str(_ACTUAL),
        ],
    ),
    (
        3,
        "TCP kinematics: bag vs chamfer/flange/yaml EE",
        ["python", "examples/check_tcp_kinematics.py"],
    ),
    (
        4,
        "approach/contact split + EE-body twist table",
        [
            "python", "examples/segment_local_frames.py",
            "--npz", str(_ACTUAL),
        ],
    ),
    (
        5,
        "wrist cam + flange vs field axes (visual kinematics)",
        [
            "python", "examples/replay_wrist_cam.py",
            "--npz", str(_ACTUAL),
        ],
    ),
    (
        6,
        "mujoco closed-loop speedJ + DualLoop (not hardware)",
        [
            "python", "examples/replay_sim_controllers.py",
            "--npz", str(_ACTUAL),
        ],
    ),
)


def _print_catalog() -> None:
    print("Local-frame replay checks (rerun any later; none are deleted):\n")
    for n, title, argv in CHECKS:
        print(f"  {n}. {title}")
        print("     PYTHONPATH=src " + " ".join(argv))
        print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("n", nargs="?", type=int, help="check number to run")
    args = p.parse_args()
    if args.n is None:
        _print_catalog()
        return
    match = [c for c in CHECKS if c[0] == args.n]
    if not match:
        _print_catalog()
        raise SystemExit(f"unknown check {args.n}")
    n, title, argv = match[0]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    print(f"running check {n}: {title}")
    raise SystemExit(subprocess.call(argv, cwd=_REPO, env=env))


if __name__ == "__main__":
    main()
