"""Prefer the env ``diffused_fields`` over a ROS overlay (Python 3.10 install/)."""
from __future__ import annotations

import sys
from pathlib import Path

_OVERLAY_SNIPPET = "install/diffused_fields"


def is_ros_overlay_diffused_fields_path(path: str | Path | None) -> bool:
    """True for the colcon Python 3.10 install that shadows the teleop env."""
    if path is None:
        return False
    p = str(path).replace("\\", "/")
    return _OVERLAY_SNIPPET in p and "python3.10" in p


def prefer_env_diffused_fields(*, warn: bool = False) -> list[str]:
    """Drop ROS overlay ``diffused_fields`` from ``sys.path`` before import.

    ``source ~/ros2_ws/install/setup.bash`` prepends
    ``install/diffused_fields/lib/python3.10/site-packages``. Pointcloud then
    looks for ``config/pointclouds.yaml`` next to that overlay (missing).
    """
    dropped = [p for p in sys.path if is_ros_overlay_diffused_fields_path(p)]
    if dropped:
        sys.path[:] = [p for p in sys.path if p not in dropped]
        if warn:
            print(
                "warning: ignoring ROS overlay diffused_fields "
                f"({dropped[0]}); using the geodude-teleop package instead",
                flush=True,
            )
    victim = sys.modules.get("diffused_fields")
    loc = getattr(victim, "__file__", None) if victim is not None else None
    if is_ros_overlay_diffused_fields_path(loc):
        for name in list(sys.modules):
            if name == "diffused_fields" or name.startswith("diffused_fields."):
                del sys.modules[name]
    return dropped
