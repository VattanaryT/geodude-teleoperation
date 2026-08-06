"""geodude-teleoperation package.

Heavy teleop dependencies (gafropy, diffused_fields, …) load lazily on attribute
access so subpackages like ``calibration`` can be imported without them.
"""

from __future__ import annotations

__all__ = [
    "TeleoperationController",
    "AdmittanceTeleoperationController",
    "SpaceMouseTeleop",
    "KeyboardTeleop",
    "OrientationField",
    "WalkOnSpheresVirtualFixture",
    "compute_resting_orientation",
    "load_object_pointcloud",
    "load_points",
    "read_ascii_pcd",
]

_LAZY_ATTRS = {
    "TeleoperationController": ("geodude_teleoperation.controller", "TeleoperationController"),
    "AdmittanceTeleoperationController": (
        "geodude_teleoperation.controller",
        "AdmittanceTeleoperationController",
    ),
    "SpaceMouseTeleop": ("geodude_teleoperation.spacemouse", "SpaceMouseTeleop"),
    "KeyboardTeleop": ("geodude_teleoperation.keyboard", "KeyboardTeleop"),
    "OrientationField": ("geodude_teleoperation.virtual_fixture", "OrientationField"),
    "WalkOnSpheresVirtualFixture": (
        "geodude_teleoperation.virtual_fixture",
        "WalkOnSpheresVirtualFixture",
    ),
    "compute_resting_orientation": (
        "geodude_teleoperation.mesh_fixture",
        "compute_resting_orientation",
    ),
    "load_object_pointcloud": ("geodude_teleoperation.mesh_fixture", "load_object_pointcloud"),
    "load_points": ("geodude_teleoperation.mesh_fixture", "load_points"),
    "read_ascii_pcd": ("geodude_teleoperation.mesh_fixture", "read_ascii_pcd"),
}


def __getattr__(name: str):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY_ATTRS[name]
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value
