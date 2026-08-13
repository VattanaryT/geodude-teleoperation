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
    # stringer teleop — frames
    "load_T_tag0_base",
    "base_to_tag0_point",
    "tag0_rot_to_base",
    "tip_from_tcp",
    "warn_tip_offset_xy",
    "lookup_T_tag0_base_tf",
    # stringer teleop — field query
    "query_field_basis",
    "query_basis_surface",
    # stringer teleop — stringer field
    "config_defaults",
    "load_stringer_pointcloud",
    "compute_orientation_field",
    "validate_field_prints",
    "build_wos",
    # stringer teleop — contact / WoS VF
    "ContactWosOrientationField",
    "VFStatus",
    # stringer teleop — mode / status
    "TeleopState",
    "format_status_line",
    # stringer teleop — controller / keyboard
    "qdot_from_error",
    "dispatch_key",
    # stringer teleop — CLI / RTDE loop
    "parse_args",
    "validate_args",
    "run_rtde_loop",
]

_LAZY_ATTRS = {
    "TeleoperationController": ("geodude_teleoperation.teleop.controller", "TeleoperationController"),
    "AdmittanceTeleoperationController": (
        "geodude_teleoperation.teleop.controller",
        "AdmittanceTeleoperationController",
    ),
    "SpaceMouseTeleop": ("geodude_teleoperation.teleop.spacemouse", "SpaceMouseTeleop"),
    "KeyboardTeleop": ("geodude_teleoperation.teleop.keyboard", "KeyboardTeleop"),
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
    "load_T_tag0_base": ("geodude_teleoperation.frames", "load_T_tag0_base"),
    "base_to_tag0_point": ("geodude_teleoperation.frames", "base_to_tag0_point"),
    "tag0_rot_to_base": ("geodude_teleoperation.frames", "tag0_rot_to_base"),
    "tip_from_tcp": ("geodude_teleoperation.frames", "tip_from_tcp"),
    "warn_tip_offset_xy": ("geodude_teleoperation.frames", "warn_tip_offset_xy"),
    "lookup_T_tag0_base_tf": ("geodude_teleoperation.frames", "lookup_T_tag0_base_tf"),
    "query_field_basis": ("geodude_teleoperation.field.field_query", "query_field_basis"),
    "query_basis_surface": ("geodude_teleoperation.field.field_query", "query_basis_surface"),
    "config_defaults": ("geodude_teleoperation.field.stringer_field", "config_defaults"),
    "load_stringer_pointcloud": (
        "geodude_teleoperation.field.stringer_field",
        "load_stringer_pointcloud",
    ),
    "compute_orientation_field": (
        "geodude_teleoperation.field.stringer_field",
        "compute_orientation_field",
    ),
    "validate_field_prints": ("geodude_teleoperation.field.stringer_field", "validate_field_prints"),
    "build_wos": ("geodude_teleoperation.field.stringer_field", "build_wos"),
    "ContactWosOrientationField": (
        "geodude_teleoperation.teleop.contact_wos_vf",
        "ContactWosOrientationField",
    ),
    "VFStatus": ("geodude_teleoperation.teleop.contact_wos_vf", "VFStatus"),
    "TeleopState": ("geodude_teleoperation.teleop.teleop_state", "TeleopState"),
    "format_status_line": ("geodude_teleoperation.teleop.teleop_state", "format_status_line"),
    "qdot_from_error": ("geodude_teleoperation.teleop.controller", "qdot_from_error"),
    "dispatch_key": ("geodude_teleoperation.teleop.keyboard", "dispatch_key"),
    "parse_args": ("geodude_teleoperation.teleop.ur_stringer_teleop", "parse_args"),
    "validate_args": ("geodude_teleoperation.teleop.ur_stringer_teleop", "validate_args"),
    "run_rtde_loop": ("geodude_teleoperation.rtde.control_loop", "run_rtde_loop"),
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
