"""ROS overlay must not shadow the env's diffused_fields."""

import sys

from geodude_teleoperation.df_runtime import (
    is_ros_overlay_diffused_fields_path,
    prefer_env_diffused_fields,
)


def test_overlay_path_detection():
    overlay = (
        "/home/boeing/ros2_ws/install/diffused_fields/lib/python3.10/site-packages"
    )
    assert is_ros_overlay_diffused_fields_path(overlay)
    assert is_ros_overlay_diffused_fields_path(overlay + "/diffused_fields/__init__.py")
    assert not is_ros_overlay_diffused_fields_path(
        "/home/boeing/ros2_ws/src/diffused_fields/src"
    )
    assert not is_ros_overlay_diffused_fields_path(
        "/opt/ros/humble/lib/python3.10/site-packages"
    )


def test_prefer_env_drops_overlay_sys_path(monkeypatch):
    overlay = (
        "/tmp/ros2_ws/install/diffused_fields/lib/python3.10/site-packages"
    )
    local = "/tmp/diffused_fields/src"
    monkeypatch.setattr(sys, "path", [overlay, local, *sys.path])
    dropped = prefer_env_diffused_fields()
    assert overlay in dropped
    assert overlay not in sys.path
    assert local in sys.path
