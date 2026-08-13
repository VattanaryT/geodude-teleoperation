"""Startup field debug viz helpers (no Polyscope window in unit tests)."""
from __future__ import annotations

import numpy as np

from geodude_teleoperation.viz.field_viz import (
    R_tcp_base_to_viz,
    bc_clouds_tag0,
    camera_pose_in_viz,
    contact_frame_overlays,
    points_in_viz_frame,
    teleop_key_ghosts,
    tip_tcp_markers,
)


def test_points_in_viz_frame_tag0_identity():
    V = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    T = np.eye(4)
    T[:3, 3] = [10.0, 0.0, 0.0]
    out = points_in_viz_frame(
        V, T_tag0_vention=T, T_rtde_base_vention=None, viz_frame="tag0"
    )
    assert np.allclose(out, V)


def test_points_in_viz_frame_vention_applies_T():
    V = np.array([[1.0, 0.0, 0.0]])
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 0.0]
    out = points_in_viz_frame(
        V, T_tag0_vention=T, T_rtde_base_vention=None, viz_frame="vention"
    )
    assert np.allclose(out, [[1.5, 0.0, 0.0]])


def test_points_rtde_uses_mount():
    V = np.array([[0.0, 0.0, 0.0]])
    T_tag = np.eye(4)
    T_tag[:3, 3] = [1.0, 0.0, 0.0]
    T_rv = np.eye(4)
    T_rv[:3, 3] = [0.3, 0.0, 0.0]
    # p_rtde = inv(T_rv) @ T_tag @ p_tag0 = [0.7,0,0]
    out = points_in_viz_frame(
        V, T_tag0_vention=T_tag, T_rtde_base_vention=T_rv, viz_frame="rtde"
    )
    assert np.allclose(out, [[0.7, 0.0, 0.0]])


def test_bc_clouds_end_lines():
    class _PC:
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    class _SD:
        source_vertices = [2]
        sink_vertices = [0]

    clouds = bc_clouds_tag0(_PC(), _SD())
    assert "source_line" in clouds and "sink_line" in clouds


def test_tip_tcp_markers_vention_with_mount():
    T_tag = np.eye(4)
    T_rv = np.eye(4)
    T_rv[:3, 3] = [0.34, -0.828, 1.65]
    tcp_pose = [0.1, 0.0, 0.2, 0.0, 0.0, 0.0]
    markers = tip_tcp_markers(
        tcp_pose,
        np.zeros(3),
        T_tag0_vention=T_tag,
        T_rtde_base_vention=T_rv,
        viz_frame="vention",
    )
    assert np.allclose(markers["tcp"], [0.44, -0.828, 1.85])
    assert "T_tcp" in markers
    assert np.allclose(markers["T_tcp"][:3, 3], markers["tcp"])


def test_camera_pose_in_viz_vention_with_mount():
    T_tag = np.eye(4)
    T_rv = np.eye(4)
    T_rv[:3, 3] = [0.34, -0.828, 1.65]
    T_cam_EE = np.eye(4)
    T_cam_EE[:3, 3] = [0.0, 0.0, 0.1]
    tcp_pose = [0.1, 0.0, 0.2, 0.0, 0.0, 0.0]
    T_cam = camera_pose_in_viz(
        tcp_pose,
        T_cam_EE,
        T_tag0_vention=T_tag,
        T_rtde_base_vention=T_rv,
        viz_frame="vention",
    )
    assert T_cam is not None
    # TCP at [0.44,-0.828,1.85] + cam z offset 0.1 → [0.44,-0.828,1.95]
    assert np.allclose(T_cam[:3, 3], [0.44, -0.828, 1.95])


def test_show_field_debug_builds_scene(monkeypatch):
    from geodude_teleoperation.viz import field_viz

    calls = {"show": 0, "clouds": [], "enabled": {}, "framed": False}

    class _FakePS:
        _is_fake_for_tests = True

        def init(self):
            pass

        def set_up_dir(self, *_a, **_k):
            calls["up_dir"] = _a[0] if _a else None

        def reset_camera_to_home_view(self):
            calls["reset_home"] = True

        def set_ground_plane_mode(self, *_a, **_k):
            pass

        def register_point_cloud(self, name, pts, **kwargs):
            calls["clouds"].append(name)
            calls["enabled"][name] = kwargs.get("enabled", True)

            class _Cloud:
                def add_scalar_quantity(self, *_a, **_k):
                    pass

            return _Cloud()

        def register_curve_network(self, name, *_a, **kwargs):
            calls["clouds"].append(name)
            calls["enabled"][name] = kwargs.get("enabled", True)
            return object()

        def show(self):
            calls["show"] += 1

    monkeypatch.setattr(field_viz, "ps", _FakePS())
    monkeypatch.setattr(
        field_viz,
        "plot_orientation_field",
        lambda *a, **k: calls["clouds"].append("field"),
    )
    monkeypatch.setattr(
        field_viz,
        "set_camera_and_plane",
        lambda **k: calls.update({"up_dir": k.get("up", "z_up")}),
    )
    monkeypatch.setattr(
        field_viz,
        "frame_camera_on_points",
        lambda *_a, **_k: calls.update({"framed": True}),
    )

    class _PC:
        vertices = np.zeros((10, 3))
        local_bases = np.tile(np.eye(3), (10, 1, 1))

    class _SD:
        source_vertices = [0, 1]
        sink_vertices = [8, 9]
        ut = np.linspace(0, 1, 10)

    T_cam_EE = np.eye(4)
    T_cam_EE[:3, 3] = [0.02, 0.0, 0.05]
    field_viz.show_field_debug(
        _PC(),
        _SD(),
        T_tag0_vention=np.eye(4),
        T_rtde_base_vention=np.eye(4),
        viz_frame="vention",
        frame_stride=2,
        tcp_pose_rtde=[0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        tip_offset=np.zeros(3),
        T_cam_EE=T_cam_EE,
    )
    assert calls["show"] == 1
    assert calls.get("up_dir") == "z_up"
    assert calls.get("framed") is True
    assert calls["enabled"].get("rtde_base") is False
    assert "object" in calls["clouds"]
    assert "tcp" in calls["clouds"]
    assert "tcp_x" in calls["clouds"]
    assert "tip" in calls["clouds"]
    assert "camera" in calls["clouds"]
    assert "camera_frustum" in calls["clouds"]


def test_show_field_debug_laser_overlay(monkeypatch):
    from geodude_teleoperation.viz import field_viz

    calls = {"clouds": []}

    class _FakePS:
        _is_fake_for_tests = True

        def init(self):
            pass

        def set_up_dir(self, *_a, **_k):
            pass

        def reset_camera_to_home_view(self):
            pass

        def set_ground_plane_mode(self, *_a, **_k):
            pass

        def register_point_cloud(self, name, pts, **kwargs):
            calls["clouds"].append(name)

            class _Cloud:
                def add_scalar_quantity(self, *_a, **_k):
                    pass

            return _Cloud()

        def register_curve_network(self, name, *_a, **_k):
            return object()

        def show(self):
            pass

    monkeypatch.setattr(field_viz, "ps", _FakePS())
    monkeypatch.setattr(
        field_viz, "plot_orientation_field", lambda *a, **k: None
    )
    monkeypatch.setattr(field_viz, "set_camera_and_plane", lambda **_k: None)
    monkeypatch.setattr(field_viz, "frame_camera_on_points", lambda *_a, **_k: None)

    class _PC:
        vertices = np.zeros((4, 3))
        local_bases = np.tile(np.eye(3), (4, 1, 1))

    class _SD:
        source_vertices = [0]
        sink_vertices = [1]
        ut = None

    field_viz.show_field_debug(
        _PC(),
        _SD(),
        T_tag0_vention=np.eye(4),
        T_rtde_base_vention=np.eye(4),
        viz_frame="vention",
        frame_stride=2,
        laser_vention=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
    )
    assert "laser" in calls["clouds"]


def test_teleop_key_ghosts_field_frame():
    tip = np.array([1.0, 2.0, 3.0])
    R = np.eye(3)
    ghosts = teleop_key_ghosts(tip, R, keyboard_step=0.04, linear_scale=0.25)
    step = 0.01
    assert np.allclose(ghosts["w"], tip + [step, 0, 0])
    assert np.allclose(ghosts["s"], tip - [step, 0, 0])
    assert np.allclose(ghosts["d"], tip + [0, step, 0])
    assert np.allclose(ghosts["a"], tip - [0, step, 0])
    assert np.allclose(ghosts["x"], tip + [0, 0, step])
    assert np.allclose(ghosts["y"], tip - [0, 0, step])


def test_teleop_key_ghosts_rotated_field():
    tip = np.zeros(3)
    # Field +y along world +x (right-handed with +z).
    R = np.column_stack([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    ghosts = teleop_key_ghosts(tip, R, keyboard_step=0.04, linear_scale=0.25)
    assert np.allclose(ghosts["d"], [0.01, 0.0, 0.0])


def test_R_tcp_base_to_viz_vention_with_mount():
    R = np.eye(3)
    T_rv = np.eye(4)
    T_rv[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    out = R_tcp_base_to_viz(
        R, T_tag0_vention=np.eye(4), T_rtde_base_vention=T_rv, viz_frame="vention"
    )
    assert np.allclose(out, T_rv[:3, :3])


def test_show_field_debug_teleop_keys(monkeypatch):
    from geodude_teleoperation.viz import field_viz

    calls = {"clouds": [], "show": 0, "interactive": 0}

    class _FakePS:
        _is_fake_for_tests = True

        def init(self):
            pass

        def set_up_dir(self, *_a, **_k):
            pass

        def reset_camera_to_home_view(self):
            pass

        def set_ground_plane_mode(self, *_a, **_k):
            pass

        def register_point_cloud(self, name, pts, **kwargs):
            calls["clouds"].append(name)

            class _Cloud:
                def add_scalar_quantity(self, *_a, **_k):
                    pass

            return _Cloud()

        def register_curve_network(self, name, *_a, **_k):
            calls["clouds"].append(name)
            return object()

        def show(self):
            calls["show"] += 1

        def set_user_callback(self, *_a, **_k):
            pass

        def clear_user_callback(self):
            pass

    monkeypatch.setattr(field_viz, "ps", _FakePS())
    monkeypatch.setattr(
        field_viz, "plot_orientation_field", lambda *a, **k: calls["clouds"].append("field")
    )
    monkeypatch.setattr(field_viz, "set_camera_and_plane", lambda **_k: None)
    monkeypatch.setattr(field_viz, "frame_camera_on_points", lambda *_a, **_k: None)
    monkeypatch.setattr(
        field_viz,
        "query_R_field_at_tcp",
        lambda *a, **k: (np.eye(3), 0.05, False, True),
    )

    def _fake_interactive(**kwargs):
        calls["interactive"] += 1
        assert "tip0" in kwargs

    monkeypatch.setattr(field_viz, "_run_interactive_teleop_keys", _fake_interactive)

    class _PC:
        vertices = np.zeros((10, 3))
        local_bases = np.tile(np.eye(3), (10, 1, 1))

    class _SD:
        source_vertices = [0]
        sink_vertices = [9]
        ut = np.linspace(0, 1, 10)

    field_viz.show_field_debug(
        _PC(),
        _SD(),
        T_tag0_vention=np.eye(4),
        T_rtde_base_vention=np.eye(4),
        viz_frame="vention",
        tcp_pose_rtde=[0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        tip_offset=np.zeros(3),
        show_teleop_keys=True,
        wos=object(),
        keyboard_step=0.04,
    )
    assert calls["interactive"] == 1


def test_point_viz_to_tag0_roundtrip():
    from geodude_teleoperation.viz.field_viz import point_viz_to_tag0, points_in_viz_frame

    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    p_tag = np.array([0.1, 0.2, 0.3])
    p_v = points_in_viz_frame(
        p_tag.reshape(1, 3),
        T_tag0_vention=T,
        T_rtde_base_vention=None,
        viz_frame="vention",
    )[0]
    back = point_viz_to_tag0(
        p_v, T_tag0_vention=T, T_rtde_base_vention=None, viz_frame="vention"
    )
    assert np.allclose(back, p_tag)


def test_contact_frame_overlays_geometry():
    tip = np.array([1.0, 2.0, 3.0])
    R = np.eye(3)
    segs = contact_frame_overlays(tip, R, tangent_len=0.1, approach_len=0.05)
    assert np.allclose(segs["tangent_line"][0], tip - [0.1, 0, 0])
    assert np.allclose(segs["tangent_line"][1], tip + [0.1, 0, 0])
    assert np.allclose(segs["approach_n"][1], tip + [0, 0, 0.05])
    mid = 0.5 * (segs["tangent_line"][0] + segs["tangent_line"][1])
    assert abs(np.dot(mid - tip, R[:, 2])) < 1e-12


def test_live_viz_state_publish_snapshot():
    from geodude_teleoperation.viz.field_viz import LiveVizState

    live = LiveVizState()
    assert live.viz_open is True
    seq0, tcp0, *_ = live.snapshot()
    assert seq0 == 0 and tcp0 is None
    live.publish([0.1, 0.2, 0.3, 0, 0, 0], signed_d=-0.01, in_contact=True)
    seq1, tcp1, _R, d, c = live.snapshot()
    assert seq1 == 1
    assert np.allclose(tcp1[:3], [0.1, 0.2, 0.3])
    assert d == -0.01 and c is True
    live.push_teleop_key("\n")
    live.push_teleop_key("w")
    assert live.drain_teleop_keys() == ["\n", "w"]
    assert live.drain_teleop_keys() == []
    live.mark_viz_closed()
    assert live.viz_open is False


def test_show_field_debug_live_rtde_callback(monkeypatch):
    from geodude_teleoperation.viz import field_viz
    from geodude_teleoperation.viz.field_viz import LiveVizState

    calls = {"show": 0, "clouds": [], "callback": None, "updates": 0}

    class _Updatable:
        def __init__(self, name):
            self.name = name

        def update_point_positions(self, *_a, **_k):
            calls["updates"] += 1

        def update_node_positions(self, *_a, **_k):
            calls["updates"] += 1

        def add_scalar_quantity(self, *_a, **_k):
            pass

    class _FakePS:
        _is_fake_for_tests = True

        def init(self):
            pass

        def set_up_dir(self, *_a, **_k):
            pass

        def reset_camera_to_home_view(self):
            pass

        def set_ground_plane_mode(self, *_a, **_k):
            pass

        def register_point_cloud(self, name, pts, **kwargs):
            calls["clouds"].append(name)
            return _Updatable(name)

        def register_curve_network(self, name, *_a, **_k):
            calls["clouds"].append(name)
            return _Updatable(name)

        def set_user_callback(self, cb):
            calls["callback"] = cb

        def clear_user_callback(self):
            pass

        def show(self):
            calls["show"] += 1
            # Simulate one RTDE publish then a Polyscope frame.
            live.publish([0.05, 0.0, 0.4, 0, 0, 0], R_field_tcp=np.eye(3))
            assert calls["callback"] is not None
            calls["callback"]()

    live = LiveVizState()
    monkeypatch.setattr(field_viz, "ps", _FakePS())
    monkeypatch.setattr(
        field_viz,
        "plot_orientation_field",
        lambda *a, **k: calls["clouds"].append("field"),
    )
    monkeypatch.setattr(field_viz, "set_camera_and_plane", lambda **_k: None)
    monkeypatch.setattr(field_viz, "frame_camera_on_points", lambda *_a, **_k: None)

    class _PC:
        vertices = np.zeros((10, 3))
        local_bases = np.tile(np.eye(3), (10, 1, 1))

    class _SD:
        source_vertices = [0]
        sink_vertices = [9]
        ut = np.linspace(0, 1, 10)

    field_viz.show_field_debug(
        _PC(),
        _SD(),
        T_tag0_vention=np.eye(4),
        T_rtde_base_vention=np.eye(4),
        viz_frame="vention",
        live_viz=live,
        tip_offset=np.zeros(3),
    )
    assert calls["show"] == 1
    assert live.viz_open is False
    assert "tip" in calls["clouds"]
    assert "tcp" in calls["clouds"]
    assert "tcp_x" in calls["clouds"]
    assert calls["callback"] is not None

