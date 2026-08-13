"""Tests for RTDE teleop session and host loop (fake ur_rtde interfaces)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

import geodude_teleoperation.rtde.control_loop as rcl


class FakeControl:
    instances: list["FakeControl"] = []

    def __init__(self, ip: str):
        self.ip = ip
        self.speedj_calls: list[tuple[list[float], float, float]] = []
        self.speed_stop_calls: list[tuple] = []
        self.stopj_calls: list[list[float]] = []
        FakeControl.instances.append(self)

    def speedJ(self, qdot, accel, dt):
        self.speedj_calls.append((list(qdot), float(accel), float(dt)))

    def speedStop(self, *args):
        self.speed_stop_calls.append(args)

    def stopJ(self, acc):
        self.stopj_calls.append(list(acc))

    def disconnect(self):
        pass


class FakeReceive:
    instances: list["FakeReceive"] = []

    def __init__(self, ip: str):
        self.ip = ip
        self._q = [0.0] * 6
        self._tcp = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
        self._fault = False
        FakeReceive.instances.append(self)

    def getActualQ(self):
        return list(self._q)

    def getActualTCPPose(self):
        return list(self._tcp)

    def getRobotMode(self):
        return 7 if not self._fault else 4

    def isProtectiveStopped(self):
        return False

    def isEmergencyStopped(self):
        return False

    def disconnect(self):
        pass


@pytest.fixture
def fake_ur_rtde(monkeypatch):
    FakeControl.instances.clear()
    FakeReceive.instances.clear()
    control_mod = types.ModuleType("rtde_control")
    control_mod.RTDEControlInterface = FakeControl
    receive_mod = types.ModuleType("rtde_receive")
    receive_mod.RTDEReceiveInterface = FakeReceive
    monkeypatch.setitem(sys.modules, "rtde_control", control_mod)
    monkeypatch.setitem(sys.modules, "rtde_receive", receive_mod)
    return FakeControl, FakeReceive


def test_dry_run_opens_receive_only(fake_ur_rtde):
    FakeControl, FakeReceive = fake_ur_rtde
    session = rcl.open("192.168.1.20", dry_run=True)
    try:
        assert len(FakeReceive.instances) == 1
        assert len(FakeControl.instances) == 0
        assert session.control_open is False
        state = session.read_state()
        assert "q" in state and "tcp_pose" in state and "fault" in state
        with pytest.raises(RuntimeError, match="dry-run"):
            session.speedj([0.1] * 6, 0.5)
    finally:
        session.close()


def test_live_opens_control_and_receive(fake_ur_rtde):
    FakeControl, FakeReceive = fake_ur_rtde
    session = rcl.open("192.168.1.20", dry_run=False)
    try:
        assert len(FakeReceive.instances) == 1
        assert len(FakeControl.instances) == 1
        assert session.control_open is True
        session.speedj(np.zeros(6), 0.5)
        assert FakeControl.instances[0].speedj_calls
    finally:
        session.close()


def test_close_dry_run_no_speedj(fake_ur_rtde):
    FakeControl, FakeReceive = fake_ur_rtde
    session = rcl.open("192.168.1.20", dry_run=True)
    session.close()
    assert len(FakeControl.instances) == 0


def test_close_live_zeros_then_disconnects(fake_ur_rtde):
    FakeControl, FakeReceive = fake_ur_rtde
    session = rcl.open("192.168.1.20", dry_run=False)
    session.close()
    ctrl = FakeControl.instances[0]
    assert ctrl.speed_stop_calls or ctrl.stopj_calls or ctrl.speedj_calls
    if ctrl.speedj_calls:
        assert np.allclose(ctrl.speedj_calls[-1][0], [0.0] * 6)


def test_stop_motion_calls_speedStop(fake_ur_rtde):
    FakeControl, FakeReceive = fake_ur_rtde
    session = rcl.open("192.168.1.20", dry_run=False)
    session.speedj([0.2] * 6, 0.5)
    session.stop_motion()
    ctrl = FakeControl.instances[0]
    assert ctrl.speed_stop_calls, "Ctrl+C must speedStop, not rely on last speedJ"
    session.close()


def test_zero_speed_stops_without_latching_shutdown(fake_ur_rtde):
    FakeControl, FakeReceive = fake_ur_rtde
    session = rcl.open("192.168.1.20", dry_run=False)
    session.speedj([0.2] * 6, 0.5)
    session.zero_speed()
    ctrl = FakeControl.instances[0]
    assert ctrl.speed_stop_calls
    n = len(ctrl.speedj_calls)
    session.speedj([0.3] * 6, 0.5)
    assert len(ctrl.speedj_calls) == n + 1
    session.close()


def test_request_rtde_shutdown_stops_active_session(fake_ur_rtde):
    FakeControl, FakeReceive = fake_ur_rtde
    session = rcl.open("192.168.1.20", dry_run=False)
    session.speedj([0.3] * 6, 0.5)
    rcl.request_rtde_shutdown()
    ctrl = FakeControl.instances[0]
    assert ctrl.speed_stop_calls
    n_before = len(ctrl.speedj_calls)
    session.speedj([0.4] * 6, 0.5)
    assert len(ctrl.speedj_calls) == n_before
    session.close()


def test_rtde_loop_should_continue_stops_on_shutdown_or_viz_close():
    assert rcl.rtde_loop_should_continue(
        keyboard_running=True, shutdown=False, viz_open=None
    )
    assert not rcl.rtde_loop_should_continue(
        keyboard_running=True, shutdown=True, viz_open=True
    )
    assert not rcl.rtde_loop_should_continue(
        keyboard_running=True, shutdown=False, viz_open=False
    )
    assert not rcl.rtde_loop_should_continue(
        keyboard_running=False, shutdown=False, viz_open=True
    )


def test_watchdog_zeros(fake_ur_rtde, monkeypatch):
    FakeControl, FakeReceive = fake_ur_rtde
    session = rcl.open("192.168.1.20", dry_run=False)

    times = iter([0.0, 0.0, 0.6])
    monkeypatch.setattr(rcl.time, "monotonic", lambda: next(times, 0.6))

    assert rcl.watchdog_is_stale(0.0, 0.6, watchdog_s=0.5) is True
    assert rcl.watchdog_is_stale(0.55, 0.6, watchdog_s=0.5) is False

    qdot = rcl.apply_watchdog(np.ones(6), last_command_time=0.0, now=0.6, watchdog_s=0.5)
    assert np.allclose(qdot, 0.0)

    session.close()


def test_should_run_vf_query_skips_during_orient_lock():
    assert (
        rcl.should_run_vf_query(
            dry_run=False,
            sim_active=False,
            now=1.0,
            last_query_t=0.0,
            dry_period_s=0.5,
            orient_locking=True,
            has_lock_R_star=True,
        )
        is False
    )


def test_should_run_vf_query_until_R_star_captured():
    assert (
        rcl.should_run_vf_query(
            dry_run=False,
            sim_active=False,
            now=1.0,
            last_query_t=0.0,
            dry_period_s=0.5,
            orient_locking=True,
            has_lock_R_star=False,
        )
        is True
    )


def test_should_run_vf_query_live_idle_still_queries():
    assert (
        rcl.should_run_vf_query(
            dry_run=False,
            sim_active=False,
            now=1.0,
            last_query_t=0.0,
            dry_period_s=0.5,
            orient_locking=False,
            has_lock_R_star=False,
        )
        is True
    )


def test_should_run_vf_query_skips_while_awaiting_approve():
    assert (
        rcl.should_run_vf_query(
            dry_run=False,
            sim_active=False,
            now=1.0,
            last_query_t=0.0,
            dry_period_s=0.5,
            orient_locking=False,
            has_lock_R_star=True,
            awaiting_approve=True,
        )
        is False
    )


def test_should_run_vf_query_skips_when_vf_frozen():
    assert (
        rcl.should_run_vf_query(
            dry_run=False,
            sim_active=False,
            now=1.0,
            last_query_t=0.0,
            dry_period_s=0.5,
            orient_locking=False,
            has_lock_R_star=True,
            vf_frozen=True,
        )
        is False
    )


def test_should_run_vf_query_after_approve_unfrozen():
    assert (
        rcl.should_run_vf_query(
            dry_run=False,
            sim_active=False,
            now=1.0,
            last_query_t=0.0,
            dry_period_s=0.5,
            orient_locking=False,
            has_lock_R_star=True,
            awaiting_approve=False,
            vf_frozen=False,
        )
        is True
    )


def test_watchdog_not_refreshed_by_zero_speedj_ticks():
    """Gated speedJ(0) ticks must not reset the watchdog clock."""
    last_command_time = 0.0
    watchdog_s = 0.5
    fresh_command = False

    for now in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        send_qdot = np.zeros(6)
        send_qdot = rcl.apply_watchdog(
            send_qdot,
            last_command_time=last_command_time,
            now=now,
            watchdog_s=watchdog_s,
        )
        assert np.allclose(send_qdot, 0.0)
        if rcl.should_refresh_watchdog(
            commands_allowed=False,
            dry_run=False,
            kin_ok=True,
            control_open=True,
            fresh_command=fresh_command,
        ):
            last_command_time = now
        fresh_command = False

    assert rcl.watchdog_is_stale(last_command_time, 0.6, watchdog_s=watchdog_s)
    qdot = rcl.apply_watchdog(
        np.ones(6),
        last_command_time=last_command_time,
        now=0.6,
        watchdog_s=watchdog_s,
    )
    assert np.allclose(qdot, 0.0)


def test_watchdog_refreshed_by_active_control_without_keypress():
    """Pose servo between WASD keypresses must keep the watchdog alive."""
    last_command_time = 0.0
    watchdog_s = 0.5
    for now in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        if rcl.should_refresh_watchdog(
            commands_allowed=True,
            dry_run=False,
            kin_ok=True,
            control_open=True,
            fresh_command=False,
        ):
            last_command_time = now
        qdot = rcl.apply_watchdog(
            np.ones(6),
            last_command_time=last_command_time,
            now=now,
            watchdog_s=watchdog_s,
        )
        assert np.allclose(qdot, np.ones(6))
    assert not rcl.watchdog_is_stale(last_command_time, 0.6, watchdog_s=watchdog_s)


def test_watchdog_refreshed_in_dry_run_sim():
    """Dry-run Polyscope sim must keep the watchdog alive between keypresses."""
    assert rcl.should_refresh_watchdog(
        commands_allowed=True,
        dry_run=True,
        kin_ok=True,
        control_open=False,
        fresh_command=False,
    )


def test_flange_tcp_offset_roundtrip():
    from geodude_teleoperation.rtde.ur5e_kinematics import forward_kinematics
    from scipy.spatial.transform import Rotation

    q = np.array([0.1, -1.0, 1.2, -1.4, -1.57, 0.2])
    T_f, _ = forward_kinematics(q)
    tcp = np.concatenate(
        [T_f[:3, 3] + np.array([0.0, 0.0, 0.05]), Rotation.from_matrix(T_f[:3, :3]).as_rotvec()]
    )
    off = rcl.flange_tcp_offset_matrix(q, tcp)
    tcp2 = rcl.tcp_pose_from_q_and_offset(q, off)
    assert np.allclose(tcp, tcp2, atol=1e-9)


def test_resolve_T_fixed_identity():
    args = MagicMock()
    args.base_frame = None
    args.tag0_frame = None
    args.T_tag0_base = None
    args.no_rtde_base_vention = True
    args.T_rtde_base_vention = None
    T, tf_ok, tf_age_ms = rcl.resolve_T_tag0_base(args)
    assert np.allclose(T, np.eye(4))
    assert tf_ok is True
    assert tf_age_ms == 0.0


def test_resolve_T_live_tf_unavailable():
    rcl._live_tf_fallback_warned = False
    rcl._ros_tf_unavailable_warned = False
    args = MagicMock()
    args.base_frame = "base"
    args.tag0_frame = "tag_0"
    args.T_tag0_base = None
    args.no_rtde_base_vention = True
    args.T_rtde_base_vention = None
    T, tf_ok, tf_age_ms = rcl.resolve_T_tag0_base(args)
    assert np.allclose(T, np.eye(4))
    assert tf_ok is False
    assert np.isnan(tf_age_ms)


def test_resolve_T_live_tf_fallback_explicit_file(tmp_path):
    rcl._live_tf_fallback_warned = False
    rcl._ros_tf_unavailable_warned = False
    fixed = np.eye(4)
    fixed[:3, 3] = [0.1, 0.2, 0.3]
    path = tmp_path / "T_tag0_base.npy"
    np.save(path, fixed)

    args = MagicMock()
    args.base_frame = "base"
    args.tag0_frame = "tag_0"
    args.T_tag0_base = path
    args.no_rtde_base_vention = True
    args.T_rtde_base_vention = None
    T, tf_ok, tf_age_ms = rcl.resolve_T_tag0_base(args)
    assert np.allclose(T, fixed)
    assert tf_ok is True
    assert np.isnan(tf_age_ms)


def test_resolve_T_composes_rtde_mount(tmp_path):
    T_tag = np.eye(4)
    T_tag[:3, 3] = [1.0, 0.0, 0.0]
    T_rv = np.eye(4)
    T_rv[:3, 3] = [0.3, 0.0, 0.0]
    tag_path = tmp_path / "T_tag0.npy"
    rv_path = tmp_path / "T_rv.npy"
    np.save(tag_path, T_tag)
    np.save(rv_path, T_rv)

    args = MagicMock()
    args.base_frame = None
    args.tag0_frame = None
    args.T_tag0_base = tag_path
    args.T_rtde_base_vention = rv_path
    args.no_rtde_base_vention = False
    T, tf_ok, _ = rcl.resolve_T_tag0_base(args)
    assert tf_ok is True
    # p_rtde = inv(T_rv) @ T_tag @ 0 = [0.7,0,0] → translation of composed T
    assert np.allclose(T[:3, 3], [0.7, 0.0, 0.0])


def test_ee_motor_from_tcp_pose():
    motor = rcl.ee_motor_from_tcp_pose([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    t = motor.get_translator()
    assert t.x() == pytest.approx(0.1)
    assert t.y() == pytest.approx(0.2)
    assert t.z() == pytest.approx(0.3)


def test_should_run_dry_vf_query_ignores_spam():
    assert rcl.should_run_dry_vf_query(0.0, float("-inf"), 1.0) is True
    assert rcl.should_run_dry_vf_query(0.2, 0.0, 1.0) is False
    assert rcl.should_run_dry_vf_query(1.0, 0.0, 1.0) is True


def test_resolve_live_kinematics_falls_back_to_numpy(monkeypatch):
    monkeypatch.setattr(
        "geodude_teleoperation.gafropy_compat.gafropy_importable", lambda: False
    )
    monkeypatch.setattr(
        rcl._GafropyKinematicsBridge, "try_create", classmethod(lambda cls: None)
    )
    kin = rcl.resolve_live_kinematics()
    assert kin is not None
    assert rcl._is_numpy_kinematics(kin)


def test_make_twist_without_gafropy(monkeypatch):
    from geodude_teleoperation.teleop.keyboard import make_twist
    from geodude_teleoperation.se3 import NumpyTwist
    from geodude_teleoperation import gafropy_compat

    gafropy_compat.clear_gafropy_importable_cache()
    monkeypatch.setattr(gafropy_compat, "gafropy_importable", lambda: False)
    tw = make_twist([0.0, 0.0, 0.0, 0.01, 0.0, 0.0])
    assert isinstance(tw, NumpyTwist)
    assert np.allclose(tw.get_linear(), [0.01, 0.0, 0.0])


def test_numpy_live_loop_can_speedj(fake_ur_rtde, monkeypatch):
    """With gafropy unavailable, numpy backup issues non-zero speedJ after enable."""
    from geodude_teleoperation.teleop.controller import TeleoperationController
    from geodude_teleoperation.se3 import NumpyTwist
    from geodude_teleoperation.teleop.teleop_state import TeleopState
    from geodude_teleoperation.rtde.ur5e_kinematics import forward_kinematics
    from scipy.spatial.transform import Rotation

    FakeControl, FakeReceive = fake_ur_rtde
    monkeypatch.setattr(
        rcl._GafropyKinematicsBridge, "try_create", classmethod(lambda cls: None)
    )

    q = np.array([0.0, -1.2, 1.4, -1.6, -1.57, 0.0])
    T_f, _ = forward_kinematics(q)
    tcp = np.concatenate(
        [T_f[:3, 3], Rotation.from_matrix(T_f[:3, :3]).as_rotvec()]
    )

    class _KB:
        def __init__(self):
            self._running = __import__("threading").Event()
            self._running.set()
            self._n = 0

        def start(self, on_twist, on_button=None, on_enable=None, enable_key="\n"):
            self._on_twist = on_twist
            self._on_enable = on_enable
            self._running.set()

        def stop(self):
            self._running.clear()

        def pump(self):
            self._n += 1
            if self._n == 1 and self._on_enable is not None:
                self._on_enable()
            elif self._n == 2 and self._on_twist is not None:
                self._on_twist(
                    NumpyTwist.from_array([0.0, 0.0, 0.0, 0.0, 0.0, 0.02])
                )
            elif self._n >= 6:
                self._running.clear()

    kb = _KB()
    # Drive the fake keyboard from the loop by patching sleep to pump keys.
    real_sleep = rcl.time.sleep

    def _sleep(_dt):
        kb.pump()
        if not kb._running.is_set():
            return
        real_sleep(0.0)

    monkeypatch.setattr(rcl.time, "sleep", _sleep)

    recv_holder = {}

    class _Recv(FakeReceive):
        def __init__(self, ip: str):
            super().__init__(ip)
            self._q = list(q)
            self._tcp = list(tcp)
            recv_holder["r"] = self

    monkeypatch.setitem(
        sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=_Recv)
    )
    monkeypatch.setitem(
        sys.modules,
        "rtde_control",
        types.SimpleNamespace(RTDEControlInterface=FakeControl),
    )

    class _VF:
        def update_from_state(self, ee, T=None):
            pass

        def project(self, motor):
            return motor

        def get_distance(self, position):
            return 0.05

        @property
        def status(self):
            return {
                "signed_d": 0.05,
                "in_contact": False,
                "tip_nn": 0.05,
                "used_wos": True,
                "query_ms": 0.0,
                "R_field_base": np.eye(3),
            }

    args = MagicMock()
    args.dry_run = False
    args.robot_ip = "192.168.1.20"
    args.control_hz = 50.0
    args.status_hz = 0.0
    args.log_motion = None
    args.plot_motion = False
    args.record_mcap = False
    args.rtde_accel = 1.0
    args.watchdog_ms = 500.0
    args.max_joint_velocity = 1.0
    args.keyboard_step = 0.01
    args.tip_offset = (0.0, 0.0, 0.0)
    args.base_frame = None
    args.tag0_frame = None
    args.T_tag0_base = None
    args.calib_meta = None
    args.enable_key = "\n"
    args.no_rtde_base_vention = True
    args.T_rtde_base_vention = None
    args.diag_hold_zero = False
    args.line_tracking = False
    args.tactile = False
    args.in_contact = False
    args.verify_dtc_500hz = False
    args.T_tag0_base = None
    args.base_frame = None
    args.tag0_frame = None

    controller = TeleoperationController(max_joint_velocity=1.0)
    state = TeleopState()
    # enable() sets motion_enabled; teleop_active defaults True after enable
    rcl.run_rtde_loop(
        args,
        (None, None),
        _VF(),
        controller,
        state,
        keyboard=kb,
    )

    assert FakeControl.instances
    # At least one non-zero speedJ after enable + twist
    non_zero = [
        c for c in FakeControl.instances[0].speedj_calls if np.linalg.norm(c[0]) > 1e-6
    ]
    assert non_zero, FakeControl.instances[0].speedj_calls


def test_enable_alone_idle_speedj_near_zero(fake_ur_rtde, monkeypatch):
    """Enter with live VF must not chase R_field without a keyboard twist."""
    import sys
    import types

    from geodude_teleoperation.teleop.controller import TeleoperationController
    from geodude_teleoperation.se3 import _DuckMotor
    from geodude_teleoperation.teleop.teleop_state import TeleopState
    from geodude_teleoperation.rtde.ur5e_kinematics import forward_kinematics
    from scipy.spatial.transform import Rotation

    FakeControl, FakeReceive = fake_ur_rtde
    monkeypatch.setattr(
        rcl._GafropyKinematicsBridge, "try_create", classmethod(lambda cls: None)
    )

    q = np.array([0.0, -1.2, 1.4, -1.6, -1.57, 0.0])
    T_f, _ = forward_kinematics(q)
    tcp = np.concatenate(
        [T_f[:3, 3], Rotation.from_matrix(T_f[:3, :3]).as_rotvec()]
    )
    R_field = Rotation.from_euler("xyz", [30, -40, 20], degrees=True).as_matrix()

    class _KB:
        def __init__(self):
            self._running = __import__("threading").Event()
            self._running.set()
            self._n = 0

        def start(self, on_twist, on_button=None, on_enable=None, enable_key="\n"):
            self._on_enable = on_enable
            self._running.set()

        def stop(self):
            self._running.clear()

        def pump(self):
            self._n += 1
            if self._n == 1 and self._on_enable is not None:
                self._on_enable()
            elif self._n >= 12:
                self._running.clear()

    kb = _KB()
    real_sleep = rcl.time.sleep

    def _sleep(_dt):
        kb.pump()
        if kb._running.is_set():
            real_sleep(0.0)

    monkeypatch.setattr(rcl.time, "sleep", _sleep)

    class _Recv(FakeReceive):
        def __init__(self, ip: str):
            super().__init__(ip)
            self._q = list(q)
            self._tcp = list(tcp)

    monkeypatch.setitem(
        sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=_Recv)
    )
    monkeypatch.setitem(
        sys.modules,
        "rtde_control",
        types.SimpleNamespace(RTDEControlInterface=FakeControl),
    )

    class _VF:
        def update_from_state(self, ee, T=None):
            pass

        def project(self, motor):
            return _DuckMotor(motor.get_translator(), R_field)

        def motion_basis(self):
            return R_field.copy()

        def get_distance(self, position):
            return 0.05

        @property
        def status(self):
            return {
                "signed_d": 0.05,
                "in_contact": False,
                "tip_nn": 0.05,
                "used_wos": True,
                "query_ms": 0.0,
                "R_field_base": R_field,
                "R_motion_base": R_field,
            }

    args = MagicMock()
    args.dry_run = False
    args.robot_ip = "192.168.1.20"
    args.control_hz = 50.0
    args.status_hz = 0.0
    args.log_motion = None
    args.plot_motion = False
    args.record_mcap = False
    args.rtde_accel = 1.0
    args.watchdog_ms = 500.0
    args.max_joint_velocity = 1.0
    args.keyboard_step = 0.01
    args.tip_offset = (0.0, 0.0, 0.0)
    args.base_frame = None
    args.tag0_frame = None
    args.T_tag0_base = None
    args.calib_meta = None
    args.enable_key = "\n"
    args.no_rtde_base_vention = True
    args.T_rtde_base_vention = None
    args.diag_hold_zero = False
    args.line_tracking = False
    args.tactile = False
    args.in_contact = False
    args.object_offset_t = None
    args.no_object_offset = True
    args.viz = False
    args.orient_lock_on_enable = False

    rcl.run_rtde_loop(
        args,
        (None, None),
        _VF(),
        TeleoperationController(max_joint_velocity=1.0),
        TeleopState(),
        keyboard=kb,
    )

    assert FakeControl.instances
    after_enable = FakeControl.instances[0].speedj_calls
    assert after_enable
    norms = [float(np.linalg.norm(c[0])) for c in after_enable]
    assert max(norms) < 1e-3, norms


def test_enable_locks_orientation_with_startup_limits(fake_ur_rtde, monkeypatch):
    """Live Enter snaps T_ref to R_field using startup vel/accel, not general."""
    import sys
    import types

    from geodude_teleoperation.teleop.controller import TeleoperationController
    from geodude_teleoperation.se3 import _DuckMotor
    from geodude_teleoperation.teleop.teleop_state import TeleopState
    from geodude_teleoperation.rtde.ur5e_kinematics import forward_kinematics
    from scipy.spatial.transform import Rotation

    FakeControl, FakeReceive = fake_ur_rtde
    monkeypatch.setattr(
        rcl._GafropyKinematicsBridge, "try_create", classmethod(lambda cls: None)
    )

    q = np.array([0.0, -1.2, 1.4, -1.6, -1.57, 0.0])
    T_f, _ = forward_kinematics(q)
    tcp = np.concatenate(
        [T_f[:3, 3], Rotation.from_matrix(T_f[:3, :3]).as_rotvec()]
    )
    R_field = Rotation.from_euler("xyz", [30, -40, 20], degrees=True).as_matrix()

    class _KB:
        def __init__(self):
            self._running = __import__("threading").Event()
            self._running.set()
            self._n = 0

        def start(self, on_twist, on_button=None, on_enable=None, enable_key="\n"):
            self._on_enable = on_enable
            self._running.set()

        def stop(self):
            self._running.clear()

        def pump(self):
            self._n += 1
            if self._n == 1 and self._on_enable is not None:
                self._on_enable()
            elif self._n >= 12:
                self._running.clear()

    kb = _KB()
    real_sleep = rcl.time.sleep

    class _Recv(FakeReceive):
        def __init__(self, ip: str):
            super().__init__(ip)
            self._q = list(q)
            self._tcp = list(tcp)

    monkeypatch.setitem(
        sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=_Recv)
    )
    monkeypatch.setitem(
        sys.modules,
        "rtde_control",
        types.SimpleNamespace(RTDEControlInterface=FakeControl),
    )

    class _VF:
        def __init__(self):
            self.n_update = 0
            self.n_update_after_enable = 0
            self._enabled = False

        def mark_enabled(self):
            self._enabled = True

        def update_from_state(self, ee, T=None):
            self.n_update += 1
            if self._enabled:
                self.n_update_after_enable += 1

        def project(self, motor):
            return _DuckMotor(motor.get_translator(), R_field)

        def motion_basis(self):
            return R_field.copy()

        def get_distance(self, position):
            return 0.05

        @property
        def status(self):
            return {
                "signed_d": 0.05,
                "in_contact": False,
                "tip_nn": 0.05,
                "used_wos": True,
                "query_ms": 0.0,
                "R_field_base": R_field,
                "R_motion_base": R_field,
            }

    vf = _VF()

    def _sleep(_dt):
        kb.pump()
        if kb._n >= 1:
            vf.mark_enabled()
        if kb._running.is_set():
            real_sleep(0.0)

    monkeypatch.setattr(rcl.time, "sleep", _sleep)

    args = MagicMock()
    args.dry_run = False
    args.robot_ip = "192.168.1.20"
    args.control_hz = 50.0
    args.status_hz = 0.0
    args.log_motion = None
    args.plot_motion = False
    args.record_mcap = False
    args.rtde_accel = 1.0
    args.watchdog_ms = 500.0
    args.max_joint_velocity = 1.0
    args.max_linear_speed = 0.05
    args.max_angular_speed = 0.4
    args.pose_gain = 4.0
    args.keyboard_step = 0.01
    args.tip_offset = (0.0, 0.0, 0.0)
    args.base_frame = None
    args.tag0_frame = None
    args.T_tag0_base = None
    args.calib_meta = None
    args.enable_key = "\n"
    args.no_rtde_base_vention = True
    args.T_rtde_base_vention = None
    args.diag_hold_zero = False
    args.line_tracking = False
    args.tactile = False
    args.in_contact = False
    args.object_offset_t = None
    args.no_object_offset = True
    args.viz = False
    args.orient_lock_on_enable = True
    args.startup_max_joint_velocity = 0.2
    args.startup_rtde_accel = 0.25
    args.startup_max_angular_speed = 0.1
    args.startup_ramp_s = 30.0
    args.orient_lock_tol_deg = 2.0

    rcl.run_rtde_loop(
        args,
        (None, None),
        vf,
        TeleoperationController(max_joint_velocity=1.0, gain=4.0),
        TeleopState(),
        keyboard=kb,
    )

    assert FakeControl.instances
    calls = FakeControl.instances[0].speedj_calls
    assert calls
    nonzero = [c for c in calls if float(np.linalg.norm(c[0])) > 1e-6]
    assert nonzero, calls
    norms = [float(np.linalg.norm(c[0])) for c in nonzero]
    accels = [float(c[1]) for c in nonzero]
    assert max(norms) > 1e-3, norms
    assert max(norms) <= 0.2 + 1e-9, norms
    assert max(accels) < 0.35, accels
    assert min(accels) >= 0.24, accels
    # R* already available from motion_basis — no WoS during lock.
    assert vf.n_update_after_enable == 0, vf.n_update_after_enable


def test_second_enter_approves_teleop_f_unlocks_vf(fake_ur_rtde, monkeypatch):
    """Enter #2 approves (VF frozen). f holds pose; f again unlocks VF queries."""
    import sys
    import types

    from geodude_teleoperation.teleop.controller import TeleoperationController
    from geodude_teleoperation.se3 import _DuckMotor
    from geodude_teleoperation.teleop.teleop_state import TeleopState
    from geodude_teleoperation.rtde.ur5e_kinematics import forward_kinematics
    from scipy.spatial.transform import Rotation

    FakeControl, FakeReceive = fake_ur_rtde
    monkeypatch.setattr(
        rcl._GafropyKinematicsBridge, "try_create", classmethod(lambda cls: None)
    )

    q = np.array([0.0, -1.2, 1.4, -1.6, -1.57, 0.0])
    T_f, _ = forward_kinematics(q)
    tcp = np.concatenate(
        [T_f[:3, 3], Rotation.from_matrix(T_f[:3, :3]).as_rotvec()]
    )
    R_field = T_f[:3, :3].copy()  # already aligned → settle immediately

    class _KB:
        def __init__(self):
            self._running = __import__("threading").Event()
            self._running.set()
            self._n = 0
            self._on_enable = None
            self._on_button = None

        def start(self, on_twist, on_button=None, on_enable=None, enable_key="\n"):
            self._on_enable = on_enable
            self._on_button = on_button
            self._running.set()

        def stop(self):
            self._running.clear()

        def pump(self):
            self._n += 1
            if self._n == 1 and self._on_enable is not None:
                self._on_enable()
            elif self._n == 6 and self._on_enable is not None:
                self._on_enable()
            elif self._n == 10 and self._on_button is not None:
                self._on_button(2)  # f → pose hold ON, VF frozen
            elif self._n == 12 and self._on_button is not None:
                self._on_button(2)  # f → pose hold OFF, VF unlocked
            elif self._n >= 16:
                self._running.clear()

    kb = _KB()
    real_sleep = rcl.time.sleep
    query_log = []

    class _Recv(FakeReceive):
        def __init__(self, ip: str):
            super().__init__(ip)
            self._q = list(q)
            self._tcp = list(tcp)

    monkeypatch.setitem(
        sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=_Recv)
    )
    monkeypatch.setitem(
        sys.modules,
        "rtde_control",
        types.SimpleNamespace(RTDEControlInterface=FakeControl),
    )

    class _VF:
        def __init__(self):
            self.n_update = 0
            self._frozen = False

        def set_frozen(self, frozen: bool) -> None:
            self._frozen = bool(frozen)

        def update_from_state(self, ee, T=None):
            self.n_update += 1
            query_log.append(kb._n)

        def project(self, motor):
            return _DuckMotor(motor.get_translator(), R_field)

        def motion_basis(self):
            return R_field.copy()

        def get_distance(self, position):
            return 0.05

        @property
        def status(self):
            return {
                "signed_d": 0.05,
                "in_contact": False,
                "tip_nn": 0.05,
                "used_wos": True,
                "query_ms": 0.0,
                "R_field_base": R_field,
                "R_motion_base": R_field,
            }

    vf = _VF()

    def _sleep(_dt):
        kb.pump()
        if kb._running.is_set():
            real_sleep(0.0)

    monkeypatch.setattr(rcl.time, "sleep", _sleep)

    args = MagicMock()
    args.dry_run = False
    args.robot_ip = "192.168.1.20"
    args.control_hz = 50.0
    args.status_hz = 0.0
    args.log_motion = None
    args.plot_motion = False
    args.record_mcap = False
    args.rtde_accel = 1.0
    args.watchdog_ms = 500.0
    args.max_joint_velocity = 1.0
    args.max_linear_speed = 0.05
    args.max_angular_speed = 0.4
    args.pose_gain = 4.0
    args.keyboard_step = 0.01
    args.tip_offset = (0.0, 0.0, 0.0)
    args.base_frame = None
    args.tag0_frame = None
    args.T_tag0_base = None
    args.calib_meta = None
    args.enable_key = "\n"
    args.no_rtde_base_vention = True
    args.T_rtde_base_vention = None
    args.diag_hold_zero = False
    args.line_tracking = False
    args.tactile = False
    args.in_contact = False
    args.object_offset_t = None
    args.no_object_offset = True
    args.viz = False
    args.orient_lock_on_enable = True
    args.startup_max_joint_velocity = 0.2
    args.startup_rtde_accel = 0.25
    args.startup_max_angular_speed = 0.1
    args.startup_ramp_s = 0.0
    args.orient_lock_tol_deg = 2.0

    rcl.run_rtde_loop(
        args,
        (None, None),
        vf,
        TeleoperationController(max_joint_velocity=1.0, gain=4.0),
        TeleopState(),
        keyboard=kb,
    )

    # Lock / approve / first f: no WoS. Second f unlocks fields.
    queries_while_frozen = [n for n in query_log if 1 <= n < 12]
    assert queries_while_frozen == [], query_log
    queries_after_unlock = [n for n in query_log if n >= 12]
    assert queries_after_unlock, query_log
    assert vf._frozen is False
    ctrl = FakeControl.instances[0]
    assert ctrl.speed_stop_calls, "f / approve must speedStop so last speedJ does not coast"


def test_dry_run_sim_moves_viz_tcp_without_speedj(fake_ur_rtde, monkeypatch):
    """--dry-run after Enter: integrate sim TCP to live_viz; never open control."""
    from geodude_teleoperation.teleop.controller import TeleoperationController
    from geodude_teleoperation.se3 import NumpyTwist
    from geodude_teleoperation.teleop.teleop_state import TeleopState
    from geodude_teleoperation.viz.live_state import LiveVizState
    from geodude_teleoperation.rtde.ur5e_kinematics import forward_kinematics
    from scipy.spatial.transform import Rotation

    FakeControl, FakeReceive = fake_ur_rtde
    monkeypatch.setattr(
        rcl._GafropyKinematicsBridge, "try_create", classmethod(lambda cls: None)
    )

    q = np.array([0.0, -1.2, 1.4, -1.6, -1.57, 0.0])
    T_f, _ = forward_kinematics(q)
    tcp0 = np.concatenate(
        [T_f[:3, 3], Rotation.from_matrix(T_f[:3, :3]).as_rotvec()]
    )
    R_field = Rotation.from_euler("xyz", [20, -30, 10], degrees=True).as_matrix()

    class _KB:
        def __init__(self):
            self._running = __import__("threading").Event()
            self._running.set()
            self._n = 0

        def start(self, on_twist, on_button=None, on_enable=None, enable_key="\n"):
            self._on_twist = on_twist
            self._on_enable = on_enable
            self._running.set()

        def stop(self):
            self._running.clear()

        def pump(self):
            self._n += 1
            if self._n == 1 and self._on_enable is not None:
                self._on_enable()
            elif self._n == 2:
                live_viz.set_held_teleop_keys({"w"})
            elif self._n >= 15:
                live_viz.set_held_teleop_keys(set())
                self._running.clear()

    kb = _KB()
    real_sleep = rcl.time.sleep

    def _sleep(_dt):
        kb.pump()
        if kb._running.is_set():
            real_sleep(0.0)

    monkeypatch.setattr(rcl.time, "sleep", _sleep)

    class _Recv(FakeReceive):
        def __init__(self, ip: str):
            super().__init__(ip)
            self._q = list(q)
            self._tcp = list(tcp0)

    monkeypatch.setitem(
        sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=_Recv)
    )
    monkeypatch.setitem(
        sys.modules,
        "rtde_control",
        types.SimpleNamespace(RTDEControlInterface=FakeControl),
    )

    class _VF:
        def update_from_state(self, ee, T=None):
            pass

        def project(self, motor):
            return motor

        def motion_basis(self):
            return R_field.copy()

        def get_distance(self, position):
            return 0.05

        @property
        def status(self):
            return {
                "signed_d": 0.05,
                "in_contact": False,
                "tip_nn": 0.05,
                "used_wos": True,
                "query_ms": 0.0,
                "R_field_base": R_field,
                "R_motion_base": R_field,
                "R_field_raw_smoothed": R_field,
            }

    live_viz = LiveVizState()
    args = MagicMock()
    args.dry_run = True
    args.robot_ip = "192.168.1.20"
    args.control_hz = 50.0
    args.status_hz = 0.0
    args.log_motion = None
    args.plot_motion = False
    args.record_mcap = False
    args.rtde_accel = 1.0
    args.watchdog_ms = 500.0
    args.max_joint_velocity = 1.0
    args.keyboard_step = 0.01
    args.pose_gain = 4.0
    args.max_linear_speed = 0.05
    args.max_angular_speed = 0.4
    args.tip_offset = (0.0, 0.0, 0.0)
    args.base_frame = None
    args.tag0_frame = None
    args.T_tag0_base = None
    args.calib_meta = None
    args.enable_key = "\n"
    args.no_rtde_base_vention = True
    args.T_rtde_base_vention = None
    args.diag_hold_zero = False
    args.line_tracking = False
    args.tactile = False
    args.in_contact = False
    args.object_offset_t = None
    args.no_object_offset = True
    args.viz = False
    args.verify_dtc_500hz = False

    state = TeleopState()
    assert state.vf_enabled is True
    rcl.run_rtde_loop(
        args,
        (None, None),
        _VF(),
        TeleoperationController(max_joint_velocity=1.0),
        state,
        keyboard=kb,
        live_viz=live_viz,
    )

    assert len(FakeControl.instances) == 0
    _seq, tcp_pub, R_pub, _d, _c = live_viz.snapshot()
    assert tcp_pub is not None
    delta = float(np.linalg.norm(np.asarray(tcp_pub[:3]) - np.asarray(tcp0[:3])))
    assert delta > 1e-3, f"sim tip did not move (delta={delta})"
    assert state.vf_enabled is True
    # EE orientation should track R_field in dry-run.
    assert R_pub is not None
    R_tcp = Rotation.from_rotvec(tcp_pub[3:6]).as_matrix()
    assert np.allclose(R_tcp, R_field, atol=1e-5), (R_tcp, R_field)


def test_snap_T_ref_orientation_to_vf():
    class _Kin:
        def __init__(self):
            self._T_ref = np.eye(4)
            self._T_ref[:3, 3] = [0.1, 0.2, 0.3]

    class _VF:
        def motion_basis(self):
            return np.diag([1.0, -1.0, -1.0])

    kin = _Kin()
    assert rcl.snap_T_ref_orientation_to_vf(kin, _VF())
    assert np.allclose(kin._T_ref[:3, :3], np.diag([1.0, -1.0, -1.0]))
    assert np.allclose(kin._T_ref[:3, 3], [0.1, 0.2, 0.3])
