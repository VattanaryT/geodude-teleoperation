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
        FakeControl.instances.append(self)

    def speedJ(self, qdot, accel, dt):
        self.speedj_calls.append((list(qdot), float(accel), float(dt)))

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
    calls = FakeControl.instances[0].speedj_calls
    assert calls
    assert np.allclose(calls[-1][0], [0.0] * 6)


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


def test_resolve_live_kinematics_falls_back_to_numpy(monkeypatch):
    monkeypatch.setattr(
        rcl._GafropyKinematicsBridge, "try_create", classmethod(lambda cls: None)
    )
    kin = rcl.resolve_live_kinematics()
    assert kin is not None
    assert rcl._is_numpy_kinematics(kin)


def test_make_twist_without_gafropy(monkeypatch):
    import builtins

    from geodude_teleoperation.teleop.keyboard import make_twist
    from geodude_teleoperation.se3 import NumpyTwist

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "gafropy" or name.startswith("gafropy."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
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
