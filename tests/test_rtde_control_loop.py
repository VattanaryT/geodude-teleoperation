"""RTDE session safety: dry-run vs live, Ctrl+C stop, watchdog, VF freeze."""

from __future__ import annotations

import sys
import types

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
        FakeReceive.instances.append(self)

    def getActualQ(self):
        return list(self._q)

    def getActualTCPPose(self):
        return list(self._tcp)

    def getRobotMode(self):
        return 7

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
        with pytest.raises(RuntimeError, match="dry-run"):
            session.speedj([0.1] * 6, 0.5)
    finally:
        session.close()


def test_live_close_and_ctrl_c_speedstop(fake_ur_rtde):
    """Ctrl+C must speedStop; close must not leave a last non-zero speedJ latched."""
    FakeControl, _ = fake_ur_rtde
    session = rcl.open("192.168.1.20", dry_run=False)
    session.speedj([0.2] * 6, 0.5)
    session.stop_motion()
    ctrl = FakeControl.instances[0]
    assert ctrl.speed_stop_calls, "Ctrl+C must speedStop, not rely on last speedJ"
    session.close()
    assert ctrl.speed_stop_calls or ctrl.stopj_calls or ctrl.speedj_calls
    if ctrl.speedj_calls:
        assert np.allclose(ctrl.speedj_calls[-1][0], [0.0] * 6)


def test_shutdown_refuses_further_speedj(fake_ur_rtde):
    FakeControl, _ = fake_ur_rtde
    session = rcl.open("192.168.1.20", dry_run=False)
    session.speedj([0.3] * 6, 0.5)
    rcl.request_rtde_shutdown()
    n_before = len(FakeControl.instances[0].speedj_calls)
    session.speedj([0.4] * 6, 0.5)
    assert len(FakeControl.instances[0].speedj_calls) == n_before
    session.close()
    rcl._shutdown.clear()


def test_loop_stops_on_shutdown_or_viz_close():
    assert rcl.rtde_loop_should_continue(
        keyboard_running=True, shutdown=False, viz_open=None
    )
    assert not rcl.rtde_loop_should_continue(
        keyboard_running=True, shutdown=True, viz_open=True
    )
    assert not rcl.rtde_loop_should_continue(
        keyboard_running=False, shutdown=False, viz_open=True
    )


def test_watchdog_zeros_and_ignores_gated_ticks():
    """Stale watchdog → q̇=0. Gated speedJ(0) must not refresh the clock."""
    last = 0.0
    watchdog_s = 0.5
    for now in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        qdot = rcl.apply_watchdog(
            np.zeros(6), last_command_time=last, now=now, watchdog_s=watchdog_s
        )
        assert np.allclose(qdot, 0.0)
        assert not rcl.should_refresh_watchdog(
            commands_allowed=False,
            dry_run=False,
            kin_ok=True,
            control_open=True,
            fresh_command=False,
        )
    assert rcl.watchdog_is_stale(last, 0.6, watchdog_s=watchdog_s)
    assert np.allclose(
        rcl.apply_watchdog(np.ones(6), last_command_time=last, now=0.6, watchdog_s=watchdog_s),
        0.0,
    )
    # Active pose servo between WASD ticks keeps the watchdog alive.
    last = 0.0
    for now in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        if rcl.should_refresh_watchdog(
            commands_allowed=True,
            dry_run=False,
            kin_ok=True,
            control_open=True,
            fresh_command=False,
        ):
            last = now
        assert np.allclose(
            rcl.apply_watchdog(np.ones(6), last_command_time=last, now=now, watchdog_s=watchdog_s),
            np.ones(6),
        )


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"orient_locking": True, "has_lock_R_star": True}, False),
        ({"orient_locking": True, "has_lock_R_star": False}, True),
        ({"awaiting_approve": True, "has_lock_R_star": True}, False),
        ({"vf_frozen": True, "has_lock_R_star": True}, False),
        ({"has_lock_R_star": True, "awaiting_approve": False, "vf_frozen": False}, True),
    ],
)
def test_vf_query_gated_during_lock_hold_and_freeze(kwargs, expected):
    args = dict(
        dry_run=False,
        sim_active=False,
        now=1.0,
        last_query_t=0.0,
        dry_period_s=0.5,
        orient_locking=False,
        has_lock_R_star=False,
        awaiting_approve=False,
        vf_frozen=False,
    )
    args.update(kwargs)
    assert rcl.should_run_vf_query(**args) is expected


def test_sync_vf_freeze_with_n_star():
    """in_contact freeze_n_on_arm / f must freeze VF (no WoS), matching DualLoop R*."""

    class _VF:
        def __init__(self):
            self.frozen = False

        def set_frozen(self, frozen: bool) -> None:
            self.frozen = bool(frozen)

    vf = _VF()
    rcl.sync_vf_freeze_with_n_star(vf, True)
    assert vf.frozen is True
    rcl.sync_vf_freeze_with_n_star(vf, False)
    assert vf.frozen is False
    rcl.sync_vf_freeze_with_n_star(object(), True)  # no set_frozen → no-op


def test_live_kinematics_falls_back_to_numpy(monkeypatch):
    monkeypatch.setattr(
        "geodude_teleoperation.gafropy_compat.gafropy_importable", lambda: False
    )
    monkeypatch.setattr(
        rcl._GafropyKinematicsBridge, "try_create", classmethod(lambda cls: None)
    )
    kin = rcl.resolve_live_kinematics()
    assert kin is not None
    assert rcl._is_numpy_kinematics(kin)


def test_resolve_T_composes_rtde_mount(tmp_path):
    T_tag = np.eye(4)
    T_tag[:3, 3] = [1.0, 0.0, 0.0]
    T_rv = np.eye(4)
    T_rv[:3, 3] = [0.3, 0.0, 0.0]
    tag_path = tmp_path / "T_tag0.npy"
    rv_path = tmp_path / "T_rv.npy"
    np.save(tag_path, T_tag)
    np.save(rv_path, T_rv)
    from unittest.mock import MagicMock

    args = MagicMock()
    args.base_frame = None
    args.tag0_frame = None
    args.T_tag0_base = tag_path
    args.T_rtde_base_vention = rv_path
    args.no_rtde_base_vention = False
    T, tf_ok, _ = rcl.resolve_T_tag0_base(args)
    assert tf_ok is True
    assert np.allclose(T[:3, 3], [0.7, 0.0, 0.0])


def test_flange_tcp_offset_roundtrip():
    from geodude_teleoperation.rtde.ur5e_kinematics import forward_kinematics
    from scipy.spatial.transform import Rotation

    q = np.array([0.1, -1.0, 1.2, -1.4, -1.57, 0.2])
    T_f, _ = forward_kinematics(q)
    tcp = np.concatenate(
        [T_f[:3, 3] + np.array([0.0, 0.0, 0.05]), Rotation.from_matrix(T_f[:3, :3]).as_rotvec()]
    )
    off = rcl.flange_tcp_offset_matrix(q, tcp)
    assert np.allclose(tcp, rcl.tcp_pose_from_q_and_offset(q, off), atol=1e-9)
