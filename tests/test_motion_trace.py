"""MotionTraceLogger CSV + FixedOrientationField wobble-diag helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geodude_teleoperation.teleop.contact_wos_vf import FixedOrientationField
from geodude_teleoperation.teleop.controller import qdot_from_error
from geodude_teleoperation.teleop.motion_trace import MotionTraceLogger, save_and_maybe_plot


class _DuckMotor:
    def __init__(self, R):
        self._R = np.asarray(R, dtype=float).reshape(3, 3)

    def get_translator(self):
        class _T:
            def x(self):
                return 0.0

            def y(self):
                return 0.0

            def z(self):
                return 0.5

        return _T()

    def get_rotor(self):
        from scipy.spatial.transform import Rotation

        return Rotation.from_matrix(self._R)


def test_motion_trace_csv(tmp_path: Path):
    log = MotionTraceLogger()
    R0 = np.eye(3)
    R1 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    log.append(
        t_mono=1.0,
        qdot=[0.1, 0, 0, 0, 0, 0],
        qdot_raw=[2.5, 0, 0, 0, 0, 0],
        R_motion=R0,
        relocked=True,
        err_pos_m=0.01,
        err_rot_rad=0.02,
        err_norm=0.022,
        scale_factor=10.0,
        distance_m=0.23,
        twist_align_rad=0.1,
        qdot_unclipped_norm=2.5,
        qdot_clipped=True,
        tip_delta_m=0.02,
        dt_since_relock_s=0.05,
        relock_reason="rate_limit",
        key="w",
        twist_linear=[0.01, 0, 0],
        tcp_pose=[0.1, 0.2, 0.3, 0, 0, 0],
        ref_pose=[0.11, 0.2, 0.3, 0, 0, 0],
    )
    log.append(
        t_mono=1.02,
        qdot=[0.0, 0.2, 0, 0, 0, 0],
        R_motion=R1,
        relocked=False,
        query_ms=70.0,
        loop_ms=5.0,
        scale_factor=1.0,
        key="d",
    )
    csv = log.save_csv(tmp_path / "m.csv")
    text = csv.read_text()
    header = text.splitlines()[0]
    assert "scale_factor" in header
    assert "err_pos_m" in header
    assert "twist_align_rad" in header
    assert "qdot_unclipped_norm" in header
    assert "key_w" in header
    assert "ref_x" in header
    data = log.as_arrays()
    assert data["t_s"].shape == (2,)
    assert data["relocked"][0] == 1.0
    assert data["scale_factor"][0] == 10.0
    assert data["qdot_clipped"][0] == 1.0
    assert data["qdot_unclipped_norm"][0] == 2.5
    assert data["key_w"][0] == 1.0
    assert data["key_d"][1] == 1.0
    assert data["rm_delta_rad"][1] > 0.5
    assert abs(data["tcp_pose"][0, 0] - 0.1) < 1e-9
    assert abs(data["ref_pose"][0, 0] - 0.11) < 1e-9


def test_qdot_from_error_diag_reports_clip():
    err = np.ones(6)
    J = np.eye(6)
    q, diag = qdot_from_error(
        err, J, gain=100.0, damping=0.0, max_joint_velocity=1.0, return_diag=True
    )
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9
    assert diag["qdot_clipped"] is True
    assert diag["qdot_unclipped_norm"] > 1.0
    assert diag["err_pos_m"] > 0
    assert diag["err_rot_rad"] > 0
    assert diag["qdot_raw"].shape == (6,)
    assert np.linalg.norm(diag["qdot_raw"]) > 1.0


def test_tip_velocity_field_frame_projection():
    from geodude_teleoperation.teleop.motion_trace import tip_velocity_field_frame

    R = np.eye(3)
    # World +x → t1; +y → t2; +z → approach into surface → v_n = -1
    assert tip_velocity_field_frame([1, 0, 0], R) == (1.0, 0.0, 0.0)
    assert tip_velocity_field_frame([0, 1, 0], R) == (0.0, 1.0, 0.0)
    vt1, vt2, vn = tip_velocity_field_frame([0, 0, 1], R)
    assert abs(vt1) < 1e-12 and abs(vt2) < 1e-12 and abs(vn + 1.0) < 1e-12


def test_motion_trace_logs_field_tip_velocity():
    log = MotionTraceLogger()
    R = np.eye(3)
    log.append(
        t_mono=0.0,
        qdot=np.zeros(6),
        R_motion=R,
        tcp_pose=[0.0, 0.0, 0.0, 0, 0, 0],
    )
    log.append(
        t_mono=0.1,
        qdot=np.zeros(6),
        R_motion=R,
        tcp_pose=[0.05, 0.0, 0.0, 0, 0, 0],  # 0.5 m/s along +x = t1
    )
    data = log.as_arrays()
    assert "v_t1" in MotionTraceLogger.HEADER
    assert np.isnan(data["v_t1"][0])
    assert abs(data["v_t1"][1] - 0.5) < 1e-9
    assert abs(data["v_t2"][1]) < 1e-9
    assert abs(data["v_n"][1]) < 1e-9


def test_motion_trace_plot_smoke(tmp_path: Path):
    pytest.importorskip("matplotlib")
    log = MotionTraceLogger()
    for i in range(5):
        log.append(
            t_mono=0.02 * i,
            qdot=[0.1 * i, 0, 0, 0, 0, 0],
            qdot_raw=[0.5 * i, 0, 0, 0, 0, 0],
            qdot_clipped=i > 2,
            qdot_unclipped_norm=0.5 * i,
            R_motion=np.eye(3),
            tcp_pose=[0.01 * i, 0, 0.5, 0, 0, 0],
            ref_pose=[0.012 * i, 0, 0.5, 0, 0, 0],
            key="w" if i == 2 else None,
            twist_linear=[0.01, 0, 0] if i == 2 else None,
        )
    png = log.plot(tmp_path / "m.png")
    assert png is not None and png.exists()


def test_contact_motion_trace_columns_and_plot(tmp_path: Path):
    log = MotionTraceLogger()
    R = np.eye(3)
    log.append(
        t_mono=1.0,
        qdot=np.zeros(6),
        R_motion=R,
        tcp_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        Fd_N=5.0,
        Fs_N=0.0,
        Fe_N=5.0,
        n_freeze=0.0,
        force_sign=-1.0,
        k_ori=0.75,
        key="w",
        twist_linear=[0.01, 0.0, 0.0],
    )
    log.append(
        t_mono=1.05,
        qdot=[0.1, 0, 0, 0, 0, 0],
        R_motion=R,
        tcp_pose=[0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        Fd_N=5.0,
        Fs_N=4.5,
        Fe_N=0.5,
        n_freeze=1.0,
        force_sign=-1.0,
        k_ori=5.0,
        key="d",
        twist_linear=[0.0, 0.01, 0.0],
    )
    assert log.prefers_contact_plot()
    assert "Fd_N" in MotionTraceLogger.HEADER
    assert "a_t1" in MotionTraceLogger.HEADER
    data = log.as_arrays()
    assert data["Fd_N"][0] == 5.0
    assert data["Fs_N"][1] == 4.5
    assert abs(data["a_t1"][1] - 0.01) < 1e-6  # ∫ v_t1 dt ≈ 0.01/0.05 * 0.05? 
    # tip moved +0.01 in x over 0.05s → v=0.2, a_t1 += 0.2*0.05 = 0.01
    assert abs(data["a_t1"][1] - 0.01) < 1e-9
    csv = log.save_csv(tmp_path / "c.csv")
    assert "Fd_N" in csv.read_text().splitlines()[0]
    png = log.plot(tmp_path / "c.png", style="contact")
    assert png is not None and png.exists()
    csv2, png2 = save_and_maybe_plot(
        log, tmp_path / "c2.csv", plot=True, style="contact"
    )
    assert csv2.exists() and png2 is not None and png2.exists()


def test_save_and_maybe_plot_without_matplotlib(tmp_path: Path):
    log = MotionTraceLogger()
    log.append(t_mono=0.0, qdot=np.zeros(6), R_motion=np.eye(3))
    csv, png = save_and_maybe_plot(log, tmp_path / "a.csv", plot=False)
    assert csv.exists()
    assert png is None


def test_fixed_orientation_locks_once():
    R_a = np.eye(3)
    R_b = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    vf = FixedOrientationField()
    vf.update_from_state(_DuckMotor(R_a))
    assert np.allclose(vf.motion_basis(), R_a)
    assert vf.status["relocked"] is True
    vf.update_from_state(_DuckMotor(R_b))
    assert np.allclose(vf.motion_basis(), R_a)
    assert vf.status["relocked"] is False
    assert vf.status["used_wos"] is False


def test_fixed_orientation_explicit():
    R = np.diag([1.0, -1.0, -1.0])
    vf = FixedOrientationField(R_fixed=R)
    vf.update_from_state(_DuckMotor(np.eye(3)))
    assert np.allclose(vf.motion_basis(), R)
