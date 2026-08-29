import datetime
import pathlib
import queue
import tempfile
import unittest

import numpy as np

from geodude_teleoperation.speedj_plot import (
    SpeedJTrace,
    downsample_maxabs,
    sample_from_qdot,
    session_save_paths,
    write_speedj_session,
)


class SpeedJTraceTest(unittest.TestCase):
    def test_first_sample_has_zero_dt_and_accel(self):
        qdot = np.array([0.1, -0.2, 0.0])
        sample = sample_from_qdot(qdot, t=1.0, prev=None)
        self.assertAlmostEqual(sample.dt, 0.0)
        np.testing.assert_allclose(sample.qdot, qdot)
        np.testing.assert_allclose(sample.qddot, 0.0)

    def test_qdot_step_accel_is_delta_over_dt(self):
        first = sample_from_qdot(np.zeros(2), t=0.0, prev=None)
        second = sample_from_qdot(
            np.array([0.8, 0.0]), t=0.002, prev=first)
        self.assertAlmostEqual(second.dt, 0.002)
        np.testing.assert_allclose(second.qddot, [400.0, 0.0])

    def test_constant_qdot_has_zero_accel(self):
        first = sample_from_qdot(np.array([0.3, 0.1]), t=0.0, prev=None)
        second = sample_from_qdot(
            np.array([0.3, 0.1]), t=0.002, prev=first)
        np.testing.assert_allclose(second.qddot, [0.0, 0.0], atol=1e-12)

    def test_record_drops_oldest_instead_of_blocking(self):
        sink = queue.Queue(maxsize=1)
        trace = SpeedJTrace(sink=sink)
        trace.record(np.array([0.0]), t=0.0)
        trace.record(np.array([0.1]), t=0.002)
        trace.record(np.array([0.2]), t=0.004)
        self.assertEqual(sink.qsize(), 1)
        sample = sink.get_nowait()
        np.testing.assert_allclose(sample.qdot, [0.2])

    def test_downsample_keeps_accel_spike(self):
        t = np.linspace(0.0, 10.0, 10000)
        y = np.zeros(10000)
        y[4321] = 400.0
        t2, y2 = downsample_maxabs(t, y, n_bins=100)
        self.assertLessEqual(t2.size, 100)
        self.assertAlmostEqual(float(np.max(np.abs(y2))), 400.0)

    def test_session_save_paths_use_timestamp(self):
        when = datetime.datetime(2026, 8, 24, 22, 54, 3)
        png, npz = session_save_paths("speedj_logs", when=when)
        self.assertEqual(png, pathlib.Path("speedj_logs/speedj_20260824_225403.png"))
        self.assertEqual(npz, pathlib.Path("speedj_logs/speedj_20260824_225403.npz"))

    def test_write_session_saves_png_and_npz(self):
        t = np.array([0.0, 0.002, 0.004])
        qdot = np.array([[0.0, 0.0], [0.8, 0.0], [0.4, 0.1]])
        qddot = np.array([[0.0, 0.0], [400.0, 0.0], [-200.0, 50.0]])
        dt = np.array([0.0, 0.002, 0.002])
        spring = np.array([0.0, 1.0, 1.0])
        with tempfile.TemporaryDirectory() as tmp:
            png = pathlib.Path(tmp) / "speedj.png"
            npz = pathlib.Path(tmp) / "speedj.npz"
            write_speedj_session(
                png, t=t, qdot=qdot, qddot=qddot, dt=dt, spring=spring,
                path_npz=npz)
            self.assertTrue(png.is_file())
            self.assertGreater(png.stat().st_size, 1000)
            data = np.load(npz)
            np.testing.assert_allclose(data["qdot"], qdot)
            np.testing.assert_allclose(data["qddot"][1], [400.0, 0.0])
            self.assertAlmostEqual(float(data["dt"][1]), 0.002)

    def test_write_session_saves_stick(self):
        t = np.array([0.0, 0.002])
        qdot = np.zeros((2, 2))
        qddot = np.zeros((2, 2))
        dt = np.array([0.0, 0.002])
        spring = np.array([0.0, 1.0])
        stick = np.array([[0.0, 0.0, 0.0], [0.00119, 0.0, 0.0]])
        with tempfile.TemporaryDirectory() as tmp:
            png = pathlib.Path(tmp) / "speedj.png"
            npz = pathlib.Path(tmp) / "speedj.npz"
            write_speedj_session(
                png, t=t, qdot=qdot, qddot=qddot, dt=dt, spring=spring,
                stick=stick, path_npz=npz)
            data = np.load(npz)
            np.testing.assert_allclose(data["stick"][1], [0.00119, 0.0, 0.0])

    def test_write_session_trims_racy_length_mismatch(self):
        t = np.linspace(0.0, 1.0, 5)
        qdot = np.zeros((5, 2))
        qddot = np.zeros((5, 2))
        dt = np.full(6, 0.002)
        spring = np.zeros(7)
        stick = np.zeros((6, 3))
        stick[-1, 1] = 0.001
        with tempfile.TemporaryDirectory() as tmp:
            png = pathlib.Path(tmp) / "speedj.png"
            npz = pathlib.Path(tmp) / "speedj.npz"
            write_speedj_session(
                png, t=t, qdot=qdot, qddot=qddot, dt=dt, spring=spring,
                stick=stick, path_npz=npz)
            self.assertTrue(png.is_file())
            data = np.load(npz)
            self.assertEqual(data["t"].shape[0], 5)
            self.assertEqual(data["stick"].shape[0], 5)

    def test_trace_save_session_uses_recorded_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            png = pathlib.Path(tmp) / "sess.png"
            npz = pathlib.Path(tmp) / "sess.npz"
            trace = SpeedJTrace()
            trace.record(np.zeros(2), t=0.0, spring=False)
            trace.record(np.array([0.8, 0.0]), t=0.002, spring=True)
            out = trace.save_session(png, npz)
            self.assertEqual(out, png)
            self.assertTrue(png.is_file())
            data = np.load(npz)
            np.testing.assert_allclose(data["qddot"][1], [400.0, 0.0])


class WrenchTraceTest(unittest.TestCase):
    """--dtc-law wrench feeds this plot instead of speedJ.

    The wrench law never calls ``compute_control_command``, so the controller's
    trace hook never fires and the window was empty for every wrench-law run.
    The backend traces it instead, with commanded tau in the ``qdot`` slot and
    the task wrench alongside; a sample carrying a wrench flips the top two
    panels to N.m and N.
    """

    def test_sample_carries_the_wrench_and_defaults_to_none(self):
        plain = sample_from_qdot(np.zeros(6), t=0.0, prev=None)
        self.assertIsNone(plain.wrench)
        W = np.array([0.1, 0.2, 0.3, 4.0, 5.0, 6.0])
        s = sample_from_qdot(np.zeros(6), t=0.0, prev=None, wrench=W)
        np.testing.assert_allclose(s.wrench, W)

    def test_wrench_session_renders_and_round_trips(self):
        t = np.array([0.0, 0.008, 0.016])
        tau = np.array([[0.0] * 6, [6.0, 3.5, 2.9, 1.5, 1.8, 0.06], [-4.0] + [0.0] * 5])
        qddot = np.zeros((3, 6))
        dt = np.array([0.0, 0.008, 0.008])
        spring = np.array([0.0, 1.0, 1.0])
        stick = np.zeros((3, 3))
        measured = np.zeros((3, 6))
        measured[1, 0] = 0.02
        wrench = np.array([
            [0.0] * 6,
            [0.1, 0.2, 0.3, 10.0, 2.0, 1.0],
            [0.0, 0.0, 0.0, 45.0, 0.0, 0.0],
        ])
        with tempfile.TemporaryDirectory() as tmp:
            png = pathlib.Path(tmp) / "wrench.png"
            npz = pathlib.Path(tmp) / "wrench.npz"
            write_speedj_session(
                png, t=t, qdot=tau, qddot=qddot, dt=dt, spring=spring,
                measured=measured, stick=stick, wrench=wrench, path_npz=npz)
            # The PNG path swallows its own exceptions, so its existence is the
            # only evidence the wrench-mode drawing branch actually ran.
            self.assertTrue(png.is_file())
            self.assertGreater(png.stat().st_size, 1000)
            data = np.load(npz)
            np.testing.assert_allclose(data["wrench"][2], wrench[2])

    def test_a_speedj_session_still_writes_no_wrench_key(self):
        t = np.array([0.0, 0.002])
        qdot = np.zeros((2, 2))
        with tempfile.TemporaryDirectory() as tmp:
            png = pathlib.Path(tmp) / "s.png"
            npz = pathlib.Path(tmp) / "s.npz"
            write_speedj_session(
                png, t=t, qdot=qdot, qddot=np.zeros((2, 2)),
                dt=np.array([0.0, 0.002]), spring=np.zeros(2), path_npz=npz)
            self.assertNotIn("wrench", np.load(npz).files)

    def test_trace_save_session_keeps_the_wrench(self):
        with tempfile.TemporaryDirectory() as tmp:
            png = pathlib.Path(tmp) / "sess.png"
            npz = pathlib.Path(tmp) / "sess.npz"
            trace = SpeedJTrace()
            W = np.array([0.3, 0.0, 0.0, 12.0, 0.0, 0.0])
            trace.record(np.zeros(6), t=0.0, spring=False)
            trace.record(np.full(6, 2.0), t=0.008, spring=True,
                         measured=np.zeros(6), wrench=W)
            trace.save_session(png, npz)
            self.assertTrue(png.is_file())
            data = np.load(npz)
            self.assertEqual(data["wrench"].shape, (2, 6))
            # The first sample had no wrench: NaN, not a fabricated zero, or a
            # run that started before the law engaged would read as "0 N".
            self.assertTrue(np.all(np.isnan(data["wrench"][0])))
            np.testing.assert_allclose(data["wrench"][1], W)
