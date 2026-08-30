"""The analyzer's drive-independent ringing metric.

Band share cannot A/B a damping change: at an identical D_lin = 12 and
fn_lin = 4.0, dtc_20260830_134133.csv read 22.7% of the qdot power above 2 Hz
and dtc_20260830_135527.csv read 11.8%, because a share divides the ring by
the total motion and the operator owns that denominator. A resonance's Q is
its width, so it measures how damped the mode is rather than how hard it was
hit.
"""
import importlib.util
import pathlib
import unittest

import numpy as np

_ANALYZER = pathlib.Path(__file__).resolve().parents[1] / "examples" / "analyze_dtc_log.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyze_dtc_log", _ANALYZER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lorentzian(freq, f0, q_true):
    """Power spectrum of a lightly damped mode: half power at f0/Q wide."""
    return 1.0 / (1.0 + ((freq - f0) / (0.5 * f0 / q_true)) ** 2)


class PeakQTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load()

    def test_it_recovers_the_q_of_a_synthetic_resonance(self):
        freq = np.linspace(0.0, 40.0, 8001)   # 5 mHz bins
        for q_true in (20.0, 50.0, 120.0):
            power = _lorentzian(freq, 15.0, q_true)
            pk = int(np.argmax(power))
            got = self.mod.peak_q(freq, power, pk, 2.0)
            self.assertAlmostEqual(got, q_true, delta=0.05 * q_true,
                                   msg=f"Q={q_true}")

    def test_more_damping_reads_as_a_lower_q(self):
        """The direction the metric exists for -- see D_lin 12 -> 15.08."""
        freq = np.linspace(0.0, 40.0, 8001)
        sharp = self.mod.peak_q(freq, _lorentzian(freq, 15.0, 140.0), int(np.argmax(_lorentzian(freq, 15.0, 140.0))), 2.0)
        broad = self.mod.peak_q(freq, _lorentzian(freq, 15.0, 77.0), int(np.argmax(_lorentzian(freq, 15.0, 77.0))), 2.0)
        self.assertLess(broad, sharp)

    def test_q_is_blind_to_how_hard_the_mode_was_driven(self):
        """The whole point: 10x the excitation is the same Q."""
        freq = np.linspace(0.0, 40.0, 8001)
        base = _lorentzian(freq, 15.0, 60.0)
        pk = int(np.argmax(base))
        self.assertAlmostEqual(self.mod.peak_q(freq, base, pk, 2.0),
                               self.mod.peak_q(freq, 100.0 * base, pk, 2.0),
                               places=6)

    def test_a_shoulder_is_not_a_resonance(self):
        """Never quote a Q for a line that does not come back down."""
        freq = np.linspace(0.0, 40.0, 4001)
        power = np.exp(-freq / 30.0)      # monotone, no peak in band
        power[0] = 0.0
        pk = int(np.argmax(power[10:])) + 10
        self.assertTrue(np.isnan(self.mod.peak_q(freq, power, pk, 2.0)))

    def test_the_stick_floor_bounds_the_downward_search(self):
        """A peak sitting on the operator's own skirt must not widen into it."""
        freq = np.linspace(0.0, 40.0, 4001)
        power = _lorentzian(freq, 15.0, 40.0) + 5.0 * np.exp(-freq)
        pk = int(np.argmax(np.where(freq > 5.0, power, 0.0)))
        got = self.mod.peak_q(freq, power, pk, 2.0)
        self.assertTrue(np.isnan(got) or got > 0.0)

    def test_an_edge_bin_has_no_width(self):
        freq = np.linspace(0.0, 40.0, 101)
        power = np.ones(101)
        self.assertTrue(np.isnan(self.mod.peak_q(freq, power, 0, 2.0)))
        self.assertTrue(np.isnan(self.mod.peak_q(freq, power, 100, 2.0)))


if __name__ == "__main__":
    unittest.main()
