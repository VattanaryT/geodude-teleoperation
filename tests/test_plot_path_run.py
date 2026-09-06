"""The post-run plot: it must survive a real log's shape, and a partial one.

The plot is generated automatically after every executed path, from a
subprocess, which means nobody is watching it fail. These cases are the
watching: a full log, a log written before the commanded-pose columns existed,
and a run with no contact CSV at all.
"""

import csv
import importlib.util
import pathlib
import unittest

import numpy as np

_SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
           / "examples" / "plot_path_run.py")

DTC_COLS = (
    ["t", "dt_ms", "phase", "safety"]
    + [f"w{i}" for i in range(6)] + [f"sm{i}" for i in range(6)]
    + [f"tau{i}" for i in range(6)] + [f"q{i}" for i in range(6)]
    + [f"qd{i}" for i in range(6)] + [f"clip{i}" for i in range(6)]
    + ["erx", "ery", "erz", "ex", "ey", "ez", "n_deg", "act",
       "lam_ang", "lam_dir", "lam_hi", "lev",
       "ee_x", "ee_y", "ee_z", "tgt_x", "tgt_y", "tgt_z",
       "ee_qw", "ee_qx", "ee_qy", "ee_qz",
       "tgt_qw", "tgt_qx", "tgt_qy", "tgt_qz"]
    + [f"qc{i}" for i in range(6)]
)

CONTACT_COLS = [
    "t", "elapsed", "dt", "mode", "press_loop", "fz_raw", "Fs", "Fd",
    "F_press", "F_emit", "engaged", "aborted", "d_surf", "phase",
    "ft_fx", "ft_fy", "ft_fz", "ft_tx", "ft_ty", "ft_tz", "Fs_n",
]


def _load():
    spec = importlib.util.spec_from_file_location("plot_path_run", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_dtc(path, n=40, cols=None):
    cols = list(DTC_COLS if cols is None else cols)
    with open(path, "w", newline="") as fh:
        fh.write("# argv: fake\n# law=wrench hz=500 K_lin=355.3\n")
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(n):
            row = dict.fromkeys(cols, "")
            row["t"] = f"{0.002 * i:.4f}"
            row["dt_ms"] = "2.06"
            row["phase"] = "drive"
            row["safety"] = "NORMAL"
            for j in range(6):
                row[f"q{j}"] = f"{0.01 * i + j:.5f}"
                row[f"qd{j}"] = "0.001"
                row[f"tau{j}"] = "1.5"
                row[f"clip{j}"] = "0"
                if f"qc{j}" in row:
                    row[f"qc{j}"] = f"{0.01 * i + j + 0.002:.5f}"
            for k, a in enumerate("xyz"):
                row[f"ee_{a}"] = f"{0.3 + 0.001 * i:.6f}"
                row[f"tgt_{a}"] = f"{0.3 + 0.001 * i + 0.002:.6f}"
                row[f"e{a}"] = "0.002"
            row["erx"] = "0.01"
            row["n_deg"] = "1.2"
            row["act"] = "1"
            for a in "wxyz":
                if f"ee_q{a}" in row:
                    row[f"ee_q{a}"] = "0.5"
                    row[f"tgt_q{a}"] = "0.5"
            w.writerow([row[c] for c in cols])


def _write_contact(path, n=30):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CONTACT_COLS)
        for i in range(n):
            t = 0.005 * i
            phase = "approach" if i < 10 else "traverse" if i < 25 else "retract"
            eng = 1 if phase == "traverse" else 0
            w.writerow([f"{t:.4f}", f"{t:.4f}", "0.005", "contact", "dtc",
                        "-12.0", f"{15.0 * eng:.3f}", "15.0",
                        f"{15.0 * eng:.3f}", f"{15.0 * eng:.3f}", str(eng), "0",
                        "0.001", phase, "0.4", "-0.2", "-27.0", "0.1", "0.05",
                        "0.0", f"{15.0 * eng:.3f}"])


class PlotPathRunTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.mod = _load()
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def _run(self, argv):
        return self.mod.main(argv)

    def test_renders_a_full_run(self):
        dtc = self.tmp / "dtc_20260906_120000.csv"
        contact = self.tmp / "contact_20260906_120001.csv"
        _write_dtc(dtc)
        _write_contact(contact)
        out = self.tmp / "run.png"
        self.assertEqual(self._run([str(dtc), "--contact-log", str(contact),
                                    "--out", str(out), "--no-show"]), 0)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 10_000)

    def test_renders_without_a_contact_log(self):
        # Aborting before contact is entered leaves no force record; the plot
        # still has to draw the half that decides smoothness.
        dtc = self.tmp / "dtc_20260906_130000.csv"
        _write_dtc(dtc)
        out = self.tmp / "nocontact.png"
        self.assertEqual(self._run([str(dtc), "--contact-log",
                                    str(self.tmp / "missing.csv"),
                                    "--out", str(out), "--no-show"]), 0)
        self.assertTrue(out.exists())

    def test_renders_a_log_predating_the_commanded_pose_columns(self):
        old_cols = [c for c in DTC_COLS
                    if not c.startswith(("ee_q", "tgt_q", "qc"))]
        dtc = self.tmp / "dtc_20260901_100000.csv"
        _write_dtc(dtc, cols=old_cols)
        out = self.tmp / "old.png"
        self.assertEqual(self._run([str(dtc), "--out", str(out), "--no-show"]), 0)
        self.assertTrue(out.exists())

    def test_reads_the_columns_it_plots(self):
        dtc = self.tmp / "dtc_20260906_140000.csv"
        _write_dtc(dtc, n=5)
        rows, meta = self.mod.read_csv(dtc)
        self.assertEqual(len(rows), 5)
        self.assertTrue(any("law=wrench" in m for m in meta))
        np.testing.assert_allclose(self.mod.col(rows, "n_deg"), 1.2)
        # A missing value must read as NaN, not 0: a zero is a measurement.
        self.assertTrue(np.isnan(self.mod.col(rows, "w0")).all())

    def test_phase_resample_never_invents_a_transition(self):
        contact = self.tmp / "contact_20260906_150000.csv"
        _write_contact(contact, n=30)
        rows = self.mod.read_csv(contact)[0]
        phases = self.mod._phase_on_dtc_clock(np.linspace(0, 0.15, 60), rows)
        self.assertEqual(len(phases), 60)
        self.assertTrue(set(phases) <= {"approach", "traverse", "retract"})

    def test_phase_resample_survives_an_empty_contact_log(self):
        self.assertEqual(self.mod._phase_on_dtc_clock(np.zeros(3), []), [])


if __name__ == "__main__":
    unittest.main()
