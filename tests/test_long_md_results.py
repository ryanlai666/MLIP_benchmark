"""Consistency checks on the committed long-MD and reference-validation artifacts.

These read only the saved CSV/JSON results, so they run without any model
environment and fail if a manifest or report stops matching its own raw data.
"""
import csv
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "results/long-md"
VAL = ROOT / "results/validation-etch"
BACKENDS = ["mace", "nequip", "deepmd"]


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


class TrajectoryTests(unittest.TestCase):
    def test_manifest_diagnostics_match_the_saved_steps(self):
        for name in BACKENDS + ["interface-gpu"]:
            with self.subTest(name=name):
                manifest = json.loads((MD / name / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "completed")
                table = rows(MD / name / "thermodynamics.csv")
                self.assertEqual(len(table), manifest["steps"] + 1)
                total = [float(r["total_eV"]) for r in table]
                drift = [(e - total[0]) / manifest["natoms"] * 1000 for e in total]
                self.assertAlmostEqual(max(abs(d) for d in drift),
                                       manifest["max_abs_energy_change_meV_atom"], places=9)
                self.assertAlmostEqual(drift[-1], manifest["final_energy_change_meV_atom"], places=9)
                self.assertAlmostEqual(float(table[-1]["time_fs"]), manifest["duration_fs"], places=9)

    def test_saved_frame_count_follows_the_recording_interval(self):
        for name in BACKENDS + ["interface-gpu"]:
            with self.subTest(name=name):
                manifest = json.loads((MD / name / "manifest.json").read_text())
                self.assertEqual(manifest["saved_frames"], manifest["steps"] // manifest["record_every"] + 1)

    def test_longer_runs_exceed_the_committed_short_runs(self):
        for name in BACKENDS + ["interface-gpu"]:
            with self.subTest(name=name):
                short = json.loads((ROOT / "results/short-md" / name / "manifest.json").read_text())
                long_run = json.loads((MD / name / "manifest.json").read_text())
                self.assertEqual(long_run["timestep_fs"], short["timestep_fs"])
                self.assertGreater(long_run["duration_fs"], short["duration_fs"])


class ValidationTests(unittest.TestCase):
    def test_pooled_metrics_match_the_per_frame_table(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                summary = json.loads((VAL / backend / "summary.json").read_text())
                table = rows(VAL / backend / "frames.csv")
                self.assertEqual(len(table), summary["frames"])
                components = [3 * int(r["natoms"]) for r in table]
                total = sum(components)
                self.assertEqual(sum(int(r["natoms"]) for r in table), summary["total_atoms"])
                mae = sum(float(r["force_mae_eV_A"]) * n for r, n in zip(table, components)) / total
                mse = sum(float(r["force_rmse_eV_A"]) ** 2 * n for r, n in zip(table, components)) / total
                self.assertAlmostEqual(mae, summary["force_component_mae_eV_A"], places=9)
                self.assertAlmostEqual(math.sqrt(mse), summary["force_component_rmse_eV_A"], places=9)
                raw = [float(r["raw_energy_error_meV_atom"]) for r in table]
                self.assertAlmostEqual(sum(abs(e) for e in raw) / len(raw),
                                       summary["raw_energy_mae_meV_atom"], places=9)

    def test_every_archived_frame_was_evaluated_once(self):
        expected = None
        for backend in BACKENDS:
            manifest = json.loads((VAL / backend / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["stride"], 1)
            frames = [int(r["frame"]) for r in rows(VAL / backend / "frames.csv")]
            self.assertEqual(frames, sorted(set(frames)))
            self.assertEqual(frames, list(range(len(frames))))
            if expected is None:
                expected = frames
            self.assertEqual(frames, expected, "backends must cover the same reference frames")

    def test_reported_csv_matches_the_summaries(self):
        reported = {r["checkpoint"]: r for r in rows(ROOT / "reports/etch_validation_metrics.csv")}
        labels = {"mace": "MACE-MP-0b3 medium", "nequip": "NequIP-OAM-S 0.1", "deepmd": "DPA-3.3-1M / OMat24"}
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                summary = json.loads((VAL / backend / "summary.json").read_text())
                row = reported[labels[backend]]
                self.assertAlmostEqual(float(row["force_component_mae_eV_A"]),
                                       summary["force_component_mae_eV_A"], places=9)
                self.assertAlmostEqual(float(row["offset_fitted_energy_mae_meV_atom"]),
                                       summary["offset_fitted_energy_mae_meV_atom"], places=9)

    def test_fitted_offset_is_reported_separately_from_raw_error(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                summary = json.loads((VAL / backend / "summary.json").read_text())
                self.assertLessEqual(summary["offset_fitted_energy_mae_meV_atom"],
                                     summary["raw_energy_mae_meV_atom"])
                self.assertEqual(sorted(summary["fitted_element_offsets_eV"]),
                                 sorted(summary["per_element_force_mae_eV_A"]))


if __name__ == "__main__":
    unittest.main()
