import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("pilot", Path(__file__).resolve().parents[1] / "scripts/run_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class MetricTests(unittest.TestCase):
    def test_component_weighting_and_per_atom_energy(self):
        rows = [dict(track="x",status="ok",natoms=1,forces_pred_eV_A=[[3,0,0]],
                     forces_ref_eV_A=[[0,0,0]],energy_pred_eV=2,energy_ref_eV=0,seconds=1),
                dict(track="x",status="ok",natoms=2,forces_pred_eV_A=[[0,0,0],[0,0,0]],
                     forces_ref_eV_A=[[0,0,0],[0,0,0]],energy_pred_eV=2,energy_ref_eV=0,seconds=3),
                dict(track="x",status="failed")]
        metrics = pilot.summarize(rows)["x"]
        self.assertAlmostEqual(metrics["force_component_mae_eV_A"], 1/3)
        self.assertAlmostEqual(metrics["force_component_rmse_eV_A"], 1)
        self.assertAlmostEqual(metrics["raw_energy_mae_eV_atom"], 1.5)
        self.assertEqual(metrics["failed"], 1)
        self.assertEqual(metrics["median_singlepoint_seconds"], 2)

    def test_all_failed_does_not_report_zero_error(self):
        metrics = pilot.summarize([dict(track="x",status="failed")])["x"]
        self.assertNotIn("force_component_mae_eV_A", metrics)
        self.assertEqual(metrics["completed"], 0)
