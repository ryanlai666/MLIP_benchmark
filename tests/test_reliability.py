"""Analytical checks for force metrics, periodic geometry and grouped drag energies."""
import sys
import json
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from reliability_data import force_stats, local_geometry, qsd_metrics, source_group, choose_uniform
from run_reliability import symmetry_check, evaluate
from report_reliability import analyze


class ReliabilityTests(unittest.TestCase):
    def test_frozen_plan_has_unique_cases_and_consistent_coverage(self):
        path = Path(__file__).resolve().parents[1] / 'configs/reliability_plan.json'
        plan = json.loads(path.read_text())
        cases = plan['cases']
        self.assertEqual(len(cases), len({r['id'] for r in cases}))
        for member in plan['inventory']:
            rows = [r for r in cases if r['member']==member['member']]
            self.assertEqual(len(rows), member['selected_frames'])
            self.assertEqual(sum(r['selection']=='coverage' for r in rows), member['coverage_frames'])
            self.assertTrue(all(0 <= r['frame'] < member['available_frames'] for r in rows))
        self.assertEqual(len(plan['archive_sha256']), 64)

    def test_force_metrics_distinguish_vectors_and_components(self):
        m = force_stats([[3,4,0], [0,0,0]])
        self.assertAlmostEqual(m['component_mae'], 7/6)
        self.assertAlmostEqual(m['component_rmse'], np.sqrt(25/6))
        self.assertEqual(m['vector_max'], 5)
        self.assertEqual(m['component_max'], 4)

    def test_periodic_nearest_contact(self):
        atoms = Atoms('CSi', positions=[[0.1,0,0],[9.9,0,0]], cell=[10,10,10], pbc=True)
        near, cross = local_geometry(atoms)
        np.testing.assert_allclose(near, [0.2,0.2])
        self.assertAlmostEqual(cross, 0.2)
        self.assertIsNone(local_geometry(Atoms('CC', positions=[[0,0,0],[1,0,0]]))[1])

    def test_drag_offsets_cancel_and_wrong_slope_is_detected(self):
        rows = [dict(distance_A=d, symbols=['Si','F'], energy_ref_eV=e,
                     energy_pred_eV=100+2*e) for d,e in [(0.5,10),(1.,2),(2.,0)]]
        m = qsd_metrics(rows)
        self.assertAlmostEqual(m['relative_energy_mae_eV'], 4)
        self.assertEqual(m['slope_sign_mismatches'], 0)
        for r in rows:
            r['energy_pred_eV'] = 100-r['energy_ref_eV']
        self.assertEqual(qsd_metrics(rows)['slope_sign_mismatches'], 2)

    def test_source_groups_keep_related_pair_curves_together(self):
        self.assertEqual(source_group('dft/qsd/x', {'group':'CF3_20eV_4_36_Si_F'}),
                         source_group('dft/qsd/x', {'group':'CF3_20eV_4_36_O_C'}))
        self.assertEqual(choose_uniform(list(range(1000)), 3), [0,500,999])

    def test_symmetry_transform_and_force_derivative(self):
        atoms = Atoms('Ar3', positions=[[0,0,0],[1.2,0.2,0],[0.3,1.1,0.2]], cell=[8,8,8], pbc=True)
        calc = LennardJones()
        for k,v in symmetry_check(atoms, calc).items():
            if k != 'reference_force_scale_eV_A':
                self.assertLess(v, 1e-10, k)
        _, forces = evaluate(atoms, calc)
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[1,0] += 1e-5
        minus.positions[1,0] -= 1e-5
        derivative = -(evaluate(plus,calc)[0]-evaluate(minus,calc)[0])/2e-5
        self.assertAlmostEqual(derivative, forces[1,0], places=6)

    def test_targeted_outliers_do_not_change_coverage_scores(self):
        def row(name, error, selection):
            delta = [[error,0,0],[0,0,0]]
            return dict(id=name, status='ok', symbols=['C','Si'], forces_ref_eV_A=[[0,0,0],[0,0,0]],
                        forces_pred_eV_A=delta, metrics=force_stats(delta), nearest_distance_A=[.8,.8],
                        cf_sio_contact_A=.8, selection=selection, track='etch', group='same', source_group='same',
                        energy_pred_eV=1, energy_ref_eV=0)
        summary, strata, outliers, frames = analyze([row('regular',1,'coverage'), row('targeted',1000,'prior_outlier')])
        self.assertEqual(summary[0]['frames'], 1)
        self.assertAlmostEqual(summary[0]['component_mae'], 1/6)
        self.assertEqual(outliers[0]['vector_error_eV_A'], 1000)
        self.assertEqual(len(frames), 2)
        corrupted = row('corrupt',1,'coverage')
        corrupted['metrics']['component_mae'] = 123
        with self.assertRaises(ValueError):
            analyze([corrupted])


if __name__ == '__main__':
    unittest.main()
