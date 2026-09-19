"""Targeted CPU reproduction of DeePMD CUDA short-range flags, not a speed test."""
import argparse
from pathlib import Path

import numpy as np
from ase import Atoms

from reliability_data import ROOT, digest, dump
from report_reliability import read_jsonl
from run_reliability import calculator, checkpoint, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-results', type=Path, default=ROOT/'results/reliability/deepmd')
    p.add_argument('--output', type=Path, default=ROOT/'reports/reliability/deepmd_cpu_gpu_spotcheck.json')
    args = p.parse_args()
    import json
    manifest = json.loads((args.gpu_results/'manifest.json').read_text())
    path = checkpoint('deepmd')
    if manifest['device'] != 'cuda' or manifest['checkpoint_sha256'] != digest(path):
        raise ValueError('Expected CUDA results for the same checkpoint')
    calc, meta = calculator('deepmd', path, 'cpu', 4)
    checks = read_jsonl(args.gpu_results/'checks.jsonl')
    rows = read_jsonl(args.gpu_results/'predictions.jsonl')
    results = []
    for r in checks:
        if r['kind']=='pair' and r['status']=='ok' and r['pair'] in ['C-F','Si-Si'] and r['distance_A'] in [.5,.65]:
            atoms = Atoms(r['pair'].split('-'), positions=[[0,0,0],[r['distance_A'],0,0]])
            _, forces = evaluate(atoms, calc)
            results.append(dict(id=r['id'], cpu_radial_force_eV_A=float(forces[1,0]),
                                gpu_radial_force_eV_A=r['radial_force_eV_A'],
                                absolute_difference_eV_A=abs(float(forces[1,0])-r['radial_force_eV_A'])))
    for r in sorted([r for r in rows if r['track']=='qsd' and r['status']=='ok'],
                    key=lambda r:r['metrics']['vector_max'], reverse=True)[:3]:
        atoms = Atoms(r['symbols'], positions=r['positions_A'], cell=r['cell_A'], pbc=r['pbc'])
        _, forces = evaluate(atoms, calc)
        results.append(dict(id=r['id'],
                            max_cpu_gpu_force_component_difference_eV_A=float(np.abs(forces-r['forces_pred_eV_A']).max()),
                            cpu_dft_force_component_max_error_eV_A=float(np.abs(forces-r['forces_ref_eV_A']).max()),
                            gpu_dft_force_component_max_error_eV_A=r['metrics']['component_max']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(args.output, dict(checkpoint_sha256=digest(path), cpu_environment=meta,
                          gpu_manifest_sha256=digest(args.gpu_results/'manifest.json'),
                          checks_sha256=digest(args.gpu_results/'checks.jsonl'),
                          predictions_sha256=digest(args.gpu_results/'predictions.jsonl'),
                          script_sha256=digest(Path(__file__)), checks=results,
                          purpose='Targeted reproduction; not a throughput or general device-equivalence test'))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
