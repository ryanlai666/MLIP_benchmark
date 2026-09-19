"""Reproducible DFT/QSD coverage, force outliers and reference-free physical checks.

Create a common plan with --prepare; execute separately in each model environment.
Existing predictions are append-only; --resume skips completed or failed case IDs.
"""
import argparse
import importlib.metadata
import itertools
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase import Atoms

from reliability_data import ROOT, ARCHIVE, digest, dump, make_plan, selected_atoms, force_stats, local_geometry


def checkpoint(backend):
    return {'mace': ROOT / 'models/cache/mace/macemp0b3mediummodel',
            'nequip': ROOT / 'models/nequip-cache/b39e883d722e2108867a0b8028e90beeccf2e50c944c9063ef511711984a17fe.nequip.zip',
            'deepmd': ROOT / 'models/deepmd/DPA-3.3-1M.pt'}[backend]


def calculator(backend, path, device, threads):
    os.environ['DEVICE'] = device
    import torch
    torch.set_num_threads(threads)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if backend == 'mace':
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_paths=str(path), device=device, default_dtype='float64')
        parameters = list(calc.models[0].parameters())
    elif backend == 'nequip':
        from nequip.integrations.ase import NequIPCalculator
        calc = NequIPCalculator._from_saved_model(model_path=str(path), device=device,
                                                 chemical_species_to_atom_type_map=True)
        parameters = list(calc.model.parameters())
    else:
        from deepmd.calculator import DP
        calc = DP(model=str(path), head='OMat24')
        parameters = list(calc.dp.deep_eval.dp.parameters())
    devices = sorted({str(p.device) for p in parameters})
    if not devices or not all(d.startswith(device) for d in devices):
        raise RuntimeError(f'Requested {device}; actual parameter devices {devices}')
    return calc, dict(parameter_devices=devices, parameter_dtypes=sorted({str(p.dtype) for p in parameters}),
                      packages={p: importlib.metadata.version(p) for p in
                                ['torch', 'ase', 'numpy', {'mace':'mace-torch', 'nequip':'nequip', 'deepmd':'deepmd-kit'}[backend]]})


def evaluate(atoms, calc):
    atoms = atoms.copy()
    atoms.calc = calc
    energy, forces = float(atoms.get_potential_energy()), atoms.get_forces()
    if not np.isfinite(energy) or forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
        raise ValueError('Nonfinite or incorrectly shaped prediction')
    return energy, forces


def physical_checks(calc, append, done):
    distances = [0.5, 0.65, 0.8, 1., 1.2, 1.5, 2., 2.5, 3., 4., 6.]
    for a, b in itertools.combinations_with_replacement(['C', 'F', 'O', 'Si'], 2):
        for r in distances:
            key = f'pair:{a}-{b}:{r:g}'
            if key in done:
                continue
            row = dict(id=key, kind='pair', pair=f'{a}-{b}', distance_A=r,
                       reference='none; qualitative diagnostic, not chemical accuracy')
            try:
                atoms = Atoms([a,b], positions=[[0,0,0], [r,0,0]], pbc=False)
                e, f = evaluate(atoms, calc)
                row.update(status='ok', energy_eV=e, radial_force_eV_A=float(f[1,0]),
                           net_force_eV_A=float(np.linalg.norm(f.sum(axis=0))))
                if r in [0.8, 1.5, 3.]:
                    errors = []
                    for h in [1e-3, 5e-4]:
                        plus, minus = atoms.copy(), atoms.copy()
                        plus.positions[1,0] += h
                        minus.positions[1,0] -= h
                        ep, _ = evaluate(plus, calc)
                        em, _ = evaluate(minus, calc)
                        errors.append(float(abs(-(ep-em)/(2*h)-f[1,0])))
                    row.update(fd_steps_A=[1e-3, 5e-4], fd_force_errors_eV_A=errors)
            except Exception as exc:
                row.update(status='failed', error=repr(exc))
            append(row)


def symmetry_check(atoms, calc):
    # Rotate both the periodic cell and positions: row-vector convention.
    angle = 0.713
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0,0,1]])
    e, f = evaluate(atoms, calc)
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation.T
    rotated.set_cell(atoms.cell.array @ rotation.T)
    er, fr = evaluate(rotated, calc)
    shifted = atoms.copy()
    shifted.positions += [0.217, -0.349, 0.413]
    et, ft = evaluate(shifted, calc)
    perm = np.arange(len(atoms))[::-1]
    ep, fp = evaluate(atoms[perm], calc)
    return dict(rotation_energy_error_eV=abs(er-e), rotation_force_max_eV_A=float(np.abs(fr-f @ rotation.T).max()),
                translation_energy_error_eV=abs(et-e), translation_force_max_eV_A=float(np.abs(ft-f).max()),
                permutation_energy_error_eV=abs(ep-e), permutation_force_max_eV_A=float(np.abs(fp-f[perm]).max()),
                reference_force_scale_eV_A=float(np.abs(f).max()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--full', action='store_true', help='Select every archived frame instead of the screening subset')
    p.add_argument('--plan', type=Path, default=ROOT / 'configs/reliability_plan.json')
    p.add_argument('--backend', choices=['mace', 'nequip', 'deepmd'])
    p.add_argument('--model', type=Path)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--output', type=Path)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.prepare:
        if args.plan.exists():
            p.error('Plan already exists; choose a new --plan path')
        plan = make_plan(full=args.full)
        dump(args.plan, plan)
        print(json.dumps(dict(cases=len(plan['cases']), inventory=plan['inventory']), indent=2))
        return
    if not args.backend or args.output is None:
        p.error('--backend and --output required for execution')
    if args.threads <= 0:
        p.error('--threads must be positive')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if digest(ARCHIVE) != plan['archive_sha256']:
        raise ValueError('Archive hash differs from the frozen evaluation plan')
    model = (args.model or checkpoint(args.backend)).resolve()
    identity = dict(backend=args.backend, model_path=str(model), checkpoint_sha256=digest(model),
                    head='OMat24' if args.backend == 'deepmd' else None, device=args.device,
                    plan_sha256=digest(args.plan), runner_sha256=digest(Path(__file__)),
                    helpers_sha256=digest(ROOT/'scripts/reliability_data.py'), threads=args.threads)
    out = args.output
    done = set()
    if args.resume:
        old = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if any(old.get(k) != v for k,v in identity.items()):
            raise ValueError('Resume identity differs from the original run')
        for name in ['predictions.jsonl', 'checks.jsonl']:
            if (out/name).exists():
                for line in (out/name).read_text(encoding='utf-8').splitlines():
                    done.add(json.loads(line)['id'])
    else:
        out.mkdir(parents=True, exist_ok=False)
    manifest = dict(**identity, status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                    platform=platform.platform(), python=platform.python_version(), archive_sha256=plan['archive_sha256'],
                    expected_reference_cases=len(plan['cases']), protocol=plan['protocol'],
                    uncertainty=plan['uncertainty'], training_overlap='unknown',
                    limitations=['Source groups are not established independent replicates',
                                 'QSD energy spans are constrained-path diagnostics, not NEB activation barriers',
                                 'Chemical C/F and Si/O subsets do not identify projectile provenance',
                                 'Pair scans have no DFT references; no electronic charge or stopping model'])
    dump(out/'manifest.json', manifest)
    start = time.perf_counter()
    try:
        calc, meta = calculator(args.backend, model, args.device, args.threads)
        manifest.update(meta)
        dump(out/'manifest.json', manifest)
        with (out/'predictions.jsonl').open('a', encoding='utf-8') as predictions, (out/'checks.jsonl').open('a', encoding='utf-8') as checks:
            def save(stream, row):
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
            symmetry_tracks = set()
            for number, (case, atoms) in enumerate(selected_atoms(plan), 1):
                if case['id'] not in done:
                    row = dict(case, symbols=atoms.get_chemical_symbols(), positions_A=atoms.positions.tolist(),
                               cell_A=atoms.cell.tolist(), pbc=atoms.pbc.tolist())
                    try:
                        row['energy_ref_eV'] = float(atoms.get_potential_energy())
                        ref = atoms.get_forces()
                        if not np.isfinite(row['energy_ref_eV']) or not np.isfinite(ref).all():
                            raise ValueError('Nonfinite DFT reference')
                        if atoms.info.get('converged') is False or atoms.info.get('converged') == np.bool_(False):
                            raise ValueError('Reference marked unconverged')
                        tick = time.perf_counter()
                        e, f = evaluate(atoms, calc)
                        near, contact = local_geometry(atoms)
                        row.update(status='ok', energy_pred_eV=e, forces_ref_eV_A=ref.tolist(), forces_pred_eV_A=f.tolist(),
                                   nearest_distance_A=near.tolist(), cf_sio_contact_A=contact, seconds=time.perf_counter()-tick,
                                   metrics=force_stats(f-ref))
                        if case['track'] == 'qsd':
                            row['distance_A'] = float(atoms.info['dist'])
                            row['archive_e_diff_eV'] = float(atoms.info['e_diff'])
                    except Exception as exc:
                        row.update(status='failed', error=repr(exc))
                    save(predictions, row)
                if case['track'] not in symmetry_tracks:
                    symmetry_tracks.add(case['track'])
                    key = 'symmetry:' + case['id']
                    if key not in done:
                        check = dict(id=key, kind='symmetry', case=case['id'])
                        try:
                            check.update(status='ok', **symmetry_check(atoms, calc))
                        except Exception as exc:
                            check.update(status='failed', error=repr(exc))
                        save(checks, check)
                if number % 20 == 0:
                    print(f'{args.backend} reference {number}/{len(plan["cases"])} elapsed {time.perf_counter()-start:.1f}s', flush=True)
            physical_checks(calc, lambda row: save(checks, row), done)
        rows = [json.loads(line) for line in (out/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        checks = [json.loads(line) for line in (out/'checks.jsonl').read_text(encoding='utf-8').splitlines()]
        failures = sum(r['status'] != 'ok' for r in rows + checks)
        manifest.update(status='completed_with_failures' if failures else 'completed',
                        reference_completed=sum(r['status']=='ok' for r in rows), failed_cases=failures,
                        checks_attempted=len(checks))
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest.update(wall_seconds_this_invocation=time.perf_counter()-start, finished_utc=datetime.now(timezone.utc).isoformat())
        dump(out/'manifest.json', manifest)


if __name__ == '__main__':
    main()
