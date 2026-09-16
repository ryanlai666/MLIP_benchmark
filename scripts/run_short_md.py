"""Small sequentially launched Si NVE checks; no thermostat or equilibration."""
import argparse
import csv
import json
import time
import importlib.metadata
from pathlib import Path
from run_pilot import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['mace', 'nequip', 'deepmd'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from ase import units
    from ase.build import bulk
    from ase.io import write
    from ase.md.verlet import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    torch.set_num_threads(4)
    lattice = {'mace': 5.4709021605, 'nequip': 5.4702, 'deepmd': 5.4504}[args.backend]
    atoms = bulk('Si', 'diamond', a=lattice, cubic=True)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, force_temp=True, rng=np.random.RandomState(20260915))
    Stationary(atoms, preserve_temperature=True)
    manifest = dict(backend=args.backend, status='running', natoms=len(atoms), lattice_A=lattice,
                    ensemble='NVE', timestep_fs=0.5, steps=200, duration_fs=100, initial_temperature_K=300,
                    seed=20260915, device='cpu', torch_threads=4, equilibration_steps=0,
                    temperature_convention='ASE 3N degrees of freedom; center-of-mass motion removed',
                    packages={n: importlib.metadata.version(n) for n in ['ase', 'torch', 'numpy']})
    out = args.output / 'manifest.json'
    out.write_text(json.dumps(manifest, indent=2))
    start = time.perf_counter()
    try:
        if args.backend == 'mace':
            from mace.calculators import mace_mp
            atoms.calc = mace_mp(model='medium-0b3', device='cpu', default_dtype='float64', dispersion=False)
            checkpoint = Path('models/cache/mace/macemp0b3mediummodel')
            manifest['model'] = 'MACE-MP-0b3 medium'
        elif args.backend == 'nequip':
            from nequip.integrations.ase import NequIPCalculator
            atoms.calc = NequIPCalculator._from_saved_model(model_path='nequip.net:mir-group/NequIP-OAM-S:0.1', device='cpu', chemical_species_to_atom_type_map=True)
            checkpoint = Path('models/nequip-cache/b39e883d722e2108867a0b8028e90beeccf2e50c944c9063ef511711984a17fe.nequip.zip')
            manifest['model'] = 'NequIP-OAM-S 0.1'
        else:
            from deepmd.calculator import DP
            checkpoint = Path('models/deepmd/DPA-3.3-1M.pt')
            atoms.calc = DP(model=str(checkpoint.resolve()), head='OMat24')
            manifest['model'] = 'DPA-3.3-1M / OMat24'
        manifest['checkpoint_sha256'] = sha(checkpoint)
        rows, frames = [], []
        dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)
        def record():
            ep, ek = atoms.get_potential_energy(), atoms.get_kinetic_energy()
            forces = atoms.get_forces()
            if not np.isfinite([ep, ek]).all() or not np.isfinite(forces).all():
                raise ValueError('Nonfinite MD state')
            rows.append(dict(step=dyn.nsteps, time_fs=dyn.nsteps * 0.5, temperature_K=atoms.get_temperature(),
                             potential_eV=ep, kinetic_eV=ek, total_eV=ep+ek,
                             max_force_eV_A=float(np.linalg.norm(forces, axis=1).max())))
            if dyn.nsteps % 5 == 0:
                frame = atoms.copy()
                frame.info.update(time_fs=dyn.nsteps * 0.5, temperature_K=atoms.get_temperature())
                frames.append(frame)
        dyn.attach(record, interval=1)
        dyn.run(200)
        with (args.output / 'thermodynamics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        write(args.output / 'trajectory.extxyz', frames)
        delta = (np.array([r['total_eV'] for r in rows]) - rows[0]['total_eV']) / len(atoms) * 1000
        manifest.update(status='completed', saved_frames=len(frames), wall_seconds=time.perf_counter()-start,
                        final_temperature_K=rows[-1]['temperature_K'], final_energy_change_meV_atom=float(delta[-1]),
                        max_abs_energy_change_meV_atom=float(np.abs(delta).max()))
    except Exception as e:
        manifest.update(status='failed', error=repr(e)); raise
    finally:
        out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
