"""Constructed neutral CF2 impact on an archived silica slab; exploratory only."""
import argparse
import os
import importlib.metadata
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from ase import Atoms, units
from ase.constraints import FixAtoms
from ase.io import write
from ase.md.verlet import VelocityVerlet
from run_pilot import samples, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, default=Path('results/short-md/interface'))
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--timestep-fs', type=float, default=0.25)
    parser.add_argument('--record-every', type=int, default=4, help='Steps between saved trajectory frames')
    args = parser.parse_args()
    os.environ['DEVICE'] = args.device
    from deepmd.calculator import DP
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    ref, slab = next((r,a) for r,a in samples() if r['id'] == 'trj_CF2_30eV-0')
    slab.calc = None
    surface = float(slab.positions[:,2].max())
    angle = np.deg2rad(105/2)
    fragment = Atoms('CF2', positions=[[5,5,surface+3], [5+1.3*np.cos(angle),5+1.3*np.sin(angle),surface+3], [5+1.3*np.cos(angle),5-1.3*np.sin(angle),surface+3]])
    atoms = slab + fragment
    fixed = np.where(atoms.positions[:len(slab),2] < slab.positions[:,2].min()+2)[0]
    atoms.set_constraint(FixAtoms(indices=fixed))
    velocities = np.zeros((len(atoms),3))
    velocities[len(slab):,2] = -np.sqrt(2*30/fragment.get_masses().sum())
    atoms.set_velocities(velocities)
    checkpoint = Path('models/deepmd/DPA-3.3-1M.pt')
    manifest = dict(status='running', model='DPA-3.3-1M / OMat24', checkpoint_sha256=sha(checkpoint),
                    source=ref, natoms=len(atoms), substrate_atoms=len(slab), projectile='neutral CF2',
                    projectile_translation_eV=30, initial_gap_A=3, CF_bond_A=1.3, FCF_angle_deg=105,
                    fixed_indices=fixed.tolist(), timestep_fs=args.timestep_fs, steps=args.steps,
                    duration_fs=args.steps*args.timestep_fs, record_every=args.record_every,
                    device=args.device, torch_threads=4,
                    packages={p:importlib.metadata.version(p) for p in ['torch','deepmd-kit','ase','numpy']},
                    cuda_runtime=torch.version.cuda,
                    gpu_name=torch.cuda.get_device_name(0) if args.device=='cuda' else None, substrate_initial_temperature_K=0,
                    limitations=['Constructed initial projectile, not an archived dynamical continuation',
                                 'Cold, unrelaxed substrate; frozen bottom; no thermostat',
                                 'Periodic source cell retained including vacuum; no electrons or charge exchange',
                                 'One short exploratory trajectory; not validated etch yield or reaction dynamics'])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    start = time.perf_counter()
    try:
        atoms.calc = DP(model=str(checkpoint.resolve()), head='OMat24')
        devices = sorted({str(p.device) for p in atoms.calc.dp.deep_eval.dp.parameters()})
        manifest['parameter_devices'] = devices
        assert devices and all(d.startswith(args.device) for d in devices), devices
        if args.device == 'cuda': torch.cuda.synchronize()
        dyn = VelocityVerlet(atoms, timestep=args.timestep_fs*units.fs)
        rows, frames = [], []
        def record():
            ep, ek = atoms.get_potential_energy(), atoms.get_kinetic_energy()
            if not np.isfinite([ep,ek]).all() or not np.isfinite(atoms.positions).all():
                raise ValueError('Nonfinite trajectory')
            distances = atoms.get_all_distances(mic=True)[len(slab):,:len(slab)]
            rows.append(dict(step=dyn.nsteps, time_fs=dyn.nsteps*args.timestep_fs, potential_eV=ep,
                             kinetic_eV=ek, total_eV=ep+ek, projectile_min_substrate_distance_A=float(distances.min()),
                             carbon_z_A=float(atoms.positions[len(slab),2])))
            if dyn.nsteps % args.record_every == 0:
                frame=atoms.copy(); frame.info['time_fs']=dyn.nsteps*args.timestep_fs; frames.append(frame)
            if dyn.nsteps % max(40, args.steps//20) == 0: print('Interface step',dyn.nsteps,flush=True)
        dyn.attach(record,interval=1); dyn.run(args.steps)
        with (out/'thermodynamics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        write(out/'trajectory.extxyz',frames)
        if args.device == 'cuda': torch.cuda.synchronize()
        delta=(np.array([r['total_eV'] for r in rows])-rows[0]['total_eV'])/len(atoms)*1000
        manifest.update(status='completed', saved_frames=len(frames),wall_seconds=time.perf_counter()-start,
                        max_abs_energy_change_meV_atom=float(np.abs(delta).max()),
                        final_energy_change_meV_atom=float(delta[-1]),
                        minimum_projectile_substrate_distance_A=min(r['projectile_min_substrate_distance_A'] for r in rows))
    except Exception as e:
        manifest.update(status='failed',error=repr(e)); raise
    finally: (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='source'},indent=2))


if __name__=='__main__': main()
