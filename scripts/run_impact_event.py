"""Rerun the neutral CF2 impact in a taller periodic cell to follow departure."""
import argparse
import csv
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
from ase import units
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.md.verlet import VelocityVerlet

from run_pilot import ROOT, sha


def expanded_initial(source, height):
    atoms = read(source/'trajectory.extxyz', index=0)
    previous = json.loads((source/'manifest.json').read_text())
    if not np.all(atoms.pbc) or not np.allclose(atoms.cell.array, np.diag(atoms.cell.lengths())):
        raise ValueError('This slab extension requires an orthorhombic fully periodic source')
    if height <= atoms.cell[2,2] or height <= atoms.positions[:,2].max()+24:
        raise ValueError('New cell must increase the vacuum and leave at least 24 A headroom')
    if 'momenta' not in atoms.arrays:
        raise ValueError('Source must contain the original velocities')
    cell = atoms.cell.array.copy()
    cell[2,2] = height
    atoms.set_cell(cell, scale_atoms=False)
    atoms.set_constraint(FixAtoms(indices=previous['fixed_indices']))
    atoms.info.clear()
    atoms.set_array('origin_atom_id', np.arange(len(atoms)))
    return atoms, previous


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'results/short-md/interface-gpu')
    p.add_argument('--output', type=Path, default=ROOT/'results/long-md/interface-event-gpu')
    p.add_argument('--duration-fs', type=float, default=2000)
    p.add_argument('--timestep-fs', type=float, default=.25)
    p.add_argument('--cell-height-A', type=float, default=100)
    p.add_argument('--record-every', type=int, default=20)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    args = p.parse_args()
    if args.duration_fs <= 0 or args.timestep_fs <= 0 or args.record_every < 1:
        p.error('Duration, timestep and recording interval must be positive')
    steps = round(args.duration_fs/args.timestep_fs)
    if not np.isclose(steps*args.timestep_fs, args.duration_fs):
        p.error('Duration must be an integer number of timesteps')
    os.environ['DEVICE'] = args.device
    import torch
    from deepmd.calculator import DP
    torch.set_num_threads(4)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    atoms, previous = expanded_initial(args.source, args.cell_height_A)
    checkpoint = ROOT/'models/deepmd/DPA-3.3-1M.pt'
    if sha(checkpoint) != previous['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the original impact')
    args.output.mkdir(parents=True, exist_ok=False)
    substrate = previous['substrate_atoms']
    manifest = dict(status='running', model=previous['model'], device=args.device,
                    source_trajectory_sha256=sha(args.source/'trajectory.extxyz'),
                    source_initial_frame=0, checkpoint_sha256=sha(checkpoint), source_script_sha256=sha(Path(__file__)),
                    natoms=len(atoms), substrate_atoms=substrate, carbon_index=substrate,
                    fixed_indices=previous['fixed_indices'], initial_cell_A=atoms.cell.tolist(), pbc=atoms.pbc.tolist(),
                    original_cell_height_A=float(read(args.source/'trajectory.extxyz',index=0).cell[2,2]),
                    requested_duration_fs=args.duration_fs, steps=steps, timestep_fs=args.timestep_fs,
                    record_every=args.record_every, initial_surface_z_A=float(atoms.positions[:substrate,2].max()),
                    boundary_guard_A=12., saved_frames=0,
                    packages={v:importlib.metadata.version(v) for v in ['ase','torch','deepmd-kit','numpy']},
                    protocol='Rerun from identical initial coordinates and velocities; only vacuum height changed. NVE with fixed bottom.',
                    limitations=['Cold unrelaxed substrate; one neutral impact; no charge exchange or electronic stopping',
                                 'The installed DeePMD ASE adapter treats any periodic direction as fully periodic; all three remain periodic',
                                 'Carbon-fragment departure is an event diagnostic, not complete substrate equilibration or an etch yield'])
    mpath = args.output/'manifest.json'
    mpath.write_text(json.dumps(manifest,indent=2))
    start = time.perf_counter()
    totals, saved = [], []
    try:
        atoms.calc = DP(model=str(checkpoint), head='OMat24')
        devices = sorted({str(p.device) for p in atoms.calc.dp.deep_eval.dp.parameters()})
        if not devices or not all(d.startswith(args.device) for d in devices):
            raise RuntimeError(f'Wrong device: {devices}')
        manifest['parameter_devices'] = devices
        cutoff = float(atoms.calc.dp.get_rcut())
        guard = max(12., 2*cutoff)
        manifest.update(cutoff_A=cutoff, boundary_guard_A=guard)
        dyn = VelocityVerlet(atoms, timestep=args.timestep_fs*units.fs)
        with (args.output/'thermodynamics.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['step','time_fs','potential_eV','kinetic_eV','total_eV',
                'carbon_z_A','carbon_vz_A_fs','carbon_min_original_substrate_distance_A','top_clearance_A'])
            writer.writeheader()
            def record():
                ep, ek = atoms.get_potential_energy(), atoms.get_kinetic_energy()
                if not np.isfinite([ep,ek]).all() or not np.isfinite(atoms.positions).all():
                    raise ValueError('Nonfinite MD state')
                clearance = float(atoms.cell[2,2]-atoms.positions[:,2].max())
                totals.append(float(ep+ek))
                row = dict(step=dyn.nsteps,time_fs=dyn.nsteps*args.timestep_fs,potential_eV=ep,kinetic_eV=ek,total_eV=ep+ek,
                           carbon_z_A=float(atoms.positions[substrate,2]),
                           carbon_vz_A_fs=float(atoms.get_velocities()[substrate,2]*units.fs),
                           carbon_min_original_substrate_distance_A=float(atoms.get_distances(substrate,range(substrate),mic=True).min()),
                           top_clearance_A=clearance)
                writer.writerow(row)
                if dyn.nsteps % args.record_every==0 or dyn.nsteps==steps or clearance < guard:
                    frame=atoms.copy()
                    frame.info.update(time_fs=row['time_fs'],carbon_index=substrate,substrate_atoms=substrate)
                    write(args.output/'trajectory.extxyz',frame,append=bool(saved))
                    saved.append(dyn.nsteps)
                    stream.flush()
                if dyn.nsteps % 400==0:
                    write(args.output/'restart.extxyz',atoms)
                    manifest.update(last_step=dyn.nsteps,saved_frames=len(saved))
                    mpath.write_text(json.dumps(manifest,indent=2))
                    print(f"Step {dyn.nsteps}/{steps}: {row['time_fs']:.0f} fs; carbon z={row['carbon_z_A']:.2f} A; top clearance={clearance:.2f} A",flush=True)
                if clearance < guard:
                    raise RuntimeError('Stopped before periodic re-entry: enlarge the vacuum before extending further')
            dyn.attach(record,interval=1)
            dyn.run(steps)
        delta=(np.array(totals)-totals[0])/len(atoms)*1000
        manifest.update(status='completed',duration_fs=steps*args.timestep_fs,saved_frames=len(saved),
                        max_abs_energy_change_meV_atom=float(np.abs(delta).max()),final_energy_change_meV_atom=float(delta[-1]),
                        final_top_clearance_A=float(atoms.cell[2,2]-atoms.positions[:,2].max()))
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc),saved_frames=len(saved))
        raise
    finally:
        manifest['wall_seconds']=time.perf_counter()-start
        mpath.write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
