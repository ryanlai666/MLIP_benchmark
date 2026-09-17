"""Render actual saved MD coordinates and report their numerical diagnostics."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.io import read

ROOT=Path('results/short-md')
COLORS={'Si':'#e5ae59','O':'#e15b64','C':'#545f70','F':'#57c9a5'}


from render_atomistic import render


def main():
    names=['mace','nequip','deepmd','interface']
    manifests=[]
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for name in names:
        folder=ROOT/name
        m=json.loads((folder/'manifest.json').read_text())
        assert m['status']=='completed'
        with (folder/'thermodynamics.csv').open() as f: rows=list(csv.DictReader(f))
        t=np.array([float(r['time_fs']) for r in rows])
        e=np.array([float(r['total_eV']) for r in rows])
        assert len(rows)==m['steps']+1 and np.isfinite(e).all()
        drift=(e-e[0])/m['natoms']*1000
        assert np.isclose(np.abs(drift).max(),m['max_abs_energy_change_meV_atom'])
        assert np.isclose(t[-1],m['duration_fs'])
        axes[1 if name=='interface' else 0].plot(t,drift,label=name)
        render(name); manifests.append(m)
    for ax,label in zip(axes,['Si crystals','CF2 / silica interface']):
        ax.set(title=label,xlabel='Time (fs)',ylabel='Energy change (meV/atom)'); ax.grid(alpha=.2); ax.legend()
    fig.savefig(ROOT/'energy_conservation.png',dpi=160); plt.close(fig)
    text=['# Short molecular dynamics results','',
          'Three crystal checks and one surface-interface trajectory were run sequentially on CPU. These are short numerical/visual demonstrations, not equilibrium sampling or validated etching predictions.','',
          '| System | Model | Atoms | Time (fs) | Step (fs) | Max absolute energy change (meV/atom) |',
          '|---|---|---:|---:|---:|---:|']
    for name,m in zip(names,manifests):
        text.append(f"| {name} | {m['model']} | {m['natoms']} | {m['duration_fs']} | {m['timestep_fs']} | {m['max_abs_energy_change_meV_atom']:.5f} |")
    text += ['', '[Matched CPU/GPU comparison and GPU animation](CPU_GPU.md)', '',
             'Twenty-times-longer runs of these same systems, and a comparison against every archived DFT frame of the etching reference trajectory, are in [the longer-trajectory report](../long-md/README.md).', '',
             '## Surface-interface trajectory','',
             'A constructed neutral CF2 projectile starts 3 Å above the archived silica slab, with 30 eV translational energy toward the surface. The substrate initially has zero velocity and is not relaxed; atoms within 2 Å of its bottom are frozen. Original periodic boundaries and vacuum are retained. CF2 starts at 1.3 Å C–F distance and 105° F–C–F angle. This is a new constructed impact, not a continuation with the original dataset velocities.', '',
             f"Minimum distance between an original projectile atom and substrate during this trajectory: {manifests[-1]['minimum_projectile_substrate_distance_A']:.3f} Å. Atom identity is tracked even if bonding changes; no etch yield or reaction assignment is inferred.", '',
             '![CPU / GPU interface comparison](interface_comparison.gif)', '',
             '## Crystal trajectories','',
             'Eight-atom periodic diamond Si cells use each model’s earlier EOS lattice prediction, the same random seed, 300 K initial kinetic temperature, and removed center-of-mass motion. No thermostat or equilibration is applied. ASE temperature uses its 3N convention. Visualizations repeat the simulated cell 2×2×2 without magnifying motion.','',
             '![Crystal model comparison](crystal_comparison.gif)', '',
             '[NequIP animation](nequip/animation.gif) · [DeePMD animation](deepmd/animation.gif)', '',
             '## Numerical diagnostics and reproduction','',
             '![Energy conservation](energy_conservation.png)','',
             'Each subfolder includes the actual saved trajectory (`trajectory.extxyz`), per-step energy CSV, manifest, initial/final PNGs and GIF. Energy changes are relative to the first recorded step, not fitted slopes. Crystal wall times include loading and cannot establish a speed ranking.','',
             'Integration uses [ASE Velocity Verlet](https://docs.ase-lib.org/_modules/ase/md/verlet.html). The surface reference is from the [etching dataset](https://zenodo.org/records/19491140); archive and checkpoint hashes are recorded in the manifest. Short-range, timestep, cell-size and charge-state validation remain necessary for quantitative interface predictions. OMol25 is not applied to periodic surfaces.','',
             'Run from the repository root with the existing isolated environments, in this order (output folders must not already exist):','', '```powershell',
             '.venv-mace/Scripts/python.exe scripts/run_short_md.py --backend mace --output results/short-md/mace',
             '.venv-nequip/Scripts/python.exe scripts/run_short_md.py --backend nequip --output results/short-md/nequip',
             '.venv-deepmd/Scripts/python.exe scripts/run_short_md.py --backend deepmd --output results/short-md/deepmd',
             '.venv-deepmd/Scripts/python.exe scripts/run_interface_md.py',
             '.venv-render/Scripts/python.exe scripts/render_md.py', '```','']
    (ROOT/'README.md').write_text('\n'.join(text),encoding='utf-8')
    print('Verified four trajectories and rendered results/short-md')


if __name__=='__main__': main()
