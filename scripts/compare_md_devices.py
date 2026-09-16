"""Compare matched CPU/CUDA impact trajectories without asserting speed rankings."""
import csv
import json
from pathlib import Path
import numpy as np
from ase.io import read
from render_md import render

root=Path('results/short-md')
a=root/'interface'
b=root/'interface-gpu'
mc=json.loads((a/'manifest.json').read_text())
mg=json.loads((b/'manifest.json').read_text())
assert mc['status']==mg['status']=='completed'
assert mg['device']=='cuda' and all(x.startswith('cuda') for x in mg['parameter_devices'])
assert mc['checkpoint_sha256']==mg['checkpoint_sha256']
fc=read(a/'trajectory.extxyz',index=':')
fg=read(b/'trajectory.extxyz',index=':')
assert len(fc)==len(fg)==51
assert np.array_equal(fc[0].positions,fg[0].positions)
assert np.allclose(fc[0].get_velocities(),fg[0].get_velocities(),rtol=0,atol=1e-12)
assert all(np.allclose(x.positions[mg['fixed_indices']],fg[0].positions[mg['fixed_indices']]) for x in fg)
with (a/'thermodynamics.csv').open() as f: rc=list(csv.DictReader(f))
with (b/'thermodynamics.csv').open() as f: rg=list(csv.DictReader(f))
assert len(rc)==len(rg)==201
energies=np.array([float(r['total_eV']) for r in rg])
assert np.isfinite(energies).all()
assert np.isclose(np.max(np.abs(energies-energies[0]))/mg['natoms']*1000,mg['max_abs_energy_change_meV_atom'])
comparison=dict(checkpoint_sha256=mc['checkpoint_sha256'],natoms=153,duration_fs=50,
    cpu_wall_seconds=mc['wall_seconds'],gpu_wall_seconds=mg['wall_seconds'],
    initial_energy_absolute_difference_eV=abs(float(rc[0]['total_eV'])-float(rg[0]['total_eV'])),
    final_position_rms_difference_A=float(np.sqrt(np.mean((fc[-1].positions-fg[-1].positions)**2))),
    max_saved_coordinate_difference_A=float(max(np.abs(x.positions-y.positions).max() for x,y in zip(fc,fg))),
    gpu_parameter_devices=mg['parameter_devices'])
(root/'cpu_gpu_comparison.json').write_text(json.dumps(comparison,indent=2))
render('interface'); render('interface-gpu')
lines=['# CPU and GPU surface-impact comparison','',
    'The same 153-atom CF2/silica initial positions, velocities, checkpoint and 50 fs protocol were evaluated sequentially. CUDA parameter placement was checked explicitly.','',
    '| Device | End-to-end wall time (s) | Maximum energy change (meV/atom) |',
    '|---|---:|---:|',
    f"| CPU, 4 Torch threads | {mc['wall_seconds']:.2f} | {mc['max_abs_energy_change_meV_atom']:.5f} |",
    f"| {mg['gpu_name']} | {mg['wall_seconds']:.2f} | {mg['max_abs_energy_change_meV_atom']:.5f} |",'',
    f"Initial total-energy difference: {comparison['initial_energy_absolute_difference_eV']:.6g} eV. Final Cartesian coordinate RMS difference: {comparison['final_position_rms_difference_A']:.6g} Angstrom.",'',
    'Wall times include model initialization and output writing; CUDA is synchronized before final timing. These are single-run timings, not controlled steady-state throughput measurements. CPU/GPU trajectories can diverge numerically, so compare identical input frames for force-accuracy validation.','',
    '![GPU interface trajectory](interface-gpu/animation.gif)','',
    '[CPU animation](interface/animation.gif) / [GPU manifest](interface-gpu/manifest.json) / [Comparison JSON](cpu_gpu_comparison.json)','',
    '## Reproduction','',
    'Use the separate GPU environment; do not replace the tested CPU environment. The PyTorch wheel requires the CUDA index as well as its version pin. See the [official PyTorch installation archive](https://pytorch.org/get-started/previous-versions/).','',
    '```powershell',
    'uv venv --python 3.11.14 .venv-deepmd-gpu',
    'uv pip install --python .venv-deepmd-gpu/Scripts/python.exe -r environments/deepmd-windows-py311.lock.txt',
    'uv pip install --python .venv-deepmd-gpu/Scripts/python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch',
    '.venv-deepmd-gpu/Scripts/python.exe scripts/run_interface_md.py --device cuda --output results/short-md/interface-gpu',
    '.venv-mace/Scripts/python.exe scripts/compare_md_devices.py','```','',
    'Use new output folders for repeat trajectories. The installed GPU package snapshot is in `environments/deepmd-gpu-windows-py311.lock.txt`. All physical caveats in the [MD report](README.md) apply to both devices.','']
(root/'CPU_GPU.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(comparison,indent=2))
