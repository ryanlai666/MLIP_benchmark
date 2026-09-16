# CPU and GPU surface-impact comparison

The same 153-atom CF2/silica initial positions, velocities, checkpoint and 50 fs protocol were evaluated sequentially. CUDA parameter placement was checked explicitly.

| Device | End-to-end wall time (s) | Maximum energy change (meV/atom) |
|---|---:|---:|
| CPU, 4 Torch threads | 193.95 | 0.00862 |
| NVIDIA GeForce RTX 3070 Laptop GPU | 45.11 | 0.00867 |

Initial total-energy difference: 1.81198e-05 eV. Final Cartesian coordinate RMS difference: 4.68786e-08 Angstrom.

Wall times include model initialization and output writing; CUDA is synchronized before final timing. These are single-run timings, not controlled steady-state throughput measurements. CPU/GPU trajectories can diverge numerically, so compare identical input frames for force-accuracy validation.

![Synchronized CPU / GPU interface trajectories](interface_comparison.gif)

[CPU animation](interface/animation.gif) / [GPU manifest](interface-gpu/manifest.json) / [Comparison JSON](cpu_gpu_comparison.json)

## Reproduction

Use the separate GPU environment; do not replace the tested CPU environment. The PyTorch wheel requires the CUDA index as well as its version pin. See the [official PyTorch installation archive](https://pytorch.org/get-started/previous-versions/).

```powershell
uv venv --python 3.11.14 .venv-deepmd-gpu
uv pip install --python .venv-deepmd-gpu/Scripts/python.exe -r environments/deepmd-windows-py311.lock.txt
uv pip install --python .venv-deepmd-gpu/Scripts/python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch
.venv-deepmd-gpu/Scripts/python.exe scripts/run_interface_md.py --device cuda --output results/short-md/interface-gpu
.venv-render/Scripts/python.exe scripts/compare_md_devices.py
```

Use new output folders for repeat trajectories. The installed GPU package snapshot is in `environments/deepmd-gpu-windows-py311.lock.txt`. All physical caveats in the [MD report](README.md) apply to both devices.

Rendering uses ASE + PyVista/VTK. See [visualization setup and conventions](../../docs/VISUALIZATION.md). MP4 videos are available next to each GIF.
