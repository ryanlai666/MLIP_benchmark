# Atomistic visualization

Trajectories are read with ASE and rendered as shaded ball-and-stick geometry using PyVista/VTK. The interface view combines a slab overview and an enlarged impact-region view. Crystal views display 2x2x2 periodic copies of the eight-atom simulated cell. Coordinates and displacements are not magnified.

- Spheres use 0.46 times ASE covalent radii for visual clarity.
- Bond guides connect displayed atoms separated by less than 1.15 times the sum of covalent radii. They are visual proximity guides, not inferred bond orders or reaction assignments. Display-boundary bonds are omitted.
- The interface close-up uses a fixed selection of atoms whose initial z is above 17 Angstrom; the overview shows all atoms. Vacuum is cropped from the display. The physical simulation cell is unchanged.
- Colors: Si gold, O red, C slate, F green. Camera position is fixed throughout each animation.
- GIF and MP4 frames use actual saved MD configurations at 10 display frames/second. Playback time is unrelated to physical elapsed time. Timestamps show the physical time.
- CPU/GPU and model comparison panels are synchronized by saved physical timestamps. The CPU/GPU comparison focuses on the impact region; each full-size video also includes the whole slab.

The MD trajectories and numerical results are unchanged. Each rendering manifest records the source trajectory SHA256 and rendering package versions.

```powershell
uv venv --python 3.11.14 .venv-render
uv pip install --python .venv-render/Scripts/python.exe -r environments/render-windows-py311.lock.txt
.venv-render/Scripts/python.exe scripts/render_atomistic.py
.venv-render/Scripts/python.exe scripts/render_comparisons.py
```

Sources: [ASE trajectory I/O](https://docs.ase-lib.org/ase/io/io.html), [PyVista mesh rendering](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_mesh), [PyVista animation guidance](https://docs.pyvista.org/examples/02-plot/gif).
