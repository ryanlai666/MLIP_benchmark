# Atomistic visualization and periodic boundaries

The primary movies now cover the [2 ps impact event](../results/long-md/interface-event-gpu/README.md) and the [2 ps crystal comparison](../results/long-md/crystal_comparison.mp4). The earlier 50 fs impact and 100 fs crystal movies are archived smoke tests, not the main event view.

ASE reads the actual trajectories; PyVista/VTK renders ball-and-stick geometry. Every view identifies the periodic axes and cell dimensions. A blue wireframe uses all twelve edges of the actual cell vectors, including support for triclinic cells. Red/green/blue arrows indicate the a/b/c vectors.

- Solid atoms belong to one wrapped simulation cell. Faded atoms are real periodic images in a 2.8 A display halo. Crystals show images in x/y/z. Slabs show lateral x/y images; the z-periodic continuation lies across the vacuum and the complete z cell boundary is visible in the overview.
- Bonds use 1.15 times summed ASE covalent radii. Image atoms allow genuine short bonds across periodic faces to be drawn without drawing a long line through the box. Faded bonds include image atoms. These are proximity guides, not reaction or bond-order assignments.
- Spheres use 0.46 times ASE covalent radii. Colors are Si gold, O red, C slate and F green. Coordinates/displacements are never magnified; wrapping is for display only and trajectory files are unchanged.
- The interface overview shows the whole periodic cell, including vacuum. The second camera follows the original projectile carbon, retaining the departing fragment in view instead of clipping it at the surface close-up. Periodic images are visualization copies, not additional simulated atoms.
- MP4/GIF playback is 10 fps. Interface rendering samples the first 250 fs densely, then uses the selected stride; playback speed is therefore not a constant conversion to physical time. Every frame carries its physical timestamp, and every rendering manifest records the exact frame indices and timestamps.
- Comparison panels are synchronized from rendering manifests, not from an assumed one-to-one correspondence with saved trajectory frames.

The new impact simulation starts from the earlier saved initial positions and velocities, increases the cell height from 43.74 to 100 A, and runs to 2 ps at 0.25 fs per step. The installed DeePMD ASE adapter passes a full periodic cell when any PBC flag is true, so this run deliberately retains all three periodic directions rather than pretending z is nonperiodic. A guard stops the simulation before any atom approaches the top periodic boundary within max(12 A, twice the model cutoff). The event report checks sustained geometric separation and outward motion; it does not claim full substrate equilibration or an etch yield.

```powershell
# New run; use a new --output directory if results already exist.
.venv-deepmd-gpu/Scripts/python.exe scripts/run_impact_event.py
.venv-render/Scripts/python.exe scripts/report_impact_event.py

# Main 2 ps movies and synchronized crystal comparison.
.venv-render/Scripts/python.exe scripts/render_periodic_md.py --names interface-event-gpu mace nequip deepmd --stride 4
.venv-render/Scripts/python.exe scripts/render_comparisons.py

# Optional archived previews, with the same PBC-aware display.
.venv-render/Scripts/python.exe scripts/render_periodic_md.py --root results/short-md --names interface interface-gpu mace nequip deepmd --stride 1
.venv-render/Scripts/python.exe scripts/render_comparisons.py --root results/short-md --interface
.venv-render/Scripts/python.exe scripts/render_comparisons.py --root results/short-md
```

`scripts/render_atomistic.py` is a compatibility entry point for the new renderer and defaults to the long trajectories. Use the existing `.venv-render` environment described by `environments/render-windows-py311.lock.txt`. Manifests record source trajectory hashes, cell/PBC settings, periodic-image axes, camera behavior, timestamps and package versions.
