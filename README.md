# Semiconductor MLIP benchmark

A completed small CPU pilot compares MACE-MP-0b3, NequIP-OAM-S and DeePMD DPA-3.3-1M on Si crystals, Si surfaces and SiO2/CFx etching DFT snapshots. The DPA OMol25 branch separately optimizes F2 and SiF4. Models ran sequentially in isolated environments; no training was performed.

Start with [measured results](reports/PILOT_RESULTS.md), the [longer trajectories and full reference-trajectory validation](results/long-md/README.md), [reproduction instructions](docs/RUN_PILOT.md), [published performance and limitations](docs/PUBLISHED_BASELINES.md), and [development notes](docs/DEVELOPMENT.md). [Model identities](docs/MODELS.md) distinguish architectures, checkpoints and dataset heads.

## Reactive reliability: short-range checks and broader coverage

The [reactive reliability report](reports/reliability/README.md) compares the same three checkpoints on a frozen 448-frame screening plan: 48 complete DFT quasi-static drag curves, CF2/CF3 etching sequences, fluorocarbon bulk, and Si/SiC/SiO2 bulk configurations. Additional pair scans, symmetry checks, and force/energy finite differences probe physical consistency. This is a sampled screen, not the full archive evaluation.

[Force-error tails and worst structures](reports/reliability/README.md#largest-etching-force-errors), [DFT drag-curve comparisons](reports/reliability/qsd_curves.csv), and [per-element/contact-distance metrics](reports/reliability/stratified_forces.csv) expose failures hidden by average errors. Previously identified worst frames are explicitly labeled and excluded from coverage averages. QSD energy differences are evaluated without fitting energy offsets; they are constrained paths, not reaction activation barriers. Source grouping is preserved, and no confidence intervals or independent-replicate claims are made.

[Reproduction and methodology](docs/RELIABILITY.md) / [frozen frame-selection plan](configs/reliability_plan.json). Earlier pilot and MD results are preserved.

## Longer trajectories and DFT ground truth

The short runs were extended twentyfold and the snapshot comparison was widened from six frames to a whole reference trajectory. A **1 ps CF2 impact on silica** (153 atoms, DPA-3.3-1M/OMat24, CUDA, 4000 steps in 433 s) and **2 ps Si NVE trajectories** for MACE, NequIP and DeePMD are in [longer trajectories and reference validation](results/long-md/README.md), together with single points from every checkpoint against **all 1000 archived DFT frames** of the CF2 30 eV etching reference trajectory.

| Checkpoint | Force MAE vs DFT (eV/A) | Force RMSE (eV/A) | Raw energy MAE (meV/atom) | Offset-fitted energy MAE (meV/atom) |
|---|---:|---:|---:|---:|
| MACE-MP-0b3 medium | 0.1743 | 0.2994 | 12.16 | 6.71 |
| NequIP-OAM-S 0.1 | 0.2359 | 0.3566 | 23.23 | 7.87 |
| DPA-3.3-1M / OMat24 | 0.1255 | 0.2217 | 9.37 | 2.94 |

Pooled over 1000 frames and 150,037 atoms, these confirm the six-frame pilot ordering on this one etching sequence; the six pilot frames are a subset, so the two are consistent rather than independent. The archived labels are DFT single points on snapshots another model generated, so they measure agreement along the reference chemistry, not that any of these models would produce that trajectory. Per-frame metrics, per-element errors and both energy conventions are in [the report](results/long-md/README.md) and [`reports/etch_validation_metrics.csv`](reports/etch_validation_metrics.csv).

![CF2 impact on silica, 1 ps](results/long-md/interface-gpu/animation.gif)

The 1 ps impact reaches 4.25 A below the initial surface height at 113 fs and then returns upward; by 1000 fs the projectile carbon is near the top of the retained periodic cell, which bounds how much further this cell can honestly be run.

## Surface-interface and crystal MD

A **50 fs neutral CF2 impact on silica** (153 atoms, 30 eV, DeePMD DPA-3.3/OMat24, CPU and CUDA) and **100 fs Si crystal trajectories** for MACE, NequIP and DeePMD are saved in [short MD results](results/short-md/README.md), with actual trajectories, energy logs, manifests and animations. These are exploratory short runs; the impact uses a cold, unrelaxed slab with a fixed bottom and does not establish etch yields or plasma accuracy.

[CPU/GPU comparison](results/short-md/CPU_GPU.md) / [GPU animation](results/short-md/interface-gpu/animation.gif).

![Synchronized CPU and GPU interface comparison](results/short-md/interface_comparison.gif)

The side-by-side interface animation compares CPU and GPU at identical physical timestamps. ASE/PyVista ball-and-stick renderings use actual atom coordinates; the full-size videos include both the whole slab and the impact close-up. Bond lines are distance-based visual guides, not chemical reaction assignments. Colors: Si gold, O red, C gray, F green. [Side-by-side MP4](results/short-md/interface_comparison.mp4) / [CPU MP4 video](results/short-md/interface/animation.mp4) / [GPU MP4 video](results/short-md/interface-gpu/animation.mp4) / [Visualization details](docs/VISUALIZATION.md).

[Initial structure](results/short-md/interface/initial.png) / [Final structure](results/short-md/interface/final.png) / [Energy diagnostics](results/short-md/energy_conservation.png).

![Synchronized MACE, NequIP and DeePMD crystal comparison](results/short-md/crystal_comparison.gif)

[Crystal comparison MP4](results/short-md/crystal_comparison.mp4). The three panels show the same physical timestamp. Crystal display: eight simulated atoms with 2x2x2 periodic copies, without magnifying displacements. [NequIP animation](results/short-md/nequip/animation.gif) / [DeePMD animation](results/short-md/deepmd/animation.gif).

The pilot contains four crystal, two surface and six etching frames per materials model, plus a two-atom Si equation of state. Snapshot force accuracy is measured against archived DFT labels. Equation-of-state values are predictions without matched DFT property validation. These small samples do not establish production plasma accuracy, etch rates or a general model ranking.

Native Windows CPU inference succeeded with separate Python 3.11.14 environments. Tested package snapshots are in `environments/`; CUDA inference is also tested for the DeePMD surface-impact case on an RTX 3070 Laptop GPU. Raw predictions, manifests, checkpoint/data hashes and optimization trajectories are under `results/`. The compact `results/short-md/` collection is committed to Git. Other raw benchmark results, downloaded data, weights and environments remain local and need separate transfer or regeneration.

The pinned, unmodified CHIPS-FF repository at `upstream/chipsff` supports the broader research plan. The measured pilot uses direct ASE adapters in `scripts/run_pilot.py`, not CHIPS-FF. Original `configs/`, `model_configs/`, `smoke_model.py` and `requirements-mace.txt` are earlier scaffolds, not the executed pilot environment specification.

```powershell
python -m unittest discover -s tests -v
python scripts/report_pilot.py
```

See [broader research design](docs/RESEARCH_PLAN.md) and [initial repository sources](docs/SOURCES.md). Preserve upstream code, data and checkpoint attribution and license terms.
