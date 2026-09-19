# Reactive reliability evaluation

This extension covers short-range response, force-error tails and broader reference conditions using the existing MACE-MP-0b3 medium, NequIP-OAM-S 0.1 and DPA-3.3-1M/OMat24 checkpoints. It does not train models or run new impact MD.

## Frozen reference selection

`configs/reliability_plan.json` records exact archive members/frame IDs, group labels, sampling seed (20260918), selection purpose and the archive SHA-256. The source is the [authors' DFT archive](https://zenodo.org/records/19491140); its embedded README describes each subdirectory. Original references are preserved. Training overlap and exact dispersion treatment remain unresolved.

The screening plan selects 448 frames:

- 249 frames from 48 complete quasi-static drag (QSD) curves. One curve is chosen reproducibly per available projectile/energy/ordered-pair stratum. CF and CF3 source labels cover 10, 20 and 30 eV. Distances within a curve are never split or sampled independently.
- 32 uniformly spaced frames from each CF2/CF3 30 eV etching sequence (64 total).
- 96 fluorocarbon frames: four composition/density combinations, eight frames per archived melt/quench/anneal group.
- 24 melt-quench-anneal bulk frames: eight each for Si, SiC and SiO2.
- 15 additional CF2 frames from the union of the previous three models' top-ten vector-force-error frames. These are an explicitly biased diagnostic set excluded from coverage averages. Prior outliers already in the coverage sample remain labeled as such.

The plan groups related QSD pair curves by source event as well as by complete curve. Etch snapshots remain grouped by their source sequence. Source labels support grouping but do not establish statistical independence. There is no random frame train/test split, test-label calibration, confidence interval, or claim that these data were absent from pretraining.

## Run

Use the existing isolated model environments from the repository root. Output directories must be new. The committed plan is already prepared; do not overwrite it.

```powershell
.venv-mace/Scripts/python.exe scripts/run_reliability.py --backend mace --output results/reliability/mace
.venv-nequip/Scripts/python.exe scripts/run_reliability.py --backend nequip --output results/reliability/nequip
.venv-deepmd-gpu/Scripts/python.exe scripts/run_reliability.py --backend deepmd --device cuda --output results/reliability/deepmd
.venv-deepmd/Scripts/python.exe scripts/check_reliability_devices.py
.venv-mace/Scripts/python.exe scripts/report_reliability.py
.venv-render/Scripts/python.exe scripts/plot_reliability.py
.venv-mace/Scripts/python.exe -m unittest discover -s tests -v
```

DeePMD can alternatively run on CPU using `.venv-deepmd/Scripts/python.exe` and `--device cpu`, with a new output directory. Explicit checkpoint paths are accepted through `--model`. The runner uses local checkpoints only, verifies parameter devices, and records checkpoint/plan/source hashes and actual parameter dtypes. Model inference times are diagnostic and are not controlled speed measurements, especially when jobs overlap.

Each evaluated frame is immediately appended to `predictions.jsonl`, including coordinates, cell, element identities, reference and predicted forces/energies, nearest-neighbor distances, group metadata and status. Physical diagnostics are appended to `checks.jsonl`. Errors remain as failed rows and in report denominators.

To continue a stopped run, repeat its exact command with `--resume`. The runner rejects changed checkpoint, source, device, thread count or plan identities. Already recorded failures are preserved and skipped; rerun failures into a new output directory after diagnosing them. A truncated final JSON line is rejected, rather than silently discarded. Manifest `wall_seconds_this_invocation` covers only the latest invocation.

For a full archive evaluation, prepare a separate plan and separate result directories:

```powershell
.venv-mace/Scripts/python.exe scripts/run_reliability.py --prepare --full --plan configs/reliability_full_plan.json
.venv-mace/Scripts/python.exe scripts/run_reliability.py --plan configs/reliability_full_plan.json --backend mace --output results/reliability-full/mace
# Run the other two backends against that same plan, then:
.venv-mace/Scripts/python.exe scripts/report_reliability.py --plan configs/reliability_full_plan.json --root results/reliability-full --output reports/reliability-full
```

The full archive sweep is implemented but is not the screening run reported here.

## Metrics and interpretation

- **Force tails:** component MAE/RMSE/P95/P99/max and atom-vector P95/max, all recomputed from saved arrays. Track tables use common successful coverage frames across models. Targeted old outliers appear separately. Per-element, chemical-subset and nearest-distance bins expose localized failures. Distances use minimum-image conventions. QSD also has whole-frame errors binned by the archived drag coordinate; that coordinate is not necessarily the global nearest distance.
- **Chemical subsets:** C/F and Si/O are chemistry labels, not projectile/substrate identities. The archive lacks reliable projectile provenance. Nearest-distance bins describe each atom's nearest neighbor; C/F-to-Si/O contact is an additional frame-level distance.
- **QSD:** compare energies relative to the largest-distance frame of each same-composition curve. Report relative-energy MAE/max, energy spans and adjacent energy-slope sign mismatches. No energy offsets are fitted. These constrained paths are not independently validated reaction barriers or minimum-energy pathways. The shortest selected drag distance is 0.142 A; extreme errors should not be conflated with typical thermal force errors. The 249 selected QSD frames have no DFT-convergence flag in the archive, unlike the other selected references. Missing metadata does not prove unconverged DFT. The reporter audits these flags when the local archive is available.
- **Diatomics:** ten unordered Si/O/C/F pairs, eleven separations from 0.5 to 6 A. Record radial force, net force and two-step finite-difference energy/force residuals at 0.8, 1.5 and 3 A. There are no matching DFT pair references; short-range attraction is an inspection flag, not an accuracy score.
- **Symmetries:** rotate both cell and coordinates, translate atoms, and reverse atom order on the first selected frame of each track. Report energy and force residuals with actual precision metadata. These four configurations are plumbing/symmetry checks, not exhaustive invariance validation.
- **Failures:** completing a calculation means finite output, not passing a physics tolerance. Reports show failed attempts and use only matched successes for numerical comparisons. Incomplete QSD curves are omitted from curve metrics.

Raw result directories are local/ignored, as with the earlier pilot. The frozen plan, scripts, compact report tables, plots, worst-frame exports and hashes are versionable. Regenerate or separately transfer raw arrays for independent reproduction. The report and plots consume saved results and never rerun inference. The optional `check_reliability_devices.py` command reruns four pair geometries and three worst QSD frames on CPU to inspect numerical sensitivity of the CUDA DeePMD results.

Actual adsorption/desorption energies, NEB barriers and etch yields require additional matched references and independent dynamics. The current archive supports a meaningful short-range and constrained-path screen; it does not establish those reaction observables.
