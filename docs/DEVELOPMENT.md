# Development notes

## Current state

The initial sequential benchmark suite uses OMol25 and is complete. Read `reports/PILOT_RESULTS.md`, `docs/RUN_PILOT.md`, `docs/PUBLISHED_BASELINES.md` and `docs/MODELS.md` before extending the project.

- MACE-MP-0b3 medium, NequIP-OAM-S 0.1 and DeePMD DPA-3.3-1M/OMat24 each evaluated the same 12 archived DFT configurations: four Si crystal, two Si surface and six 150-atom SiO2/CFx etching snapshots. Each also completed a nine-point two-atom Si equation of state.
- DPA-3.3-1M/OMol25 separately converged neutral singlet F2 and SiF4 optimizations. This is a molecular geometry check, not a UMA/eSEN run or molecular force-accuracy benchmark.
- Direct ASE evaluation is implemented in `scripts/run_pilot.py`; molecular checks in `scripts/run_omol_pilot.py`. `scripts/report_pilot.py` independently recomputes force aggregates from raw arrays and generates the report/CSV.
- Three isolated Python 3.11.14 environments and exact installed-package snapshots exist. Intel i7-12700H CPU, four Torch threads; RTX 3070 laptop detected but unused. Five unit tests passed.
- Raw successful outputs: `results/pilot-mace-mp-0b3`, `results/pilot-nequip-oam-s`, `results/pilot-deepmd-dpa33-omat-fixed`, `results/pilot-omol25-fragments`. Preserve failed runs as provenance.
- CHIPS-FF remains unmodified at commit `7ab37a5e1ec65ad540dd7c6caaa555bda738d595`. MLearn reference checkout is `10c427a5480c6281c15c64efaf869b03be04818f`. Etch archive checksum is verified by the downloader.

## Next implementation task

Extend the existing evaluator, retaining identical reference frames across models and recording every failure. DPA/OMat24 is the first candidate to extend because it had the lowest force errors on this subset; this is not a conclusion about entire model families.

1. Audit pretraining overlap, dataset licensing, reference energy conventions and whether archived etch labels include D3. Preserve original splits and trajectory grouping. Do not describe these samples as confirmed unseen data.
2. Expand Si and etch reference coverage with trajectory-grouped sampling, per-element and impact-distance force errors. Keep thermal Si separate from nearly force-free strained cells. Add uncertainty intervals only with defensible independent sampling units.
3. Evaluate the archive's quasi-static drag/short-range repulsion configurations and reaction barriers before etching MD. Add hotter impacts, slab/timestep convergence and independent trajectories. Neutral projectile snapshots do not model plasma electrons, charge exchange or stopping.
4. Add matched DFT lattice, elastic, surface and defect references before claiming property accuracy. Extend to SiC or GaN with verified chemistry and reference consistency.
5. Repeat timing after shape-specific warmup with matched precision, repeat counts and explicit CPU/GPU synchronization. Current single-call timings include lazy initialization and are not controlled speed rankings.
6. If training is needed, use identical fixed train/validation/test groups, declared compute/tuning budgets and multiple seeds for all families. Select models only on validation data.

## Compatibility notes

- DeePMD 3.2.0 Windows wheel requires Torch 2.11.0 here; Torch 2.14 failed with an ABI mismatch. e3nn 0.6.0 is also required by this checkpoint.
- NequIP 0.19.1 pilot uses private `_from_saved_model` for eager packaged-model inference. Preserve the version pin; move production inference to the documented public compiled workflow and validate numerical agreement.
- CHIPS-FF's pinned MACE/CHGNet adapters hardcode some calculator settings and initialization fetches JARVIS even for local structures. Its local inputs have no independent property references. Original wrappers are preparation scaffolds; they did not produce this report.
- Version snapshots are Windows environment records, not portable hashed lockfiles. Preserve raw manifests and checkpoint hashes when moving platforms. Use new output directories; do not overwrite earlier runs.

Acceptance for each extension: reproducible commands, reference provenance, finite predictions, explicit convergence/failure reporting, independently checked metrics and an updated report separating measured results from published claims.

## Short MD extension

See `results/short-md/README.md` for the committed crystal and surface-interface trajectories, animations and energy diagnostics. `scripts/run_short_md.py` runs the three crystal models separately; `scripts/run_interface_md.py` constructs a neutral 30 eV CF2 impact on silica with DPA/OMat24; `scripts/render_md.py` checks saved energy aggregates and renders actual coordinates. The impact is exploratory with a cold, unrelaxed slab and fixed bottom. Validate timestep, substrate equilibration, projectile orientation, cell size and reference chemistry before extending its duration or interpreting etching mechanisms.

The interface trajectory was also completed with CUDA in `.venv-deepmd-gpu`, with verified `cuda:0` model parameters. See `results/short-md/CPU_GPU.md` and `scripts/compare_md_devices.py` for matching-input checks, numerical differences and the GPU environment snapshot. Crystal MD remains CPU-only.
