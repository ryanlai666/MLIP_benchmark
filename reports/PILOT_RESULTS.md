# Small semiconductor MLIP pilot

Completed sequentially on 2026-09-15 America/Chicago (2026-09-16 UTC). All three materials checkpoints completed the same 12 single-point reference configurations. The DPA OMol25 branch separately completed two gas-phase optimizations. No models were trained or fine-tuned.

## Force accuracy measured here

Component-wise MAE/RMSE pool all 3N Cartesian components within each subset; units are eV/Å. These diagnostic subsets are too small for a general ranking or statistical confidence intervals.

| Checkpoint | Subset | Frames | Atoms/frame | Force MAE | Force RMSE |
|---|---|---:|---|---:|---:|
| MACE-MP-0b3 medium | Si crystal | 4 | 64 | 0.0453 | 0.0798 |
| MACE-MP-0b3 medium | Si surfaces | 2 | 24/36 | 0.1421 | 0.1967 |
| MACE-MP-0b3 medium | SiO2/CFx etching | 6 | 150 | 0.1819 | 0.2964 |
| NequIP-OAM-S 0.1 | Si crystal | 4 | 64 | 0.0735 | 0.1322 |
| NequIP-OAM-S 0.1 | Si surfaces | 2 | 24/36 | 0.1734 | 0.2525 |
| NequIP-OAM-S 0.1 | SiO2/CFx etching | 6 | 150 | 0.2237 | 0.3366 |
| DPA-3.3-1M / OMat24 | Si crystal | 4 | 64 | 0.0223 | 0.0392 |
| DPA-3.3-1M / OMat24 | Si surfaces | 2 | 24/36 | 0.1072 | 0.1336 |
| DPA-3.3-1M / OMat24 | SiO2/CFx etching | 6 | 150 | 0.1191 | 0.1914 |

The crystal subset comprises two 300 K AIMD frames and two strained cells. The nearly force-free strained cells lower the pooled force error; the thermal-only values are:

- MACE-MP-0b3 medium: thermal-only Si force MAE **0.0905 eV/Å** (two frames).
- NequIP-OAM-S 0.1: thermal-only Si force MAE **0.1470 eV/Å** (two frames).
- DPA-3.3-1M / OMat24: thermal-only Si force MAE **0.0445 eV/Å** (two frames).

The crystal indices are 13, 14, 19 and 20 of the original MLearn Si test split; surface indices are 7 and 8. Etching frames are 0, 500 and 999 from each of the CF2/CF3 30 eV trajectories. These are archived DFT single points on author-generated trajectories, not new plasma MD or etch-yield calculations.

## Timing and raw energy differences

| Checkpoint | Subset | Median wall time/frame (s) | Raw energy MAE (meV/atom) |
|---|---|---:|---:|
| MACE-MP-0b3 medium | Si crystal | 1.1355 | 38.56 |
| MACE-MP-0b3 medium | Si surfaces | 0.4602 | 21.42 |
| MACE-MP-0b3 medium | SiO2/CFx etching | 2.4974 | 22.73 |
| NequIP-OAM-S 0.1 | Si crystal | 0.0133 | 65.88 |
| NequIP-OAM-S 0.1 | Si surfaces | 0.0100 | 67.34 |
| NequIP-OAM-S 0.1 | SiO2/CFx etching | 0.0251 | 27.85 |
| DPA-3.3-1M / OMat24 | Si crystal | 0.9787 | 6.29 |
| DPA-3.3-1M / OMat24 | Si surfaces | 0.0985 | 5.52 |
| DPA-3.3-1M / OMat24 | SiO2/CFx etching | 0.7359 | 8.46 |

All runs used CPU inference with torch.set_num_threads(4); the detected RTX 3070 was not used. MACE used float64, NequIP checkpoint metadata specify float32, and DeePMD retained native checkpoint precision. Each calculator was warmed on a two-atom Si cell; each reference frame was evaluated once. These times can include shape-dependent lazy initialization, especially DeePMD's first larger cells. They are end-to-end pilot timings, not matched-precision steady-state speed rankings. Load/download time is recorded separately in manifests.

Raw energy errors have no fitted offset. They may include XC, pseudopotential, dispersion and energy-zero differences and should not be the sole selection metric. The etching paper describes spin-polarized PBE/PAW, 520 eV and Gamma-point calculations; exact archived D3 inclusion still needs confirmation. MACE and NequIP here do not explicitly condition on the reference magnetic moment. See [reference audit and published results](../docs/PUBLISHED_BASELINES.md).

## Two-atom Si equation of state

Nine unrelaxed diamond-cell volume points, isotropic lattice scale 0.96–1.04 about 5.43 Å. These are **model predictions**, not errors against an independently converged DFT equation of state.

| Checkpoint | Equilibrium a (Å) | Bulk modulus (GPa) | Minimum bracketed |
|---|---:|---:|---|
| MACE-MP-0b3 medium | 5.4709 | 81.28 | True |
| NequIP-OAM-S 0.1 | 5.4702 | 84.51 | True |
| DPA-3.3-1M / OMat24 | 5.4504 | 82.36 | True |

## OMol25 gas-phase check

Used **DPA-3.3-1M / OMol25**, not UMA or eSEN. F2 and SiF4 were optimized sequentially with nonperiodic boundaries, charge 0, multiplicity 1, BFGS and fmax=0.005 eV/Å. This is a fragment geometry comparison, not periodic surface or reaction-barrier validation.

| Molecule | Atoms | Model bond (Å) | NIST bond (Å) | Absolute difference (Å) | Converged |
|---|---:|---:|---:|---:|---|
| F2 | 2 | 1.3828 | 1.4119 | 0.0291 | True |
| SiF4 | 5 | 1.5651 | 1.5540 | 0.0111 | True |

NIST sources: [F2](https://cccbdb.nist.gov/exp2x.asp?casno=7782414&charge=0), [SiF4](https://cccbdb.nist.gov/exp2x.asp?casno=7783611&charge=0). SiF4's listed distance is derived from B0 and differs in convention from a model equilibrium geometry. No molecular force/energy accuracy is claimed without matching DFT labels.

## Starting-point decision

DPA-3.3-1M / OMat24 has the lowest force errors on this selected subset, so it is the first candidate to extend. Keep MACE-MP-0b3 as a comparison and NequIP-OAM-S as a compact speed-oriented candidate. This conclusion applies only to these checkpoints and frames, not to whole model families. Preserve the original dataset splits and audit pretraining overlap before calling a larger evaluation held-out.

Before production etching MD, evaluate short-distance repulsion and quasi-static drag references, reactions/barriers, hotter impacts, charge/stopping assumptions, slab-size/timestep convergence and multiple trajectories. A low error at 30 eV does not establish accuracy at hundreds of eV. The supplied archive already includes qsd data for the next extension.

## Reproduction and artifacts

- [Run instructions](../docs/RUN_PILOT.md); [tested package snapshots](../environments); [hardware](hardware.json).
- [MACE raw predictions and manifest](../results/pilot-mace-mp-0b3).
- [NequIP raw predictions and manifest](../results/pilot-nequip-oam-s).
- [DeePMD raw predictions and manifest](../results/pilot-deepmd-dpa33-omat-fixed).
- [OMol25 output and optimization trajectories](../results/pilot-omol25-fragments).
- [Machine-readable metrics](pilot_metrics.csv); [published baselines and limits](../docs/PUBLISHED_BASELINES.md).

The two earlier DeePMD attempts remain in results/: Torch 2.14 ABI incompatibility, then a missing e3nn import after matching Torch 2.11.0. The successful environment uses deepmd-kit 3.2.0, torch 2.11.0 and e3nn 0.6.0. Native Windows worked for this CPU pilot; GPU and compiled production inference were not tested.
