# Research scope and acceptance criteria

## Crystal track

Start with Si diamond. Add Ge, 3C/4H-SiC and wurtzite GaN only after selecting explicit polymorphs and verifying reference structures. Compare MACE-MP-0 medium and CHGNet first; later add a supported SevenNet model and a material-specific classical baseline. Do not use EMT as a silicon accuracy baseline.

1. Verify energy, forces and stress are finite on the supplied Si seed. This is a plumbing check only.
2. Relax the structure and obtain an E-V curve. Require optimizer convergence, bracketed EOS minimum, physically meaningful positive volume/modulus, and consistent units. Converge tolerances and sampling before comparing models.
3. Curate held-out DFT structures covering strain, thermal displacement, defects and surfaces. Preserve original splits and group related trajectory frames. Audit overlap with each pretrained model's training data; label unknown overlap explicitly.
4. Report energy MAE/RMSE in eV/atom, component-wise force MAE/RMSE in eV/Angstrom, stress errors in GPa with an explicit sign/Voigt convention, and property errors. For energy, use consistent reference conventions or independently fitted training-only offsets; never fit offsets on test labels.
5. Add elastic constants, phonons, vacancy formation and surface energies after cell-size, slab, vacuum and displacement convergence. Record failed/unconverged cases in the denominator.

Keep reference XC functional, pseudopotential, cutoff, k-point mesh, charge and spin metadata. JARVIS reference calculations and pretrained-model training references may differ; separate this systematic mismatch from model error. Surface/defect chemical potentials must use a consistent model and composition convention. General MLIPs do not directly predict band gaps or carrier transport.

Benchmark speed separately: fixed atom counts, dtype, hardware, package versions, warmup, repeats and GPU synchronization. Exclude model load/download time from steady-state throughput but report startup separately. Report per-system scores and aggregate with declared weights rather than mixing units into an unexplained ranking.

## Plasma-surface extension

First target a documented SiO2/CFx dataset if its contents and license support evaluation; alternatively scope neutral F/Si reactions with new reference data. This track is pending data inspection, not implemented by the crystal runner.

Validate reaction energies/barriers, adsorption/desorption, close-contact repulsion and force errors on collision snapshots before production impact MD. Split by trajectory and impact conditions. Use independent seeds and confidence intervals for sputter/etch yield, reflection, implantation depth, species and damaged-layer thickness. Verify atom/energy accounting, slab thickness, boundaries, heat removal, impact energy/angle and timestep convergence.

An ordinary neutral ground-state MLIP does not model plasma electrons, sheath transport, ionization, charge transfer or electronic stopping automatically. Specify which processes the atomistic approximation covers; add explicit charge/stopping treatment if needed. Do not interpret a bulk crystal benchmark as validation for plasma chemistry. A short-range repulsive correction needs smooth matching and independent validation.

## Reproducibility record

Every production run needs source commit, resolved config, model identity and weight SHA256, dataset DOI/version and SHA256, split IDs, environment lock, device/driver, dtype, random seeds, convergence/failure status, raw predictions, reference labels, units, and timing protocol. The starter manifest captures only a subset; extend it before production runs.
