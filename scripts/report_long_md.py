"""Verify the longer trajectories and the full-reference-trajectory validation, then report."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MD = Path('results/long-md')
VAL = Path('results/validation-etch')
REPORTS = Path('reports')
BACKENDS = ['mace', 'nequip', 'deepmd']
LABELS = {'mace': 'MACE-MP-0b3 medium', 'nequip': 'NequIP-OAM-S 0.1', 'deepmd': 'DPA-3.3-1M / OMat24'}
COLORS = {'mace': '#2f6fb2', 'nequip': '#c2571a', 'deepmd': '#2e8b57'}


def trajectory(folder):
    """Manifest plus per-step rows, with the manifest diagnostics re-derived from the CSV."""
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['status'] == 'completed', (folder, manifest['status'])
    with (folder / 'thermodynamics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    table = {key: np.array([float(r[key]) for r in rows]) for key in rows[0] if key != 'formula'}
    assert len(rows) == manifest['steps'] + 1
    assert np.isfinite(table['total_eV']).all()
    assert np.isclose(table['time_fs'][-1], manifest['duration_fs'])
    drift = (table['total_eV'] - table['total_eV'][0]) / manifest['natoms'] * 1000
    assert np.isclose(np.abs(drift).max(), manifest['max_abs_energy_change_meV_atom'])
    table['drift_meV_atom'] = drift
    return manifest, table


def validation(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['status'] == 'completed', (folder, manifest['status'])
    summary = json.loads((folder / 'summary.json').read_text())
    with (folder / 'frames.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == summary['frames']
    return manifest, summary, rows


def projectile(folder, manifest, table):
    """Geometric diagnostics for the projectile, from the saved coordinates only."""
    from ase.io import read
    frames = read(folder / 'trajectory.extxyz', index=':')
    first, last = frames[0], frames[-1]
    substrate = manifest['substrate_atoms']
    carbon = substrate
    symbols = last.get_chemical_symbols()
    distances = last.get_distances(carbon, list(range(len(last))), mic=True)
    distances[carbon] = np.inf
    nearest = int(np.argmin(distances))
    step = frames[-1].info['time_fs'] - frames[-2].info['time_fs']
    velocity = (last.positions[carbon, 2] - frames[-2].positions[carbon, 2]) / step
    top = float(first.cell[2, 2])
    deepest = int(np.argmin(table['carbon_z_A']))
    return dict(substrate_top_A=float(first.positions[:substrate, 2].max()),
                initial_carbon_z_A=float(first.positions[carbon, 2]),
                deepest_carbon_z_A=float(table['carbon_z_A'][deepest]),
                deepest_time_fs=float(table['time_fs'][deepest]),
                final_carbon_z_A=float(last.positions[carbon, 2]),
                final_fluorine_z_A=[float(z) for z in last.positions[substrate + 1:, 2]],
                nearest_symbol=symbols[nearest], nearest_distance_A=float(distances[nearest]),
                nearest_is_substrate=bool(nearest < substrate),
                cell_top_A=top, final_carbon_velocity_A_fs=float(velocity),
                headroom_fs=float((top - last.positions[carbon, 2]) / velocity) if velocity > 0 else None)


def drift_rate(time_fs, drift):
    """Least-squares slope of the total-energy change, in meV/atom/ps."""
    slope = np.polyfit(time_fs, drift, 1)[0]
    return float(slope * 1000)


def main():
    crystals = {b: trajectory(MD / b) for b in BACKENDS}
    interface_manifest, interface = trajectory(MD / 'interface-gpu')
    impact = projectile(MD / 'interface-gpu', interface_manifest, interface)
    validations = {b: validation(VAL / b) for b in BACKENDS}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for backend, (manifest, table) in crystals.items():
        axes[0].plot(table['time_fs'] / 1000, table['drift_meV_atom'], color=COLORS[backend], lw=1,
                     label=f"{backend} ({drift_rate(table['time_fs'], table['drift_meV_atom']):+.4f} meV/atom/ps)")
    axes[1].plot(interface['time_fs'] / 1000, interface['drift_meV_atom'], color='#7d3c98', lw=1,
                 label=f"interface-gpu ({drift_rate(interface['time_fs'], interface['drift_meV_atom']):+.4f} meV/atom/ps)")
    for ax, title in zip(axes, [f"Si crystals, {crystals['mace'][0]['duration_fs']/1000:g} ps NVE",
                                f"CF2 / silica, {interface_manifest['duration_fs']/1000:g} ps"]):
        ax.set(title=title, xlabel='Time (ps)', ylabel='Total-energy change (meV/atom)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(MD / 'energy_conservation.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout='constrained')
    axes[0].plot(interface['time_fs'], interface['potential_eV'] - interface['potential_eV'][0], color='#b2472f', lw=1)
    axes[0].set(ylabel='Potential change (eV)', title='CF2 / silica impact, DPA-3.3-1M / OMat24, CUDA')
    axes[1].plot(interface['time_fs'], interface['carbon_z_A'], color='#36465e', lw=1, label='projectile C height z')
    axes[1].plot(interface['time_fs'], interface['projectile_min_substrate_distance_A'], color='#22bc92', lw=1,
                 label='min projectile-substrate distance')
    axes[1].set(ylabel='Distance (A)')
    axes[1].legend(fontsize=8)
    axes[2].plot(interface['time_fs'], interface['drift_meV_atom'], color='#7d3c98', lw=1)
    axes[2].set(ylabel='Energy change (meV/atom)', xlabel='Time (fs)')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(MD / 'interface_dynamics.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for backend, (manifest, table) in crystals.items():
        ax.plot(table['time_fs'] / 1000, table['temperature_K'], color=COLORS[backend], lw=.9, alpha=.85, label=backend)
    ax.set(xlabel='Time (ps)', ylabel='Instantaneous temperature (K)',
           title='Eight-atom Si NVE, 300 K initial kinetic temperature, no thermostat')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(MD / 'crystal_temperature.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout='constrained')
    window = 25
    for backend, (manifest, summary, rows) in validations.items():
        frame = np.array([float(r['frame']) for r in rows])
        rmse = np.array([float(r['force_rmse_eV_A']) for r in rows])
        smooth = np.convolve(rmse, np.ones(window) / window, mode='valid')
        axes[0].plot(frame, rmse, color=COLORS[backend], lw=.5, alpha=.25)
        axes[0].plot(frame[window - 1:], smooth, color=COLORS[backend], lw=1.6, label=f'{backend} ({window}-frame mean)')
        axes[1].plot(frame, np.array([float(r['raw_energy_error_meV_atom']) for r in rows]),
                     color=COLORS[backend], lw=.8, label=backend)
    axes[0].set(ylabel='Force RMSE vs DFT (eV/A)', title='Archived CF2 30 eV etching trajectory, 1000 DFT-labelled frames')
    axes[1].set(xlabel='Reference frame index (cumulative etch progress)', ylabel='Raw energy error (meV/atom)')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(VAL / 'force_error_vs_frame.png', dpi=160)
    plt.close(fig)

    elements = sorted({e for _, s, _ in validations.values() for e in s['per_element_force_mae_eV_A']})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    width = 0.26
    for offset, (backend, (manifest, summary, rows)) in zip([-width, 0, width], validations.items()):
        axes[0].bar(np.arange(len(elements)) + offset,
                    [summary['per_element_force_mae_eV_A'][e] for e in elements],
                    width=width, color=COLORS[backend], label=backend)
        axes[1].hist([float(r['force_rmse_eV_A']) for r in rows], bins=60, histtype='step',
                     color=COLORS[backend], label=backend)
    axes[0].set(xticks=np.arange(len(elements)), ylabel='Force component MAE (eV/A)', title='Per-element force error')
    axes[0].set_xticklabels(elements)
    axes[1].set(xlabel='Per-frame force RMSE (eV/A)', ylabel='Frames', title='Per-frame force RMSE distribution')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(VAL / 'error_distributions.png', dpi=160)
    plt.close(fig)

    REPORTS.mkdir(exist_ok=True)
    with (REPORTS / 'etch_validation_metrics.csv').open('w', newline='') as stream:
        fields = ['checkpoint', 'frames', 'total_atoms', 'force_component_mae_eV_A', 'force_component_rmse_eV_A',
                  'force_component_max_abs_eV_A', 'raw_energy_mae_meV_atom', 'raw_energy_mean_signed_error_meV_atom',
                  'offset_fitted_energy_mae_meV_atom', 'offset_fitted_energy_rmse_meV_atom',
                  'median_singlepoint_seconds']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for backend, (manifest, summary, rows) in validations.items():
            writer.writerow({'checkpoint': LABELS[backend], **{k: summary[k] for k in fields[1:]}})

    text = ['# Longer trajectories and full-reference-trajectory validation', '',
            'Two measurements were added to the earlier short pilot: molecular dynamics runs twenty times longer than '
            'the committed short ones, and single points against every archived DFT frame of one etching reference '
            'trajectory instead of six sampled frames. No model was trained or fine-tuned.', '',
            '## Longer molecular dynamics', '',
            '| Run | Model | Device | Atoms | Time (ps) | Step (fs) | Max energy change (meV/atom) | Drift slope (meV/atom/ps) | Wall (s) |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for backend in BACKENDS:
        manifest, table = crystals[backend]
        text.append(f"| {backend} | {manifest['model']} | {manifest['device']} | {manifest['natoms']} | "
                    f"{manifest['duration_fs']/1000:g} | {manifest['timestep_fs']} | "
                    f"{manifest['max_abs_energy_change_meV_atom']:.5f} | "
                    f"{drift_rate(table['time_fs'], table['drift_meV_atom']):+.5f} | {manifest['wall_seconds']:.1f} |")
    text.append(f"| interface-gpu | {interface_manifest['model']} | {interface_manifest['device']} | "
                f"{interface_manifest['natoms']} | {interface_manifest['duration_fs']/1000:g} | "
                f"{interface_manifest['timestep_fs']} | {interface_manifest['max_abs_energy_change_meV_atom']:.5f} | "
                f"{drift_rate(interface['time_fs'], interface['drift_meV_atom']):+.5f} | "
                f"{interface_manifest['wall_seconds']:.1f} |")
    text += ['', '![Energy conservation over the longer runs](energy_conservation.png)', '',
             'The drift slope is a least-squares fit of the total-energy change against time. It is a single '
             'unthermostatted run per system, so it measures integrator and model smoothness over this window, not '
             'ensemble-averaged conservation.', '',
             '### Si crystal runs', '',
             f"Eight-atom periodic diamond cells, {crystals['mace'][0]['duration_fs']/1000:g} ps NVE with the same "
             '300 K initial velocity seed, model-specific equation-of-state lattice constants and no thermostat or '
             'equilibration. Instantaneous temperature oscillates because an eight-atom microcanonical cell '
             'exchanges kinetic and potential energy; these are not converged thermal averages.', '',
             '| Model | Final temperature (K) | Mean temperature (K) | Std. dev. (K) |', '|---|---:|---:|---:|']
    for backend in BACKENDS:
        manifest, table = crystals[backend]
        text.append(f"| {LABELS[backend]} | {manifest['final_temperature_K']:.1f} | "
                    f"{table['temperature_K'].mean():.1f} | {table['temperature_K'].std():.1f} |")
    text += ['', '![Crystal temperatures](crystal_temperature.png)', '',
             '### CF2 / silica impact', '',
             f"The constructed 30 eV neutral CF2 impact was extended to {interface_manifest['duration_fs']:g} fs "
             f"({interface_manifest['steps']} steps of {interface_manifest['timestep_fs']} fs) on CUDA, from the same "
             'initial coordinates, velocities and checkpoint as the committed 50 fs run. All caveats of that run still '
             'apply: cold unrelaxed substrate, frozen bottom layer, retained periodic vacuum, no charge state and no '
             'thermostat.', '',
             f"Minimum distance between an original projectile atom and the substrate over the run: "
             f"{interface_manifest['minimum_projectile_substrate_distance_A']:.3f} A. That minimum is taken over all "
             'three original projectile atoms, so it stays near its floor for the whole run even after the carbon '
             'leaves the surface region, because the two fluorines do not.', '',
             f"The projectile carbon starts at z = {impact['initial_carbon_z_A']:.2f} A, above a substrate whose "
             f"highest atom is at {impact['substrate_top_A']:.2f} A. It reaches its lowest point, "
             f"{impact['deepest_carbon_z_A']:.2f} A, at {impact['deepest_time_fs']:.2f} fs, about "
             f"{impact['substrate_top_A'] - impact['deepest_carbon_z_A']:.2f} A below that initial surface height, "
             f"then reverses and rises to {impact['final_carbon_z_A']:.2f} A by the end of the run. Its nearest "
             f"neighbour in the final frame is a substrate {impact['nearest_symbol']} atom at "
             f"{impact['nearest_distance_A']:.3f} A, and the two original fluorine atoms remain at z = "
             + ' and '.join(f"{z:.2f}" for z in impact['final_fluorine_z_A']) + ' A. These are measured distances; '
             'no bond, desorption product or reaction channel is assigned from them.', '',
             f"This also bounds the usable run length. The carbon is still moving upward at "
             f"{impact['final_carbon_velocity_A_fs']:.3f} A/fs at {interface_manifest['duration_fs']:g} fs and the "
             f"retained periodic cell ends at {impact['cell_top_A']:.2f} A, so it would cross the top boundary and "
             f"re-enter from below roughly {impact['headroom_fs']:.0f} fs later. Running this cell materially longer "
             'would need added vacuum or a different boundary treatment.', '',
             '![Interface dynamics](interface_dynamics.png)', '',
             '![CF2 impact animation](interface-gpu/animation.gif)', '',
             'Atom identity is tracked through the trajectory; no bond breaking, sputter yield or reaction product is '
             'assigned. One trajectory cannot establish an etch yield.', '',
             '## Ground-truth comparison against the reference DFT trajectory', '',
             f"Each checkpoint was evaluated on all {validations['mace'][1]['frames']} archived DFT frames of "
             '`dft/etch/trj_CF2_30eV.extxyz`, sequentially on CPU. These are the authors\' DFT single points on '
             'snapshots generated by a different model (7net-Omni), ordered by cumulative etch progress. Consecutive '
             'frames are separate impact snapshots with changing composition, not adjacent MD steps, so this measures '
             'label accuracy along the reference chemistry rather than dynamical agreement.', '',
             '| Checkpoint | Force MAE (eV/A) | Force RMSE (eV/A) | Raw energy MAE (meV/atom) | Offset-fitted energy MAE (meV/atom) | Median s/frame |',
             '|---|---:|---:|---:|---:|---:|']
    for backend in BACKENDS:
        summary = validations[backend][1]
        text.append(f"| {LABELS[backend]} | {summary['force_component_mae_eV_A']:.4f} | "
                    f"{summary['force_component_rmse_eV_A']:.4f} | {summary['raw_energy_mae_meV_atom']:.2f} | "
                    f"{summary['offset_fitted_energy_mae_meV_atom']:.2f} | {summary['median_singlepoint_seconds']:.4f} |")
    text += ['', 'Force errors pool every Cartesian component of every frame. The raw energy column has no fitted '
             'offset and therefore still contains reference-convention and energy-zero differences. The '
             'offset-fitted column removes one least-squares energy per element, fitted on these same frames; it is a '
             'diagnostic of shape agreement, not an independently validated formation energy, and it cannot be used '
             'as a held-out accuracy estimate.', '',
             '![Force error along the reference trajectory](../validation-etch/force_error_vs_frame.png)', '',
             '| Checkpoint | ' + ' | '.join(f'{e} force MAE (eV/A)' for e in elements) + ' |',
             '|---|' + '---:|' * len(elements)]
    for backend in BACKENDS:
        summary = validations[backend][1]
        text.append(f"| {LABELS[backend]} | " + ' | '.join(
            f"{summary['per_element_force_mae_eV_A'][e]:.4f}" for e in elements) + ' |')
    text += ['', '![Per-element and per-frame error distributions](../validation-etch/error_distributions.png)', '',
             'Per-frame metrics are in `../validation-etch/<backend>/frames.csv`; pooled metrics are in '
             '`../../reports/etch_validation_metrics.csv`. The six-frame subset of the earlier pilot is a subset of '
             'these frames, so the two reports are consistent but not independent.', '',
             '## Limitations', '',
             '- One trajectory per system and one reference etch sequence; no repeats, no error bars.',
             '- The reference labels come from one DFT setup whose exact dispersion treatment is still unconfirmed; '
             'see [published baselines](../../docs/PUBLISHED_BASELINES.md).',
             '- Snapshot accuracy on an externally generated trajectory does not establish that any of these models '
             'would generate the same trajectory, nor an etch rate, yield or selectivity.',
             '- Wall times include per-frame overheads and are not matched-precision throughput measurements.',
             '- The interface cell keeps its original periodic vacuum, so the run ends just before the rising '
             'projectile would wrap through the top boundary; this is a run-length limit, not a converged result.', '',
             '## Reproduction', '',
             'Run from the repository root; output folders must not already exist.', '', '```powershell',
             f".venv-deepmd-gpu/Scripts/python.exe scripts/run_interface_md.py --device cuda --output results/long-md/interface-gpu --steps {interface_manifest['steps']} --timestep-fs {interface_manifest['timestep_fs']} --record-every {interface_manifest['record_every']}"]
    for backend in BACKENDS:
        manifest = crystals[backend][0]
        text.append(f".venv-{backend}/Scripts/python.exe scripts/run_short_md.py --backend {backend} --output "
                    f"results/long-md/{backend} --steps {manifest['steps']} --timestep-fs {manifest['timestep_fs']} "
                    f"--record-every {manifest['record_every']}")
    for backend in BACKENDS:
        text.append(f".venv-{backend}/Scripts/python.exe scripts/validate_reference_trajectory.py --backend {backend} "
                    f"--output results/validation-etch/{backend}")
    text += ['.venv-render/Scripts/python.exe scripts/render_atomistic.py --root results/long-md --names interface-gpu --stride 4',
             '.venv-render/Scripts/python.exe scripts/report_long_md.py', '```', '',
             'The reference archive is the [etching dataset](https://zenodo.org/records/19491140); its SHA-256 and '
             'each checkpoint hash are recorded in the manifests next to every result.', '']
    (MD / 'README.md').write_text('\n'.join(text), encoding='utf-8')
    print('Verified', len(crystals) + 1, 'trajectories and', len(validations), 'reference validations')


if __name__ == '__main__':
    main()
