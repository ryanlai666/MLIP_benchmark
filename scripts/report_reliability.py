"""Recompute reliability metrics from saved per-atom predictions; no model needed."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from reliability_data import ROOT, ARCHIVE, digest, dump, force_stats, qsd_metrics, selected_atoms

BACKENDS = ['mace', 'nequip', 'deepmd']
DISTANCE_BINS = [(0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, float('inf'))]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def csv_write(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(rows):
    """All component percentiles are computed from atoms, not frame percentiles."""
    by_track, buckets, outliers, per_frame = defaultdict(list), defaultdict(list), [], []
    for row in rows:
        if row['status'] != 'ok':
            continue
        delta = np.asarray(row['forces_pred_eV_A']) - row['forces_ref_eV_A']
        metrics = force_stats(delta)
        # Independently recompute the stored metrics before using an artifact.
        for key, value in metrics.items():
            if not np.isclose(value, row['metrics'][key], rtol=1e-10, atol=1e-10):
                raise ValueError(f'Saved force summary disagrees with raw arrays: {row["id"]}, {key}')
        symbols = np.array(row['symbols'])
        nearest = np.array(row['nearest_distance_A'])
        norms = np.linalg.norm(delta, axis=1)
        for i in np.argsort(norms)[-3:][::-1]:
            outliers.append(dict(id=row['id'], track=row['track'], selection=row['selection'], group=row['group'],
                                 atom_index=int(i), element=str(symbols[i]), vector_error_eV_A=float(norms[i]),
                                 component_max_eV_A=float(np.abs(delta[i]).max()), nearest_distance_A=float(nearest[i]),
                                 cf_sio_contact_A=row['cf_sio_contact_A'],
                                 ref_force_norm_eV_A=float(np.linalg.norm(row['forces_ref_eV_A'][i]))))
        per_frame.append(dict(id=row['id'], track=row['track'], selection=row['selection'], group=row['group'],
                              **metrics, cf_sio_contact_A=row['cf_sio_contact_A']))
        if row['selection'] != 'coverage':
            continue
        by_track[row['track']].append(row)
        for s in sorted(set(symbols)):
            buckets[(row['track'], 'element', str(s))].append(delta[symbols == s])
        for name, mask in [('C/F', np.isin(symbols, ['C','F'])), ('Si/O', np.isin(symbols, ['Si','O']))]:
            if mask.any():
                buckets[(row['track'], 'chemical_subset', name)].append(delta[mask])
        for low, high in DISTANCE_BINS:
            mask = (nearest >= low) & (nearest < high)
            if mask.any():
                buckets[(row['track'], 'nearest_distance_A', f'[{low:g},{high:g})')].append(delta[mask])
            if row['track']=='qsd' and low <= row['distance_A'] < high:
                buckets[(row['track'], 'archived_drag_distance_A', f'[{low:g},{high:g})')].append(delta)
    summary = []
    for track, selected in sorted(by_track.items()):
        d = np.concatenate([np.asarray(r['forces_pred_eV_A']) - r['forces_ref_eV_A'] for r in selected])
        summary.append(dict(track=track, frames=len(selected), source_groups=len({r['source_group'] for r in selected}),
                            **force_stats(d), raw_energy_mae_meV_atom=float(np.mean([
                                abs(r['energy_pred_eV']-r['energy_ref_eV'])/len(r['symbols'])*1000 for r in selected]))))
    stratified = [dict(track=k[0], category=k[1], subset=k[2], **force_stats(np.concatenate(values)))
                  for k, values in sorted(buckets.items())]
    return summary, stratified, sorted(outliers, key=lambda r:r['vector_error_eV_A'], reverse=True), per_frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'results/reliability')
    p.add_argument('--plan', type=Path, default=ROOT/'configs/reliability_plan.json')
    p.add_argument('--output', type=Path, default=ROOT/'reports/reliability')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    reference_audit = None
    if ARCHIVE.exists():
        if digest(ARCHIVE) != plan['archive_sha256']:
            raise ValueError('Reference audit archive differs from the frozen plan')
        flags = defaultdict(lambda: defaultdict(int))
        distances = []
        for case, atoms in selected_atoms(plan):
            value = atoms.info.get('converged')
            flag = 'missing' if value is None else ('true' if bool(value) else 'false')
            flags[case['track']][flag] += 1
            if case['track']=='qsd':
                distances.append(float(atoms.info['dist']))
        reference_audit = dict(archive_sha256=plan['archive_sha256'], plan_sha256=digest(args.plan),
                               reference_convergence_flags={k:dict(v) for k,v in flags.items()},
                               minimum_archived_qsd_distance_A=min(distances) if distances else None,
                               interpretation='Missing convergence metadata does not prove unconverged DFT; finite labels alone do not independently validate extreme-contact references.')
        dump(args.output/'reference_audit.json', reference_audit)
    expected = {c['id'] for c in plan['cases']}
    data, checks, manifests = {}, {}, {}
    common = expected.copy()
    provenance = dict(plan_sha256=digest(args.plan), report_script_sha256=digest(Path(__file__)), artifacts={})
    for backend in BACKENDS:
        directory = args.root/backend
        manifests[backend] = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        if manifests[backend]['status'] not in ['completed', 'completed_with_failures']:
            raise ValueError(f'{backend}: run is not complete')
        if manifests[backend]['plan_sha256'] != provenance['plan_sha256']:
            raise ValueError('Models used different plans')
        rows = read_jsonl(directory/'predictions.jsonl')
        ids = [r['id'] for r in rows]
        if set(ids) != expected or len(ids) != len(expected):
            raise ValueError(f'{backend}: missing, duplicate or unexpected frames')
        data[backend] = rows
        checks[backend] = read_jsonl(directory/'checks.jsonl')
        check_ids = [r['id'] for r in checks[backend]]
        if len(check_ids) != len(set(check_ids)):
            raise ValueError('Duplicate physical check IDs')
        common &= {r['id'] for r in rows if r['status']=='ok'}
        provenance['artifacts'][backend] = {name:digest(directory/name) for name in ['manifest.json','predictions.jsonl','checks.jsonl']}
    summaries, strata, outliers, per_frame, curves, source_scores, failures = [], [], [], [], [], [], []
    for backend, rows in data.items():
        summaries_b, strata_b, outliers_b, frames_b = analyze([r for r in rows if r['id'] in common])
        for destination, incoming in [(summaries,summaries_b), (strata,strata_b), (outliers,outliers_b), (per_frame,frames_b)]:
            destination.extend(dict(backend=backend, **r) for r in incoming)
        groups, sources = defaultdict(list), defaultdict(list)
        for r in rows:
            if r['id'] in common and r['selection']=='coverage':
                sources[(r['member'], r['source_group'])].append(r)
                if r['track']=='qsd':
                    groups[r['group']].append(r)
            if r['status'] != 'ok':
                failures.append(dict(backend=backend, kind='reference', id=r['id'], error=r.get('error')))
        for group, selected in sorted(groups.items()):
            requested = sum(c['track']=='qsd' and c['group']==group for c in plan['cases'])
            if len(selected) != requested:
                continue
            curves.append(dict(backend=backend, group=group, source_group=selected[0]['source_group'], **qsd_metrics(selected)))
        for (member, group), selected in sorted(sources.items()):
            delta = np.concatenate([np.asarray(r['forces_pred_eV_A'])-r['forces_ref_eV_A'] for r in selected])
            source_scores.append(dict(backend=backend, member=member, group=group, frames=len(selected), **force_stats(delta)))
        for r in checks[backend]:
            if r['status'] != 'ok':
                failures.append(dict(backend=backend, kind=r['kind'], id=r['id'], error=r.get('error')))
    csv_write(args.output/'track_metrics.csv', summaries)
    csv_write(args.output/'stratified_forces.csv', strata)
    csv_write(args.output/'worst_atoms.csv', outliers)
    csv_write(args.output/'frame_metrics.csv', per_frame)
    csv_write(args.output/'qsd_curves.csv', curves)
    csv_write(args.output/'source_groups.csv', source_scores)
    csv_write(args.output/'failures.csv', failures)
    if not failures:
        (args.output/'failures.csv').write_text('backend,kind,id,error\n', encoding='utf-8')
    csv_write(args.output/'physical_checks.csv', [dict(backend=b, **r) for b in BACKENDS for r in checks[b]])
    if reference_audit is not None:
        provenance['reference_audit_sha256'] = digest(args.output/'reference_audit.json')
    provenance['models'] = manifests
    dump(args.output/'provenance.json', provenance)
    lines = ['# Reactive reliability benchmark', '',
             'MACE-MP-0b3 medium, NequIP-OAM-S 0.1 and DPA-3.3-1M/OMat24 evaluated on one frozen selection from the authors\' DFT archive. '
             'No training, energy-offset fitting, or new MD was performed.', '',
             'Source: [SevenNet-Nano supporting DFT archive](https://zenodo.org/records/19491140). '
             'The archive README identifies etching single points, fluorocarbon bulk, melt-quench-anneal bulk, '
             'and quasi-static drag series. Pretraining overlap and exact DFT dispersion treatment remain unresolved.', '',
             '## Coverage and failures', '',
             '| Archive member | Available | Coverage sample | Including targeted outliers |',
             '|---|---:|---:|---:|']
    for item in plan['inventory']:
        lines.append(f"| {item['member']} | {item['available_frames']} | {item['coverage_frames']} | {item['selected_frames']} |")
    lines += ['', f"{len(expected)} reference frames per model; {len(common)} succeeded in all three models. "
              'Metrics below use the common successful coverage frames. Additional frames chosen from previous worst errors '
              'are excluded from coverage aggregates and retained in the outlier tables.', '',
              '| Model | Reference successes / attempted | Physical checks successful / attempted |', '|---|---:|---:|']
    for b in BACKENDS:
        lines.append(f"| {b} | {sum(r['status']=='ok' for r in data[b])}/{len(data[b])} | {sum(r['status']=='ok' for r in checks[b])}/{len(checks[b])} |")
    lines += ['', 'A successful check means it produced finite results, not that the model passed a physical acceptance threshold. '
              'Failed evaluations remain in the denominators and [failure table](failures.csv).', '',
              '## Force errors against DFT', '',
              'All errors below are Cartesian component errors in eV/Angstrom, pooled over atoms within each track. '
              'The maximum is a component maximum, not a vector norm. QSD contains deliberately extreme close contacts; '
              'its error scale must be read together with distance and reference-force magnitude.', '',
              '| Track | Model | Frames | MAE | RMSE | P95 | P99 | Maximum |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(summaries, key=lambda r:(r['track'],r['backend'])):
        lines.append(f"| {r['track']} | {r['backend']} | {r['frames']} | {r['component_mae']:.4g} | {r['component_rmse']:.4g} | {r['component_p95']:.4g} | {r['component_p99']:.4g} | {r['component_max']:.4g} |")
    lines += ['', '[Per-element, chemical-subset and nearest-distance metrics](stratified_forces.csv) / '
              '[per-source-group metrics](source_groups.csv) / [per-frame metrics](frame_metrics.csv).', '',
              '![Force error distributions](force_error_distributions.png)', '',
              '## Quasi-static drag: reference-backed short-range response', '',
              'Each selected curve is complete. Energies are differenced from the largest-distance frame in that same curve. '
              'This cancels a constant energy zero without fitting test labels. Slopes compare adjacent distance points; '
              'DFT energy changes below 1e-6 eV are excluded. These are constrained-path energy differences, not reaction activation barriers.', '',
              '| Model | Complete curves | Mean curve relative-energy MAE (eV) | Wrong/zero slope signs / comparisons |', '|---|---:|---:|---:|']
    for b in BACKENDS:
        selected = [r for r in curves if r['backend']==b]
        if selected:
            lines.append(f"| {b} | {len(selected)} | {np.mean([r['relative_energy_mae_eV'] for r in selected]):.4g} | {sum(r['slope_sign_mismatches'] for r in selected)}/{sum(r['slope_comparisons'] for r in selected)} |")
    if reference_audit is not None:
        qsd_flags = reference_audit['reference_convergence_flags'].get('qsd', {})
        lines += ['', f"Reference quality: {qsd_flags.get('missing', 0)} selected QSD frames lack a DFT-convergence flag. "
                  'This does not prove unconverged DFT, but finite labels alone do not independently establish reference quality '
                  'at extreme overlaps. See the [reference audit](reference_audit.json).', '']
    lines += ['', '[Every curve and its energy span](qsd_curves.csv). Equal curve weighting is descriptive; related curves '
              'share source events and are not independent replicates.', '',
              '![Selected complete QSD curves](qsd_examples.png)', '', '## Largest etching force errors', '',
              'These include deliberately selected prior outliers. Atom indices are zero-based within the archived frame. '
              'C/F and Si/O labels indicate chemical composition only; projectile provenance is unavailable.', '',
              '| Model | Frame | Atom / element | Vector error (eV/A) | Reference force norm (eV/A) | Nearest distance (A) |', '|---|---|---|---:|---:|---:|']
    for b in BACKENDS:
        worst = sorted([r for r in outliers if r['backend']==b and r['track']=='etch'], key=lambda r:r['vector_error_eV_A'], reverse=True)[:3]
        for r in worst:
            lines.append(f"| {b} | {r['id'].replace('dft/etch/','')} | {r['atom_index']} / {r['element']} | {r['vector_error_eV_A']:.4g} | {r['ref_force_norm_eV_A']:.4g} | {r['nearest_distance_A']:.3f} |")
    lines += ['', '[Worst atoms across all tracks](worst_atoms.csv) includes the top three atoms per frame. '
              'Three worst etching frames per model are exported with reference/predicted forces: '
              '[MACE](worst_etch_mace.extxyz), [NequIP](worst_etch_nequip.extxyz), [DeePMD](worst_etch_deepmd.extxyz).', '',
              '## Reference-free physical diagnostics', '',
              'Ten unordered Si/O/C/F pairs, eleven separations from 0.5 to 6 A, nonperiodic. '
              'Positive force on the right-hand atom indicates repulsion. Finite differences use two step sizes '
              '(0.001 and 0.0005 A) at 0.8, 1.5 and 3 A. Short-range attraction is flagged for inspection; '
              'there are no DFT dimer labels here. Precision-dependent residuals are reported without a universal pass threshold.', '',
              '| Model | Attractive samples at r <= 0.65 A | Max finite-difference force residual (eV/A) | Max rotation force residual (eV/A) |', '|---|---:|---:|---:|']
    for b in BACKENDS:
        pairs = [r for r in checks[b] if r['kind']=='pair' and r['status']=='ok']
        sym = [r for r in checks[b] if r['kind']=='symmetry' and r['status']=='ok']
        short = [r for r in pairs if r['distance_A'] <= .65]
        fd = [v for r in pairs for v in r.get('fd_force_errors_eV_A', [])]
        lines.append(f"| {b} | {sum(r['radial_force_eV_A']<0 for r in short)}/{len(short)} | {max(fd, default=float('nan')):.4g} | {max((r['rotation_force_max_eV_A'] for r in sym), default=float('nan')):.4g} |")
    spot_path = args.output/'deepmd_cpu_gpu_spotcheck.json'
    if spot_path.exists():
        spot = json.loads(spot_path.read_text(encoding='utf-8'))
        if spot['gpu_manifest_sha256'] != provenance['artifacts']['deepmd']['manifest.json']:
            raise ValueError('Device spot-check belongs to a different run')
        differences = [r['max_cpu_gpu_force_component_difference_eV_A'] for r in spot['checks']
                       if 'max_cpu_gpu_force_component_difference_eV_A' in r]
        lines += ['', 'A [targeted CPU/GPU reproduction](deepmd_cpu_gpu_spotcheck.json) checks C-F and Si-Si '
                  'pairs at 0.5/0.65 A and the three worst DeePMD QSD frames. '
                  f"The maximum CPU/GPU component difference on those QSD frames is {max(differences):.3g} eV/A. "
                  'The very close contacts are numerically sensitive; the saved file also reports CPU and GPU errors '
                  'against DFT. This is not a general device-equivalence test.', '']
        provenance['device_spotcheck_sha256'] = digest(spot_path)
        dump(args.output/'provenance.json', provenance)
    lines += ['', '[Full physical checks](physical_checks.csv) include translation, rotation, permutation and net pair forces.', '',
              '![Reference-free pair forces](pair_scans.png)', '',
              '## Statistical interpretation and remaining work', '',
              '- The selected QSD curves span CF/CF3 labels at 10/20/30 eV and available ordered contact pairs. '
              'Those labels describe source conditions; QSD is not an impact trajectory at those energies.',
              '- CF2 and CF3 provide two etching sequences at 30 eV; they are not proven independent replicas. '
              'Bulk conditions broaden chemistry and temperature-history coverage, not etching-yield validation.',
              '- The plan stores source groups to prevent future random frame splits. No train/test claim or confidence interval '
              'is made because source independence and pretraining overlap are unresolved.',
              '- Reaction barriers, adsorption/desorption energies, independent impact replicas and new matched DFT references '
              'remain future work. No neutral MLIP plasma-charge or stopping accuracy is inferred.',
              '- Runtime here includes diagnostics, initialization and mixed checkpoint precisions; it is not a speed ranking.', '',
              '## Reproduction', '', '[Execution instructions](../../docs/RELIABILITY.md). Raw arrays and structures remain '
              'under `results/reliability/<backend>/predictions.jsonl`; [provenance](provenance.json) includes model manifests, '
              'checkpoint hashes, actual devices/precisions and input/output hashes. Compact tables and plots are in this directory.', '']
    (args.output/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {args.output}/README.md')


if __name__ == '__main__':
    main()
