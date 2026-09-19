"""Shared data selection and diagnostics for the reactive reliability benchmark."""
import csv
import hashlib
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.io import read
from ase.io.extxyz import key_val_str_to_dict

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'data/downloads/etch_dft.tar'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def blocks(text):
    lines = text.splitlines(keepends=True)
    cursor, index = 0, 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        n = int(lines[cursor])
        if cursor + n + 2 > len(lines):
            raise ValueError('Truncated extxyz frame')
        yield index, ''.join(lines[cursor:cursor + n + 2]), key_val_str_to_dict(lines[cursor + 1])
        index += 1
        cursor += n + 2


def qsd_stratum(group):
    parts = group.split('_')
    return '_'.join(parts[:2] + parts[-2:])


def source_group(member, info):
    if '/qsd/' in member:
        # All distances and pair perturbations from the same source event stay together.
        return '_'.join(str(info['group']).split('_')[:-2])
    if '/etch/' in member:
        return str(info.get('etch', member))
    return member + ':' + str(info.get('run_type', 'unknown'))


def choose_uniform(indices, count):
    if count <= 0:
        raise ValueError('Sample count must be positive')
    return [indices[i] for i in np.unique(np.linspace(0, len(indices) - 1, min(count, len(indices))).round().astype(int))]


def make_plan(full=False, seed=20260918, etch_count=32, bulk_count=8):
    inventory, selected = [], []
    rng = np.random.default_rng(seed)
    # A separate, explicitly biased diagnostic set reproduces old worst cases.
    outliers = set()
    for backend in ['mace', 'nequip', 'deepmd']:
        path = ROOT / 'results/validation-etch' / backend / 'frames.csv'
        if path.exists():
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            outliers.update(int(r['frame']) for r in sorted(rows, key=lambda r: float(r['max_force_error_eV_A']), reverse=True)[:10])
    with tarfile.open(ARCHIVE) as tar:
        for member in sorted(m.name for m in tar.getmembers() if m.isfile() and m.name.endswith('.extxyz')):
            entries = [(i, info) for i, _, info in blocks(tar.extractfile(member).read().decode())]
            groups = defaultdict(list)
            for i, info in entries:
                key = str(info['group']) if '/qsd/' in member else source_group(member, info)
                groups[key].append(i)
            chosen = set()
            if full:
                chosen.update(i for i, _ in entries)
            elif '/qsd/' in member:
                strata = defaultdict(list)
                for group in sorted(groups):
                    strata[qsd_stratum(group)].append(group)
                for candidates in strata.values():
                    chosen.update(groups[candidates[int(rng.integers(len(candidates)))]] )
            else:
                for indices in groups.values():
                    chosen.update(choose_uniform(indices, etch_count if '/etch/' in member else bulk_count))
            regular = chosen.copy()
            if member == 'dft/etch/trj_CF2_30eV.extxyz':
                chosen.update(outliers)
            inventory.append(dict(member=member, available_frames=len(entries), available_groups=len(groups),
                                  selected_frames=len(chosen), coverage_frames=len(regular)))
            for i, info in entries:
                if i not in chosen:
                    continue
                selected.append(dict(id=f'{member}:{i}', member=member, frame=i,
                                     track=member.split('/')[1], group=str(info.get('group', source_group(member, info))),
                                     source_group=source_group(member, info),
                                     selection='coverage' if i in regular else 'prior_outlier',
                                     prior_outlier=i in outliers and member == 'dft/etch/trj_CF2_30eV.extxyz'))
    return dict(schema_version=1, seed=seed, mode='full' if full else 'stratified_screen',
                archive_sha256=digest(ARCHIVE), archive_doi='10.5281/zenodo.19491140',
                protocol='All selected QSD curves intact; uniform sampling within other source groups; no fitting.',
                uncertainty='No confidence intervals: independence between source groups is not established.',
                training_overlap='unknown', inventory=inventory, cases=selected)


def selected_atoms(plan):
    by_member = defaultdict(dict)
    for case in plan['cases']:
        by_member[case['member']][case['frame']] = case
    with tarfile.open(ARCHIVE) as tar:
        for member, wanted in by_member.items():
            for i, block, _ in blocks(tar.extractfile(member).read().decode()):
                if i in wanted:
                    yield wanted[i], read(io.StringIO(block), format='extxyz')


def force_stats(delta):
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 2 or delta.shape[1] != 3 or len(delta) == 0 or not np.isfinite(delta).all():
        raise ValueError('Expected nonempty finite N x 3 force errors')
    absolute = np.abs(delta).ravel()
    norms = np.linalg.norm(delta, axis=1)
    return dict(atoms=len(delta), component_mae=float(absolute.mean()), component_rmse=float(np.sqrt((delta**2).mean())),
                component_p95=float(np.percentile(absolute, 95)), component_p99=float(np.percentile(absolute, 99)),
                component_max=float(absolute.max()), vector_p95=float(np.percentile(norms, 95)), vector_max=float(norms.max()))


def local_geometry(atoms):
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    # No projectile identity is present in the archive. Chemical subsets are not provenance labels.
    symbols = np.array(atoms.get_chemical_symbols())
    cf = np.isin(symbols, ['C', 'F'])
    sio = np.isin(symbols, ['Si', 'O'])
    contact = float(distances[np.ix_(cf, sio)].min()) if cf.any() and sio.any() else None
    return nearest, contact


def qsd_metrics(rows):
    """Same-composition energy differences relative to largest sampled distance, never a fitted offset."""
    if len(rows) < 2:
        raise ValueError('A drag curve needs at least two points')
    rows = sorted(rows, key=lambda r: r['distance_A'])
    if len({tuple(r['symbols']) for r in rows}) != 1:
        raise ValueError('QSD curve changed composition or atom ordering')
    if len({r['distance_A'] for r in rows}) != len(rows):
        raise ValueError('Duplicate drag distance')
    ref = np.array([r['energy_ref_eV'] for r in rows])
    pred = np.array([r['energy_pred_eV'] for r in rows])
    ref -= ref[-1]
    pred -= pred[-1]
    dr, dp = np.diff(ref), np.diff(pred)
    meaningful = np.abs(dr) > 1e-6
    return dict(points=len(rows), relative_energy_mae_eV=float(np.abs(pred-ref).mean()),
                relative_energy_max_error_eV=float(np.abs(pred-ref).max()),
                reference_span_eV=float(np.ptp(ref)), predicted_span_eV=float(np.ptp(pred)),
                reference_short_range_rise_eV=float(ref[0]), predicted_short_range_rise_eV=float(pred[0]),
                slope_comparisons=int(meaningful.sum()),
                slope_sign_mismatches=int(((dr * dp <= 0) & meaningful).sum()))
