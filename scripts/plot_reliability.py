"""Plot saved reliability predictions and export the worst etching structures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import write

from reliability_data import ROOT
from report_reliability import BACKENDS, read_jsonl

COLORS = dict(mace='#3366aa', nequip='#d47721', deepmd='#26845b')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'results/reliability')
    p.add_argument('--output', type=Path, default=ROOT/'reports/reliability')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = {b:read_jsonl(args.root/b/'predictions.jsonl') for b in BACKENDS}
    checks = {b:read_jsonl(args.root/b/'checks.jsonl') for b in BACKENDS}
    common = set.intersection(*[{r['id'] for r in data[b] if r['status']=='ok'} for b in BACKENDS])
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 3, figsize=(13,4), constrained_layout=True)
    for ax, track, title in zip(axes, ['etch','FC','qsd'], ['Etching sequences (coverage sample)', 'Fluorocarbon bulk', 'Quasi-static drag (extreme contacts)']):
        for b in BACKENDS:
            errors = [np.abs(np.asarray(r['forces_pred_eV_A'])-r['forces_ref_eV_A']).ravel() for r in data[b]
                      if r['id'] in common and r['selection']=='coverage' and r['track']==track]
            if not errors:
                continue
            values = np.sort(np.concatenate(errors))
            ax.plot(values, np.arange(1,len(values)+1)/len(values), label=b, color=COLORS[b])
        ax.set(xscale='log', xlabel='Absolute force component error (eV/A)', ylabel='Cumulative fraction', title=title, ylim=(0,1.01))
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.savefig(args.output/'force_error_distributions.png', dpi=170)
    plt.close(fig)
    # Select one complete curve per source projectile/energy condition, deterministically.
    grouped = {}
    for r in data['mace']:
        if r['id'] in common and r['track']=='qsd':
            grouped.setdefault(r['group'], []).append(r)
    names = []
    for condition in ['CF_10eV','CF_20eV','CF_30eV','CF3_10eV','CF3_20eV','CF3_30eV']:
        candidates = sorted(g for g in grouped if g.startswith(condition+'_') and g.endswith('_Si_F'))
        if candidates:
            names.append(candidates[0])
    fig, axes = plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ax, group in zip(axes.ravel(), names):
        ref = sorted(grouped[group], key=lambda r:r['distance_A'])
        ax.plot([r['distance_A'] for r in ref], [r['energy_ref_eV']-ref[-1]['energy_ref_eV'] for r in ref], 'o-', color='black', label='DFT')
        for b in BACKENDS:
            rows = sorted([r for r in data[b] if r['id'] in common and r['group']==group], key=lambda r:r['distance_A'])
            ax.plot([r['distance_A'] for r in rows], [r['energy_pred_eV']-rows[-1]['energy_pred_eV'] for r in rows], '.--', color=COLORS[b], label=b)
        ax.set(title=group, xlabel='Archived drag distance (A)', ylabel='Energy relative to largest distance (eV)', yscale='symlog')
        ax.grid(alpha=.2)
    axes.ravel()[0].legend(fontsize=8)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    fig.suptitle('Selected complete Si-F drag curves: constrained paths, not activation barriers')
    fig.savefig(args.output/'qsd_examples.png', dpi=170)
    plt.close(fig)
    pair_names = sorted({r['pair'] for rows in checks.values() for r in rows if r['kind']=='pair'})
    fig, axes = plt.subplots(2,5,figsize=(15,6),constrained_layout=True)
    for ax, pair in zip(axes.ravel(), pair_names):
        for b in BACKENDS:
            rows = sorted([r for r in checks[b] if r['kind']=='pair' and r['pair']==pair and r['status']=='ok'], key=lambda r:r['distance_A'])
            if rows:
                ax.plot([r['distance_A'] for r in rows], [r['radial_force_eV_A'] for r in rows], '.-', color=COLORS[b], label=b)
        ax.axhline(0, color='black', lw=.6)
        ax.set(title=pair, xlabel='Separation (A)', ylabel='Radial force (eV/A)', yscale='symlog')
        ax.grid(alpha=.2)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle('Reference-free pair scans: positive radial force is repulsive')
    fig.savefig(args.output/'pair_scans.png', dpi=170)
    plt.close(fig)
    for b in BACKENDS:
        rows = sorted([r for r in data[b] if r['id'] in common and r['track']=='etch'],
                      key=lambda r:r['metrics']['vector_max'], reverse=True)[:3]
        structures = []
        for r in rows:
            atoms = Atoms(r['symbols'], positions=r['positions_A'], cell=r['cell_A'], pbc=r['pbc'])
            atoms.info.update(case_id=r['id'], model=b, selection=r['selection'])
            atoms.arrays['reference_force'] = np.asarray(r['forces_ref_eV_A'])
            atoms.arrays['predicted_force'] = np.asarray(r['forces_pred_eV_A'])
            atoms.arrays['force_error_norm'] = np.linalg.norm(atoms.arrays['predicted_force']-atoms.arrays['reference_force'], axis=1)
            structures.append(atoms)
        if structures:
            write(args.output/f'worst_etch_{b}.extxyz', structures)


if __name__ == '__main__':
    main()
