"""Audit the extended trajectory and report geometric impact/departure milestones."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ase import units
from ase.io import read

from periodic_geometry import carbon_departure,departure_state
from run_pilot import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,default=Path('results/long-md/interface-event-gpu'))
    args=p.parse_args();folder=args.folder
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['status']!='completed':raise ValueError('Simulation is not completed')
    with (folder/'thermodynamics.csv').open() as f: rows=list(csv.DictReader(f))
    table={k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
    if len(rows)!=manifest['steps']+1:raise ValueError('Missing integration steps')
    frames=read(folder/'trajectory.extxyz',index=':')
    carbon=manifest['carbon_index'];surface=manifest['initial_surface_z_A']
    times=np.array([float(a.info['time_fs']) for a in frames])
    if len(frames)!=manifest['saved_frames'] or times[-1]!=manifest['duration_fs']:
        raise ValueError('Trajectory length mismatch')
    delta=(table['total_eV']-table['total_eV'][0])/manifest['natoms']*1000
    if not np.isclose(abs(delta).max(),manifest['max_abs_energy_change_meV_atom']):
        raise ValueError('Energy summary differs from raw data')
    if table['top_clearance_A'].min()<manifest['boundary_guard_A']:
        raise ValueError('Vacuum boundary guard violated')
    phases=[];tail_start=None
    for t,atoms in zip(times,frames):
        cluster,separation,com_z,vz,away=departure_state(atoms,carbon,surface)
        tail_start=(t if tail_start is None else tail_start) if away else None
        phases.append(dict(time_fs=float(t),carbon_component_size=len(cluster),
                           carbon_component_indices=' '.join(map(str,cluster)),
                           distance_to_largest_component_A=separation,component_com_z_A=com_z,component_com_vz_A_fs=vz,
                           outward_and_separated=bool(away)))
    complete=tail_start is not None and times[-1]-tail_start>=200
    deepest=int(np.argmin(table['carbon_z_A']))
    final_cluster,final_separation=carbon_departure(frames[-1],carbon)
    summary=dict(trajectory_sha256=sha(folder/'trajectory.extxyz'),thermodynamics_sha256=sha(folder/'thermodynamics.csv'),
                 report_script_sha256=sha(Path(__file__)),duration_fs=float(times[-1]),
                 deepest_time_fs=float(table['time_fs'][deepest]),deepest_carbon_z_A=float(table['carbon_z_A'][deepest]),
                 depth_below_initial_surface_A=float(surface-table['carbon_z_A'][deepest]),
                 final_carbon_z_A=float(frames[-1].positions[carbon,2]),final_carbon_component_indices=final_cluster,
                 final_carbon_component_elements=[frames[-1][i].symbol for i in final_cluster],
                 final_component_separation_A=final_separation,
                 departure_confirmed=bool(complete),final_sustained_departure_start_fs=float(tail_start) if tail_start is not None else None,
                 departure_confirmation_time_fs=float(tail_start+200) if complete else None,
                 minimum_top_clearance_A=float(table['top_clearance_A'].min()),
                 max_abs_energy_change_meV_atom=float(abs(delta).max()),
                 final_fluorine_z_A=[float(z) for z in frames[-1].positions[carbon+1:,2]],
                 criterion='Carbon-containing proximity component >=8 A from largest slab component, component center of mass >10 A above initial surface and moving outward, sustained for >=200 fs through final frame.',
                 limitations='Connectivity uses 1.2 x covalent radii and is geometric, not a chemical product or yield assignment. Departure does not imply substrate equilibration.')
    (folder/'event_summary.json').write_text(json.dumps(summary,indent=2))
    with (folder/'event_phases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(phases[0]));writer.writeheader();writer.writerows(phases)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    t=table['time_fs']/1000
    axes[0,0].plot(t,table['carbon_z_A'],label='Original projectile carbon',color='#36465e')
    axes[0,0].axhline(surface,color='#b77f30',ls='--',label='Initial surface height')
    axes[0,0].axhline(manifest['initial_cell_A'][2][2],color='#235c91',label='Periodic cell top')
    axes[0,0].axhline(manifest['initial_cell_A'][2][2]-manifest['boundary_guard_A'],color='#235c91',ls=':',label='Re-entry guard')
    axes[0,0].set(ylabel='Unwrapped height z (A)',title='Full event and periodic-cell boundary');axes[0,0].legend(fontsize=8)
    axes[0,1].plot(times/1000,[r['distance_to_largest_component_A'] for r in phases],color='#26845b')
    axes[0,1].axhline(8,color='gray',ls='--',label='Departure separation criterion')
    axes[0,1].set(ylabel='Minimum periodic separation (A)',title='Carbon-containing component / main slab');axes[0,1].legend(fontsize=8)
    axes[1,0].plot(t,delta,color='#663f91');axes[1,0].set(ylabel='Total-energy change (meV/atom)',title='NVE diagnostic')
    axes[1,1].plot(t,table['top_clearance_A'],color='#235c91')
    axes[1,1].axhline(manifest['boundary_guard_A'],color='gray',ls='--',label='Guard distance')
    axes[1,1].set(ylabel='Top clearance of highest atom (A)',title='No z-periodic re-entry');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel('Physical time (ps)');ax.grid(alpha=.2)
        ax.axvline(summary['deepest_time_fs']/1000,color='#b77f30',ls=':',alpha=.7)
        if complete:ax.axvline(summary['departure_confirmation_time_fs']/1000,color='#26845b',ls=':',alpha=.7)
    fig.suptitle('30 eV neutral CF2 impact | PBC x/y/z | enlarged vacuum | dotted milestones: deepest point / sustained departure')
    fig.savefig(folder/'event_diagnostics.png',dpi=170);plt.close(fig)
    lines=['# Extended impact with explicit periodic boundaries','',
           f"The impact runs for **{times[-1]/1000:g} ps**, compared with the earlier 50 fs preview. "
           f"The cell is 10 x 10 x {manifest['initial_cell_A'][2][2]:g} A with PBC in x/y/z. "
           'It starts from the earlier saved initial positions and velocities; only the vacuum height changes.', '',
           '![Full event](animation.gif)','', '[Full-size MP4](animation.mp4) / [final view](final.png) / [event measurements](event_summary.json)', '',
           f"The carbon reaches its deepest point at **{summary['deepest_time_fs']:.2f} fs**, "
           f"{summary['depth_below_initial_surface_A']:.2f} A below the initial surface. "
           f"Its final height is {summary['final_carbon_z_A']:.2f} A. "
           f"The minimum top clearance throughout the run is {summary['minimum_top_clearance_A']:.2f} A "
           f"against a {manifest['boundary_guard_A']:g} A guard.", '',
           (f"**Sustained departure is confirmed at {summary['departure_confirmation_time_fs']:.0f} fs** and remains satisfied through "
            f"{times[-1]:.0f} fs." if complete else '**A sustained departure endpoint was not established within this run.**'), '',
           summary['criterion'], '',
           'The separation diagnostic uses periodic minimum images. It can decrease late in the run as the departing '
           'fragment approaches the next periodic slab image, while its unwrapped height continues increasing. '
           'The boundary guard keeps it outside the next periodic image interaction range.', '',
           '![Event diagnostics and periodic-boundary clearance](event_diagnostics.png)', '',
           'The left animation panel shows the entire periodic cell, including vacuum. The right panel follows the original '
           'projectile carbon. Faded image atoms and their bonds show lateral periodic continuation; the z direction repeats '
           'across the vacuum. The blue outline is the actual simulation cell, not a display supercell.', '',
           f"Maximum absolute total-energy change: {summary['max_abs_energy_change_meV_atom']:.5f} meV/atom. "
           'This is a single cold, unrelaxed-slab impact with a fixed bottom and neutral ground-state MLIP. '
           'The geometric departure criterion establishes an endpoint for the primary impact/departure event; '
           'it does not establish an etch yield, chemical product identity, complete substrate relaxation, or plasma accuracy.', '',
           '## Reproduction','', '```powershell',
           '.venv-deepmd-gpu/Scripts/python.exe scripts/run_impact_event.py',
           '.venv-render/Scripts/python.exe scripts/report_impact_event.py',
           '.venv-render/Scripts/python.exe scripts/render_periodic_md.py --names interface-event-gpu',
           '```','', 'Output directories for new simulations must not already exist. The old 1 ps and 50 fs runs remain separate historical results.','']
    (folder/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
