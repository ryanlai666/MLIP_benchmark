"""PBC-aware trajectory rendering: true cell vectors, image atoms and boundary bonds."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyvista as pv
from ase.data import chemical_symbols, covalent_radii
from ase.io import read
from PIL import Image, ImageDraw, ImageFont

from periodic_geometry import periodic_cloud, cell_edges, bond_segments

W,H=1120,700
COLORS={'Si':'#d7a04a','O':'#e44c5c','C':'#36465e','F':'#22bc92'}
BG='#f4f7fb'
ROOT=Path('results/long-md')


def font(size,bold=False):
    return ImageFont.truetype(str(Path('C:/Windows/Fonts')/('segoeuib.ttf' if bold else 'segoeui.ttf')),size)


def lines_mesh(segments,radius=None):
    points=np.asarray(segments).reshape(-1,3)
    lines=np.column_stack([np.full(len(segments),2),np.arange(len(segments))*2,np.arange(len(segments))*2+1]).ravel()
    mesh=pv.PolyData(points,lines=lines)
    return mesh.tube(radius=radius,n_sides=8) if radius else mesh


def scene(atoms,plotter,interface):
    axes=np.array([True,True,False]) if interface else atoms.pbc
    positions,numbers,ghost,origins=periodic_cloud(atoms,halo=2.8,image_axes=axes)
    sphere=pv.Sphere(radius=1,theta_resolution=16,phi_resolution=16)
    for is_ghost in [True,False]:
        for number in np.unique(numbers):
            mask=(numbers==number)&(ghost==is_ghost)
            if not mask.any():continue
            points=pv.PolyData(positions[mask]);points['radius']=covalent_radii[numbers[mask]]*.46
            plotter.add_mesh(points.glyph(scale='radius',orient=False,geom=sphere),
                             color=COLORS.get(chemical_symbols[number],'#8b82b6'),opacity=.20 if is_ghost else 1.,
                             smooth_shading=True,specular=.35,ambient=.28)
    solid,periodic=bond_segments(positions,numbers,ghost)
    if len(solid):plotter.add_mesh(lines_mesh(solid,.075),color='#9da9b8',smooth_shading=True)
    if len(periodic):plotter.add_mesh(lines_mesh(periodic,.055),color='#7297c4',opacity=.28,smooth_shading=True)
    if atoms.cell.rank==3:
        plotter.add_mesh(lines_mesh(cell_edges(atoms.cell.array)),color='#176aa4',line_width=3)
        # Explicit coordinate-vector arrows use the real (including triclinic) cell vectors.
        for axis,color in enumerate(['#bc4850','#348966','#3d68b3']):
            vector=atoms.cell.array[axis]
            arrow=vector/np.linalg.norm(vector)*min(np.linalg.norm(vector)*.22,2.)
            plotter.add_arrows(np.zeros((1,3)),arrow[None],mag=1,color=color)
    return positions[:len(atoms)]


def frame_image(atoms,name,interface,duration,carbon,subtitle=None):
    pl=pv.Plotter(off_screen=True,window_size=(W,H),shape=(1,2) if interface else (1,1),border=False)
    pl.set_background(BG);pl.enable_anti_aliasing('ssaa')
    for col in range(2 if interface else 1):
        pl.subplot(0,col);pl.set_background(BG)
        central=scene(atoms,pl,interface)
        if interface:
            height=float(atoms.cell[2,2])
            if col==0:
                target=np.array([5,5,height/2]);scale=height*.54
            else:
                target=central[carbon].copy();target[2]=max(22.,target[2]-2)
                scale=9.5
            pl.camera_position=[target+np.array([18,-48,12]),target,(0,0,1)]
            pl.enable_parallel_projection();pl.camera.parallel_scale=scale
        else:
            center=atoms.cell.array.sum(axis=0)/2
            pl.camera_position=[center+np.array([16,-24,17]),center,(0,0,1)]
            pl.enable_parallel_projection();pl.camera.parallel_scale=9.5
    raw=pl.screenshot(return_img=True);pl.close()
    canvas=Image.new('RGB',(W,H+170),BG);canvas.paste(Image.fromarray(raw).convert('RGB'),(0,105))
    d=ImageDraw.Draw(canvas)
    title='CF2 impact: approach, collision and departure' if interface else 'Periodic silicon | '+name.upper()
    d.text((25,12),title,font=font(25,True),fill='#162a43')
    t=float(atoms.info['time_fs'])
    d.text((895,15),f'{t:7.1f} fs',font=font(25,True),fill='#162a43')
    lengths=' x '.join(f'{v:.2f}' for v in atoms.cell.lengths())
    pbc='/'.join(axis if flag else '-' for axis,flag in zip('xyz',atoms.pbc))
    d.text((27,50),f'PBC: {pbc} | Cell: {lengths} A | {len(atoms)} simulated atoms | 0-{duration:g} fs',font=font(17),fill='#506178')
    if interface:
        d.text((28,82),'FULL PERIODIC CELL',font=font(15,True),fill='#235c91')
        d.text((590,82),'FOLLOWING ORIGINAL PROJECTILE CARBON',font=font(15,True),fill='#235c91')
    elif subtitle:
        d.text((27,80),subtitle,font=font(14),fill='#506178')
    x=28
    for symbol,color in COLORS.items():
        if not interface and symbol!='Si':continue
        d.ellipse((x,H+113,x+13,H+126),fill=color)
        d.text((x+20,H+107),symbol,font=font(16),fill='#263a52');x+=73
    d.text((340 if interface else 110,H+108),'Blue outline: simulation cell | Faded atoms/bonds: periodic images',font=font(16),fill='#506178')
    d.text((28,H+140),'Image copies: x/y (z repeats across the vacuum). Proximity bonds are visual guides.' if interface else
           'Image copies: x/y/z. True coordinates; no displacement magnification. Red/green/blue arrows: a/b/c.',font=font(14),fill='#506178')
    return canvas


def render(name,preview=False,root=ROOT,stride=4,subtitle=None):
    if stride<1:raise ValueError('Stride must be positive')
    folder=root/name;source=folder/'trajectory.extxyz'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    frames=read(source,index=':')
    manifest=json.loads((folder/'manifest.json').read_text())
    interface='substrate_atoms' in manifest
    carbon=manifest.get('carbon_index',manifest.get('substrate_atoms',0))
    times=[float(a.info['time_fs']) for a in frames]
    chosen=[0] if preview else sorted(set([i for i,t in enumerate(times) if (interface and t<=250) or i%stride==0]+[len(frames)-1]))
    images=[]
    movie=None if preview else imageio.get_writer(folder/'animation.mp4',fps=10,codec='libx264',quality=8,macro_block_size=2)
    try:
        for order,index in enumerate(chosen):
            im=frame_image(frames[index],name,interface,times[-1],carbon,subtitle)
            if preview:im.save(folder/'preview.png')
            else:
                movie.append_data(np.asarray(im))
                if order==0:im.save(folder/'initial.png')
                if order==len(chosen)-1:im.save(folder/'final.png')
                gif=im.copy();gif.thumbnail((840,654));images.append(gif)
            if order%20==0:print(name,'render',order+1,'/',len(chosen),flush=True)
    finally:
        if movie is not None:movie.close()
    if preview:return
    images[0].save(folder/'animation.gif',save_all=True,append_images=images[1:],duration=100,loop=0,optimize=False)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise RuntimeError('Trajectory changed during rendering')
    metadata=dict(renderer='ASE + PyVista/VTK; explicit PBC',trajectory_sha256=digest,frames=len(chosen),fps=10,
                  saved_frames=len(frames),frame_indices=chosen,physical_times_fs=[times[i] for i in chosen],
                  duration_fs=times[-1],pbc=frames[0].pbc.tolist(),cell_A=frames[0].cell.tolist(),
                  periodic_image_axes=['x','y'] if interface else ['x','y','z'],image_halo_A=2.8,
                  bond_rule='Euclidean proximity in the expanded periodic image cloud; 1.15 x covalent-radius sum; includes boundary-crossing bonds',
                  cell_rule='12 edges from the true cell vectors; wrapped display only, source unchanged',
                  camera='Full cell plus camera following original carbon' if interface else 'Fixed camera around the simulated cell',
                  playback='10 fps; dense first 250 fs for impact, then configured stride; physical timestamps are authoritative',
                  packages={p:importlib.metadata.version(p) for p in ['ase','pyvista','vtk','pillow','imageio']})
    (folder/'render_manifest.json').write_text(json.dumps(metadata,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--names',nargs='+',default=['interface-event-gpu','mace','nequip','deepmd'])
    p.add_argument('--stride',type=int,default=4)
    p.add_argument('--subtitle')
    p.add_argument('--preview',action='store_true')
    a=p.parse_args()
    for name in a.names:render(name,a.preview,a.root,a.stride,a.subtitle)


if __name__=='__main__':main()
