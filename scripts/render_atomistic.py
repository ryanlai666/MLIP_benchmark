"""ASE trajectory ingestion with PyVista/VTK ball-and-stick rendering."""
import argparse
import hashlib
import json
from pathlib import Path
import importlib.metadata
import numpy as np
import pyvista as pv
from ase.io import read
from ase.data import covalent_radii
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

ROOT=Path('results/short-md')
COLORS={'Si':'#d7a04a','O':'#e44c5c','C':'#36465e','F':'#22bc92'}
W,H=1120,700
BG='#f4f7fb'

def font(size,bold=False):
    path=Path('C:/Windows/Fonts')/('segoeuib.ttf' if bold else 'segoeui.ttf')
    return ImageFont.truetype(str(path),size)

def panel(atoms,ids,plotter,cell=False):
    positions=atoms.positions[ids]
    numbers=atoms.numbers[ids]
    symbols=np.array(atoms.get_chemical_symbols())[ids]
    for symbol in sorted(set(symbols)):
        mask=symbols==symbol
        points=pv.PolyData(positions[mask])
        points['radius']=covalent_radii[numbers[mask]]*.46
        sphere=pv.Sphere(radius=1,theta_resolution=24,phi_resolution=24)
        mesh=points.glyph(scale='radius',orient=False,geom=sphere)
        plotter.add_mesh(mesh,color=COLORS[symbol],smooth_shading=True,specular=.45,specular_power=30,ambient=.22)
    delta=positions[:,None,:]-positions[None,:,:]
    distances=np.linalg.norm(delta,axis=-1)
    cutoff=1.15*(covalent_radii[numbers,None]+covalent_radii[numbers][None,:])
    pairs=np.argwhere(np.triu((distances<cutoff)&(distances>.2),1))
    if len(pairs):
        points=positions[pairs].reshape(-1,3)
        lines=np.column_stack([np.full(len(pairs),2),np.arange(len(pairs))*2,np.arange(len(pairs))*2+1]).ravel()
        bonds=pv.PolyData(points,lines=lines).tube(radius=.085,n_sides=10)
        plotter.add_mesh(bonds,color='#a6b1bf',smooth_shading=True,specular=.15)
    if cell:
        lengths=atoms.cell.lengths()
        box=pv.Box(bounds=(0,lengths[0],0,lengths[1],0,lengths[2])).extract_all_edges()
        plotter.add_mesh(box,color='#a5b2c4',line_width=1.4)


def compose(raw,name,time,interface):
    canvas=Image.new('RGB',(W,H+150),'#f4f7fb')
    canvas.paste(Image.fromarray(raw).convert('RGB'),(0,100))
    d=ImageDraw.Draw(canvas)
    title='CF2 impact on silica' if interface else 'Diamond silicon | '+name.upper()
    d.text((32,15),title,font=font(30,True),fill='#162a43')
    device='CUDA / RTX 3070' if name.endswith('gpu') else 'CPU'
    sub=f'30 eV neutral impact | 153 atoms | DPA-3.3 / OMat24 | {device}' if interface else '100 fs NVE | 8 simulated atoms | 2 x 2 x 2 periodic display'
    d.text((34,56),sub,font=font(17),fill='#506178')
    d.text((925,25),f'{time:5.1f} fs',font=font(27,True),fill='#162a43')
    if interface:
        d.text((40,102),'SLAB OVERVIEW',font=font(16,True),fill='#506178')
        d.text((W//2+30,102),'IMPACT REGION',font=font(16,True),fill='#506178')
        footer='Vacuum cropped; bottom fixed. Bonds are distance-based visual guides, not reaction assignments.'
    else:
        footer='Actual atomic displacements, no magnification. Bonds use 1.15 x summed covalent radii.'
    x=35
    for symbol,color in COLORS.items():
        if not interface and symbol!='Si':continue
        d.ellipse((x,H+108,x+15,H+123),fill=color)
        d.text((x+22,H+102),symbol,font=font(17),fill='#263a52'); x+=83
    d.text((390 if interface else 120,H+105),footer,font=font(13),fill='#506178')
    return canvas


def render(name,preview=False):
    folder=ROOT/name
    trajectory=folder/'trajectory.extxyz'
    digest=hashlib.sha256(trajectory.read_bytes()).hexdigest()
    frames=read(trajectory,index=':')
    interface=name.startswith('interface')
    if not interface:frames=[a.repeat((2,2,2)) for a in frames]
    all_ids=np.arange(len(frames[0]))
    zoom_ids=np.where(frames[0].positions[:,2]>17)[0] if interface else all_ids
    images=[]
    selected=[0] if preview else range(len(frames))
    for index in selected:
        atoms=frames[index]
        pl=pv.Plotter(off_screen=True,window_size=(W,H),shape=(1,2) if interface else (1,1),border=False)
        pl.set_background(BG)
        pl.enable_anti_aliasing('ssaa')
        for col,ids in enumerate([all_ids,zoom_ids] if interface else [all_ids]):
            pl.subplot(0,col)
            pl.set_background(BG)
            panel(atoms,ids,pl,cell=not interface)
            if interface:
                target=(5,5,13) if col==0 else (5,5,22.2)
                position=(32,-48,25) if col==0 else (17,-29,29)
                pl.camera_position=[position,target,(0,0,1)]
                pl.enable_parallel_projection()
                pl.camera.parallel_scale=14.3 if col==0 else 7.0
            else:
                pl.camera_position=[(22,-27,21),(5.3,5.3,5.3),(0,0,1)]
                pl.enable_parallel_projection(); pl.camera.parallel_scale=9
        raw=pl.screenshot(return_img=True); pl.close()
        images.append(compose(raw,name,atoms.info['time_fs'],interface))
        if index%10==0:print(name,'frame',index,flush=True)
    if preview:
        images[0].save(folder/'preview.png'); return
    images[0].save(folder/'initial.png')
    images[-1].save(folder/'final.png')
    images[0].save(folder/'animation.gif',save_all=True,append_images=images[1:],duration=100,loop=0,optimize=False)
    with imageio.get_writer(folder/'animation.mp4',fps=10,codec='libx264',quality=8,macro_block_size=2) as writer:
        for im in images:writer.append_data(np.asarray(im))
    assert hashlib.sha256(trajectory.read_bytes()).hexdigest()==digest
    metadata=dict(renderer='ASE + PyVista/VTK',trajectory_sha256=digest,frames=len(images),fps=10,
                  packages={p:importlib.metadata.version(p) for p in ['ase','pyvista','vtk','pillow','imageio','imageio-ffmpeg']},
                  bond_rule='Euclidean distance < 1.15 * sum of ASE covalent radii; no periodic bonds across display boundary',
                  atom_radius='0.46 * ASE covalent radius; visualization only',
                  views='Overview plus fixed atom selection with initial z > 17 A; vacuum cropped' if interface else '2x2x2 periodic copies of the 8-atom cell',
                  coordinate_scaling=1)
    (folder/'render_manifest.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--names',nargs='+',default=['interface','interface-gpu','mace','nequip','deepmd'])
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    for name in args.names:render(name,args.preview)
