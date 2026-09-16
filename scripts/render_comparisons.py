"""Synchronized comparison layouts from atomistic renderings and ASE timestamps."""
from pathlib import Path
import json
import numpy as np
from ase.io import read
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from render_atomistic import font
ROOT=Path('results/short-md')

def comparison(names,output,interface):
    movies=[imageio.get_reader(ROOT/name/'animation.mp4') for name in names]
    times=[[float(a.info['time_fs']) for a in read(ROOT/name/'trajectory.extxyz',index=':')] for name in names]
    assert all(t==times[0] for t in times)
    assert all(m.count_frames()==len(times[0]) for m in movies)
    images=[]
    width,height=(1120,800) if interface else (1260,420)
    for i,t in enumerate(times[0]):
        canvas=Image.new('RGB',(width,height),'#f4f7fb'); d=ImageDraw.Draw(canvas)
        d.text((25,14),'CF2 / silica: CPU vs GPU' if interface else 'Silicon crystal: model comparison',font=font(26,True),fill='#162a43')
        d.text((width-155,18),f'{t:5.1f} fs',font=font(24,True),fill='#162a43')
        for j,(name,movie) in enumerate(zip(names,movies)):
            im=Image.fromarray(movie.get_data(i)).convert('RGB')
            if interface:
                tile=im.crop((560,140,1120,800))
                x=j*560; y=110; label='CPU | 4 threads' if j==0 else 'GPU | RTX 3070'
            else:
                tile=im.crop((0,100,1120,800)).resize((420,263),Image.Resampling.LANCZOS)
                x=j*420; y=110; label={'mace':'MACE-MP-0b3','nequip':'NequIP-OAM-S','deepmd':'DPA-3.3 / OMat24'}[name]
            canvas.paste(tile,(x,y)); d.text((x+25,70),label,font=font(21,True),fill='#506178')
            if j:d.line((x,66,x,height-28),fill='#d7e0eb',width=2)
        footer='Same initial coordinates, velocities and model; impact-region close-up. No displacement magnification.' if interface else 'Same velocity seed; model-specific EOS lattice. 8 atoms simulated; 2 x 2 x 2 copies displayed.'
        d.text((25,height-27),footer,font=font(15),fill='#506178'); images.append(canvas)
    images[0].save(ROOT/(output+'.gif'),save_all=True,append_images=images[1:],duration=100,loop=0,optimize=False)
    images[0].save(ROOT/(output+'.png'))
    with imageio.get_writer(ROOT/(output+'.mp4'),fps=10,codec='libx264',quality=8,macro_block_size=2) as writer:
        for im in images:writer.append_data(np.asarray(im))
    (ROOT/(output+'.json')).write_text(json.dumps(dict(panels=names,frames=len(images),physical_times_fs=times[0],fps=10),indent=2))
    for movie in movies:movie.close()
    print(output,len(images),'synchronized frames')

if __name__=='__main__':
    comparison(['interface','interface-gpu'],'interface_comparison',True)
    comparison(['mace','nequip','deepmd'],'crystal_comparison',False)
