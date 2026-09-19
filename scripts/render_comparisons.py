"""Compare rendered movies using their recorded physical timestamps."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
from render_periodic_md import font


def comparison(names,output,interface,root=Path('results/long-md')):
    metadata=[json.loads((root/name/'render_manifest.json').read_text()) for name in names]
    times=[m['physical_times_fs'] for m in metadata]
    if not all(t==times[0] for t in times):
        raise ValueError('Comparison movies must use identical physical timestamps')
    movies=[imageio.get_reader(root/name/'animation.mp4') for name in names]
    if not all(m.count_frames()==len(times[0]) for m in movies):
        raise ValueError('Movie frames differ from render manifests')
    width=560*len(names);height=626
    images=[]
    hashes={name:hashlib.sha256((root/name/'animation.mp4').read_bytes()).hexdigest() for name in names}
    try:
        with imageio.get_writer(root/(output+'.mp4'),fps=10,codec='libx264',quality=8,macro_block_size=2) as writer:
            for i,t in enumerate(times[0]):
                canvas=Image.new('RGB',(width,height),'#f4f7fb');d=ImageDraw.Draw(canvas)
                d.text((25,12),'Periodic interface: CPU / GPU' if interface else f'Periodic silicon: {times[0][-1]:g} fs comparison',font=font(27,True),fill='#162a43')
                d.text((width-180,16),f'{t:7.1f} fs',font=font(24,True),fill='#162a43')
                for j,(name,movie) in enumerate(zip(names,movies)):
                    image=Image.fromarray(movie.get_data(i)).convert('RGB')
                    image.thumbnail((560,475),Image.Resampling.LANCZOS)
                    canvas.paste(image,(j*560,90))
                    d.text((j*560+25,57),{'mace':'MACE-MP-0b3','nequip':'NequIP-OAM-S','deepmd':'DPA-3.3 / OMat24','interface':'CPU','interface-gpu':'CUDA / GPU'}.get(name,name),font=font(22,True),fill='#235c91')
                    if j:d.line((j*560,60,j*560,height-35),fill='#cad6e4',width=2)
                d.text((25,height-30),'Blue: simulated cell. Faded: periodic images. Panels show the same physical time.',font=font(18),fill='#506178')
                writer.append_data(np.asarray(canvas))
                thumb=canvas.copy();thumb.thumbnail((1260,585));images.append(thumb)
        images[0].save(root/(output+'.png'))
        images[0].save(root/(output+'.gif'),save_all=True,append_images=images[1:],duration=100,loop=0,optimize=False)
    finally:
        for m in movies:m.close()
    (root/(output+'.json')).write_text(json.dumps(dict(panels=names,frames=len(images),physical_times_fs=times[0],fps=10,source_movie_sha256=hashes),indent=2))
    print(output,len(images),'synchronized frames')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/long-md'))
    p.add_argument('--interface',action='store_true')
    a=p.parse_args()
    if a.interface:
        comparison(['interface','interface-gpu'],'interface_comparison',True,a.root)
    else:
        comparison(['mace','nequip','deepmd'],'crystal_comparison',False,a.root)
