"""Periodic display geometry without changing trajectory coordinates."""
from itertools import product
import numpy as np
from ase.data import covalent_radii


def cell_edges(cell, origin=None):
    cell=np.asarray(cell,dtype=float)
    origin=np.zeros(3) if origin is None else np.asarray(origin)
    edges=[]
    for corner in product([0,1],repeat=3):
        for axis in range(3):
            if corner[axis]==0:
                other=list(corner);other[axis]=1
                edges.append([origin+np.array(corner)@cell,origin+np.array(other)@cell])
    return np.asarray(edges)


def periodic_cloud(atoms, halo=2.8, image_axes=None):
    """Central wrapped cell plus a clipped halo of real periodic image atoms."""
    axes=np.array(atoms.pbc if image_axes is None else image_axes,dtype=bool) & atoms.pbc
    central=atoms.copy()
    if atoms.cell.rank==3:
        central.wrap()
        fractional=central.get_scaled_positions(wrap=False)
        margin=halo*np.linalg.norm(np.linalg.inv(atoms.cell.array),axis=0)
    else:
        if axes.any(): raise ValueError('Periodic display requires a full-rank cell')
        return central.positions.copy(),central.numbers.copy(),np.zeros(len(atoms),dtype=bool),np.arange(len(atoms))
    positions=[central.positions.copy()];numbers=[atoms.numbers];ghost=[np.zeros(len(atoms),bool)];origins=[np.arange(len(atoms))]
    for shift in product(*[[-1,0,1] if flag else [0] for flag in axes]):
        if shift==(0,0,0):continue
        frac=fractional+shift
        keep=np.all((frac>=-margin-1e-10)&(frac<=1+margin+1e-10),axis=1)
        positions.append((frac[keep]@atoms.cell.array));numbers.append(atoms.numbers[keep])
        ghost.append(np.ones(keep.sum(),bool));origins.append(np.arange(len(atoms))[keep])
    return np.concatenate(positions),np.concatenate(numbers),np.concatenate(ghost),np.concatenate(origins)


def bond_segments(positions,numbers,ghost):
    d=np.linalg.norm(positions[:,None]-positions[None,:],axis=-1)
    cutoff=1.15*(covalent_radii[numbers,None]+covalent_radii[numbers][None,:])
    pairs=np.argwhere(np.triu((d<cutoff)&(d>.2),1))
    solid=~ghost[pairs].any(axis=1)
    return positions[pairs[solid]],positions[pairs[~solid]]


def carbon_departure(atoms,carbon,threshold=1.2):
    """Geometric component separation; proximity bonds are not chemical assignments."""
    distances=atoms.get_all_distances(mic=True)
    cut=threshold*(covalent_radii[atoms.numbers,None]+covalent_radii[atoms.numbers][None,:])
    adjacent=(distances<cut)&(distances>.2)
    components=[];unseen=set(range(len(atoms)))
    while unseen:
        stack=[unseen.pop()];component=set(stack)
        while stack:
            i=stack.pop()
            new=set(np.where(adjacent[i])[0]) & unseen
            unseen-=new;component|=new;stack.extend(new)
        components.append(component)
    slab=max(components,key=len)
    fragment=next(c for c in components if carbon in c)
    if fragment==slab:
        return sorted(fragment),0.
    separation=float(distances[np.ix_(sorted(fragment),sorted(slab))].min())
    return sorted(fragment),separation


def departure_state(atoms,carbon,surface):
    """A separated component must move outward as a whole, despite internal vibrations."""
    from ase import units
    cluster,separation=carbon_departure(atoms,carbon)
    masses=atoms.get_masses()[cluster]
    com_z=float(np.average(atoms.positions[cluster,2],weights=masses))
    com_vz=float(np.average(atoms.get_velocities()[cluster,2],weights=masses)*units.fs)
    away=separation>=8. and com_z>surface+10. and com_vz>0
    return cluster,separation,com_z,com_vz,bool(away)
