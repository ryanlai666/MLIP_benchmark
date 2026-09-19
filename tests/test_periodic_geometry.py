"""Boundary-crossing display and vacuum-extension invariants."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from periodic_geometry import cell_edges,periodic_cloud,bond_segments,carbon_departure,departure_state
from run_impact_event import expanded_initial


class PeriodicGeometryTests(unittest.TestCase):
    def test_boundary_bond_uses_nearby_image_not_long_line(self):
        atoms=Atoms('CC',positions=[[.1,5,5],[9.9,5,5]],cell=[10,10,10],pbc=True)
        original=atoms.positions.copy()
        pos,nums,ghost,ids=periodic_cloud(atoms,halo=1)
        solid,periodic=bond_segments(pos,nums,ghost)
        self.assertEqual(len(solid),0)
        # 0.2 A is the explicit visual lower cutoff; use a larger separation below.
        atoms.positions[1,0]=9.5
        pos,nums,ghost,ids=periodic_cloud(atoms,halo=1)
        solid,periodic=bond_segments(pos,nums,ghost)
        lengths=np.linalg.norm(periodic[:,0]-periodic[:,1],axis=1)
        self.assertTrue(np.any(np.isclose(lengths,.6)))
        self.assertTrue((lengths<2).all())
        np.testing.assert_allclose(original[0],atoms.positions[0])

    def test_nonperiodic_directions_have_no_images(self):
        atoms=Atoms('C',positions=[[.1,.1,.1]],cell=[10,10,10],pbc=[True,False,False])
        original=atoms.positions.copy()
        pos,nums,ghost,ids=periodic_cloud(atoms,halo=1)
        np.testing.assert_allclose(pos[:,1:],np.tile([.1,.1],(len(pos),1)))
        self.assertTrue(ghost.any())
        np.testing.assert_array_equal(atoms.positions,original)
        atoms.pbc=False
        self.assertEqual(len(periodic_cloud(atoms)[0]),1)

    def test_triclinic_edges_follow_cell_vectors(self):
        cell=np.array([[4,0,0],[1,5,0],[.2,.3,6]])
        edges=cell_edges(cell)
        self.assertEqual(edges.shape,(12,2,3))
        for vector in edges[:,1]-edges[:,0]:
            self.assertTrue(any(np.allclose(vector,v) for v in cell))
        self.assertTrue(any(np.allclose(v,cell.sum(axis=0)) for v in edges.reshape(-1,3)))

    def test_departure_excludes_substrate_atom_leaving_with_carbon(self):
        atoms=Atoms('SiSiSiCO',positions=[[0,0,0],[2,0,0],[0,2,0],[0,0,12],[0,0,13]],cell=[20,20,40],pbc=True)
        ids,distance=carbon_departure(atoms,3)
        self.assertEqual(ids,[3,4])
        self.assertAlmostEqual(distance,12.)

    def test_departure_uses_fragment_com_not_internal_vibration(self):
        atoms=Atoms('SiSiSiCO',positions=[[0,0,0],[2,0,0],[0,2,0],[0,0,15],[0,0,16]],cell=[20,20,50],pbc=True)
        velocities=np.zeros((5,3));velocities[3,2]=-.1;velocities[4,2]=.3;atoms.set_velocities(velocities)
        ids,distance,z,vz,away=departure_state(atoms,3,0)
        self.assertEqual(ids,[3,4])
        self.assertGreater(vz,0)
        self.assertTrue(away)

    def test_vacuum_extension_preserves_coordinates_and_velocities(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            atoms=Atoms('SiCF2',positions=[[1,1,1],[2,2,10],[3,2,10],[2,3,10]],cell=[10,10,30],pbc=True)
            velocities=np.zeros((4,3));velocities[1:,2]=-.2;atoms.set_velocities(velocities)
            write(folder/'trajectory.extxyz',atoms)
            (folder/'manifest.json').write_text(json.dumps({'fixed_indices':[0]}))
            new,_=expanded_initial(folder,100)
            np.testing.assert_allclose(new.positions,atoms.positions)
            np.testing.assert_allclose(new.get_velocities(),velocities)
            self.assertEqual(new.cell[2,2],100)
            np.testing.assert_array_equal(new.pbc,[True,True,True])
            with self.assertRaises(ValueError):expanded_initial(folder,30)


if __name__=='__main__':unittest.main()
