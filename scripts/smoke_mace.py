"""Single point on ideal Si; tests plumbing, not accuracy. Downloads MACE weights."""
import argparse
import json
from pathlib import Path
import numpy as np
from ase.io import read
from mace.calculators import mace_mp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
args = parser.parse_args()
atoms = read(Path(__file__).resolve().parents[1] / "data/structures/Si.vasp")
atoms.calc = mace_mp(model="medium", device=args.device, default_dtype="float64", dispersion=False)
energy = float(atoms.get_potential_energy())
forces = atoms.get_forces()
stress = atoms.get_stress()
if not (np.isfinite(energy) and np.isfinite(forces).all() and np.isfinite(stress).all()):
    raise RuntimeError("Nonfinite model output")
print(json.dumps({"purpose": "smoke_only", "model": "MACE-MP-0 medium", "device": args.device,
                  "energy_eV": energy, "forces_eV_per_A": forces.tolist(),
                  "stress_eV_per_A3": stress.tolist()}, indent=2))
