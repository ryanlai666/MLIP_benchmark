"""Tiny gas-phase fragment geometry check with the DPA OMol25 branch."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time
from run_pilot import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/deepmd/DPA-3.3-1M.pt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from ase import Atoms
    from ase.optimize import BFGS
    from deepmd.calculator import DP
    torch.set_num_threads(4)
    manifest = {"started_utc":datetime.now(timezone.utc).isoformat(), "checkpoint":str(args.model),
                "checkpoint_sha256":sha(args.model), "head":"OMol25", "charge":0, "multiplicity":1,
                "device":"cpu", "threads":4, "status":"running",
                "packages":{p:importlib.metadata.version(p) for p in ["deepmd-kit", "torch", "ase", "numpy"]},
                "limitations":["Two gas-phase fragments only; not surface or plasma validation",
                               "Experimental comparison includes reference DFT error and vibrational effects",
                               "SiF4 reference is derived from B0, not an exact equilibrium bond length"]}
    path = args.output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    try:
        calc = DP(model=str(args.model.resolve()), head="OMol25")
        tetra = np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]]) / np.sqrt(3)
        systems = [
            ("F2", Atoms("F2", positions=[[0,0,0],[0,0,1.55]]), 1.4119,
             "https://cccbdb.nist.gov/exp2x.asp?casno=7782414&charge=0", "equilibrium re from listed Cartesian coordinates"),
            ("SiF4", Atoms("SiF4", positions=np.vstack([np.zeros(3),tetra*1.65])), 1.554,
             "https://cccbdb.nist.gov/exp2x.asp?casno=7783611&charge=0", "bond length derived from B0"),
        ]
        rows = []
        for name, atoms, reference, source, convention in systems:
            atoms.set_cell([20,20,20]); atoms.center(); atoms.pbc = False
            atoms.info["charge_spin"] = np.array([0, 1])
            atoms.calc = calc
            start = time.perf_counter()
            optimizer = BFGS(atoms, logfile=str(args.output / f"{name}-opt.log"),
                             trajectory=str(args.output / f"{name}.traj"))
            converged = bool(optimizer.run(fmax=0.005, steps=100))
            bonds = atoms.get_distances(0, range(1,len(atoms)))
            forces = atoms.get_forces()
            row = {"molecule":name, "natoms":len(atoms), "converged":converged,
                   "steps":optimizer.nsteps, "seconds":time.perf_counter()-start,
                   "mean_bond_A":float(np.mean(bonds)), "reference_bond_A":reference,
                   "absolute_difference_A":float(abs(np.mean(bonds)-reference)),
                   "reference":source, "reference_convention":convention,
                   "energy_eV":float(atoms.get_potential_energy()),
                   "max_force_norm_eV_A":float(np.linalg.norm(forces,axis=1).max()),
                   "positions_A":atoms.positions.tolist(), "forces_eV_A":forces.tolist()}
            if not np.isfinite(row["energy_eV"]) or not np.isfinite(forces).all():
                raise ValueError("Nonfinite output")
            rows.append(row)
            (args.output / "summary.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
        manifest["status"] = "completed" if all(r["converged"] for r in rows) else "unconverged"
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
