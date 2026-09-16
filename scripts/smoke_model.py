"""ASE single-point scaffold for user-supplied models; not an accuracy benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_calculator(config):
    backend = config["backend"]
    if backend not in {"mace", "nequip", "deepmd"}:
        raise ValueError(f"Unsupported backend: {backend}")
    path = (ROOT / config["checkpoint"]).resolve(strict=True)
    if backend == "mace":
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=str(path), device=config["device"],
                              default_dtype=config["default_dtype"])
    if backend == "nequip":
        from nequip.integrations.ase import NequIPCalculator
        return NequIPCalculator.from_compiled_model(
            compile_path=str(path), device=config["device"],
            chemical_species_to_atom_type_map=config["species_map"],
            energy_units_to_eV=config["energy_units_to_eV"],
            length_units_to_A=config["length_units_to_A"])
    from deepmd.calculator import DP
    # Device selection is backend/environment specific; do not invent a DP device argument.
    return DP(model=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--structure", type=Path, default=ROOT / "data/structures/Si.vasp")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    calc = make_calculator(config)
    import numpy as np
    from ase.io import read
    atoms = read(args.structure)
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    if forces.shape != (len(atoms), 3) or stress.shape != (6,):
        raise ValueError("Unexpected force/stress shape")
    if not (np.isfinite(energy) and np.isfinite(forces).all() and np.isfinite(stress).all()):
        raise ValueError("Nonfinite predictions")
    checkpoint = (ROOT / config["checkpoint"]).resolve()
    with checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"purpose": "smoke_only", "config": config,
              "checkpoint_sha256": checkpoint_hash,
              "structure_sha256": hashlib.sha256(args.structure.read_bytes()).hexdigest(),
              "energy_eV": energy, "forces_eV_per_A": forces.tolist(),
              "stress_eV_per_A3_voigt": stress.tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
