"""Sequential, tiny DFT-reference pilot. No training or etch-rate simulation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "models/cache"))
os.environ.setdefault("NEQUIP_CACHE_DIR", str(ROOT / "models/nequip-cache"))
os.environ.setdefault("MPLBACKEND", "Agg")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summarize(rows):
    import numpy as np
    result = {}
    for track in sorted({r["track"] for r in rows}):
        selected = [r for r in rows if r["track"] == track]
        good = [r for r in selected if r["status"] == "ok"]
        metrics = {"attempted": len(selected), "completed": len(good), "failed": len(selected)-len(good)}
        if good:
            delta = np.concatenate([np.array(r["forces_pred_eV_A"])-r["forces_ref_eV_A"] for r in good]).ravel()
            e = np.array([(r["energy_pred_eV"]-r["energy_ref_eV"])/r["natoms"] for r in good])
            metrics.update(force_component_mae_eV_A=float(np.abs(delta).mean()),
                           force_component_rmse_eV_A=float(np.sqrt(np.mean(delta**2))),
                           raw_energy_mae_eV_atom=float(np.abs(e).mean()),
                           raw_energy_mean_signed_error_eV_atom=float(e.mean()),
                           median_singlepoint_seconds=float(np.median([r["seconds"] for r in good])))
        result[track] = metrics
    return result


def samples():
    from ase import Atoms
    from ase.io import read
    source = ROOT / "upstream/mlearn/data/Si/test.json"
    refs = json.loads(source.read_text())
    for index in [13, 14, 19, 20, 7, 8]:
        ref = refs[index]
        s = ref["structure"]
        atoms = Atoms(symbols=[x["species"][0]["element"] for x in s["sites"]],
                      scaled_positions=[x["abc"] for x in s["sites"]], cell=s["lattice"]["matrix"], pbc=True)
        yield {"id": f"mlearn-Si-test-{index}", "track": "Si_crystal" if index not in [7, 8] else "Si_surface",
               "source": str(source.relative_to(ROOT)), "source_sha256": sha(source),
               "description": ref["description"], "reference_method": "DFT PBE; original MLearn test split",
               "energy": ref["outputs"]["energy"], "forces": ref["outputs"]["forces"]}, atoms
    archive = ROOT / "data/downloads/etch_dft.tar"
    archive_hash = sha(archive)
    with tarfile.open(archive) as tar:
        for member in ["dft/etch/trj_CF2_30eV.extxyz", "dft/etch/trj_CF3_30eV.extxyz"]:
            # Only read known regular members; no archive paths are extracted.
            lines = tar.extractfile(member).read().decode().splitlines(keepends=True)
            offsets = []
            cursor = 0
            while cursor < len(lines):
                if not lines[cursor].strip():
                    cursor += 1
                    continue
                n = int(lines[cursor]); offsets.append((cursor, n)); cursor += n + 2
            for index in sorted({0, len(offsets)//2, len(offsets)-1}):
                cursor, n = offsets[index]
                atoms = read(io.StringIO("".join(lines[cursor:cursor+n+2])), format="extxyz")
                energy = float(atoms.get_potential_energy())
                forces = atoms.get_forces().tolist()
                yield {"id": f"{Path(member).stem}-{index}", "track": "SiO2_CFx_etch_snapshots",
                       "source": member, "source_sha256": archive_hash, "source_frame": index,
                       "source_nframes": len(offsets), "reference_method": "Authors' DFT labels; detailed XC/pseudopotential audit pending",
                       "metadata": {k:str(v) for k,v in atoms.info.items()}, "energy": energy, "forces": forces}, atoms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="medium-0b3")
    parser.add_argument("--backend", choices=["mace", "nequip", "deepmd"], default="mace")
    parser.add_argument("--head", default="OMat24")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "deepmd" and args.device != "cpu":
        parser.error("The DeePMD pilot adapter is validated only for CPU; configure and validate GPU placement before enabling it.")
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from ase.build import bulk
    from ase.eos import EquationOfState
    torch.set_num_threads(args.threads)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "model": args.model,
                "backend": args.backend, "device": args.device, "dtype": "float64" if args.backend == "mace" else "checkpoint native", "threads": args.threads,
                "platform": platform.platform(), "python": platform.python_version(),
                "packages": {p:importlib.metadata.version(p) for p in ["torch", {"mace":"mace-torch", "nequip":"nequip", "deepmd":"deepmd-kit"}[args.backend], "ase", "numpy"]},
                "protocol": "Si crystal, Si surfaces, then six CFx etching frames; sequential single points; no fitting",
                "limitations": ["Tiny diagnostic subset, not a representative ranking", "Training overlap unknown",
                                "Raw energy errors include reference convention differences; no test-set offset fit",
                                "Etching DFT method audit pending; not etch yield or plasma dynamics accuracy"],
                "status": "running"}
    mpath = args.output / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    start = time.perf_counter()
    try:
        if args.backend == "mace":
            from mace.calculators import mace_mp
            calc = mace_mp(model=args.model, device=args.device, default_dtype="float64", dispersion=False)
        elif args.backend == "nequip":
            from nequip.integrations.ase import NequIPCalculator
            # Verified in installed NequIP 0.19.1. Private eager loader avoids a native compiler;
            # pin the environment and use compiled production integration for speed studies.
            calc = NequIPCalculator._from_saved_model(model_path=args.model, device=args.device,
                                                       chemical_species_to_atom_type_map=True)
            manifest["model_metadata"] = {k:str(v) for k,v in calc.model.metadata.items()}
        else:
            from deepmd.calculator import DP
            calc = DP(model=str(Path(args.model).resolve()), head=args.head)
            manifest["head"] = args.head
            manifest["checkpoint_sha256"] = sha(Path(args.model))
            manifest["type_map"] = calc.dp.get_type_map()
    except Exception as exc:
        manifest.update(status="model_load_failed", error=repr(exc))
        mpath.write_text(json.dumps(manifest, indent=2))
        raise
    manifest["load_seconds_including_possible_download"] = time.perf_counter()-start
    warm = bulk("Si", "diamond", a=5.43); warm.calc = calc
    warm.get_forces()
    rows = []
    for ref, atoms in samples():
        row = {k:v for k,v in ref.items() if k not in {"energy", "forces"}}
        row.update(natoms=len(atoms), formula=atoms.get_chemical_formula(), pbc=atoms.pbc.tolist(),
                   cell_A=atoms.cell.tolist(), positions_A=atoms.positions.tolist(), symbols=atoms.get_chemical_symbols())
        try:
            atoms.calc = calc
            if args.device == "cuda": torch.cuda.synchronize()
            start = time.perf_counter()
            energy = float(atoms.get_potential_energy()); forces = atoms.get_forces()
            if args.device == "cuda": torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            if not np.isfinite(energy) or not np.isfinite(forces).all(): raise ValueError("Nonfinite prediction")
            row.update(status="ok", energy_pred_eV=energy, energy_ref_eV=ref["energy"],
                       forces_pred_eV_A=forces.tolist(), forces_ref_eV_A=ref["forces"], seconds=seconds)
            print(ref["id"], len(atoms), "atoms", f"{seconds:.3f}s", flush=True)
        except Exception as exc:
            row.update(status="failed", error=repr(exc)); print(row, flush=True)
        rows.append(row)
        (args.output / "predictions.json").write_text(json.dumps(rows, indent=2))
        (args.output / "summary.json").write_text(json.dumps(summarize(rows), indent=2))
    # Small two-atom E-V scan, reported as model prediction, not measured DFT accuracy.
    volumes, energies = [], []
    for scale in np.linspace(0.96, 1.04, 9):
        atoms = bulk("Si", "diamond", a=5.43*scale); atoms.calc = calc
        volumes.append(atoms.get_volume()); energies.append(float(atoms.get_potential_energy()))
    eos = {"purpose": "model_prediction_only", "volume_A3": volumes, "energy_eV": energies}
    try:
        v0, e0, modulus = EquationOfState(volumes, energies).fit()
        eos.update(equilibrium_a_A=float((v0*4)**(1/3)), bulk_modulus_GPa=float(modulus*160.21766208),
                   minimum_bracketed=bool(min(volumes)<v0<max(volumes)))
    except Exception as exc: eos["fit_error"] = repr(exc)
    (args.output / "eos.json").write_text(json.dumps(eos, indent=2))
    cache = ROOT / "models/cache"
    cache = {"mace":cache, "nequip":ROOT / "models/nequip-cache", "deepmd":ROOT / "models/deepmd"}[args.backend]
    manifest["checkpoint_files"] = [{"path":str(p.relative_to(ROOT)), "sha256":sha(p)} for p in cache.rglob("*") if p.is_file() and p.stat().st_size>1000000]
    manifest.update(status="completed" if all(r["status"]=="ok" for r in rows) else "completed_with_failures",
                    finished_utc=datetime.now(timezone.utc).isoformat())
    mpath.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summarize(rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
