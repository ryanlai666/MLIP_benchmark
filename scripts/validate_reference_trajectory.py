"""Single points against every archived DFT frame of an etching reference trajectory.

The archive stores DFT labels for 1000 consecutive snapshots of one CFx etching
sequence. Each snapshot is an independent DFT single point, not an adjacent MD
step, so this measures label accuracy along the reference chemistry, not
dynamical agreement.
"""
import argparse
import csv
import importlib.metadata
import io
import json
import platform
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

from run_pilot import ROOT, sha

ARCHIVE = ROOT / "data/downloads/etch_dft.tar"


def frames(member, stride, limit):
    """Yield (index, frame_count, ase.Atoms) for the requested archive member."""
    from ase.io import read
    with tarfile.open(ARCHIVE) as tar:
        # Only a known regular member is read; no archive path is extracted to disk.
        lines = tar.extractfile(member).read().decode().splitlines(keepends=True)
    offsets, cursor = [], 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        n = int(lines[cursor])
        offsets.append((cursor, n))
        cursor += n + 2
    selected = list(range(0, len(offsets), stride))[:limit]
    for index in selected:
        cursor, n = offsets[index]
        yield index, len(offsets), read(io.StringIO("".join(lines[cursor:cursor + n + 2])), format="extxyz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mace", "nequip", "deepmd"], required=True)
    parser.add_argument("--member", default="dft/etch/trj_CF2_30eV.extxyz")
    parser.add_argument("--model", default=None, help="Checkpoint path or hub id; backend default otherwise")
    parser.add_argument("--head", default="OMat24")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), status="running", backend=args.backend,
                    reference_member=args.member, archive_sha256=sha(ARCHIVE), stride=args.stride,
                    device=args.device, threads=args.threads, platform=platform.platform(),
                    python=platform.python_version(),
                    reference_method="Authors' DFT labels on 7net-Omni generated etching snapshots",
                    protocol="One single point per archived frame; no fitting, no relaxation, no dynamics",
                    limitations=["Consecutive archived frames are separate impact snapshots, not adjacent MD steps",
                                 "Raw energy differences include reference-convention and energy-zero offsets",
                                 "A fitted per-element offset is reported separately and is not a DFT-validated formation energy",
                                 "Accuracy on archived snapshots does not establish etch yields or reaction rates"])
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    start = time.perf_counter()
    try:
        if args.backend == "mace":
            from mace.calculators import mace_mp
            calc = mace_mp(model=args.model or "medium-0b3", device=args.device, default_dtype="float64", dispersion=False)
            manifest.update(model="MACE-MP-0b3 medium", dtype="float64")
        elif args.backend == "nequip":
            from nequip.integrations.ase import NequIPCalculator
            calc = NequIPCalculator._from_saved_model(model_path=args.model or "nequip.net:mir-group/NequIP-OAM-S:0.1",
                                                      device=args.device, chemical_species_to_atom_type_map=True)
            manifest.update(model="NequIP-OAM-S 0.1", dtype="checkpoint native")
        else:
            from deepmd.calculator import DP
            checkpoint = Path(args.model or "models/deepmd/DPA-3.3-1M.pt")
            calc = DP(model=str(checkpoint.resolve()), head=args.head)
            manifest.update(model="DPA-3.3-1M / OMat24", head=args.head, dtype="checkpoint native",
                            checkpoint_sha256=sha(checkpoint))
        manifest["packages"] = {p: importlib.metadata.version(p) for p in
                                ["torch", {"mace": "mace-torch", "nequip": "nequip", "deepmd": "deepmd-kit"}[args.backend], "ase", "numpy"]}
    except Exception as exc:
        manifest.update(status="model_load_failed", error=repr(exc))
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        raise
    manifest["load_seconds"] = time.perf_counter() - start
    rows, pooled, per_element, counts, energies = [], [], {}, [], []
    started = time.perf_counter()
    try:
        for index, total, atoms in frames(args.member, args.stride, args.limit):
            symbols = np.array(atoms.get_chemical_symbols())
            reference_energy = float(atoms.get_potential_energy())
            reference_forces = atoms.get_forces()
            atoms.calc = calc
            if args.device == "cuda":
                torch.cuda.synchronize()
            tick = time.perf_counter()
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            if args.device == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - tick
            if not np.isfinite(energy) or not np.isfinite(forces).all():
                raise ValueError(f"Nonfinite prediction at frame {index}")
            delta = forces - reference_forces
            rows.append(dict(frame=index, natoms=len(atoms), formula=atoms.get_chemical_formula(),
                             energy_ref_eV=reference_energy, energy_pred_eV=energy,
                             raw_energy_error_meV_atom=(energy - reference_energy) / len(atoms) * 1000,
                             force_mae_eV_A=float(np.abs(delta).mean()),
                             force_rmse_eV_A=float(np.sqrt((delta ** 2).mean())),
                             max_force_error_eV_A=float(np.linalg.norm(delta, axis=1).max()),
                             max_ref_force_eV_A=float(np.linalg.norm(reference_forces, axis=1).max()),
                             seconds=seconds))
            pooled.append(delta.ravel())
            for symbol in sorted(set(symbols)):
                per_element.setdefault(symbol, []).append(delta[symbols == symbol].ravel())
            counts.append({s: int((symbols == s).sum()) for s in sorted(set(symbols))})
            energies.append((energy, reference_energy))
            if index % (50 * args.stride) == 0:
                print(f"{args.backend} frame {index}/{total} {seconds:.3f}s", flush=True)
        pooled = np.concatenate(pooled)
        elements = sorted({s for c in counts for s in c})
        composition = np.array([[c.get(s, 0) for s in elements] for c in counts], dtype=float)
        predicted = np.array([e[0] for e in energies])
        reference = np.array([e[1] for e in energies])
        natoms = np.array([r["natoms"] for r in rows], dtype=float)
        raw = (predicted - reference) / natoms * 1000
        # Least-squares per-element energy offset, reported separately from the raw errors.
        shift, *_ = np.linalg.lstsq(composition, predicted - reference, rcond=None)
        aligned = (predicted - reference - composition @ shift) / natoms * 1000
        summary = dict(frames=len(rows), total_atoms=int(natoms.sum()),
                       force_component_mae_eV_A=float(np.abs(pooled).mean()),
                       force_component_rmse_eV_A=float(np.sqrt((pooled ** 2).mean())),
                       force_component_max_abs_eV_A=float(np.abs(pooled).max()),
                       raw_energy_mae_meV_atom=float(np.abs(raw).mean()),
                       raw_energy_mean_signed_error_meV_atom=float(raw.mean()),
                       offset_fitted_energy_mae_meV_atom=float(np.abs(aligned).mean()),
                       offset_fitted_energy_rmse_meV_atom=float(np.sqrt((aligned ** 2).mean())),
                       fitted_element_offsets_eV={s: float(v) for s, v in zip(elements, shift)},
                       median_singlepoint_seconds=float(np.median([r["seconds"] for r in rows])),
                       per_element_force_mae_eV_A={s: float(np.abs(np.concatenate(v)).mean()) for s, v in sorted(per_element.items())},
                       per_element_force_rmse_eV_A={s: float(np.sqrt((np.concatenate(v) ** 2).mean())) for s, v in sorted(per_element.items())})
        with (args.output / "frames.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        manifest.update(status="completed", wall_seconds=time.perf_counter() - started,
                        finished_utc=datetime.now(timezone.utc).isoformat(), summary=summary)
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
