"""Build an auditable Markdown/CSV report from completed pilot outputs."""
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("MACE-MP-0b3 medium", "pilot-mace-mp-0b3"),
    ("NequIP-OAM-S 0.1", "pilot-nequip-oam-s"),
    ("DPA-3.3-1M / OMat24", "pilot-deepmd-dpa33-omat-fixed"),
]


def main():
    report = ROOT / "reports"
    report.mkdir(exist_ok=True)
    lines = ["# Small semiconductor MLIP pilot", "",
             "Completed sequentially on 2026-09-15 America/Chicago (2026-09-16 UTC). All three materials checkpoints completed the same 12 single-point reference configurations. The DPA OMol25 branch separately completed two gas-phase optimizations. No models were trained or fine-tuned.", "",
             "## Force accuracy measured here", "",
             "Component-wise MAE/RMSE pool all 3N Cartesian components within each subset; units are eV/Å. These diagnostic subsets are too small for a general ranking or statistical confidence intervals.", "",
             "| Checkpoint | Subset | Frames | Atoms/frame | Force MAE | Force RMSE |", "|---|---|---:|---|---:|---:|"]
    csv_rows, timings, eos_rows, details = [], [], [], []
    track_names = {"Si_crystal":"Si crystal", "Si_surface":"Si surfaces", "SiO2_CFx_etch_snapshots":"SiO2/CFx etching"}
    for label, folder in RUNS:
        base = ROOT / "results" / folder
        manifest = json.loads((base / "manifest.json").read_text())
        assert manifest["status"] == "completed", (folder,manifest["status"])
        summary = json.loads((base / "summary.json").read_text())
        rows = json.loads((base / "predictions.json").read_text())
        assert len(rows) == 12 and all(r["status"] == "ok" for r in rows)
        for track in ["Si_crystal", "Si_surface", "SiO2_CFx_etch_snapshots"]:
            s = summary[track]
            selected = [r for r in rows if r["track"] == track]
            assert s["completed"] == len(selected)
            # Independent recomputation from saved raw arrays checks the exported aggregate.
            errors = [float(p)-float(t) for r in selected for ps,ts in zip(r["forces_pred_eV_A"],r["forces_ref_eV_A"]) for p,t in zip(ps,ts)]
            mae = statistics.mean(abs(x) for x in errors)
            rmse = math.sqrt(statistics.mean(x*x for x in errors))
            assert math.isclose(mae, s["force_component_mae_eV_A"], abs_tol=1e-12)
            assert math.isclose(rmse, s["force_component_rmse_eV_A"], abs_tol=1e-12)
            counts = "/".join(str(n) for n in sorted({r["natoms"] for r in selected}))
            lines.append(f"| {label} | {track_names[track]} | {len(selected)} | {counts} | {mae:.4f} | {rmse:.4f} |")
            csv_rows.append({"checkpoint":label,"track":track,"natoms":counts,**s})
            timings.append(f"| {label} | {track_names[track]} | {s['median_singlepoint_seconds']:.4f} | {1000*s['raw_energy_mae_eV_atom']:.2f} |")
        thermal = [r for r in rows if r["id"] in {"mlearn-Si-test-13", "mlearn-Si-test-14"}]
        e = [abs(p-t) for r in thermal for ps,ts in zip(r["forces_pred_eV_A"],r["forces_ref_eV_A"]) for p,t in zip(ps,ts)]
        details.append(f"- {label}: thermal-only Si force MAE **{statistics.mean(e):.4f} eV/Å** (two frames).")
        eos = json.loads((base / "eos.json").read_text())
        eos_rows.append(f"| {label} | {eos.get('equilibrium_a_A',float('nan')):.4f} | {eos.get('bulk_modulus_GPa',float('nan')):.2f} | {eos.get('minimum_bracketed')} |")
    lines += ["", "The crystal subset comprises two 300 K AIMD frames and two strained cells. The nearly force-free strained cells lower the pooled force error; the thermal-only values are:", "", *details, "",
              "The crystal indices are 13, 14, 19 and 20 of the original MLearn Si test split; surface indices are 7 and 8. Etching frames are 0, 500 and 999 from each of the CF2/CF3 30 eV trajectories. These are archived DFT single points on author-generated trajectories, not new plasma MD or etch-yield calculations.", "",
              "## Timing and raw energy differences", "",
              "| Checkpoint | Subset | Median wall time/frame (s) | Raw energy MAE (meV/atom) |", "|---|---|---:|---:|", *timings, "",
              "All runs used CPU inference with torch.set_num_threads(4); the detected RTX 3070 was not used. MACE used float64, NequIP checkpoint metadata specify float32, and DeePMD retained native checkpoint precision. Each calculator was warmed on a two-atom Si cell; each reference frame was evaluated once. These times can include shape-dependent lazy initialization, especially DeePMD's first larger cells. They are end-to-end pilot timings, not matched-precision steady-state speed rankings. Load/download time is recorded separately in manifests.", "",
              "Raw energy errors have no fitted offset. They may include XC, pseudopotential, dispersion and energy-zero differences and should not be the sole selection metric. The etching paper describes spin-polarized PBE/PAW, 520 eV and Gamma-point calculations; exact archived D3 inclusion still needs confirmation. MACE and NequIP here do not explicitly condition on the reference magnetic moment. See [reference audit and published results](../docs/PUBLISHED_BASELINES.md).", "",
              "## Two-atom Si equation of state", "",
              "Nine unrelaxed diamond-cell volume points, isotropic lattice scale 0.96–1.04 about 5.43 Å. These are **model predictions**, not errors against an independently converged DFT equation of state.", "",
              "| Checkpoint | Equilibrium a (Å) | Bulk modulus (GPa) | Minimum bracketed |", "|---|---:|---:|---|", *eos_rows, "",
              "## OMol25 gas-phase check", "",
              "Used **DPA-3.3-1M / OMol25**, not UMA or eSEN. F2 and SiF4 were optimized sequentially with nonperiodic boundaries, charge 0, multiplicity 1, BFGS and fmax=0.005 eV/Å. This is a fragment geometry comparison, not periodic surface or reaction-barrier validation.", "",
              "| Molecule | Atoms | Model bond (Å) | NIST bond (Å) | Absolute difference (Å) | Converged |", "|---|---:|---:|---:|---:|---|"]
    molecules = json.loads((ROOT / "results/pilot-omol25-fragments/summary.json").read_text())
    for r in molecules:
        lines.append(f"| {r['molecule']} | {r['natoms']} | {r['mean_bond_A']:.4f} | {r['reference_bond_A']:.4f} | {r['absolute_difference_A']:.4f} | {r['converged']} |")
    lines += ["", "NIST sources: [F2](https://cccbdb.nist.gov/exp2x.asp?casno=7782414&charge=0), [SiF4](https://cccbdb.nist.gov/exp2x.asp?casno=7783611&charge=0). SiF4's listed distance is derived from B0 and differs in convention from a model equilibrium geometry. No molecular force/energy accuracy is claimed without matching DFT labels.", "",
              "## Starting-point decision", "",
              "DPA-3.3-1M / OMat24 has the lowest force errors on this selected subset, so it is the first candidate to extend. Keep MACE-MP-0b3 as a comparison and NequIP-OAM-S as a compact speed-oriented candidate. This conclusion applies only to these checkpoints and frames, not to whole model families. Preserve the original dataset splits and audit pretraining overlap before calling a larger evaluation held-out.", "",
              "Before production etching MD, evaluate short-distance repulsion and quasi-static drag references, reactions/barriers, hotter impacts, charge/stopping assumptions, slab-size/timestep convergence and multiple trajectories. A low error at 30 eV does not establish accuracy at hundreds of eV. The supplied archive already includes qsd data for the next extension.", "",
              "## Reproduction and artifacts", "",
              "- [Run instructions](../docs/RUN_PILOT.md); [tested package snapshots](../environments); [hardware](hardware.json).",
              "- [MACE raw predictions and manifest](../results/pilot-mace-mp-0b3).",
              "- [NequIP raw predictions and manifest](../results/pilot-nequip-oam-s).",
              "- [DeePMD raw predictions and manifest](../results/pilot-deepmd-dpa33-omat-fixed).",
              "- [OMol25 output and optimization trajectories](../results/pilot-omol25-fragments).",
              "- [Machine-readable metrics](pilot_metrics.csv); [published baselines and limits](../docs/PUBLISHED_BASELINES.md).", "",
              "The two earlier DeePMD attempts remain in results/: Torch 2.14 ABI incompatibility, then a missing e3nn import after matching Torch 2.11.0. The successful environment uses deepmd-kit 3.2.0, torch 2.11.0 and e3nn 0.6.0. Native Windows worked for this CPU pilot; GPU and compiled production inference were not tested."]
    (report / "PILOT_RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    with (report / "pilot_metrics.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    print(report / "PILOT_RESULTS.md")


if __name__ == "__main__":
    main()
