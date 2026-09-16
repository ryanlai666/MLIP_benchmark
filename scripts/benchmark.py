"""Prepare or launch a pinned CHIPS-FF run; preparation needs only Python."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(config_path):
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    upstream = ROOT / lock["path"]
    git = ["git", "-c", f"safe.directory={upstream.as_posix()}", "-C", str(upstream)]
    commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    if commit != lock["commit"]:
        raise ValueError("Upstream revision differs from upstream.lock.json")
    if subprocess.check_output(git + ["status", "--porcelain"], text=True).strip():
        raise ValueError("Upstream has local changes; record a reviewed revision first")
    config = json.loads(config_path.read_text())
    tree = ast.parse((upstream / "chipsff/config.py").read_text())
    fields = {n.target.id for n in ast.walk(tree)
              if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    if set(config) - fields:
        raise ValueError(f"Unknown configuration fields: {set(config) - fields}")
    if config.get("calculator_type") not in {"mace", "chgnet"}:
        raise ValueError("Starter supports mace or chgnet; review adapter before extending")
    allowed = {"relax_structure", "calculate_ev_curve"}
    props = config.get("properties_to_calculate", [])
    if not props or set(props) - allowed or "relax_structure" not in props:
        raise ValueError("Starter requires relaxation and optionally an E-V curve")
    if config.get("calculator_settings"):
        raise ValueError("Pinned upstream ignores settings for these adapters")
    structure = (ROOT / config["structure_path"]).resolve(strict=True)
    config["structure_path"] = str(structure)
    return config, upstream, {
        "upstream": lock, "config_sha256": digest(config_path),
        "structure_sha256": digest(structure), "python": sys.version,
        "platform": platform.platform(), "scientific_validation": "pending",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--execute", action="store_true", help="Run calculations; may download data and weights")
    args = parser.parse_args()
    config, upstream, manifest = prepare(args.config.resolve())
    if not args.execute:
        print(json.dumps({"status": "prepared_only", "config": config, **manifest}, indent=2))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"{stamp}-{config['calculator_type']}-{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True)
    config["chemical_potentials_file"] = str(out / "chemical_potentials.json")
    input_file = out / "input.json"
    input_file.write_text(json.dumps(config, indent=2))
    manifest.update(status="running", started_utc=stamp)
    metadata = out / "manifest.json"
    metadata.write_text(json.dumps(manifest, indent=2))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(upstream) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    (out / "pip-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    command = [sys.executable, "-m", "chipsff.run_chipsff", "--input_file", str(input_file)]
    try:
        with (out / "run.log").open("w") as log:
            result = subprocess.run(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
        manifest.update(status="process_completed" if result.returncode == 0 else "failed", returncode=result.returncode)
    except BaseException as exc:
        manifest.update(status="interrupted_or_failed", error=str(exc))
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata.write_text(json.dumps(manifest, indent=2))
        print(f"Run artifacts: {out}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
