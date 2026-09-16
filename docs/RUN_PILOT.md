# Reproduce the sequential pilot

Run commands from the workspace root in PowerShell. Existing environments, references and model caches are already present on this machine. Every output directory must be new. Run each command to completion before starting the next.

```powershell
.venv-mace/Scripts/python.exe scripts/run_pilot.py --model medium-0b3 --output results/repeat-mace
.venv-nequip/Scripts/python.exe scripts/run_pilot.py --backend nequip --model nequip.net:mir-group/NequIP-OAM-S:0.1 --output results/repeat-nequip
.venv-deepmd/Scripts/python.exe scripts/run_pilot.py --backend deepmd --model models/deepmd/DPA-3.3-1M.pt --head OMat24 --output results/repeat-deepmd
.venv-deepmd/Scripts/python.exe scripts/run_omol_pilot.py --output results/repeat-omol25
```

Defaults are CPU and four Torch threads. GPU performance and numerical agreement have not been tested. DeePMD's pilot adapter rejects a CUDA request because it does not configure GPU placement.

## Recreate environments on another Windows machine

The snapshots record installed versions from successful Python 3.11.14 runs. They are not hashed, cross-platform dependency locks. Use independent environments to avoid incompatible Torch/e3nn requirements.

```powershell
uv venv --python 3.11.14 .venv-mace
uv pip install --python .venv-mace/Scripts/python.exe -r environments/mace-windows-py311.lock.txt
uv venv --python 3.11.14 .venv-nequip
uv pip install --python .venv-nequip/Scripts/python.exe -r environments/nequip-windows-py311.lock.txt
uv venv --python 3.11.14 .venv-deepmd
uv pip install --python .venv-deepmd/Scripts/python.exe -r environments/deepmd-windows-py311.lock.txt
```

DeePMD 3.2.0 required Torch 2.11.0 and e3nn 0.6.0 here. The NequIP eager loader is private API pinned to 0.19.1. Consult manifests and snapshots before changing versions.

## References and weights

Skip cloning if the pinned checkouts already exist. The CHIPS-FF checkout supports the broader project and is not required by the direct pilot.

```powershell
git clone https://github.com/materialyzeai/mlearn.git upstream/mlearn
git -C upstream/mlearn checkout 10c427a5480c6281c15c64efaf869b03be04818f
python scripts/fetch_etch_data.py
.venv-deepmd/Scripts/dp.exe pretrained download DPA-3.3-1M --cache-dir models/deepmd
```

The etch downloader verifies the archive's published MD5. MACE and NequIP download their named checkpoints on first use; their local caches reside under `models/`. Run manifests record hashes to detect upstream changes. Do not redistribute weights or data without checking their licenses.

## Validate and regenerate the report

```powershell
python -m unittest discover -s tests -v
python scripts/report_pilot.py
```

The report generator reads the four original successful run folders, independently verifies aggregate force metrics against raw predictions, and writes `reports/PILOT_RESULTS.md` and `reports/pilot_metrics.csv`. To report repeat runs, update its explicit run-folder mapping first. Do not mix measurements from different versions without labeling them.

Each material run stores raw per-frame reference/predicted energies and forces, timings, EOS predictions and a manifest. Molecular outputs include optimization trajectories and convergence flags. Earlier failed DeePMD attempts remain preserved. See [results and limitations](../reports/PILOT_RESULTS.md) and [development tasks](DEVELOPMENT.md).
