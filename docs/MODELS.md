# Model identities and integrations

The molecular benchmark uses OMol25. OMol25 names a dataset/task; the actual molecular checkpoint used here is DPA-3.3-1M with its OMol25 head.

| Family | Executed checkpoint | Scope | Official repository |
|---|---|---|---|
| MACE | MACE-MP-0b3 medium | Periodic reference frames and Si EOS | [MACE](https://github.com/ACEsuit/mace) |
| NequIP | NequIP-OAM-S 0.1 | Same periodic frames and EOS | [NequIP](https://github.com/mir-group/nequip) |
| DeePMD | DPA-3.3-1M, OMat24 head | Same periodic frames and EOS | [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) |
| OMol25 task | DPA-3.3-1M, OMol25 head | Nonperiodic neutral singlet F2 and SiF4 | [DPA checkpoint card](https://huggingface.co/deepmodelingcommunity/DPA-3.3-1M) |

MACE used float64; NequIP checkpoint metadata specify float32; DeePMD retained checkpoint-native precision. All measured runs used CPU. No fine-tuning was performed. Checkpoint hashes and model/data provenance are recorded in run manifests.

The installed NequIP 0.19.1 adapter uses the private eager `_from_saved_model` method. Production deployment should use its [public ASE workflow](https://nequip.readthedocs.io/en/latest/api/ase.html), with separate validation of compiled results. DeePMD uses the ASE DP calculator and an explicitly selected head.

UMA/eSEN and CHGNet were not run. Published OMol25 eSEN results are literature context only and must not be attributed to this DPA run. Molecular heads require appropriate charge/spin and boundary conditions; do not apply their molecular energy conventions to periodic bulk rankings.

See [run instructions](RUN_PILOT.md), [measured results](../reports/PILOT_RESULTS.md), and [published baselines and limits](PUBLISHED_BASELINES.md). Earlier `model_configs/` and smoke scripts are generic scaffolds, separate from the executed evaluator.
