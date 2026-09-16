# Sources inspected on 2026-09-15

| Resource | Role and decision |
|---|---|
| [CHIPS-FF](https://github.com/atomgptlab/chipsff) | Selected orchestration base; existing crystal, surface, defect, and property workflows. Local commit is in `upstream.lock.json`. |
| [CHIPS-FF paper](https://doi.org/10.1021/acsmaterialslett.5c00093) | Semiconductor-oriented benchmark motivation and reference methodology. |
| [MACE](https://github.com/ACEsuit/mace) | Initial potential family; pinned CHIPS-FF selects the `medium` MACE-MP model. |
| [MACE foundation documentation](https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html) | Explicitly specify model identity; defaults have changed across versions. |
| [Silicon cleavage benchmark](https://github.com/d2r2group/mlip-cleavage-benchmark) | Candidate second benchmark for Si surface stability; not downloaded or validated here. Dataset linked by authors at https://zenodo.org/records/16970768. |
| [SevenNet-Nano data](https://zenodo.org/records/19491140) | Candidate plasma extension: authors list SiO2/CFx etching DFT data, checkpoints, and modified code. Inspect actual labels, splits, license, and model compatibility before use; not downloaded here. |
| [Silicon radiation damage](https://github.com/hongweiniu/primary_radiation_damage_of_silicon) | Candidate Si collision-potential comparison; not a reactive fluorine potential or a verified etching dataset. |
| [ViennaPS etching model](https://viennatools.github.io/ViennaPS/models/prebuilt/SF6O2Etching.html) | Possible later feature-scale coupling, separate from atomistic MLIP evaluation. |

The repository selection is a project recommendation. Availability of a checkpoint or elemental coverage does not establish validity for a particular reaction or collision regime.
