"""Record selected installed API details used to set up model integrations."""
import argparse
import inspect
import json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("backend", choices=["nequip", "mace", "deepmd"])
args = parser.parse_args()
if args.backend == "nequip":
    from nequip.model import ModelFromPackage
    from nequip.integrations.ase import NequIPCalculator
    from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper, NeighborListTransform
    print(inspect.getsource(ModelFromPackage))
    print(inspect.getsource(NequIPCalculator.__bases__[0]))
    print(inspect.signature(ChemicalSpeciesToAtomTypeMapper))
    print(inspect.signature(NeighborListTransform))
elif args.backend == "mace":
    from mace.calculators import mace_omol
    print(inspect.getsource(mace_omol))
else:
    import deepmd
    base = Path(deepmd.__file__).parent
    print((base / "pt/cxx_op.py").read_text())
    for path in (base / "lib").glob("*.ini"):
        print(path.read_text())
