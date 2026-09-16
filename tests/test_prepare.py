import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts/benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class PreparationTests(unittest.TestCase):
    def test_configs_resolve_structure_and_record_provenance(self):
        for path in (ROOT / "configs").glob("*.json"):
            config, _, manifest = benchmark.prepare(path)
            self.assertTrue(Path(config["structure_path"]).is_file())
            self.assertEqual(len(manifest["structure_sha256"]), 64)
            self.assertEqual(manifest["scientific_validation"], "pending")

    def test_ignored_device_settings_are_rejected(self):
        config = json.loads((ROOT / "configs/si_mace.json").read_text())
        config["calculator_settings"] = {"mace": {"device": "cuda"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "ignores settings"):
                benchmark.prepare(path)

    def test_misspelled_field_is_rejected(self):
        config = json.loads((ROOT / "configs/si_mace.json").read_text())
        config["calculator_typo"] = "mace"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Unknown configuration"):
                benchmark.prepare(path)


if __name__ == "__main__":
    unittest.main()
