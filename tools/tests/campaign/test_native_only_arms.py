"""Campaign radiance arms run through the native owner only: the Python second planner and its writer flow are gone."""
import importlib
import os
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bigcherry.core import paths
from bigcherry.engine_adapters import radiance_inventory
from bigcherry.engine_adapters.radiance import RadianceAdapter
from bigcherry.engine_adapters import adapter as engine_adapter
from bigcherry.engine_adapters.base import EngineAdapter

# Each pattern names something only the removed planner or the old exit-10/exit-11 writer flow carried.
REMOVED = (
    r"radiance_plan\b", r"write_serving_manifest", r"HostPoolRemeasureRefused", r"REMEASURE_REQUIRED",
)
SCANNED_SUFFIXES = {".py", ".sh", ".json", ".toml"}


def _sources():
    roots = [paths.REPO_ROOT / "tools"]
    this = Path(__file__).resolve()
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.suffix in SCANNED_SUFFIXES and path.is_file() and path.resolve() != this and "__pycache__" not in path.parts:
                yield path


class RemovedPlannerTests(unittest.TestCase):
    def test_the_planner_module_does_not_import(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("bigcherry.engine_adapters.radiance_plan")

    def test_no_tool_or_test_refers_to_the_removed_flow(self):
        hits = []
        for path in _sources():
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in REMOVED:
                if re.search(pattern, text):
                    hits.append(f"{path.relative_to(paths.REPO_ROOT)}: {pattern}")
        self.assertEqual(hits, [])

    def test_no_adapter_offers_a_manifest_writer(self):
        self.assertFalse(hasattr(EngineAdapter, "write_serving_manifest"))
        self.assertFalse(hasattr(engine_adapter("radiance"), "write_serving_manifest"))

    def test_the_parity_script_has_no_v2_branch(self):
        text = (paths.REPO_ROOT / "tools/lab/radiance/flashnext-parity.sh").read_text(encoding="utf-8")
        for pattern in (r"EDITION", r"(?i)\bv2\b", r"--configs", r"--help", r'--set "(engine|handoff)\.'):
            self.assertIsNone(re.search(pattern, text), pattern)
        # the one start is the native owner with the arm's own YAML
        self.assertIn("--native-config", text)


class NativeArmLayoutTests(unittest.TestCase):
    """A native arm's model and cards come from the owner's offline explain of its own YAML and the live detector."""
    FACTS = {"gpus": [
        {"bdf": "0000:03:00.0", "hip_ordinal": 8, "arch": "gfx1100", "vram_total_mib": 24560},
        {"bdf": "0000:09:00.0", "hip_ordinal": 2, "arch": "gfx1201", "vram_total_mib": 32624}]}

    def resolve(self, explained):
        calls = []

        def fake_run(binary, args):
            calls.append(args)
            return explained

        with patch.object(radiance_inventory, "_run", fake_run), patch.object(
                radiance_inventory, "detect", return_value=radiance_inventory.Inventory.from_facts(self.FACTS)), patch(
                "bigcherry.engine_adapters.radiance._host", return_value=SimpleNamespace(model_root="/models")), patch.dict(
                os.environ, {"BC_MODEL_ROOT": "/models"}):
            layout = RadianceAdapter().serving_layout("cfg", Path("arm.yaml"), Path("/b/bin/radiance"))
        return layout, calls

    def test_cards_and_model_come_from_the_explain_of_the_arm_yaml(self):
        explained = {"values": {"engine": {"model": "/models/m.rad"}, "layout": {"engine_cards": ["0000:09:00.0"]}}}
        layout, calls = self.resolve(explained)
        self.assertEqual(layout.model, Path("/models/m.rad"))
        self.assertEqual(layout.environment, {"ROCR_VISIBLE_DEVICES": "2", "HIP_VISIBLE_DEVICES": "0"})
        self.assertEqual(layout.expected_execution["locators"], ["0000:09:00.0"])
        self.assertEqual(calls[0][:4], ["explain", "cfg", "--native-config", "arm.yaml"])
        self.assertIn("--engine-binary", calls[0])

    def test_an_explain_without_the_structured_fields_is_refused(self):
        with self.assertRaisesRegex(ValueError, "structured field"):
            self.resolve({"values": {"engine": {}}})


if __name__ == "__main__":
    unittest.main()
