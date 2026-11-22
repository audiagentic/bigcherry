"""Campaign radiance arms run through the native owner only: the Python second planner and its writer flow are gone."""
import importlib
import re
import unittest
from pathlib import Path

from bigcherry.core import paths
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


if __name__ == "__main__":
    unittest.main()
