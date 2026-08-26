"""RA38/TR13 inventory, identity, and non-reintroduction guards."""

from __future__ import annotations

import ast
import importlib
import re
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PRODUCT_ROOT = REPO_ROOT / "tools" / "bigcherry"
sys.path.insert(0, str(REPO_ROOT / "tools"))

# These root modules were pure aliases to the canonical domain modules.  Their
# repository consumers were migrated to the canonical paths before removal.
RETIRED_ROOT_FACADES = (
    "ab_benchmark", "artifacts", "autotune_catalog", "autotune_schema",
    "builds", "campaign_build", "campaign_execution", "campaign_lane",
    "campaign_plan", "campaign_planner", "campaign_resolution",
    "campaign_source", "campaign_workers", "comparisons", "compile_check",
    "config", "context", "correctness_evidence", "csource",
    "experiment_bundle", "experiment_contract", "focal_source_plans",
    "generated_tree", "multi_gpu_validate", "patch_activation",
    "patch_catalog", "patch_lifecycle", "patch_registry",
    "patch_source_isolation", "patch_validation_campaign",
    "patch_validation_evidence", "patch_validation", "patchset", "paths",
    "pin_status", "pipeline", "promotion_correctness_gate", "promotion",
    "provenance", "ranking_policy", "reduce_correctness", "releases",
    "replay_inspect", "resources", "runtime_smoke",
    "signature_correctness_mapping", "source_audit", "source_identity",
    "sources", "telemetry", "toolchain", "tune_journal", "tune_promotion",
    "upstream", "workspace",
)

# Supported external surfaces remain aliases until their explicit contracts
# have canonical replacements: the packaged patch API and two documented
# ``python -m`` entry points.
RETAINED_ROOT_FACADES = {
    "patcher": "bigcherry.patch.apply",
    "inventory": "bigcherry.tuning.inventory",
    "replay_cache": "bigcherry.tuning.replay",
}


def _python_files(root: Path):
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


class CompatibilityFacadeTests(unittest.TestCase):
    def test_retired_root_facades_stay_removed(self) -> None:
        for name in RETIRED_ROOT_FACADES:
            with self.subTest(name=name):
                self.assertFalse(
                    (PRODUCT_ROOT / f"{name}.py").exists(),
                    f"retired root facade {name}.py must not return",
                )

    def test_retained_facades_are_true_module_aliases(self) -> None:
        for name, canonical_name in RETAINED_ROOT_FACADES.items():
            with self.subTest(name=name):
                legacy = importlib.import_module(f"bigcherry.{name}")
                canonical = importlib.import_module(canonical_name)
                self.assertIs(legacy, canonical)
                self.assertEqual(legacy.__name__, canonical_name)

    def test_no_static_or_dynamic_consumer_uses_retired_root_paths(self) -> None:
        retired = set(RETIRED_ROOT_FACADES)
        violations: list[str] = []
        for root in (REPO_ROOT / "tools", REPO_ROOT / "patches", REPO_ROOT / "config"):
            for path in _python_files(root):
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    imported: str | None = None
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imported = alias.name
                            if imported in retired or imported.startswith("bigcherry.") and imported.split(".")[1] in retired:
                                violations.append(f"{path}:{node.lineno}: {imported}")
                    elif isinstance(node, ast.ImportFrom) and node.level == 0:
                        imported = node.module
                        if imported in retired or imported and imported.startswith("bigcherry.") and imported.split(".")[1] in retired:
                            violations.append(f"{path}:{node.lineno}: {imported}")
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"import_module", "__import__"}:
                        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                            imported = node.args[0].value
                            if imported in retired or imported.startswith("bigcherry.") and imported.split(".")[1] in retired:
                                violations.append(f"{path}:{node.lineno}: {imported}")
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        value = node.value
                        stale_import_string = any(
                            re.search(rf"from bigcherry import\s+{name}\b", value)
                            for name in retired
                        )
                        if stale_import_string or any(
                            value == f"bigcherry.{name}"
                            or value.startswith(f"bigcherry.{name}.")
                            for name in retired
                        ):
                            violations.append(f"{path}:{node.lineno}: {value}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
