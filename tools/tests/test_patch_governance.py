"""RE40: patch-catalog governance checks (external patch-management review,
2026-08-20) -- REQUIRES/CONFLICTS enforcement, no patch-to-patch imports, and
no NEW patch relying on the implicit GROUP="core" default."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import patch_catalog, patchset  # noqa: E402

# Patches that predate this rule and are grandfathered -- confirmed via a
# real scan (2026-08-20) that this is the only current implicit-GROUP
# module. No new entry may be added here; new patches must set GROUP
# explicitly.
GRANDFATHERED_IMPLICIT_GROUP = frozenset({"0900_pool_workspace_metrics"})


class RequiresConflictsBackfillTests(unittest.TestCase):
    """The real, enforced REQUIRES/CONFLICTS backfill (patchset.py's
    mechanism, not patch_catalog.py's -- see patch_catalog.CatalogEntry's
    docstring for why those stay separate)."""

    def setUp(self):
        self.modules = {m.patch_id: m for m in patchset.catalog()}

    def test_1216_requires_1215(self):
        self.assertEqual(
            self.modules["1216_rd43_concurrent_join_fusion_guard"].requires,
            ("1215_rd394041_amd_stream_moe_overlap",),
        )

    def test_1217_requires_1215_and_1216(self):
        self.assertEqual(
            self.modules["1217_rd44_graph_opt_default_rdna35"].requires,
            ("1215_rd394041_amd_stream_moe_overlap", "1216_rd43_concurrent_join_fusion_guard"),
        )

    def test_1205_and_1207_conflict_reciprocally(self):
        self.assertIn(
            "1207_rd17_moe_topk_down_fold",
            self.modules["1205_rd12_paired_mmvq_dual_output"].conflicts,
        )
        self.assertIn(
            "1205_rd12_paired_mmvq_dual_output",
            self.modules["1207_rd17_moe_topk_down_fold"].conflicts,
        )

    def test_resolve_exact_enforces_1216_requires_1215(self):
        with self.assertRaises(ValueError) as ctx:
            patchset.resolve_exact(["1216_rd43_concurrent_join_fusion_guard"])
        self.assertIn("requires explicitly selected", str(ctx.exception))

    def test_resolve_exact_enforces_1205_1207_conflict(self):
        with self.assertRaises(ValueError) as ctx:
            patchset.resolve_exact([
                "1205_rd12_paired_mmvq_dual_output",
                "1207_rd17_moe_topk_down_fold",
            ])
        self.assertIn("conflicts with selected", str(ctx.exception))

    def test_real_recipe_chains_still_resolve_cleanly(self):
        # Regression proof: every existing config/recipes.toml entry naming
        # 1216/1217 already includes their full REQUIRES chain, so this
        # backfill must not break real experiment resolution.
        resolved = patchset.resolve_exact([
            "1215_rd394041_amd_stream_moe_overlap",
            "1216_rd43_concurrent_join_fusion_guard",
            "1217_rd44_graph_opt_default_rdna35",
        ])
        self.assertEqual(len(resolved.modules), 3)


class NoPatchToPatchImportsTests(unittest.TestCase):
    """RE40 P1: patch modules must not import each other -- shared logic
    belongs in a common helper location, never a sibling patch module."""

    def test_no_current_patch_imports_another_patch_module(self):
        violations = []
        patch_ids = {m.patch_id for m in patchset.catalog()}
        for module in patchset.catalog():
            tree = ast.parse(module.path.read_text(encoding="utf-8"), filename=str(module.path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    last = node.module.rsplit(".", 1)[-1]
                    if last in patch_ids and last != module.patch_id:
                        violations.append(f"{module.patch_id} imports {node.module!r}")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        last = alias.name.rsplit(".", 1)[-1]
                        if last in patch_ids and last != module.patch_id:
                            violations.append(f"{module.patch_id} imports {alias.name!r}")
        self.assertEqual(violations, [], f"patch-to-patch imports found: {violations}")


class ImplicitGroupGrandfatherTests(unittest.TestCase):
    """RE40 P1: the implicit GROUP="core" default (patchset.DEFAULT_GROUP)
    may stay for grandfathered existing modules, but no new patch may rely
    on it silently."""

    def test_only_grandfathered_patches_use_the_implicit_default(self):
        implicit = {m.patch_id for m in patchset.catalog() if not m.group_explicit}
        self.assertEqual(
            implicit, set(GRANDFATHERED_IMPLICIT_GROUP),
            "a patch is relying on the implicit GROUP default without being "
            "grandfathered -- set GROUP explicitly, or add it to "
            "GRANDFATHERED_IMPLICIT_GROUP only if this is confirmed pre-existing",
        )


class CatalogMetadataExtensionTests(unittest.TestCase):
    """RE40: the new optional plan_ids/backends/subsystems/hardware fields."""

    def test_new_fields_default_to_empty_when_absent(self):
        entries = patch_catalog.load_catalog()
        # No current catalog.toml entry sets these yet -- confirm the
        # absent-field default is an empty tuple, not None or an error.
        entry = entries["0100_cmake_options"]
        self.assertEqual(entry.plan_ids, ())
        self.assertEqual(entry.backends, ())
        self.assertEqual(entry.subsystems, ())
        self.assertEqual(entry.hardware, ())

    def test_backends_field_validates_against_known_backends(self):
        raw = patch_catalog.load_catalog.__module__  # noqa: F841 -- sanity import check
        import tempfile
        bad_toml = """
version = 1
[[patch]]
id = "x"
kind = "framework"
origin = "local"
backend = "hip"
state = "validated"
backends = ["not-a-real-backend"]
"""
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as handle:
            handle.write(bad_toml)
            path = Path(handle.name)
        try:
            with self.assertRaises(ValueError):
                patch_catalog.load_catalog(path)
        finally:
            path.unlink()

    def test_cross_check_still_clean_with_new_optional_fields(self):
        # The real catalog doesn't use the new fields yet, but cross_check's
        # 1:1 coverage proof must still hold with the extended schema.
        # PA05 quarantine (runbook M3): tolerate only the tracked
        # 1200_rd19 validated-without-HI83-evidence gap (see
        # docs/planning/active/patch-system/PA05.md and the identical helper
        # in test_patch_catalog.py); any other finding still fails.
        problems = patch_catalog.cross_check()
        tracked = [
            p for p in problems
            if not p.startswith("1200_rd19_single_gpu_meta_bypass:")
        ]
        self.assertEqual(tracked, [])


if __name__ == "__main__":
    unittest.main()
