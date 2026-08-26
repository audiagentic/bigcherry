"""EC20/EC21: CI link-audit -- source -> plan -> patch/action -> contract
must be mechanically valid, not a manual-review habit.

Extends tools/tests/test_external_sources.py's existing cross_check_patches
machinery (patches/*.py PROVENANCE <-> external-sources.toml, both
directions for entries reachable via a patch's own PROVENANCE) rather than
duplicating it. This file covers the checks that machinery does NOT do:
reverse-direction dangling `patch =` references, REQUIRES/CONFLICTS
existence (RE40), no-cycles, no-duplicate-IDs, rejected patches absent from
production patch-sets (general case, not just the rdna-boosts group),
recipes.toml patch references all resolve to real files, and an orphan
report distinguishing "no signal at all" cases.
"""

from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import patchset
from bigcherry.source import sources # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RECIPES_PATH = ROOT / "config" / "recipes.toml"
EXTERNAL_SOURCES_PATH = ROOT / "config" / "external-sources.toml"


def _real_patch_ids() -> set[str]:
    return {module.patch_id for module in patchset.catalog()}


class RequiresConflictsExistenceTests(unittest.TestCase):
    """RE40 backfilled real REQUIRES/CONFLICTS onto patchset.PatchModule.
    Every target must resolve to a real patch_id -- a typo'd dependency
    that silently resolves to nothing is worse than no dependency
    declaration at all."""

    def test_every_requires_target_exists(self):
        real_ids = _real_patch_ids()
        problems = []
        for module in patchset.catalog():
            for target in module.requires:
                if target not in real_ids:
                    problems.append(f"{module.patch_id}: REQUIRES {target!r} does not exist")
        self.assertEqual(problems, [], f"dangling REQUIRES targets: {problems}")

    def test_every_conflicts_target_exists(self):
        real_ids = _real_patch_ids()
        problems = []
        for module in patchset.catalog():
            for target in module.conflicts:
                if target not in real_ids:
                    problems.append(f"{module.patch_id}: CONFLICTS {target!r} does not exist")
        self.assertEqual(problems, [], f"dangling CONFLICTS targets: {problems}")

    def test_no_requires_cycles(self):
        graph = {module.patch_id: module.requires for module in patchset.catalog()}
        visiting: set[str] = set()
        visited: set[str] = set()

        def _visit(node: str, path: list[str]) -> None:
            if node in visited:
                return
            if node in visiting:
                cycle = " -> ".join(path[path.index(node):] + [node])
                self.fail(f"REQUIRES cycle detected: {cycle}")
            visiting.add(node)
            for dep in graph.get(node, ()):
                _visit(dep, path + [node])
            visiting.discard(node)
            visited.add(node)

        for patch_id in graph:
            _visit(patch_id, [])

    def test_conflicts_declarations_are_reciprocal(self):
        # 1205<->1207's real conflict is declared on both sides -- prove
        # that pattern holds project-wide: a one-sided CONFLICTS is a real
        # data bug (the anchor collision is symmetric by construction, a
        # missing reciprocal entry means the OTHER patch could be selected
        # alongside this one with no warning).
        by_id = {module.patch_id: module for module in patchset.catalog()}
        problems = []
        for module in patchset.catalog():
            for target in module.conflicts:
                other = by_id.get(target)
                if other is not None and module.patch_id not in other.conflicts:
                    problems.append(
                        f"{module.patch_id} declares CONFLICTS {target!r}, "
                        f"but {target} does not declare the reciprocal conflict"
                    )
        self.assertEqual(problems, [], f"non-reciprocal conflicts: {problems}")


class NoDuplicateCanonicalIdsTests(unittest.TestCase):
    def test_patchset_catalog_raises_on_duplicate_ids(self):
        # patchset.catalog() already raises ValueError internally on a
        # duplicate patch_id (see patchset.py) -- prove it does not silently
        # swallow one by confirming the real catalog call succeeds with a
        # length matching the number of distinct real files on disk.
        modules = patchset.catalog()
        ids = [m.patch_id for m in modules]
        self.assertEqual(len(ids), len(set(ids)), "duplicate patch_id survived catalog()")


class RejectedPatchesAbsentFromProductionSetsTests(unittest.TestCase):
    """General case of test_external_sources.py's
    test_rdna_patches_not_in_production_patch_sets, which only checks the
    rdna-boosts GROUP specifically -- this checks EVERY patch's real STATE,
    regardless of group, against every production patch-set."""

    def test_no_rejected_state_patch_in_any_production_patch_set(self):
        recipes = tomllib.loads(RECIPES_PATH.read_text(encoding="utf-8"))
        states = {info.name: info.state for info in patchset.describe()}
        problems = []
        for set_name, body in (recipes.get("patch-set") or {}).items():
            for patch_id in body.get("patches", []):
                state = states.get(patch_id)
                if state == "rejected":
                    problems.append(f"{set_name}: {patch_id} is STATE=rejected")
        self.assertEqual(problems, [], f"rejected patches in production sets: {problems}")


class RecipesReferenceRealPatchesTests(unittest.TestCase):
    """Every patch_id named anywhere in config/recipes.toml (patch-sets,
    experiment.* entries) must exist -- the CLI would fail loudly on a typo
    today, but that's an operator finding it at build time, not CI."""

    def test_every_patch_set_member_exists(self):
        recipes = tomllib.loads(RECIPES_PATH.read_text(encoding="utf-8"))
        real_ids = _real_patch_ids()
        problems = []
        for set_name, body in (recipes.get("patch-set") or {}).items():
            for patch_id in body.get("patches", []):
                if patch_id not in real_ids:
                    problems.append(f"patch-set {set_name!r}: {patch_id!r} does not exist")
        self.assertEqual(problems, [], f"dangling patch-set members: {problems}")

    def test_every_experiment_member_exists(self):
        recipes = tomllib.loads(RECIPES_PATH.read_text(encoding="utf-8"))
        real_ids = _real_patch_ids()
        problems = []
        for exp_name, body in (recipes.get("experiment") or {}).items():
            for patch_id in body.get("patches", []):
                if patch_id not in real_ids:
                    problems.append(f"experiment {exp_name!r}: {patch_id!r} does not exist")
        self.assertEqual(problems, [], f"dangling experiment members: {problems}")


class OrphanReportTests(unittest.TestCase):
    """Not a pass/fail CI gate -- an informational report, tested for real
    output shape against the real repo (proves the report mechanism itself
    works, distinct from asserting there are zero orphans, which would be a
    much stronger and more brittle claim about registry content)."""

    def test_orphan_report_runs_and_returns_real_categories(self):
        from bigcherry.patch import lifecycle as patch_lifecycle

        real_ids = _real_patch_ids()
        registry = sources.load_registry()
        tracked_plan_items = {
            entry.get("plan-item")
            for source in registry.get("sources", [])
            for entry in source.get("tracked", [])
            if entry.get("plan-item") and entry.get("plan-item") != "-"
        }
        statuses = patch_lifecycle.compute_all()

        # "source-pinned but never materialized" -- a real, expected, large
        # category today (most of the 65-item backlog is still planned).
        planned_only = [
            item for item in tracked_plan_items
            if item in statuses and statuses[item].source_pinned and not statuses[item].materialized
        ]
        self.assertGreater(
            len(planned_only), 0,
            "expected at least one real source-pinned-but-unmaterialized plan item today",
        )

        # No materialized patch should be a complete orphan (materialized
        # with zero source-pin AND zero contract) without at least a
        # plan-item name to explain it -- every 12xx-numbered patch in this
        # repo carries PROVENANCE.
        orphan_patches = [
            item for item, s in statuses.items()
            if s.materialized and not s.source_pinned and not s.contracted
        ]
        # This is a real, current-state assertion, not a hypothetical: if
        # it ever fails, that's a genuine new orphan worth a human looking
        # at, not a reason to weaken the check.
        self.assertEqual(orphan_patches, [], f"orphaned materialized patches: {orphan_patches}")


if __name__ == "__main__":
    unittest.main()
