"""PRQ05: source admission by declared class and patch state (card-free)."""
from __future__ import annotations

import copy
import tomllib
import unittest
from unittest import mock
from dataclasses import replace
from pathlib import Path

from bigcherry.core import config, engines, paths
from bigcherry.patch import profile_evidence, registry, source_policy

_LAYOUT = engines.load(paths.REPO_ROOT, "radiance")
_RECIPES = _LAYOUT.recipes_path(paths.REPO_ROOT)
_PATCHES = _LAYOUT.patches_root(paths.REPO_ROOT)
_EXCEPTIONS = paths.REPO_ROOT / "engines" / "radiance" / "untested-in-production.toml"
_PENDING = {
    "rad0004_moe_remote_diagnostics": "diagnostic",
    "rad0010_routing_evidence": "diagnostic",
    "rad0006_heat_freeze_diagnostic": "superseded",
}


def _data():
    raw = tomllib.loads(_RECIPES.read_text(encoding="utf-8"))
    with mock.patch.object(config, "_sdk_root", return_value=Path("/tmp/sdk")):
        config_recipes = config.load(_RECIPES)
    states = {d.patch_id: d.state for d in registry.load_registry("radiance", _PATCHES).descriptors}
    exceptions = frozenset(
        row["patch-id"]
        for row in tomllib.loads(_EXCEPTIONS.read_text(encoding="utf-8"))["exception"]
    )
    return raw, config_recipes, states, exceptions


def _diagnostic_record(patch_id: str):
    return profile_evidence.ProfileEvidence(
        patch_id, "fixture-pin", "fixture-digest", False, (),
        {"mechanics": "mechanics test passed", "activation": "own marker recorded"},
        profile_evidence.DIAGNOSTIC_TIER,
    )


def _migration_fixture():
    raw, recipes, states, exceptions = _data()
    future_states = dict(states)
    records = {}
    for patch_id, target in _PENDING.items():
        if states[patch_id] not in ("untested", target):
            raise AssertionError(f"{patch_id}: unexpected pre-migration state {states[patch_id]}")
        future_states[patch_id] = target
        if target == "diagnostic":
            if states[patch_id] == target:
                loaded = profile_evidence.load(_PATCHES / patch_id)
                if loaded is not None:
                    records[patch_id] = loaded
            else:
                records[patch_id] = _diagnostic_record(patch_id)
    future_recipes = copy.deepcopy(recipes)
    # A retired diagnostic must be removed from its patch set in the migration.
    sets = future_recipes.patch_sets
    for name, patch_set in list(sets.items()):
        kept = tuple(p for p in patch_set.patches if p != "rad0006_heat_freeze_diagnostic")
        state = "diagnostic" if name in ("moe-remote-diag", "routing-evidence") else patch_set.required_state
        sets[name] = replace(patch_set, patches=kept, required_state=state)
    return raw, future_recipes, future_states, exceptions, records


class RadianceSourceClassesTests(unittest.TestCase):
    def test_all_sources_admit_only_their_declared_states_after_migration_fixture(self):
        raw, recipes, states, exceptions, records = _migration_fixture()
        self.assertEqual(
            set(raw["source"]),
            {name for name, source in recipes.sources.items() if source.source_class in registry.SOURCE_ALLOWED_STATES},
        )
        self.assertTrue(all(
            source["class"] in registry.SOURCE_ALLOWED_STATES
            for source in raw["source"].values()
        ))
        self.assertEqual(
            source_policy.source_state_problems(recipes, states, production_exceptions=exceptions,
                                                diagnostic_records=records), (),
        )

    def test_production_rejects_diagnostic_and_unlisted_untested(self):
        _, recipes, states, exceptions, records = _migration_fixture()
        bad = copy.deepcopy(recipes)
        core = bad.patch_sets["serving-core"]
        bad.patch_sets["serving-core"] = replace(
            core, patches=(*core.patches, "rad0004_moe_remote_diagnostics"),
        )
        errors = source_policy.source_state_problems(
            bad, states, production_exceptions=exceptions, diagnostic_records=records,
        )
        self.assertTrue(any("source radiance (production)" in error and "rad0004" in error for error in errors), errors)

        states["rad0002_moe_remote_handoff"] = "untested"
        errors = source_policy.source_state_problems(
            recipes, states, production_exceptions=exceptions, diagnostic_records=records,
        )
        self.assertTrue(any("source radiance (production)" in error and "rad0002" in error for error in errors), errors)

    def test_diagnostic_rejects_untested_and_requires_marker(self):
        _, recipes, states, exceptions, records = _migration_fixture()
        states["rad0004_moe_remote_diagnostics"] = "untested"
        errors = source_policy.source_state_problems(
            recipes, states, production_exceptions=exceptions, diagnostic_records=records,
        )
        self.assertTrue(any("source radiance-diag (diagnostic)" in error for error in errors), errors)

        states["rad0004_moe_remote_diagnostics"] = "diagnostic"
        records.pop("rad0004_moe_remote_diagnostics")
        errors = source_policy.source_state_problems(
            recipes, states, production_exceptions=exceptions, diagnostic_records=records,
        )
        self.assertTrue(any("activation record missing" in error for error in errors), errors)
        records["rad0004_moe_remote_diagnostics"] = replace(
            _diagnostic_record("rad0004_moe_remote_diagnostics"),
            checks={"mechanics": "passes"},
        )
        errors = source_policy.source_state_problems(
            recipes, states, production_exceptions=exceptions, diagnostic_records=records,
        )
        self.assertTrue(any("check 'activation' is missing" in error for error in errors), errors)

    def test_experiments_exclude_retired_and_allow_untested(self):
        _, recipes, states, exceptions, records = _migration_fixture()
        self.assertNotIn("rad0003_mtp_full_head_switch", source_policy.source_state_problems(
            recipes, states, production_exceptions=exceptions, diagnostic_records=records,
        ))
        states["rad0003_mtp_full_head_switch"] = "superseded"
        errors = source_policy.source_state_problems(
            recipes, states, production_exceptions=exceptions, diagnostic_records=records,
        )
        self.assertTrue(any("experiment mtp-full-head-switch" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
