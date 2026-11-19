"""PRQ05: real Radiance source admission by declared class and lifecycle state."""
from __future__ import annotations

import copy
import tomllib
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from bigcherry.core import config, engines, paths
from bigcherry.patch import profile_evidence, registry, source_policy

_LAYOUT = engines.load(paths.REPO_ROOT, "radiance")
_RECIPES = _LAYOUT.recipes_path(paths.REPO_ROOT)
_PATCHES = _LAYOUT.patches_root(paths.REPO_ROOT)


def _data():
    raw = tomllib.loads(_RECIPES.read_text(encoding="utf-8"))
    with mock.patch.object(config, "_sdk_root", return_value=Path("/tmp/sdk")):
        recipes = config.load(_RECIPES)
    patch_registry = registry.load_registry("radiance", _PATCHES)
    states = {descriptor.patch_id: descriptor.state for descriptor in patch_registry.descriptors}
    records = {
        descriptor.patch_id: evidence
        for descriptor in patch_registry.descriptors
        if descriptor.state == "diagnostic"
        if (evidence := profile_evidence.load(_PATCHES / descriptor.package_root)) is not None
    }
    return raw, recipes, patch_registry, states, records


def _problems(recipes, states, records):
    return source_policy.source_state_problems(recipes, states, diagnostic_records=records)


class RadianceSourceClassesTests(unittest.TestCase):
    def test_real_sources_admit_exactly_the_allowed_states(self):
        raw, recipes, _, states, records = _data()
        self.assertEqual(
            set(raw["source"]),
            set(recipes.sources),
        )
        self.assertTrue(all(
            source["class"] in registry.SOURCE_ALLOWED_STATES
            for source in raw["source"].values()
        ))
        self.assertEqual(_problems(recipes, states, records), ())

    def test_migrated_package_state_kind_records_and_retirement(self):
        _, recipes, patches, states, records = _data()
        self.assertEqual(states["rad0004_moe_remote_diagnostics"], "diagnostic")
        self.assertEqual(states["rad0010_routing_evidence"], "diagnostic")
        self.assertNotIn("rad0006_heat_freeze_diagnostic", states)    # archived, never loaded
        self.assertEqual(patches.get("rad0010_routing_evidence").kind, "diagnostic")
        self.assertEqual(recipes.patch_sets["moe-remote-diag"].patches, ("rad0004_moe_remote_diagnostics",))
        self.assertEqual(recipes.patch_sets["moe-remote-diag"].required_state, "diagnostic")
        self.assertEqual(recipes.patch_sets["routing-evidence"].required_state, "diagnostic")
        self.assertEqual(set(records), {"rad0004_moe_remote_diagnostics", "rad0010_routing_evidence"})
        for patch_id, record in records.items():
            self.assertEqual(record.patch_id, patch_id)
            self.assertEqual(record.tier, profile_evidence.DIAGNOSTIC_TIER)
            self.assertEqual(profile_evidence.problems(record), ())
        for patch_set in recipes.patch_sets.values():
            self.assertNotIn("rad0006_heat_freeze_diagnostic", patch_set.patches)
        for experiment in recipes.experiments.values():
            self.assertNotIn("rad0006_heat_freeze_diagnostic", experiment.patches)

    def test_production_rejects_every_nonvalidated_state(self):
        _, recipes, _, states, records = _data()
        for state in ("untested", "evaluated", "diagnostic", "superseded", "rejected"):
            with self.subTest(state=state):
                broken = copy.deepcopy(recipes)
                core = broken.patch_sets["serving-core"]
                broken.patch_sets["serving-core"] = replace(
                    core, patches=(*core.patches, "rad0004_moe_remote_diagnostics"),
                )
                changed = {**states, "rad0004_moe_remote_diagnostics": state}
                errors = _problems(broken, changed, records)
                self.assertTrue(any(
                    "source radiance (production)" in error and "rad0004" in error
                    for error in errors
                ), errors)

    def test_diagnostic_source_rejects_untested_and_requires_own_activation(self):
        _, recipes, _, states, records = _data()
        bad_states = {**states, "rad0004_moe_remote_diagnostics": "untested"}
        errors = _problems(recipes, bad_states, records)
        self.assertTrue(any("source radiance-diag (diagnostic)" in error for error in errors), errors)

        missing = {k: v for k, v in records.items() if k != "rad0004_moe_remote_diagnostics"}
        errors = _problems(recipes, states, missing)
        self.assertTrue(any("rad0004" in error and "activation record missing" in error for error in errors), errors)

        malformed = dict(records)
        malformed["rad0004_moe_remote_diagnostics"] = replace(
            records["rad0004_moe_remote_diagnostics"], checks={"mechanics": "passed"},
        )
        errors = _problems(recipes, states, malformed)
        self.assertTrue(any("check 'activation' is missing" in error for error in errors), errors)

    def test_source_name_does_not_override_declared_class(self):
        _, recipes, _, states, records = _data()
        changed = copy.deepcopy(recipes)
        changed.sources["radiance-diag"] = replace(
            changed.sources["radiance-diag"], source_class="production",
        )
        self.assertTrue(any(
            "source radiance-diag (production)" in error and "rad0004" in error
            for error in _problems(changed, states, records)
        ))

    def test_experiment_accepts_unqualified_but_not_retired(self):
        _, recipes, _, states, records = _data()
        self.assertEqual(_problems(recipes, states, records), ())
        probe = config.Experiment("probe", ("rad9999_probe",), (), (), (), ())
        with_probe = replace(recipes, experiments={**recipes.experiments, "probe": probe})
        self.assertEqual(_problems(with_probe, {**states, "rad9999_probe": "untested"}, records), ())
        errors = _problems(with_probe, {**states, "rad9999_probe": "superseded"}, records)
        self.assertTrue(any("experiment probe" in error for error in errors), errors)

    def test_macro_build_is_separate_from_lifecycle_admission(self):
        _, recipes, patches, _, _ = _data()
        self.assertEqual(patches.get("rad0004_moe_remote_diagnostics").requires, ("rad0002_moe_remote_handoff",))
        diag = recipes.sources["radiance-diag"]
        prod = recipes.sources["radiance"]
        self.assertTrue(set(prod.patch_sets) <= set(diag.patch_sets))
        self.assertEqual(set(diag.patch_sets) - set(prod.patch_sets), {"moe-remote-diag"})
        self.assertIn(("CMAKE_CXX_FLAGS", "-DBC_MOE_REMOTE_DIAG"), recipes.builds["moe-diag"].options)


if __name__ == "__main__":
    unittest.main()
