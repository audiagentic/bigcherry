"""Guard Radiance production patch composition and source inheritance (PRQ04).

Production exceptions are explicit, typed and temporary; a validated patch
must never remain on the exception list.
"""

from __future__ import annotations

import copy
import tomllib
import unittest
from pathlib import Path

RADIANCE = Path(__file__).resolve().parents[3] / "engines" / "radiance"
RECIPES = RADIANCE / "recipes.toml"
EXCEPTIONS = RADIANCE / "untested-in-production.toml"
REASON_TYPES = frozenset({"build-enabler", "qualification-pending"})
EXCEPTION_FIELDS = frozenset({"patch-id", "reason-type", "reason", "plan-id"})


def production_guard_errors(recipes: dict, exceptions: dict, patch_states: dict[str, str]) -> list[str]:
    """Return all deviations from the declared production qualification contract."""
    errors: list[str] = []
    sources = recipes["source"]
    production_sets = sources["radiance"]["patch-sets"]
    production_patches: set[str] = set()
    for set_name in production_sets:
        patch_set = recipes["patch-set"].get(set_name)
        if patch_set is None:
            errors.append(f"source.radiance references unknown patch-set {set_name}")
            continue
        production_patches.update(patch_set["patches"])

    allowed: set[str] = set()
    entries = exceptions.get("exception", [])
    if not set(exceptions) <= {"exception"} or not isinstance(entries, list):
        return errors + ["exception file must contain only [[exception]] entries"]
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != EXCEPTION_FIELDS:
            errors.append(f"exception must have exactly {sorted(EXCEPTION_FIELDS)}: {entry!r}")
            continue
        patch_id = entry["patch-id"]
        if not isinstance(patch_id, str) or not patch_id.strip():
            errors.append(f"invalid exception patch-id: {patch_id!r}")
            continue
        if patch_id in allowed:
            errors.append(f"duplicate exception for {patch_id}")
        allowed.add(patch_id)
        for field in ("reason", "plan-id"):
            if not isinstance(entry[field], str) or not entry[field].strip():
                errors.append(f"{patch_id}: {field} must be non-empty")
        if not isinstance(entry["reason-type"], str) or entry["reason-type"] not in REASON_TYPES:
            errors.append(f"{patch_id}: unknown reason-type {entry['reason-type']!r}")
        if patch_id not in production_patches:
            errors.append(f"{patch_id}: stale exception, patch not in source.radiance")
        if patch_states.get(patch_id) == "validated":
            errors.append(f"{patch_id}: stale exception, already validated")

    for patch_id in sorted(production_patches):
        state = patch_states.get(patch_id)
        if state is None:
            errors.append(f"{patch_id}: missing patch metadata")
        elif state != "validated" and patch_id not in allowed:
            errors.append(f"{patch_id}: state {state!r} is not validated or allowlisted")
    return errors


def _load_real_data() -> tuple[dict, dict, dict[str, str]]:
    recipes = tomllib.loads(RECIPES.read_text(encoding="utf-8"))
    exceptions = tomllib.loads(EXCEPTIONS.read_text(encoding="utf-8"))
    patch_ids = {
        patch_id
        for set_name in recipes["source"]["radiance"]["patch-sets"]
        for patch_id in recipes["patch-set"][set_name]["patches"]
    }
    patch_ids.update(
        entry.get("patch-id") for entry in exceptions.get("exception", [])
        if isinstance(entry, dict) and isinstance(entry.get("patch-id"), str)
    )
    states: dict[str, str] = {}
    for patch_id in patch_ids:
        manifest = RADIANCE / "patches" / patch_id / "patch.toml"
        if manifest.is_file():
            raw = tomllib.loads(manifest.read_text(encoding="utf-8"))
            if raw.get("id") == patch_id:
                states[patch_id] = raw.get("state")
    return recipes, exceptions, states


class RadianceProductionQualificationTests(unittest.TestCase):
    def test_source_radiance_has_only_validated_or_explained_patches(self) -> None:
        recipes, exceptions, states = _load_real_data()
        self.assertEqual(production_guard_errors(recipes, exceptions, states), [])

    def test_guard_rejects_new_unvalidated_patch_without_exception(self) -> None:
        recipes, exceptions, states = _load_real_data()
        broken = copy.deepcopy(recipes)
        broken["patch-set"]["serving-core"]["patches"].append("unqualified_test_patch")
        states["unqualified_test_patch"] = "untested"
        self.assertIn(
            "unqualified_test_patch: state 'untested' is not validated or allowlisted",
            production_guard_errors(broken, exceptions, states),
        )

    def test_guard_rejects_stale_or_unexplained_exceptions(self) -> None:
        recipes, exceptions, states = _load_real_data()
        patch_id = exceptions["exception"][0]["patch-id"]
        validated_states = {**states, patch_id: "validated"}
        self.assertTrue(any("already validated" in err for err in
                            production_guard_errors(recipes, exceptions, validated_states)))
        for field in ("reason", "reason-type", "plan-id"):
            broken = copy.deepcopy(exceptions)
            broken["exception"][0][field] = ""
            self.assertTrue(any(field in err for err in
                                production_guard_errors(recipes, broken, states)), field)


if __name__ == "__main__":
    unittest.main()
