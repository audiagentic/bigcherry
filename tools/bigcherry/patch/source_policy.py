"""State-based source admission for patch recipes.

The source class is explicit recipe metadata. Patch kinds, compile-time
macros and recipe names never determine lifecycle admission.
"""
from __future__ import annotations
from collections.abc import Mapping
from ..core import config
from . import profile_evidence, registry

def source_state_problems(
    recipes: config.Config,
    states: Mapping[str, str],
    *,
    diagnostic_records: Mapping[str, profile_evidence.ProfileEvidence] | None = None,
) -> tuple[str, ...]:
    """Check every source/experiment by declared class and lifecycle state."""
    records = diagnostic_records or {}
    problems: list[str] = []
    for name, source in sorted(recipes.sources.items()):
        allowed = registry.SOURCE_ALLOWED_STATES[source.source_class]
        selected = {
            patch_id
            for set_name in source.patch_sets
            for patch_id in recipes.patch_sets[set_name].patches
        }
        for patch_id in sorted(selected):
            state = states.get(patch_id)
            if state in allowed:
                if state == "diagnostic":
                    record = records.get(patch_id)
                    if record is None:
                        problems.append(f"source {name}: {patch_id} diagnostic activation record missing")
                    else:
                        issues = profile_evidence.problems(record)
                        if record.patch_id != patch_id or record.tier != profile_evidence.DIAGNOSTIC_TIER:
                            issues += ("record identity or tier mismatch",)
                        problems.extend(f"source {name}: {patch_id}: {issue}" for issue in issues)
                continue
            problems.append(f"source {name} ({source.source_class}): {patch_id} state {state!r} not allowed")
    for name, experiment in sorted(recipes.experiments.items()):
        for patch_id in experiment.patches:
            state = states.get(patch_id)
            if state not in registry.EXPERIMENT_ALLOWED_STATES:
                problems.append(f"experiment {name}: {patch_id} state {state!r} not allowed")
    return tuple(problems)
