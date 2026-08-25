"""Isolated upstream patch-compatibility probe (HI46).

Materialises a candidate ref into its own isolated source slice and runs it
through the canonical campaign build lane (RE13/RV48) -- never the pinned
checkout, never the shared `artifacts/<revision>/` tree, and never the
mutable-checkout mechanics RE23 is retiring -- so a probe against `master`
(ahead of the pin) can never be mistaken for a real release's candidate set,
and cannot silently diverge from what a real canonical build actually does.
Answers "do the patches still apply and build cleanly against this ref",
nothing more: full release validation (the record -> tune -> promote ->
replay -> coverage gate sequence) is separate, larger work.

This version has no build knowledge of its own beyond the legacy `recipe`
name -> v2 (source, builds, platform) mapping below: it runs
``execute_campaign_lane`` directly, the exact same production API a real
`build` invocation uses, so a probe and a real build can never disagree
about what building this ref actually does. Earlier versions of this file
shelled out to `bigcherry pull` + `bigcherry legacy-build` as separate
subprocesses; RE23 deletes `legacy-build` entirely, so a probe still calling
it would simply break the day that lands.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

from . import paths
from .autotune_schema import VARIANT_SETS
from .multi_gpu_validate import validate_multi_gpu_claim
from .device_state_validate import validate_device_state_report


class ReleaseGateError(ValueError):
    """A validated-release claim is missing or contradicts its evidence."""


class ProbeConfigError(ValueError):
    """A probe request names a recipe/ref combination the canonical v2
    config cannot express -- distinct from a build/patch failure: this is
    a request the probe cannot even attempt, not evidence about the ref."""


# RE13: legacy `[compat.recipe.*]` names are a groups/states-filtered patch
# selection over one mutable checkout -- a different model from v2's fixed
# named Source objects. Most compat recipes correspond to a v2 source of the
# same name directly (bigcherry, bigcherry-native); the rest are explicit
# aliases here rather than guessed, so a recipe this table cannot place
# fails closed (ProbeConfigError) instead of silently probing the wrong
# source. Recipes whose groups/states subset (e.g. "core") has no
# corresponding v2 Source are deliberately NOT aliased -- add a real
# [source.*] for them before probing, rather than approximating one here.
_RECIPE_SOURCE_ALIASES = {
    "upstream": "llama-native",
    "workstation": "bigcherry",
    "dev": "bigcherry",
}

# The v2 compat loader (recipes.py) injects a "native" alias into ITS OWN
# builds table only, pointing at the real v2 "control" build -- v2's own
# config.Config.builds has no "native" key. A compat recipe's builds list
# can legitimately contain "native"; this is the same rename applied on
# the v2 lookup side.
_LEGACY_BUILD_ALIASES = {"native": "control"}


def _v2_source_name_for_recipe(recipe: str, cfg: Any) -> str:
    if recipe in cfg.sources:
        return recipe
    alias = _RECIPE_SOURCE_ALIASES.get(recipe)
    if alias is not None and alias in cfg.sources:
        return alias
    raise ProbeConfigError(
        f"recipe {recipe!r} has no v2 source mapping for the canonical "
        f"probe -- add a [source.{recipe}] (or extend "
        f"_RECIPE_SOURCE_ALIASES) before probing it"
    )


PRODUCTION_GATE_STAGES = (
    "audit", "patch", "generate", "build_descriptor", "record", "tune",
    "promote", "replay", "coverage",
)
PRODUCTION_GATE_STATES = frozenset(("prepared", "validated", "failed"))
PRODUCTION_STAGE_STATES = frozenset(("pending", "prepared", "validated", "failed"))


def validate_production_gate(record: dict[str, Any]) -> None:
    """Validate the ordered, evidence-bearing production release state machine."""
    gate = record.get("production_gate")
    if not isinstance(gate, dict):
        raise ReleaseGateError(
            "validated release claim lacks production_gate evidence")
    state = gate.get("state")
    if state not in PRODUCTION_GATE_STATES:
        raise ReleaseGateError(
            "production_gate.state must be prepared, validated, or failed")
    declared_architectures = gate.get("required_architectures")
    required_architectures = record.get("required_architectures")
    if (not isinstance(declared_architectures, list)
            or declared_architectures != sorted(set(declared_architectures))
            or (isinstance(required_architectures, list)
                and declared_architectures != sorted(set(required_architectures)))):
        raise ReleaseGateError(
            "production_gate.required_architectures disagrees with the release claim")

    stages = gate.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(PRODUCTION_GATE_STAGES):
        raise ReleaseGateError(
            "production_gate.stages must contain every required production stage")

    unvalidated_seen = False
    for name in PRODUCTION_GATE_STAGES:
        evidence = stages[name]
        if not isinstance(evidence, dict):
            raise ReleaseGateError(f"production_gate stage {name!r} is invalid")
        stage_state = evidence.get("state")
        if stage_state not in PRODUCTION_STAGE_STATES:
            raise ReleaseGateError(
                f"production_gate stage {name!r} has an invalid state")
        if stage_state != "validated":
            unvalidated_seen = True
        elif unvalidated_seen:
            raise ReleaseGateError(
                f"production_gate stage {name!r} is validated after an incomplete stage")
        if stage_state == "validated":
            if evidence.get("ok") is not True:
                raise ReleaseGateError(
                    f"production_gate stage {name!r} lacks ok=true evidence")
            refs = evidence.get("evidence")
            if (not isinstance(refs, list) or not refs
                    or any(not isinstance(ref, str) or not ref.strip() for ref in refs)):
                raise ReleaseGateError(
                    f"production_gate stage {name!r} lacks evidence references")

    if state == "validated" and unvalidated_seen:
        raise ReleaseGateError(
            "validated production_gate contains an incomplete stage")
    if state == "failed" and not any(
            stages[name].get("state") == "failed" for name in PRODUCTION_GATE_STAGES):
        raise ReleaseGateError(
            "failed production_gate must identify a failed stage")


def _architecture_report(record: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """Return architecture evidence and the architectures required by a claim.

    New records use an explicit report with ``required``, ``observed``,
    ``validated`` and ``by_architecture`` fields.  The older flat mapping is
    still accepted for inventory diagnostics, where it is evidence about the
    observed machine rather than an optimized release-support claim.
    """
    raw = record.get("architecture_coverage")
    if not isinstance(raw, dict) or not raw:
        raise ReleaseGateError(
            "validated release claim lacks architecture_coverage evidence")

    if "by_architecture" in raw:
        by_architecture = raw.get("by_architecture")
        required = raw.get("required")
        observed = raw.get("observed")
        validated = raw.get("validated")
        for name, value in (("required", required), ("observed", observed),
                            ("validated", validated)):
            if not isinstance(value, list) or any(
                not isinstance(item, str) or not item.strip() for item in value
            ) or len(set(value)) != len(value):
                raise ReleaseGateError(
                    f"architecture_coverage.{name} must be a unique list")
        if not isinstance(by_architecture, dict) or not by_architecture:
            raise ReleaseGateError(
                "architecture_coverage.by_architecture must be a non-empty object")
        declared_required = record.get("required_architectures", required)
        if (not isinstance(declared_required, list)
                or set(required) != set(declared_required)):
            raise ReleaseGateError(
                "architecture_coverage required set disagrees with claim")
        observed_by_entry = {
            name for name, evidence in by_architecture.items()
            if isinstance(evidence, dict) and evidence.get("observed") is True
        }
        validated_by_entry = {
            name for name, evidence in by_architecture.items()
            if isinstance(evidence, dict) and evidence.get("validated") is True
        }
        if (set(observed) != observed_by_entry
                or set(validated) != validated_by_entry):
            raise ReleaseGateError(
                "architecture_coverage observed/validated set disagrees with evidence")
        return by_architecture, set(required)

    # Legacy flat map: retain this path for inventory-only diagnostic claims.
    expected = record.get("required_architectures", record.get("architectures"))
    if expected is None:
        expected = list(raw)
    if (not isinstance(expected, list) or not expected
            or any(not isinstance(item, str) or not item.strip() for item in expected)):
        raise ReleaseGateError(
            "architecture_coverage lacks required architecture reporting")
    return raw, set(expected)


def validate_release_claim(record: dict[str, Any]) -> None:
    """Fail closed when a record claims validation without coverage evidence.

    ``probe`` records compatibility only and therefore does not need hardware
    evidence.  A separate producer may add ``claim: validated`` once the full
    record/tune/promote/replay pipeline has run.  At that boundary both
    architecture coverage and candidate coverage are mandatory and must agree
    with the identities they describe.
    """
    validate_multi_gpu_claim(record)
    if any(key in record for key in ("device_state_pre", "device_state_post", "device_clock_drift")):
        validate_device_state_report(record)
    if record.get("claim") != "validated" and record.get("stage") != "validated":
        return

    validate_production_gate(record)

    candidates = record.get("candidate_coverage")
    if not isinstance(candidates, dict):
        raise ReleaseGateError(
            "validated release claim lacks candidate_coverage evidence")

    supported = record.get("supported_coverage")
    if supported is not None:
        try:
            from .autotune_schema import validate_supported_coverage
            validate_supported_coverage(supported)
        except (ValueError, TypeError) as exc:
            raise ReleaseGateError(
                f"supported_coverage evidence is invalid: {exc}") from exc

    architecture, required_architectures = _architecture_report(record)
    variant_set = candidates.get("variant_set")
    if variant_set == "inventory" and "by_architecture" not in record["architecture_coverage"]:
        # Inventory is diagnostic evidence.  A legacy flat map may describe
        # what was observed without pretending that every target was tested.
        required_architectures = set()

    expected_architectures = record.get("architectures")
    if expected_architectures is not None:
        if (not isinstance(expected_architectures, list)
                or not expected_architectures
                or (variant_set != "inventory"
                    and set(expected_architectures) != required_architectures)):
            raise ReleaseGateError(
                "architecture_coverage required set does not match declared architectures")

    for arch, evidence in architecture.items():
        if not isinstance(arch, str) or not arch.strip():
            raise ReleaseGateError("architecture coverage contains an invalid key")
        if not isinstance(evidence, dict):
            raise ReleaseGateError(f"architecture coverage for {arch!r} is invalid")
        validated = evidence.get("validated", evidence.get("status") == "validated")
        observed = evidence.get(
            "observed", evidence.get("status") in {"observed", "validated"})
        if not isinstance(observed, bool) or not isinstance(validated, bool):
            raise ReleaseGateError(
                f"architecture coverage for {arch!r} lacks boolean observed/validated fields")
        if validated and not observed:
            raise ReleaseGateError(
                f"architecture coverage for {arch!r} is validated without observation")
        if arch in required_architectures and (not observed or not validated):
            raise ReleaseGateError(
                f"required architecture {arch!r} lacks observed and validated evidence")
        if evidence.get("candidate_coverage") is not True:
            raise ReleaseGateError(
                f"architecture coverage for {arch!r} lacks candidate coverage")

    missing = required_architectures - set(architecture)
    if missing:
        raise ReleaseGateError(
            f"required architecture coverage is missing: {sorted(missing)}")

    # Optimized claims must use the explicit report so a partial flat map
    # cannot accidentally be interpreted as complete architecture support.
    if variant_set != "inventory" and "by_architecture" not in record["architecture_coverage"]:
        raise ReleaseGateError(
            "optimized release claim requires explicit per-architecture coverage report")

    observed_types = candidates.get("observed_types")
    by_type = candidates.get("by_type")
    variant_set = candidates.get("variant_set")
    if variant_set not in VARIANT_SETS:
        raise ReleaseGateError(
            "candidate coverage has an invalid or missing variant_set")
    if (not isinstance(observed_types, list) or not observed_types
            or not isinstance(by_type, dict)
            or set(observed_types) != set(by_type)):
        raise ReleaseGateError(
            "candidate coverage observed_types and by_type are inconsistent")
    for type_name, evidence in by_type.items():
        if not isinstance(type_name, str) or not type_name.strip():
            raise ReleaseGateError("candidate coverage contains an invalid type")
        if not isinstance(evidence, dict) or evidence.get("observed") is not True:
            raise ReleaseGateError(
                f"candidate coverage for {type_name!r} is not observed")
        count = evidence.get("candidate_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ReleaseGateError(
                f"candidate coverage for {type_name!r} has no candidates")
        alternatives = evidence.get("alternative_count")
        if not isinstance(alternatives, int) or isinstance(alternatives, bool) \
                or alternatives < 0:
            raise ReleaseGateError(
                f"candidate coverage for {type_name!r} has an invalid "
                "alternative count")
        # Inventory is deliberately native-only: it is a diagnostic profile
        # used to establish observed signatures and must not be treated as an
        # optimization claim. Every optimized release profile, however, must
        # have at least one non-native candidate for every observed type.
        if variant_set != "inventory" and alternatives < 1:
            raise ReleaseGateError(
                f"optimized candidate coverage for {type_name!r} has zero "
                "alternatives")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-") or "upstream"


def _write(run: Path, record: dict[str, Any]) -> Path:
    validate_release_claim(record)
    record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    path = run / "run.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _failure_class(stage: str) -> str:
    return "patch-drift" if stage == "build" else f"{stage}-failed"


def probe(
    run_id: str,
    staging_root: Path,
    ref: str,
    recipe: str,
    inventory: Path | None = None,
    promoted_winners: Path | None = None,
) -> tuple[int, Path]:
    """Prove `ref` still audits, patches, and builds clean under `recipe`.

    Never touches the pinned checkout: this run gets its own isolated
    ``work_root`` (source/build cache + a dedicated ArtifactStore) under
    ``staging_root``, so a probe against an unrelated ref (typically
    `master`, ahead of the pin) cannot corrupt or be confused with the
    shared checkout or its `artifacts/<revision>/` tree. The revision
    embedded in every stage's provenance is the probed ref's own resolved
    commit, which is exactly what makes it distinguishable from a real
    release's candidates rather than requiring a separate override flag.

    Still shares the host-local upstream git mirror with real production
    builds (cloning llama.cpp fresh per probe would be needlessly slow) --
    ``fetch_ref`` only fetches into it, it is never mutated any other way
    from here.
    """
    from . import config as campaign_config
    from . import recipes as legacy_recipes
    from .artifacts import ArtifactStore
    from .campaign.lane import CampaignLaneError, CampaignLaneExecutionSpec, execute_campaign_lane
    from .context import ProjectContext
    from .workspace import UpstreamRepository, WorkspaceError

    run = staging_root / safe_name(run_id)
    if run.exists():
        raise FileExistsError(f"run already exists: {run}")
    run.mkdir(parents=True)
    record: dict[str, Any] = {
        "schema_version": 2, "run_id": run_id, "ref": ref, "recipe": recipe,
        "source_revision": ref, "bigcherry_revision": "unknown",
    }
    try:
        record["bigcherry_revision"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass

    def _config_failure(exc: Exception) -> tuple[int, Path]:
        record["outcome"] = "config-error"
        record["failure"] = {"stage": "config", "detail": str(exc),
                              "failure_class": "config-error"}
        return 1, _write(run, record)

    default_context = ProjectContext.resolve()
    context = ProjectContext.resolve(
        work_root=run / "work", upstream_repo=default_context.upstream_repo)

    try:
        legacy_recipe = legacy_recipes.get(recipe, path=context.config_path)
        cfg = campaign_config.load(context.config_path)
        source_name = _v2_source_name_for_recipe(recipe, cfg)
    except (legacy_recipes.RecipeError, campaign_config.ConfigError, ProbeConfigError) as exc:
        return _config_failure(exc)

    platform_name = legacy_recipe.platform
    if platform_name is None or platform_name not in cfg.platforms:
        return _config_failure(
            ProbeConfigError(f"recipe {recipe!r} names no usable v2 platform"))
    build_names = [_LEGACY_BUILD_ALIASES.get(name, name) for name in legacy_recipe.builds]
    unknown_builds = [name for name in build_names if name not in cfg.builds]
    if unknown_builds:
        return _config_failure(
            ProbeConfigError(f"recipe {recipe!r} names unknown v2 build(s): "
                             f"{', '.join(unknown_builds)}"))

    try:
        resolved_ref = UpstreamRepository(context.upstream_repo).fetch_ref(ref)
    except WorkspaceError as exc:
        record["outcome"] = "pull-failed"
        record["failure"] = {"stage": "pull", "detail": str(exc),
                              "failure_class": _failure_class("pull")}
        return 1, _write(run, record)
    record["source_revision"] = resolved_ref

    probe_source = dc_replace(cfg.sources[source_name], ref=resolved_ref)
    cfg = dc_replace(cfg, sources={**cfg.sources, source_name: probe_source})

    store = ArtifactStore(run / "artifacts-store")
    architectures = cfg.platforms[platform_name].targets
    # GPT-auto-agent review (RE13 follow-up, 2026-08-17): the shipped
    # default recipe ("bigcherry") includes record/tune/replay -- tune
    # needs "inventory", replay needs "inventory" AND "promoted-winners".
    # Neither is meaningful to synthesize for an arbitrary, possibly
    # untested future ref (there is no real tuning history for it yet), so
    # a build whose declared needs this probe cannot supply is SKIPPED,
    # not attempted -- attempting it always failed with a CampaignBuildError
    # that got misreported as "patch-drift-or-build-failed", when the real
    # situation is "this input does not exist to test against", not "the
    # patches/build are broken". Compilation compatibility (what this probe
    # actually answers) is still tested for every build whose needs ARE
    # satisfiable.
    available_inputs = {
        name: path for name, path in (
            ("inventory", inventory), ("promoted-winners", promoted_winners))
        if path is not None
    }
    build_records: dict[str, Any] = {}
    build_ok = True
    any_skipped = False
    for build_name in build_names:
        needs = cfg.builds[build_name].needs
        missing = needs - set(available_inputs)
        if missing:
            build_records[build_name] = {
                "ok": True, "skipped": True,
                "reason": f"missing required input(s): {sorted(missing)}",
            }
            any_skipped = True
            continue
        inputs = tuple((name, available_inputs[name]) for name in sorted(needs))
        spec = CampaignLaneExecutionSpec(
            source_name=source_name, build_name=build_name, platform_name=platform_name,
            architectures=architectures, inputs=inputs,
        )
        try:
            result = execute_campaign_lane(
                spec, cfg=cfg, context=context, store=store,
                run_id=f"{run_id}-{build_name}")
        except CampaignLaneError as exc:
            build_records[build_name] = {"ok": False, "detail": str(exc)}
            build_ok = False
            break
        build_records[build_name] = {
            "ok": True, "build_plan_id": result.build_plan_id,
            "workload_id": result.workload_id,
        }
    record["builds"] = build_records
    # GPT-auto-agent review (RE13 follow-up, 2026-08-17): "compatible" must
    # not silently mean more than was actually demonstrated. A skipped
    # build (missing inventory/promoted-winners) means its own compile
    # path was never exercised at all -- record that explicitly as
    # "compatible-partial" rather than letting an incomplete probe report
    # the same unqualified "compatible" a fully-exercised one would.
    if not build_ok:
        record["outcome"] = "patch-drift-or-build-failed"
    elif any_skipped:
        record["outcome"] = "compatible-partial"
    else:
        record["outcome"] = "compatible"
    if not build_ok:
        failed_build = next(name for name, info in build_records.items() if not info["ok"])
        record["failure"] = {"stage": "build", "build": failed_build,
                              "detail": build_records[failed_build]["detail"],
                              "failure_class": _failure_class("build")}
    return (0 if build_ok else 1), _write(run, record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bigcherry probe-release", description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--staging-root", default=str(paths.ARTIFACTS / "release-runs"))
    parser.add_argument("--ref", default="master")
    parser.add_argument("--recipe", default="bigcherry")
    parser.add_argument(
        "--inventory", default=None,
        help="record-mode inventory JSON for recipes whose build includes tuning",
    )
    parser.add_argument(
        "--promoted-winners", default=None,
        help="promoted-winners JSONL for recipes whose build includes replay",
    )
    args = parser.parse_args(argv)
    code, path = probe(
        args.run_id, Path(args.staging_root), args.ref, args.recipe,
        Path(args.inventory) if args.inventory else None,
        Path(args.promoted_winners) if args.promoted_winners else None,
    )
    print(f"record: {path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
