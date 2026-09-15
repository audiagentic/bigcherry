"""PA26 stage 2: the two-arm HIP hardware-run harness.

Extends ``replay_equivalence.py``'s offline composition/corpus preflight
with the staged pipeline GPT's design review (dev-gpt-agent session
ses_c2892cdae7f14feb, req_19d7a4cde3e84bd8 then req_a4974e46d4d643e0) called
for: materialization/build identity binding, then real replay execution and
exact comparison.

Design (per req_a4974e46d4d643e0's review of the first draft of this
module):

1. Composition/corpus preflight -- ``replay_equivalence.py`` (unchanged).
2. Build identity -- both arms are compiled through the REAL production
   pipeline, ``campaign.lane.execute_campaign_lane`` (never a second
   materialize/build implementation), via an injectable ``lane_executor``
   parameter that defaults to the real function. Identity fairness is
   checked via the ``BuildPlan`` PROJECTION excluding ``source_slice_id``
   (which legitimately differs -- composition is the variable under test)
   plus ``CampaignLaneResult.effective_build_id`` equality (the actual
   resolved configure/toolchain identity, not merely the requested plan).
3. TWO paired lane specs, not one:
   - the PRODUCTION pair (exact PA26 control vs exact serving-core
     candidate) proves process/output/correctness equivalence, but is
     intentionally diagnostics-free on both arms (the production candidate
     never carries 0810_replay_hit_diagnostics, and the production build
     itself carries no GGML_HIP_REPLAY_DIAGNOSTICS instrumentation) --
     it cannot emit a per-dispatch hit log.
   - the DIAGNOSTIC companion pair re-adds 0810 to the candidate ONLY as an
     ephemeral PA26 probe (``replay_equivalence.build_serving_core_diagnostic_config``)
     under an ephemeral ``replay-diagnostic``-shaped build, so both
     diagnostic arms share the same per-dispatch observer.
     ``require_diagnostic_matches_production`` explicitly binds the
     diagnostic candidate's composition back to the production candidate's
     (diagnostic == production + 0810, nothing else different) before any
     diagnostic result is trusted.
4. Runtime execution -- an injectable ``runtime_runner`` callable, called
   once per arm of whichever pair (production or diagnostic) is meant to
   supply the comparison data, returning one ``ArmRuntimeResult`` (arm-level
   fields: process success, clean shutdown, output digest, correctness,
   model/GPU/runtime-args identity -- plus the ordered per-dispatch entries).
   The default, ``real_hardware_runtime_runner``, is a deliberately
   unimplemented real-integration stub that raises ``RuntimeNotEvaluated``
   (never a bare ``NotImplementedError``, so an unrelated bug elsewhere
   cannot be silently swallowed as "not evaluated") and is NEVER called by
   any test in this repo.
5. Comparison -- ``compare_runtime_results`` requires: (a) EXACT equality of
   every arm-level field; (b) the OBSERVED dispatch set on both arms exactly
   equals ``expected_dispatches`` (derived from the frozen corpus) -- two
   empty result sets can no longer read as equivalent; (c) EXACT per-entry
   equality of dispatch/signature/winner/config_binding/transform_id/
   match_kind/call_count/outcome (GPT: "Require exact equality; do not
   normalize away ordering or identities").

``build_hardware_receipt`` composes all of this into one receipt. Only
``RuntimeNotEvaluated`` is caught to produce a ``NOT_EVALUATED`` receipt --
any other exception (a parser bug, an SSH/process error after execution
started, a malformed log, a runner defect) propagates, never silently
downgraded to "not evaluated".
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable

from ..core.artifacts import ArtifactStore
from ..core.context import ProjectContext
from ..core import config
from ..patch import patchset
from ..tuning.execution_audit import HitRecord
from . import replay_equivalence as offline
from .lane import CampaignLaneExecutionSpec, CampaignLaneResult, execute_campaign_lane


class ReplayEquivalenceHardwareError(offline.ReplayEquivalenceError):
    pass


class RuntimeNotEvaluated(Exception):
    """Raised by a ``RuntimeRunner`` ONLY for a real, known reason runtime
    execution cannot happen right now (no hardware available, the stub is
    unimplemented). Never a stand-in for an unexpected error -- a parser
    bug, a process crash mid-run, a malformed log, or any other runner
    defect must raise something else so it propagates as a real failure."""


#: Matches execute_campaign_lane's call shape exactly -- the default IS
#: that real function; tests inject a fake with the same call shape.
LaneExecutor = Callable[..., CampaignLaneResult]


# GPT design correction (req_579777e899454ed9), following req_9cd5b140ca544ef8:
# raw generated_compile_inputs_hash exact equality is ALSO over-strong for
# PA26 -- candidate_implementation_digest() hashes raw source-file bytes for
# each candidate, and hip-autotune-build-hash.h embeds the manifest/build
# identity itself, so ANY expected source-composition removal touching an
# implementation-mapped file legitimately changes this hash even when
# registry/config semantics are identical. generated_tree.compile_inputs_hash
# is a byte-identity/reuse mechanism, not a semantic-equivalence oracle.
#
# STRICT_GENERATED_FILES must hash byte-identical between arms (registry
# tables, arch header, MMVQ instance geometry -- the actual compiled
# candidate surface). EXPECTED_IDENTITY_ONLY_FILE (the build-hash header,
# which embeds manifest_hash/build_descriptor -- themselves derived from
# generated_at and other real per-arm identity) is the one file explicitly
# NOT required to match.
STRICT_GENERATED_FILES: frozenset[str] = frozenset({
    "hip-autotune-registry.inc",
    "hip-autotune-arch.h",
    "hip-autotune-mmvq-instances.inc",
})
EXPECTED_IDENTITY_ONLY_FILE = "hip-autotune-build-hash.h"

#: Top-level manifest fields that are real per-arm/per-run identity, not
#: candidate-registry/config semantics -- excluded from the manifest
#: projection compared below.
_MANIFEST_IDENTITY_ONLY_FIELDS = ("generated_at", "manifest_hash", "build_descriptor")
#: Per-candidate fields excluded for the same reason (raw source-byte
#: digests of files that may differ between arms for an expected removed-
#: module reason without changing runtime candidate configuration).
_CANDIDATE_IDENTITY_ONLY_FIELDS = ("implementation_digest", "implementation_source_files")


def _strip_manifest_identity_only_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    projected = dict(manifest)
    for key in _MANIFEST_IDENTITY_ONLY_FIELDS:
        projected.pop(key, None)
    candidates = projected.get("candidates")
    if isinstance(candidates, list):
        projected["candidates"] = [
            {
                field: value
                for field, value in candidate.items()
                if field not in _CANDIDATE_IDENTITY_ONLY_FIELDS
            }
            if isinstance(candidate, dict)
            else candidate
            for candidate in candidates
        ]
    return projected


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _read_json_ref(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return None
    return data if isinstance(data, dict) else None


@dataclasses.dataclass(frozen=True)
class ArmBuildIdentity:
    """Per-arm identity extracted from a real ``CampaignLaneResult`` --
    never hand-invented. ``source_slice_id``/``build_plan_id`` are recorded
    for the receipt but deliberately NOT required to match between arms
    (different patch composition is the thing under test); everything else
    here IS required to match by ``require_shared_build_inputs``."""

    resolved_revision: str
    source_slice_id: str
    build_plan_id: str
    #: The full BuildPlan.canonical() dict EXCLUDING source_slice_id, as a
    #: sorted-items tuple -- the fairness check GPT asked for: same
    #: platform/targets/cmake_options-minus-patch-composition/backend/
    #: variant_set/catalog_architectures/requested_targets/input_hashes/
    #: toolchain/environment, differing only in the one field composition
    #: necessarily changes.
    build_plan_projection: tuple[tuple[str, Any], ...]
    effective_build_id: str | None
    #: PA26 (req_9cd5b140ca544ef8): the raw post-configure CMakeCache record
    #: (not merely its digest), as a sorted-items tuple -- kept alongside
    #: ``effective_build_id`` for provenance. ``require_shared_build_inputs``
    #: no longer requires ``effective_build_id`` equality; it requires exact
    #: equality of this field AFTER the narrow, caller-supplied
    #: ``pa26_effective_configure_projection`` normalization.
    effective_configure: tuple[tuple[str, str], ...]
    #: The generated-catalog byte-reuse identity (``workers.py``'s
    #: ``compile_inputs_hash``) -- recorded for provenance only. GPT design
    #: correction (req_579777e899454ed9): raw equality is over-strong for
    #: PA26 (it folds in per-candidate source-byte digests that legitimately
    #: differ for an expected removed-module reason); ``require_shared_
    #: build_inputs`` instead requires the semantic projection fields below.
    generated_compile_inputs_hash: str | None
    #: Canonical JSON of the generated manifest with real per-run/per-arm
    #: identity fields stripped (see ``_strip_manifest_identity_only_fields``)
    #: -- required exact. None when the build has no generate stage.
    generated_manifest_projection: str | None
    #: The generated tree's declared compile-input filename set -- required
    #: exact (same files must be compiled against on both arms).
    generated_tree_compile_inputs: tuple[str, ...]
    #: Byte hashes of ``STRICT_GENERATED_FILES`` only (the actual compiled
    #: candidate-registry/arch/MMVQ-instance surface) -- required exact.
    #: ``EXPECTED_IDENTITY_ONLY_FILE`` (the build-hash header, which embeds
    #: manifest_hash/build_descriptor) is deliberately excluded here.
    generated_strict_file_hashes: tuple[tuple[str, str], ...]
    input_hashes: tuple[tuple[str, str], ...]
    binary_digest: str
    runtime_bundle_digest: str
    manifest_digest: str | None
    generated_tree_digest: str | None

    @classmethod
    def from_result(cls, result: CampaignLaneResult) -> "ArmBuildIdentity":
        canonical = dict(result.build_plan.canonical())
        canonical.pop("source_slice_id", None)
        manifest_doc = _read_json_ref(
            result.manifest_ref.path if result.manifest_ref else None
        )
        tree_doc = _read_json_ref(
            result.generated_tree_ref.path if result.generated_tree_ref else None
        )
        generated_manifest_projection = (
            _canonical_json(_strip_manifest_identity_only_fields(manifest_doc))
            if manifest_doc is not None
            else None
        )
        tree_files = tree_doc.get("files", {}) if tree_doc is not None else {}
        tree_compile_inputs = (
            tuple(sorted(tree_doc.get("compile_inputs", ())))
            if tree_doc is not None
            else ()
        )
        strict_hashes = tuple(
            sorted(
                (name, tree_files[name])
                for name in STRICT_GENERATED_FILES
                if isinstance(tree_files, dict) and name in tree_files
            )
        )
        return cls(
            resolved_revision=result.resolved_revision,
            source_slice_id=result.source_slice_id,
            build_plan_id=result.build_plan.build_plan_id,
            build_plan_projection=tuple(sorted(canonical.items())),
            effective_build_id=result.effective_build_id,
            effective_configure=tuple(sorted(result.effective_configure.items())),
            generated_compile_inputs_hash=result.generated_compile_inputs_hash,
            generated_manifest_projection=generated_manifest_projection,
            generated_tree_compile_inputs=tree_compile_inputs,
            generated_strict_file_hashes=strict_hashes,
            input_hashes=tuple(
                sorted((name, ref.content_hash) for name, ref in result.input_refs)
            ),
            binary_digest=result.binary_ref.content_hash,
            runtime_bundle_digest=result.runtime_bundle_ref.content_hash,
            manifest_digest=(
                result.manifest_ref.content_hash if result.manifest_ref else None
            ),
            generated_tree_digest=(
                result.generated_tree_ref.content_hash
                if result.generated_tree_ref
                else None
            ),
        )


@dataclasses.dataclass(frozen=True)
class BuildStageDelta:
    control: ArmBuildIdentity
    candidate: ArmBuildIdentity


# GPT design correction (req_9cd5b140ca544ef8): the exact effective_build_id
# equality check in require_shared_build_inputs() is over-strong for PA26 --
# removing the PA20 serving boundary (0110/0810) makes several GGML_HIP_*
# cmake option() declarations go entirely UNDECLARED in the candidate's
# CMakeCache.txt vs present-and-OFF in control, even though every such
# option's VALUE is inert (OFF) on both arms for a plain replay build. This
# is a PA26-LOCAL semantic-equivalence rule -- it does NOT change
# build.effective_build_id()'s global semantics, which correctly fingerprints
# exact resolved CMake state for every other caller.
#
# Maps each removed-module-owned option to the module that declares it.
# Normalization is applied ONLY for an option whose owning module is
# actually absent from the candidate for the comparison at hand (see
# ``allowed_removed_modules`` below) -- never blanket-applied.
_PA26_REMOVED_OPTION_OWNERS: dict[str, str] = {
    "GGML_HIP_AUTOTUNE": "0110_campaign_tune_record_build",
    "GGML_HIP_AUTOTUNE_RECORD": "0110_campaign_tune_record_build",
    "GGML_HIP_DISPATCH_DIAGNOSTICS": "0110_campaign_tune_record_build",
    "GGML_HIP_ROUTING_TRANSFORM": "0110_campaign_tune_record_build",
    "GGML_HIP_WORKSPACE_METRICS": "0110_campaign_tune_record_build",
    "GGML_HIP_REPLAY_DIAGNOSTICS": "0810_replay_hit_diagnostics",
}


# GPT design correction (req_13b71975cbe94c8d): a build-local path derived
# from (source_slice_id, build_plan_id) -- both of which already legitimately
# differ between arms by PA26 design -- is itself non-semantic identity
# noise, not behavioral content. A FIXED allowlist only (never a generic
# "contains source_slice_id/build_plan_id" rule -- GPT explicit: that could
# hide a future semantically important option that happens to embed one of
# those IDs). Validated-then-canonicalized, not blindly deleted: this proves
# each arm actually used ITS OWN expected content-addressed generated
# directory before discarding the pathname difference. Safe because the
# generated directory's actual CONTENTS are separately bound by the
# stronger semantic checks above (compile-input filename set, strict
# generated files, manifest projection) -- this key carries location
# identity only.
_PA26_DERIVED_PATH_CONFIGURE_KEYS: frozenset[str] = frozenset({
    "GGML_HIP_AUTOTUNE_GENERATED_DIR",
})


def _normalize_pa26_derived_path_keys(
    identity: "ArmBuildIdentity", configure: dict[str, str]
) -> None:
    """Mutates ``configure`` in place: for each key in
    ``_PA26_DERIVED_PATH_CONFIGURE_KEYS``, requires it be present and end
    with the exact expected build-local suffix
    (``/builds/<source_slice_id>/<build_plan_id>/generated-inputs``), then
    replaces its value with a fixed placeholder so the two arms' otherwise-
    identical records can compare equal. Fails closed (never silently drops
    the key) if it is missing or does not match the expected shape --
    GPT explicit: validate then canonicalize, never blindly delete."""
    expected_suffix = (
        f"/builds/{identity.source_slice_id}/{identity.build_plan_id}/generated-inputs"
    )
    for key in _PA26_DERIVED_PATH_CONFIGURE_KEYS:
        value = configure.get(key)
        if value is None:
            continue
        normalized = value.replace("\\", "/")
        if not normalized.endswith(expected_suffix):
            raise ReplayEquivalenceHardwareError(
                f"{key} is not the expected build-local generated-input path "
                f"for this arm's own (source_slice_id, build_plan_id): {value!r} "
                f"(expected suffix {expected_suffix!r})"
            )
        configure[key] = "<PA26_BUILD_LOCAL_GENERATED_DIR>"


def pa26_effective_configure_projection(
    control: dict[str, str],
    candidate: dict[str, str],
    *,
    allowed_removed_modules: frozenset[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Narrow, directional normalization of two ``effective_configure``
    records before requiring exact equality.

    The ONLY tolerated asymmetry: for an option owned by a module in
    ``allowed_removed_modules`` (i.e. a module this specific comparison
    legitimately removes from the candidate), control declares it at
    literal ``"OFF"`` and candidate does not declare it at all -- that
    combination is dropped from ``control`` so the two records can compare
    equal. Nothing else is normalized: not ``"FALSE"``/``"0"``/empty/any
    other CMake-false spelling, not ``ON``-vs-absent, and not any option
    whose owner is not in ``allowed_removed_modules`` for this call. Any
    other difference -- including an option unrelated to a removed module,
    or an owned option in the wrong direction (absent in control, OFF in
    candidate) -- is left in place so the caller's exact-equality check
    still fails on it.
    """
    control = dict(control)
    candidate = dict(candidate)

    for key, owner in _PA26_REMOVED_OPTION_OWNERS.items():
        if owner not in allowed_removed_modules:
            continue
        if control.get(key) == "OFF" and key not in candidate:
            control.pop(key)

    return control, candidate


def require_shared_build_inputs(
    delta: BuildStageDelta,
    *,
    allowed_removed_modules: frozenset[str] = frozenset(),
) -> None:
    """Fail closed unless both arms are a fair comparison: same resolved
    upstream revision, same generated inputs (inventory, promoted-winners --
    the same fixed winner corpus), the same BuildPlan in every respect
    except the composition-driven ``source_slice_id``, the same generated
    compile-inputs identity, and the same PA26-normalized effective
    configure record. ``source_slice_id``/``build_plan_id`` are deliberately
    NOT compared directly (composition is the variable under test); raw
    ``effective_build_id`` is recorded on each arm for provenance but is no
    longer required to match exactly -- ``effective_configure`` (after the
    narrow ``pa26_effective_configure_projection`` normalization) is the
    stronger, more relevant check GPT's design calls for instead."""
    if delta.control.resolved_revision != delta.candidate.resolved_revision:
        raise ReplayEquivalenceHardwareError(
            "control and candidate were materialized from different "
            f"resolved revisions: {delta.control.resolved_revision!r} vs "
            f"{delta.candidate.resolved_revision!r}"
        )
    control_inputs = dict(delta.control.input_hashes)
    candidate_inputs = dict(delta.candidate.input_hashes)
    if control_inputs != candidate_inputs:
        raise ReplayEquivalenceHardwareError(
            "control and candidate were built from different generated "
            f"inputs: {control_inputs} vs {candidate_inputs} -- both arms "
            "must replay the exact same fixed winner corpus"
        )
    if delta.control.build_plan_projection != delta.candidate.build_plan_projection:
        raise ReplayEquivalenceHardwareError(
            "control and candidate BuildPlans diverge on a field other than "
            "source_slice_id -- got "
            f"{dict(delta.control.build_plan_projection)} vs "
            f"{dict(delta.candidate.build_plan_projection)}"
        )
    # GPT design correction (req_579777e899454ed9): raw
    # generated_compile_inputs_hash equality is over-strong (it folds in
    # per-candidate source-byte digests that legitimately differ between
    # arms for an expected removed-module reason). Require instead: the
    # same compile-input filename set, byte-identical STRICT_GENERATED_FILES
    # content, and an identical manifest after stripping real per-run/per-arm
    # identity fields -- raw generated_compile_inputs_hash is still recorded
    # on each arm for provenance but no longer compared directly.
    if delta.control.generated_tree_compile_inputs != delta.candidate.generated_tree_compile_inputs:
        raise ReplayEquivalenceHardwareError(
            "control and candidate declare different generated compile-input "
            f"filename sets: {list(delta.control.generated_tree_compile_inputs)} "
            f"vs {list(delta.candidate.generated_tree_compile_inputs)}"
        )
    if delta.control.generated_strict_file_hashes != delta.candidate.generated_strict_file_hashes:
        raise ReplayEquivalenceHardwareError(
            "control and candidate compiled different STRICT_GENERATED_FILES "
            f"content: {dict(delta.control.generated_strict_file_hashes)} vs "
            f"{dict(delta.candidate.generated_strict_file_hashes)}"
        )
    if delta.control.generated_manifest_projection != delta.candidate.generated_manifest_projection:
        raise ReplayEquivalenceHardwareError(
            "control and candidate generated manifests diverge after "
            "stripping real per-run identity fields (generated_at, "
            "manifest_hash, build_descriptor, per-candidate "
            "implementation_digest/implementation_source_files)"
        )
    # Ordering (GPT design, req_13b71975cbe94c8d): raw effective_configure
    # -> validate+canonicalize known build-local path keys -> directional
    # OFF->absent removed-module normalization -> exact equality.
    control_configure = dict(delta.control.effective_configure)
    candidate_configure = dict(delta.candidate.effective_configure)
    _normalize_pa26_derived_path_keys(delta.control, control_configure)
    _normalize_pa26_derived_path_keys(delta.candidate, candidate_configure)
    projected_control, projected_candidate = pa26_effective_configure_projection(
        control_configure,
        candidate_configure,
        allowed_removed_modules=allowed_removed_modules,
    )
    if projected_control != projected_candidate:
        raise ReplayEquivalenceHardwareError(
            "control and candidate resolved different effective configure "
            f"state (after PA26 removed-module normalization): "
            f"{projected_control} vs {projected_candidate}"
        )


def build_hardware_specs(
    *,
    platform_name: str,
    architectures: tuple[str, ...],
    inventory_ref: Any,
    winners_ref: Any,
    build_name: str = "replay",
    binary_relative_path: str = "bin/llama-server",
    control_source: str = "bigcherry-native",
) -> tuple[CampaignLaneExecutionSpec, CampaignLaneExecutionSpec]:
    """The PRODUCTION pair -- exact control vs exact serving-core candidate.
    Shares every field except ``source_name``: the composition delta is the
    only thing PA26 varies here."""
    inputs = (("inventory", inventory_ref), ("promoted-winners", winners_ref))
    control_spec = CampaignLaneExecutionSpec(
        source_name=control_source,
        build_name=build_name,
        platform_name=platform_name,
        architectures=architectures,
        inputs=inputs,
        binary_relative_path=binary_relative_path,
    )
    candidate_spec = dataclasses.replace(
        control_spec, source_name=offline.SERVING_CORE_SOURCE_NAME
    )
    return control_spec, candidate_spec


#: GPT design correction (req_9cd5b140ca544ef8): the recipes.toml
#: ``[build.replay-diagnostic]`` profile turns on
#: ``GGML_HIP_DISPATCH_DIAGNOSTICS`` (0110's unrelated dispatch-diagnostics
#: instrumentation) as well as ``GGML_HIP_REPLAY_DIAGNOSTICS`` (0810's hit
#: observer). In the control arm, 0110 is present and turns that option into
#: real compile definitions/coverage sources; in the serving candidate, 0110
#: is absent, so the same cache variable cannot provide that implementation
#: -- a genuine observer asymmetry, not an OFF/absent artifact PA26's
#: narrow normalization is allowed to tolerate. PA26's diagnostic pair must
#: therefore use its OWN ephemeral build profile (never persisted to
#: ``config/recipes.toml`` -- same "ephemeral in-code override" pattern
#: ``replay_equivalence.py`` already uses for source/patch-set composition):
#: replay + 0810's hit observer ON, 0110's dispatch diagnostics explicitly
#: OFF on both arms so neither arm depends on 0110 for it.
PA26_DIAGNOSTIC_BUILD_NAME = "__pa26_replay_diagnostic"


def build_pa26_diagnostic_build_config(cfg: config.Config) -> config.Config:
    """Return an ephemeral copy of ``cfg`` carrying
    ``PA26_DIAGNOSTIC_BUILD_NAME``, cloned from ``[build.replay-diagnostic]``
    but with ``GGML_HIP_DISPATCH_DIAGNOSTICS`` forced ``OFF`` instead of
    ``ON`` (GPT design, see ``PA26_DIAGNOSTIC_BUILD_NAME``'s docstring
    above). Never persisted to ``config/recipes.toml``."""
    base = cfg.builds.get("replay-diagnostic")
    if base is None:
        raise ReplayEquivalenceHardwareError(
            "cfg carries no 'replay-diagnostic' build to derive PA26's "
            "ephemeral diagnostic build profile from"
        )
    ephemeral = dataclasses.replace(
        base,
        name=PA26_DIAGNOSTIC_BUILD_NAME,
        options=(
            ("GGML_HIP_DISPATCH_REPLAY", "ON"),
            ("GGML_HIP_REPLAY_DIAGNOSTICS", "ON"),
            ("GGML_HIP_DISPATCH_DIAGNOSTICS", "OFF"),
        ),
    )
    return dataclasses.replace(
        cfg, builds={**cfg.builds, PA26_DIAGNOSTIC_BUILD_NAME: ephemeral}
    )


def build_diagnostic_hardware_specs(
    *,
    platform_name: str,
    architectures: tuple[str, ...],
    inventory_ref: Any,
    winners_ref: Any,
    build_name: str = PA26_DIAGNOSTIC_BUILD_NAME,
    binary_relative_path: str = "bin/llama-server",
    control_source: str = "bigcherry-native",
) -> tuple[CampaignLaneExecutionSpec, CampaignLaneExecutionSpec]:
    """The DIAGNOSTIC companion pair -- same control, but the candidate is
    the serving-core composition PLUS 0810 (the hit-log observer), built
    with PA26's own ephemeral diagnostic build profile
    (``PA26_DIAGNOSTIC_BUILD_NAME`` -- see its docstring for why this is not
    simply ``[build.replay-diagnostic]``). Supplies the per-dispatch runtime
    data the production pair cannot. The caller's ``cfg`` must carry this
    build entry -- ``build_hardware_receipt`` ensures this via
    ``build_pa26_diagnostic_build_config`` before executing this pair."""
    inputs = (("inventory", inventory_ref), ("promoted-winners", winners_ref))
    control_spec = CampaignLaneExecutionSpec(
        source_name=control_source,
        build_name=build_name,
        platform_name=platform_name,
        architectures=architectures,
        inputs=inputs,
        binary_relative_path=binary_relative_path,
    )
    candidate_spec = dataclasses.replace(
        control_spec, source_name=offline.SERVING_CORE_DIAGNOSTIC_SOURCE_NAME
    )
    return control_spec, candidate_spec


def execute_two_arm_build(
    cfg: config.Config,
    catalog: list[patchset.PatchModule],
    *,
    context: ProjectContext,
    store: ArtifactStore,
    control_spec: CampaignLaneExecutionSpec,
    candidate_spec: CampaignLaneExecutionSpec,
    run_id_prefix: str,
    candidate_cfg_builder: Callable[[config.Config], config.Config] = (
        offline.build_serving_core_config
    ),
    lane_executor: LaneExecutor = execute_campaign_lane,
    allowed_removed_modules: frozenset[str] = frozenset(offline.EXPECTED_REMOVED_MODULES),
) -> tuple[BuildStageDelta, CampaignLaneResult, CampaignLaneResult]:
    """Re-proves composition (stage 1) before spending any build time, then
    materializes+builds both arms through the REAL production pipeline
    (``lane_executor``, defaulting to ``execute_campaign_lane`` -- never a
    second implementation), and binds/validates their build identities.

    ``candidate_cfg_builder`` defaults to the production serving-core
    builder; pass ``offline.build_serving_core_diagnostic_config`` to build
    the diagnostic companion pair instead. Composition proof for the
    diagnostic pair is intentionally weaker here (it is not
    ``EXPECTED_REMOVED_MODULES``-exact) -- callers building the diagnostic
    pair must separately call
    ``offline.require_diagnostic_matches_production`` against the
    production pair's resolved candidate identity.
    """
    candidate_cfg = candidate_cfg_builder(cfg)
    is_production_candidate = candidate_cfg_builder is offline.build_serving_core_config
    if is_production_candidate:
        delta = offline.resolve_composition_delta(
            cfg, catalog, control_source=control_spec.source_name,
            candidate_cfg=candidate_cfg,
        )
        offline.require_expected_composition_delta(delta)
    control_result = lane_executor(
        control_spec,
        cfg=cfg,
        context=context,
        store=store,
        run_id=f"{run_id_prefix}-control",
    )
    candidate_result = lane_executor(
        candidate_spec,
        cfg=candidate_cfg,
        context=context,
        store=store,
        run_id=f"{run_id_prefix}-candidate",
    )
    build_delta = BuildStageDelta(
        control=ArmBuildIdentity.from_result(control_result),
        candidate=ArmBuildIdentity.from_result(candidate_result),
    )
    require_shared_build_inputs(
        build_delta, allowed_removed_modules=allowed_removed_modules
    )
    return build_delta, control_result, candidate_result


@dataclasses.dataclass(frozen=True)
class ReplayExpectation:
    """One winner-corpus entry's STATIC, expected identity -- from the
    frozen cache itself (``tuning.replay.read_cache``), never from a live
    observation. GPT design correction (req_a6a387bdefbc474c): the real
    ``GGML_HIP_DISPATCH_HIT_LOG`` JSONL format does not carry
    ``config_binding``/``transform_id``/``match_kind``/``outcome`` at all --
    those belong on the corpus's own static expectation, not on a runtime
    observation."""

    dispatch: str
    signature: str
    winner: str
    transform_id: int
    match_kind: int
    manifest_hash: str


@dataclasses.dataclass(frozen=True)
class ArmRuntimeResult:
    """One arm's complete real runtime observation.

    ``entries`` reuses ``tuning.execution_audit.HitRecord`` directly (GPT
    design correction) rather than inventing a second per-dispatch type --
    it is exactly the shape ``execution_audit.load_hit_log`` already parses
    from a real ``GGML_HIP_DISPATCH_HIT_LOG`` JSONL: ``{dispatch,
    signature, candidate, from_cache, calls}``."""

    process_success: bool
    clean_shutdown: bool
    output_digest: str | None
    correctness_status: str
    model_hash: str
    gpu_identity: str
    #: Canonical digest of the resolved runtime args + environment.
    runtime_args_digest: str
    entries: tuple[HitRecord, ...]


#: Real signature: (built lane result, fixed corpus) -> this arm's full
#: runtime observation. The default is `real_hardware_runtime_runner`
#: below; tests inject a fake with this same shape.
RuntimeRunner = Callable[[CampaignLaneResult, offline.WinnersCorpus], ArmRuntimeResult]


def real_hardware_runtime_runner(
    result: CampaignLaneResult, corpus: offline.WinnersCorpus
) -> ArmRuntimeResult:
    """Real integration point -- NEVER invoked by any test in this repo.

    A real implementation SSHes to the built ``runtime_bundle``/``binary``
    ref's host, launches the server built by the DIAGNOSTIC pair (see
    HI130's replay-diagnostic build: ``GGML_HIP_DISPATCH_REPLAY=ON`` +
    ``GGML_HIP_REPLAY_DIAGNOSTICS=ON`` with
    ``GGML_HIP_DISPATCH_HIT_LOG=<path>``), replays every entry in
    ``corpus``, tears it down via a clean shutdown path (e.g.
    ``ServerRunner(..., shutdown_method="sigint")``, requiring
    ``ShutdownResult.clean``), and parses the resulting per-dispatch
    attribution JSONL into ``RuntimeEntryResult`` rows plus the arm-level
    ``ArmRuntimeResult`` fields -- not hand-parsed ad hoc stdout.

    Deliberately unimplemented: PA26's real hardware run happens on Brutus,
    outside this harness's own process, once it is free. Raises
    ``RuntimeNotEvaluated`` specifically (never a bare
    ``NotImplementedError`` that could mask an unrelated bug) so
    ``build_hardware_receipt`` can distinguish "genuinely not run yet" from
    a real failure.
    """
    raise RuntimeNotEvaluated(
        "real_hardware_runtime_runner requires real HIP hardware execution "
        "(e.g. SSH to Brutus) -- not available in this process. Supply a "
        "runner via the runtime_runner= parameter for testing."
    )


def make_real_hardware_runtime_runner(
    *,
    model_path: Path,
    devices: str,
    runtime_profile: Any,
    workdir: Path,
    winners_cache_path: Path,
) -> RuntimeRunner:
    """The real implementation of ``real_hardware_runtime_runner``'s own
    docstring, bound to a concrete model/devices/runtime-profile/workdir/
    winners-cache -- the bare two-argument ``RuntimeRunner`` shape has no
    room to carry that configuration, so it is supplied here via closure
    and the resulting callable is passed as ``runtime_runner=`` to
    ``build_hardware_receipt`` (exactly the mechanism the stub's own
    docstring names). This process is expected to run ON the real hardware
    host (e.g. invoked over SSH on Brutus, the same way ``bigcherry
    tune-campaign`` itself is invoked) -- it launches the server locally via
    ``ServerRunner``, exactly like ``tuning.workflow._stage_replay_validate``
    already does for the in-campaign behavioral gate.

    Per GPT's design correction (req_a6a387bdefbc474c): the lane result
    passed in here MUST be from the DIAGNOSTIC pair (candidate = serving-core
    + 0810_replay_hit_diagnostics, built with replay-diagnostic's
    diagnostics-enabled options) -- the production pair is diagnostics-free
    and cannot emit a hit log at all. ``build_hardware_receipt`` enforces
    this by building and passing the diagnostic pair's results to whichever
    runner is supplied.
    """
    from ..tuning import execution_audit
    from ..tuning import workflow as workflow_mod
    from ..tuning.server_runner import ServerError, ServerRunner
    import hashlib
    import json

    model_digest = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    env_unset = (
        "GGML_HIP_FORCE_CANDIDATE", "GGML_HIP_FORCE_CANDIDATE_STRICT",
        "GGML_HIP_DISPATCH_DB", "GGML_HIP_DISPATCH_CACHE",
        "GGML_HIP_DISPATCH_COVERAGE", "GGML_HIP_DISPATCH_HIT_LOG",
    )
    common_args = (
        "-ngl", "99", "-c", str(runtime_profile.production_context),
        *runtime_profile.server_args,
    )

    def _runner(result: CampaignLaneResult, corpus: offline.WinnersCorpus) -> ArmRuntimeResult:
        tag = f"{result.source_slice_id[:16]}-{result.build_plan.build_plan_id[:12]}"
        hit_log_path = Path(workdir) / f"pa26-hit-log-{tag}.jsonl"
        log_path = Path(workdir) / f"pa26-server-{tag}.log"
        if hit_log_path.exists():
            hit_log_path.unlink()
        env = {
            "HIP_VISIBLE_DEVICES": devices,
            "GGML_HIP_DISPATCH_MODE": "replay",
            "GGML_HIP_DISPATCH_CACHE": str(winners_cache_path),
            "GGML_HIP_REPLAY_DIAGNOSTICS": "1",
            "GGML_HIP_DISPATCH_HIT_LOG": str(hit_log_path),
        }
        runner = ServerRunner(
            binary=result.binary_ref.path, model=model_path, extra_args=common_args,
            env_overrides=env, env_unset=env_unset, log_path=log_path,
            shutdown_method="sigint",
        )
        process_success = True
        output_digest: str | None = None
        try:
            with runner:
                workflow_mod.run_tune_signature_workload(runner, runtime_profile)
                final = runner.run_completion(
                    "Explain how a compass works.", n_predict=64
                )
                # Digest the generated content only -- not the full response
                # (timings/other volatile fields would make two otherwise-
                # identical arms compare unequal for no real reason).
                output_digest = hashlib.sha256(
                    json.dumps(final.get("content", final), sort_keys=True).encode("utf-8")
                ).hexdigest()
        except ServerError:
            process_success = False
        shutdown = runner.last_shutdown
        clean_shutdown = bool(shutdown and shutdown.clean)
        hits = execution_audit.load_hit_log(hit_log_path if hit_log_path.is_file() else None)
        entries = tuple(sorted(hits.values(), key=lambda h: h.dispatch))
        return ArmRuntimeResult(
            process_success=process_success,
            clean_shutdown=clean_shutdown,
            output_digest=output_digest,
            correctness_status="pass" if process_success else "fail",
            model_hash=model_digest,
            gpu_identity=devices,
            runtime_args_digest=hashlib.sha256(
                repr(sorted(env.items())).encode("utf-8")
            ).hexdigest(),
            entries=entries,
        )

    return _runner


@dataclasses.dataclass(frozen=True)
class RuntimeComparison:
    equivalent: bool
    differences: tuple[dict[str, Any], ...]


_ARM_LEVEL_FIELDS = (
    "process_success",
    "clean_shutdown",
    "output_digest",
    "correctness_status",
    "model_hash",
    "gpu_identity",
    "runtime_args_digest",
)


def compare_runtime_results(
    control: ArmRuntimeResult,
    candidate: ArmRuntimeResult,
    *,
    expected: dict[str, ReplayExpectation],
) -> RuntimeComparison:
    """Arm-level exact equality, plus per-entry comparison against the
    frozen corpus's own static ``ReplayExpectation`` (GPT design correction,
    req_a6a387bdefbc474c):

    - Coverage is ``expected <= observed`` per arm, NOT exact-set equality
      -- extra/fallback dispatches appearing in the hit log are not a PA26
      failure (GPT: "extra/fallback dispatches in the hit log are NOT a
      PA26 failure"), only a MISSING expected dispatch is.
    - Per-entry comparison excludes ``recorded_calls`` (L1/L2 warm-cache
      bypass makes it a non-authoritative counter per
      ``execution_audit.py``'s own documented caveat) -- compares
      ``from_cache``, ``candidate == expected.winner``, and ``signature``
      only, for both arms against the SAME static expectation and against
      each other.
    """
    differences: list[dict[str, Any]] = []

    for field in _ARM_LEVEL_FIELDS:
        control_value = getattr(control, field)
        candidate_value = getattr(candidate, field)
        if control_value != candidate_value:
            differences.append(
                {
                    "scope": "arm",
                    "field": field,
                    "control": control_value,
                    "candidate": candidate_value,
                }
            )

    control_by_dispatch = {hit.dispatch: hit for hit in control.entries}
    candidate_by_dispatch = {hit.dispatch: hit for hit in candidate.entries}
    for label, observed in (
        ("control", control_by_dispatch),
        ("candidate", candidate_by_dispatch),
    ):
        missing = set(expected) - set(observed)
        if missing:
            differences.append(
                {"scope": "coverage", "arm": label, "reason": "missing", "dispatches": sorted(missing)}
            )

    for dispatch, exp in sorted(expected.items()):
        control_hit = control_by_dispatch.get(dispatch)
        candidate_hit = candidate_by_dispatch.get(dispatch)
        if control_hit is None or candidate_hit is None:
            continue  # already reported as a coverage "missing" difference
        for label, hit in (("control", control_hit), ("candidate", candidate_hit)):
            if not hit.from_cache or hit.candidate != exp.winner or hit.signature != exp.signature:
                differences.append(
                    {
                        "scope": "entry",
                        "dispatch": dispatch,
                        "arm": label,
                        "reason": "mismatch",
                        "expected": {"winner": exp.winner, "signature": exp.signature},
                        "observed": {
                            "from_cache": hit.from_cache,
                            "candidate": hit.candidate,
                            "signature": hit.signature,
                        },
                    }
                )
        if (
            control_hit.from_cache == candidate_hit.from_cache
            and control_hit.candidate == candidate_hit.candidate
            and control_hit.signature == candidate_hit.signature
        ):
            continue
        differences.append(
            {
                "scope": "entry",
                "dispatch": dispatch,
                "reason": "arm-divergence",
                "control": {
                    "from_cache": control_hit.from_cache,
                    "candidate": control_hit.candidate,
                    "signature": control_hit.signature,
                },
                "candidate": {
                    "from_cache": candidate_hit.from_cache,
                    "candidate": candidate_hit.candidate,
                    "signature": candidate_hit.signature,
                },
            }
        )

    return RuntimeComparison(
        equivalent=not differences, differences=tuple(differences)
    )


def build_hardware_receipt(
    cfg: config.Config,
    catalog: list[patchset.PatchModule],
    winners_cache_path: Path,
    *,
    bigcherry_revision: str,
    context: ProjectContext,
    store: ArtifactStore,
    control_spec: CampaignLaneExecutionSpec,
    candidate_spec: CampaignLaneExecutionSpec,
    run_id_prefix: str,
    diagnostic_control_spec: CampaignLaneExecutionSpec | None = None,
    diagnostic_candidate_spec: CampaignLaneExecutionSpec | None = None,
    control_source: str = "bigcherry-native",
    lane_executor: LaneExecutor = execute_campaign_lane,
    runtime_runner: RuntimeRunner = real_hardware_runtime_runner,
) -> dict[str, Any]:
    """The full staged PA26 receipt: composition/corpus preflight, build
    identity binding for the PRODUCTION pair (always built and reported --
    process/output/correctness equivalence is proven against it), and (when
    a real runner is supplied) a SEPARATE diagnostic-pair build + runtime
    execution + exact comparison.

    GPT's blocking correction (req_a6a387bdefbc474c): the production pair
    is intentionally diagnostics-free on both arms and CANNOT emit a hit
    log -- a non-default ``runtime_runner`` is therefore run against the
    DIAGNOSTIC pair (built here from ``diagnostic_control_spec``/
    ``diagnostic_candidate_spec``, which are REQUIRED whenever
    ``runtime_runner`` is not the default stub), whose candidate composition
    is validated (``offline.require_diagnostic_matches_production``) against
    the production candidate's resolved composition before any diagnostic
    result is trusted. The default stub is still exercised directly against
    the production pair (it never touches real hardware either way, so no
    diagnostic build is needed just to observe it raise
    ``RuntimeNotEvaluated``).

    Never fabricates a positive runtime result: only ``RuntimeNotEvaluated``
    (the default stub, or a real runner reporting a real but-currently-
    unreachable condition) is caught to leave ``runtime``/
    ``decision_equivalence`` honestly ``NOT_EVALUATED``. Any other exception
    (a parser bug, a process error after launch, a malformed log) propagates.
    """
    receipt = offline.build_offline_receipt(
        cfg,
        catalog,
        winners_cache_path,
        bigcherry_revision=bigcherry_revision,
        control_source=control_source,
    )
    build_delta, control_result, candidate_result = execute_two_arm_build(
        cfg,
        catalog,
        context=context,
        store=store,
        control_spec=control_spec,
        candidate_spec=candidate_spec,
        run_id_prefix=run_id_prefix,
        lane_executor=lane_executor,
    )
    receipt["build_identity"] = {
        "control": dataclasses.asdict(build_delta.control),
        "candidate": dataclasses.asdict(build_delta.candidate),
    }
    corpus = offline.load_winners_corpus(winners_cache_path)
    expected = {
        entry["dispatch"]: ReplayExpectation(
            dispatch=entry["dispatch"],
            signature=entry["signature"],
            winner=entry["winner"],
            transform_id=entry["transform_id"],
            match_kind=entry["match_kind"],
            manifest_hash=entry["manifest_hash"],
        )
        for entry in corpus.entries
    }

    is_default_runner = runtime_runner is real_hardware_runtime_runner
    if is_default_runner:
        runtime_control_result, runtime_candidate_result = control_result, candidate_result
    else:
        if diagnostic_control_spec is None or diagnostic_candidate_spec is None:
            raise ReplayEquivalenceHardwareError(
                "a non-default runtime_runner requires diagnostic_control_spec/"
                "diagnostic_candidate_spec -- the production pair is "
                "diagnostics-free and cannot emit a hit log"
            )
        if diagnostic_control_spec.build_name != PA26_DIAGNOSTIC_BUILD_NAME or (
            diagnostic_candidate_spec.build_name != PA26_DIAGNOSTIC_BUILD_NAME
        ):
            raise ReplayEquivalenceHardwareError(
                "diagnostic_control_spec/diagnostic_candidate_spec must both "
                f"use build_name={PA26_DIAGNOSTIC_BUILD_NAME!r} (PA26's own "
                "ephemeral diagnostic profile -- see build_pa26_diagnostic_"
                "build_config's docstring for why [build.replay-diagnostic] "
                f"itself is not fair here), got "
                f"{diagnostic_control_spec.build_name!r}/"
                f"{diagnostic_candidate_spec.build_name!r}"
            )
        diagnostic_build_cfg = build_pa26_diagnostic_build_config(cfg)
        diag_delta, diag_control_result, diag_candidate_result = execute_two_arm_build(
            diagnostic_build_cfg,
            catalog,
            context=context,
            store=store,
            control_spec=diagnostic_control_spec,
            candidate_spec=diagnostic_candidate_spec,
            run_id_prefix=f"{run_id_prefix}-diagnostic",
            candidate_cfg_builder=offline.build_serving_core_diagnostic_config,
            lane_executor=lane_executor,
            # The diagnostic candidate re-adds 0810 (DIAGNOSTIC_ADDBACK_MODULE)
            # -- only 0110's owned options may tolerate OFF->absent here;
            # GGML_HIP_REPLAY_DIAGNOSTICS must match identically between arms
            # since both diagnostic arms are expected to carry 0810.
            allowed_removed_modules=frozenset(offline.EXPECTED_REMOVED_MODULES)
            - {offline.DIAGNOSTIC_ADDBACK_MODULE},
        )
        diagnostic_cfg = offline.build_serving_core_diagnostic_config(cfg)
        diagnostic_selection = offline.resolution.resolve_canonical_selection(
            offline.SERVING_CORE_DIAGNOSTIC_SOURCE_NAME, diagnostic_cfg, catalog
        )
        production_delta = offline.resolve_composition_delta(
            cfg, catalog, control_source=control_source
        )
        offline.require_diagnostic_matches_production(
            production_delta.candidate.patch_ids,
            diagnostic_selection.identity.patch_ids,
        )
        receipt["diagnostic_build_identity"] = {
            "control": dataclasses.asdict(diag_delta.control),
            "candidate": dataclasses.asdict(diag_delta.candidate),
        }
        runtime_control_result, runtime_candidate_result = diag_control_result, diag_candidate_result

    try:
        control_runtime = runtime_runner(runtime_control_result, corpus)
        candidate_runtime = runtime_runner(runtime_candidate_result, corpus)
    except RuntimeNotEvaluated as exc:
        receipt["decision_equivalence"] = {
            "status": "NOT_EVALUATED",
            "differences": None,
            "reason": str(exc),
        }
        receipt["runtime"] = {"status": "NOT_EVALUATED", "reason": str(exc)}
        return receipt
    comparison = compare_runtime_results(
        control_runtime, candidate_runtime, expected=expected
    )
    receipt["decision_equivalence"] = {
        "status": "EVALUATED",
        "differences": list(comparison.differences),
    }
    receipt["runtime"] = {
        "status": "EVALUATED",
        "equivalent": comparison.equivalent,
    }
    return receipt
