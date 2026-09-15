"""PA30: gates 1-3 real-source composition/diagnostic-overlay preflight.

PA26's ``replay_equivalence.py`` proves an EPHEMERAL candidate
("framework minus EXPECTED_REMOVED_MODULES", 8 modules including 0700)
against ``bigcherry-native`` -- historical machinery, deliberately left
untouched (see that module's own docstring; PA28 kept 0700 in the real
``serving-core`` patch-set, so PA26's ephemeral composition is NOT the same
as the real ``bigcherry-serving-base``/migrated ``bigcherry`` composition).

PA30 gates 1-3 need the REAL post-PA29 named sources, not an ephemeral
in-memory composition:

- Gate 1 (serving control equivalence): ``bigcherry-native`` vs
  ``bigcherry-serving-base``.
- Gate 2/3 (release/replay-winner equivalence against the final migrated
  source): reconstructed pre-cutover release (``framework`` +
  ``upstream-fixes`` + ``validated-enhancements``, which is exactly
  ``bigcherry-native`` while ``validated-enhancements`` is empty) vs the
  NOW-migrated ``source.bigcherry`` (``serving-core`` + ``upstream-fixes``
  + ``validated-enhancements``).

Per GPT design review (dev-gpt-agent session ses_c2892cdae7f14feb,
request req_c7cc2be3aca4415d): the expected removed-module delta for the
REAL comparison is exactly 7 modules -- 0700_coverage_counters must NOT be
in it (PA28 kept 0700 in serving-core). Do not mutate
``replay_equivalence.EXPECTED_REMOVED_MODULES`` (PA26's own 8-module
record is historically correct for what PA26 actually proved); this module
owns its own, distinct constant for the real post-cutover delta.

Both real sources (``bigcherry-native``, ``bigcherry-serving-base``,
``bigcherry``) already exist in ``config/recipes.toml`` (PA28/PA29) -- no
ephemeral source/patch-set injection is needed for the production pair.
The diagnostic companion pair still needs an ephemeral overlay (0810 is
not part of any real named source), constructed directly over the
resolved real ``bigcherry-serving-base`` composition rather than over
PA26's obsolete 8-removal composition.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from ..core import config
from ..core.artifacts import ArtifactStore
from ..core.context import ProjectContext
from ..patch import patchset
from . import resolution
from . import replay_equivalence as pa26
from . import replay_equivalence_hardware as pa26_hw
from .lane import CampaignLaneExecutionSpec, CampaignLaneResult

REAL_EXPECTED_REMOVED_MODULES: tuple[str, ...] = (
    "0110_campaign_tune_record_build",
    "0800_server_shutdown_endpoint",
    "0810_replay_hit_diagnostics",
    "0820_measurement_signature_shapes",
    "0830_split_reduce_telemetry",
    "0900_pool_workspace_metrics",
    "1100_hi70_direct_op_evidence",
)

DIAGNOSTIC_ADDBACK_MODULE = pa26.DIAGNOSTIC_ADDBACK_MODULE  # "0810_replay_hit_diagnostics"
SERVING_BASE_DIAGNOSTIC_SOURCE_NAME = "__pa30_serving_base_diagnostic"
SERVING_BASE_DIAGNOSTIC_PATCH_SET_NAME = "__pa30_serving_base_diagnostic_set"


class Pa30ReplayEquivalenceError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class RealCompositionDelta:
    control: resolution.SelectorIdentity
    candidate: resolution.SelectorIdentity
    removed: tuple[str, ...]
    added: tuple[str, ...]


def resolve_real_composition_delta(
    cfg: config.Config,
    catalog: list[patchset.PatchModule],
    *,
    control_source: str = "bigcherry-native",
    candidate_source: str = "bigcherry-serving-base",
) -> RealCompositionDelta:
    """Resolve two REAL named sources (no ephemeral cfg mutation) through
    the same normal resolution path production uses, and report the exact
    ordered-module delta between them."""
    control = resolution.resolve_canonical_selection(control_source, cfg, catalog)
    candidate = resolution.resolve_canonical_selection(candidate_source, cfg, catalog)
    control_ids = set(control.identity.patch_ids)
    candidate_ids = set(candidate.identity.patch_ids)
    removed = tuple(
        pid for pid in control.identity.patch_ids if pid not in candidate_ids
    )
    added = tuple(
        pid for pid in candidate.identity.patch_ids if pid not in control_ids
    )
    return RealCompositionDelta(
        control=control.identity,
        candidate=candidate.identity,
        removed=removed,
        added=added,
    )


def require_real_expected_composition_delta(
    delta: RealCompositionDelta,
    *,
    expected_removed: tuple[str, ...] = REAL_EXPECTED_REMOVED_MODULES,
) -> None:
    """Fail closed unless the candidate is EXACTLY control minus
    ``expected_removed`` (default: the real 7-module PA30 delta -- 0700
    excluded), in the SAME order, with the SAME content hashes for every
    surviving module. Mirrors
    ``replay_equivalence.require_expected_composition_delta`` but against
    real (not ephemeral) resolved identities and the real 7-module delta.
    """
    removed = frozenset(expected_removed)
    expected_ids = tuple(
        pid for pid in delta.control.patch_ids if pid not in removed
    )
    expected_hashes = tuple(
        (pid, digest)
        for pid, digest in delta.control.module_hashes
        if pid not in removed
    )
    if delta.candidate.patch_ids != expected_ids:
        raise Pa30ReplayEquivalenceError(
            "candidate patch_ids are not control minus the expected real "
            f"7-module delta: got {list(delta.candidate.patch_ids)}, want "
            f"{list(expected_ids)} (removed={sorted(removed)})"
        )
    if delta.candidate.module_hashes != expected_hashes:
        raise Pa30ReplayEquivalenceError(
            "candidate module_hashes are not control's minus the expected "
            "real 7-module delta -- a surviving module's content differs "
            "between control and candidate"
        )
    if frozenset(delta.removed) != removed:
        raise Pa30ReplayEquivalenceError(
            f"observed removed-module set {sorted(delta.removed)} does not "
            f"equal the expected real 7-module delta {sorted(removed)}"
        )
    if delta.added:
        raise Pa30ReplayEquivalenceError(
            f"candidate adds module(s) not present in control: {list(delta.added)}"
        )


def build_serving_base_diagnostic_config(
    cfg: config.Config,
    *,
    base_source: str = "bigcherry-serving-base",
) -> config.Config:
    """Return an ephemeral copy of ``cfg`` carrying a diagnostic companion
    candidate: the REAL ``base_source`` composition PLUS
    ``DIAGNOSTIC_ADDBACK_MODULE`` (0810) re-added as a PA30-only probe.
    Never persisted to ``config/recipes.toml``. Built directly from the
    real source's own declared patch-sets, not from PA26's obsolete
    8-removal composition."""
    if base_source not in cfg.sources:
        raise Pa30ReplayEquivalenceError(f"cfg carries no {base_source!r} source")
    base = cfg.sources[base_source]
    base_patch_ids: list[str] = []
    for set_name in base.patch_sets:
        base_patch_ids.extend(cfg.patch_sets[set_name].patches)
    if DIAGNOSTIC_ADDBACK_MODULE in base_patch_ids:
        raise Pa30ReplayEquivalenceError(
            f"{base_source!r} already carries {DIAGNOSTIC_ADDBACK_MODULE!r} "
            "-- diagnostic add-back would be a duplicate"
        )
    candidate_set = config.PatchSet(
        name=SERVING_BASE_DIAGNOSTIC_PATCH_SET_NAME,
        patches=(*base_patch_ids, DIAGNOSTIC_ADDBACK_MODULE),
        required_state=cfg.patch_sets[base.patch_sets[0]].required_state,
    )
    candidate_source = config.Source(
        name=SERVING_BASE_DIAGNOSTIC_SOURCE_NAME,
        ref=base.ref,
        overlay=base.overlay,
        patch_sets=(SERVING_BASE_DIAGNOSTIC_PATCH_SET_NAME,),
        backend=base.backend,
    )
    return dataclasses.replace(
        cfg,
        patch_sets={
            **cfg.patch_sets,
            SERVING_BASE_DIAGNOSTIC_PATCH_SET_NAME: candidate_set,
        },
        sources={**cfg.sources, SERVING_BASE_DIAGNOSTIC_SOURCE_NAME: candidate_source},
    )


def require_real_diagnostic_matches_production(
    production_candidate_ids: tuple[str, ...],
    diagnostic_candidate_ids: tuple[str, ...],
) -> None:
    """Fail closed unless the diagnostic candidate is EXACTLY the real
    production candidate (``bigcherry-serving-base`` as actually resolved)
    plus ``DIAGNOSTIC_ADDBACK_MODULE``, order-preserving except for the
    appended module."""
    expected = tuple(
        pid for pid in diagnostic_candidate_ids if pid != DIAGNOSTIC_ADDBACK_MODULE
    )
    if expected != production_candidate_ids:
        raise Pa30ReplayEquivalenceError(
            "diagnostic candidate composition is not the real production "
            "candidate plus the diagnostic add-back module: got "
            f"{list(expected)} (after removing {DIAGNOSTIC_ADDBACK_MODULE!r}), "
            f"want {list(production_candidate_ids)}"
        )
    if DIAGNOSTIC_ADDBACK_MODULE not in diagnostic_candidate_ids:
        raise Pa30ReplayEquivalenceError(
            f"diagnostic candidate does not carry {DIAGNOSTIC_ADDBACK_MODULE!r}"
        )


# ---------------------------------------------------------------------------
# Real-hardware receipt (gates 1-3).
#
# GPT design review (dev-gpt-agent session ses_c2892cdae7f14feb,
# req_92fb5a4fe596449e), following req_c7cc2be3aca4415d: reuse
# replay_equivalence_hardware.py's execute_two_arm_build/
# require_corpus_candidate_semantics/compare_runtime_results/
# make_real_hardware_runtime_runner UNCHANGED (never a second
# implementation of those safeguards), but:
#
# - candidate_cfg_builder=identity (real sources already exist in
#   config/recipes.toml -- no ephemeral cfg mutation for the production
#   pair), with THIS module performing resolve_real_composition_delta() /
#   require_real_expected_composition_delta() itself immediately before
#   calling execute_two_arm_build (execute_two_arm_build's own internal
#   PA26 composition check is gated on `candidate_cfg_builder is
#   offline.build_serving_core_config`, which is false for identity, so it
#   is correctly skipped rather than silently wrong).
# - allowed_removed_modules is OVERRIDDEN to the real PA30 7-module delta
#   for the production pair (REAL_EXPECTED_REMOVED_MODULES) and the real
#   6-module delta (7 minus 0810) for the diagnostic pair -- GPT: reusing
#   PA26's 8-module allowance (which still includes 0700) would be wrong
#   for the real-source comparison, since PA28 kept 0700 in serving-core.
def require_pre_cutover_release_identity(
    cfg: config.Config,
    catalog: list[patchset.PatchModule],
    *,
    native_source: str = "bigcherry-native",
) -> resolution.SelectorIdentity:
    """Gate 2's control must be the EXPLICIT pre-cutover release identity
    (``framework`` + ``upstream-fixes`` + ``validated-enhancements``), not
    an unproven assumption that ``bigcherry-native`` (``framework`` +
    ``upstream-fixes`` only) stands in for it. Resolves that explicit
    reconstruction as an ephemeral source, asserts it against
    ``native_source``'s own real resolution, and returns the (now-proven)
    real identity to use as gate 2's control -- so provenance never rests
    on "validated-enhancements currently happens to be empty" as an
    implicit assumption (GPT design review, req_92fb5a4fe596449e)."""
    if native_source not in cfg.sources:
        raise Pa30ReplayEquivalenceError(f"cfg carries no {native_source!r} source")
    native = cfg.sources[native_source]
    reconstructed_set = config.PatchSet(
        name="__pa30_pre_cutover_release_set",
        patches=(
            *cfg.patch_sets["framework"].patches,
            *cfg.patch_sets["upstream-fixes"].patches,
            *cfg.patch_sets["validated-enhancements"].patches,
        ),
        required_state=cfg.patch_sets["framework"].required_state,
    )
    reconstructed_source = config.Source(
        name="__pa30_pre_cutover_release",
        ref=native.ref,
        overlay=native.overlay,
        patch_sets=("__pa30_pre_cutover_release_set",),
        backend=native.backend,
    )
    ephemeral_cfg = dataclasses.replace(
        cfg,
        patch_sets={
            **cfg.patch_sets,
            "__pa30_pre_cutover_release_set": reconstructed_set,
        },
        sources={**cfg.sources, "__pa30_pre_cutover_release": reconstructed_source},
    )
    reconstructed = resolution.resolve_canonical_selection(
        "__pa30_pre_cutover_release", ephemeral_cfg, catalog
    )
    native_resolved = resolution.resolve_canonical_selection(
        native_source, cfg, catalog
    )
    if reconstructed.identity.patch_ids != native_resolved.identity.patch_ids:
        raise Pa30ReplayEquivalenceError(
            "explicit pre-cutover release reconstruction (framework + "
            "upstream-fixes + validated-enhancements) does NOT equal "
            f"{native_source!r}'s own resolution -- got "
            f"{list(reconstructed.identity.patch_ids)} vs "
            f"{list(native_resolved.identity.patch_ids)}; gate 2 must use "
            "the explicit reconstruction as its control, not "
            f"{native_source!r}"
        )
    if reconstructed.identity.module_hashes != native_resolved.identity.module_hashes:
        raise Pa30ReplayEquivalenceError(
            "explicit pre-cutover release reconstruction matches "
            f"{native_source!r} on patch_ids but not module_hashes -- "
            "content divergence, not merely membership"
        )
    return native_resolved.identity


def build_real_hardware_specs(
    *,
    platform_name: str,
    architectures: tuple[str, ...],
    inventory_ref: Any,
    winners_ref: Any,
    build_name: str = "replay",
    binary_relative_path: str = "bin/llama-server",
    control_source: str = "bigcherry-native",
    candidate_source: str = "bigcherry-serving-base",
) -> tuple[CampaignLaneExecutionSpec, CampaignLaneExecutionSpec]:
    """The real-source PRODUCTION pair -- exact ``control_source`` vs exact
    ``candidate_source``, both real named sources already in
    config/recipes.toml. Mirrors
    ``replay_equivalence_hardware.build_hardware_specs`` but never resolves
    ``offline.SERVING_CORE_SOURCE_NAME``."""
    inputs = (("inventory", inventory_ref), ("promoted-winners", winners_ref))
    control_spec = CampaignLaneExecutionSpec(
        source_name=control_source,
        build_name=build_name,
        platform_name=platform_name,
        architectures=architectures,
        inputs=inputs,
        binary_relative_path=binary_relative_path,
    )
    candidate_spec = dataclasses.replace(control_spec, source_name=candidate_source)
    return control_spec, candidate_spec


def build_real_diagnostic_hardware_specs(
    *,
    platform_name: str,
    architectures: tuple[str, ...],
    inventory_ref: Any,
    winners_ref: Any,
    build_name: str = pa26_hw.PA26_DIAGNOSTIC_BUILD_NAME,
    binary_relative_path: str = "bin/llama-server",
    control_source: str = "bigcherry-native",
) -> tuple[CampaignLaneExecutionSpec, CampaignLaneExecutionSpec]:
    """The real-source DIAGNOSTIC companion pair -- same control, candidate
    is ``SERVING_BASE_DIAGNOSTIC_SOURCE_NAME`` (real ``bigcherry-serving-
    base`` + 0810). Reuses PA26's own ephemeral diagnostic build profile
    name so ``build_pa26_diagnostic_build_config`` (unchanged) still applies."""
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
        control_spec, source_name=SERVING_BASE_DIAGNOSTIC_SOURCE_NAME
    )
    return control_spec, candidate_spec


def build_real_hardware_receipt(
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
    diagnostic_control_spec: CampaignLaneExecutionSpec,
    diagnostic_candidate_spec: CampaignLaneExecutionSpec,
    corpus_producer_manifest_path: Path,
    control_source: str = "bigcherry-native",
    candidate_source: str = "bigcherry-serving-base",
    lane_executor: pa26_hw.LaneExecutor = pa26_hw.execute_campaign_lane,
    runtime_runner: pa26_hw.RuntimeRunner = pa26_hw.real_hardware_runtime_runner,
) -> dict[str, Any]:
    """PA30 gates 1-3 real-hardware receipt -- mirrors
    ``replay_equivalence_hardware.build_hardware_receipt`` exactly in
    structure and safeguards (corpus-producer semantic compatibility before
    REVISION_MATCH=0, diagnostic-vs-production composition binding, exact
    arm-level + per-dispatch comparison), but resolved against the REAL
    named sources instead of PA26's ephemeral composition, with
    ``allowed_removed_modules`` overridden to PA30's real 7-/6-module delta
    (GPT design review req_92fb5a4fe596449e)."""
    delta = resolve_real_composition_delta(
        cfg, catalog, control_source=control_source, candidate_source=candidate_source,
    )
    require_real_expected_composition_delta(delta)
    corpus = pa26.load_winners_corpus(winners_cache_path)
    receipt: dict[str, Any] = {
        "schema_version": pa26.RECEIPT_SCHEMA_VERSION,
        "bigcherry_revision": bigcherry_revision,
        "control_selector": delta.control.to_payload(),
        "candidate_selector": delta.candidate.to_payload(),
        "expected_composition_delta": list(REAL_EXPECTED_REMOVED_MODULES),
        "actual_composition_delta": list(delta.removed),
        "winners_sha256": corpus.sha256,
        "winners_header": corpus.header,
    }

    identity_cfg_builder = lambda c: c  # noqa: E731 -- real sources need no mutation
    build_delta, control_result, candidate_result = pa26_hw.execute_two_arm_build(
        cfg,
        catalog,
        context=context,
        store=store,
        control_spec=control_spec,
        candidate_spec=candidate_spec,
        run_id_prefix=run_id_prefix,
        candidate_cfg_builder=identity_cfg_builder,
        lane_executor=lane_executor,
        allowed_removed_modules=frozenset(REAL_EXPECTED_REMOVED_MODULES),
    )
    receipt["build_identity"] = {
        "control": dataclasses.asdict(build_delta.control),
        "candidate": dataclasses.asdict(build_delta.candidate),
    }

    expected = {
        entry["dispatch"]: pa26_hw.ReplayExpectation(
            dispatch=entry["dispatch"],
            signature=entry["signature"],
            winner=entry["winner"],
            transform_id=entry["transform_id"],
            match_kind=entry["match_kind"],
            manifest_hash=entry["manifest_hash"],
        )
        for entry in corpus.entries
    }

    is_default_runner = runtime_runner is pa26_hw.real_hardware_runtime_runner
    if is_default_runner:
        receipt["decision_equivalence"] = {"status": "NOT_EVALUATED", "differences": None,
                                            "reason": "default stub runtime_runner"}
        receipt["runtime"] = {"status": "NOT_EVALUATED", "reason": "default stub runtime_runner"}
        return receipt

    if diagnostic_control_spec.build_name != pa26_hw.PA26_DIAGNOSTIC_BUILD_NAME or (
        diagnostic_candidate_spec.build_name != pa26_hw.PA26_DIAGNOSTIC_BUILD_NAME
    ):
        raise Pa30ReplayEquivalenceError(
            "diagnostic specs must both use build_name="
            f"{pa26_hw.PA26_DIAGNOSTIC_BUILD_NAME!r}"
        )
    diagnostic_build_cfg = pa26_hw.build_pa26_diagnostic_build_config(cfg)
    diag_delta, diag_control_result, diag_candidate_result = pa26_hw.execute_two_arm_build(
        diagnostic_build_cfg,
        catalog,
        context=context,
        store=store,
        control_spec=diagnostic_control_spec,
        candidate_spec=diagnostic_candidate_spec,
        run_id_prefix=f"{run_id_prefix}-diagnostic",
        candidate_cfg_builder=build_serving_base_diagnostic_config,
        lane_executor=lane_executor,
        allowed_removed_modules=frozenset(REAL_EXPECTED_REMOVED_MODULES)
        - {DIAGNOSTIC_ADDBACK_MODULE},
    )
    diagnostic_cfg = build_serving_base_diagnostic_config(cfg)
    diagnostic_selection = resolution.resolve_canonical_selection(
        SERVING_BASE_DIAGNOSTIC_SOURCE_NAME, diagnostic_cfg, catalog
    )
    require_real_diagnostic_matches_production(
        delta.candidate.patch_ids, diagnostic_selection.identity.patch_ids,
    )
    receipt["diagnostic_build_identity"] = {
        "control": dataclasses.asdict(diag_delta.control),
        "candidate": dataclasses.asdict(diag_delta.candidate),
    }

    producer_manifest = pa26_hw._read_json_ref(corpus_producer_manifest_path)
    if producer_manifest is None:
        raise Pa30ReplayEquivalenceError(
            f"cannot read corpus_producer_manifest_path {corpus_producer_manifest_path!r}"
        )
    runtime_architecture = diagnostic_control_spec.architectures[0]
    if len(diagnostic_control_spec.architectures) != 1 or (
        diagnostic_candidate_spec.architectures != diagnostic_control_spec.architectures
    ):
        raise Pa30ReplayEquivalenceError(
            "both diagnostic arms must be built for exactly one shared "
            f"runtime architecture -- got control="
            f"{diagnostic_control_spec.architectures!r} candidate="
            f"{diagnostic_candidate_spec.architectures!r}"
        )
    for label, lane_result in (
        ("control", diag_control_result), ("candidate", diag_candidate_result),
    ):
        target_manifest = (
            pa26_hw._read_json_ref(Path(lane_result.manifest_ref.path))
            if lane_result.manifest_ref else None
        )
        if target_manifest is None:
            raise Pa30ReplayEquivalenceError(
                f"{label}: diagnostic arm produced no readable manifest_ref"
            )
        pa26_hw.require_corpus_candidate_semantics(
            corpus,
            producer_manifest=producer_manifest,
            target_manifest=target_manifest,
            target_label=label,
            runtime_architecture=runtime_architecture,
        )

    try:
        control_runtime = runtime_runner(diag_control_result, corpus)
        candidate_runtime = runtime_runner(diag_candidate_result, corpus)
    except pa26_hw.RuntimeNotEvaluated as exc:
        receipt["decision_equivalence"] = {
            "status": "NOT_EVALUATED", "differences": None, "reason": str(exc),
        }
        receipt["runtime"] = {"status": "NOT_EVALUATED", "reason": str(exc)}
        return receipt
    comparison = pa26_hw.compare_runtime_results(
        control_runtime, candidate_runtime, expected=expected
    )
    receipt["decision_equivalence"] = {
        "status": "EVALUATED", "differences": list(comparison.differences),
    }
    receipt["runtime"] = {"status": "EVALUATED", "equivalent": comparison.equivalent}
    return receipt
