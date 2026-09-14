"""PA26: prove serving-core replay equivalence against the current aggregate
framework, for a fixed promoted-winner corpus (evidence gate before PA28).

Design per dev-gpt-agent review (session ses_c2892cdae7f14feb,
req_accf444d1eaa4537), following PA27's mechanical split of
0100_cmake_options into serving (0100) and campaign tune/record/coverage
build (0110) halves:

- The candidate composition is NOT built via ``extra_patch_ids``/overlay
  (those are additive and cannot express "framework minus campaign/
  qualification modules"). It is a deliberately EPHEMERAL patch-set/source
  pair, constructed in-memory only -- never written to
  ``config/recipes.toml``. PA28, not PA26, owns creating a real public
  serving-core taxonomy; PA26 only proves whether removing the eight
  non-serving modules below is behaviourally inert for replay.
- ``EXPECTED_REMOVED_MODULES`` is the PA20 serving boundary. Both arms are
  resolved through the SAME normal ``campaign.resolution`` path used by
  production, and the exact ordered-module delta between them must equal
  this set exactly -- any drift (recipes.toml gaining/losing a framework
  module) fails this module closed rather than silently comparing the wrong
  compositions.
- This module is generic and reusable (PA30 is a declared future consumer),
  not a second replay implementation: it calls ``tuning.replay.read_cache``
  and production resolution APIs only.

This module supplies composition/corpus PREFLIGHT for PA26 -- not an
equivalence proof. It exactly proves the candidate composition (membership,
order, and content hash) and binds a fixed winner corpus; it makes no
runtime/dispatch/build claim at all (``build_offline_receipt``'s
``decision_equivalence`` and ``runtime`` are always ``NOT_EVALUATED``). PA26
cannot close on this alone: a real two-arm HIP build+replay run on real
hardware, comparing actual resolved dispatch/signature/winner/output per
``docs/planning/active/patching-patch-system/PA26.md``, is still required.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

from ..core import config
from ..patch import patchset
from . import resolution


class ReplayEquivalenceError(ValueError):
    pass


RECEIPT_SCHEMA_VERSION = 1

# PA20 serving boundary (GPT design review): the current `framework`
# patch-set minus these eight modules is PA26's serving-core candidate.
# 0300-0650 (the actual dispatch kernel-variant candidates the tuner/replay
# choose between) are deliberately NOT in this list -- they are identical in
# both arms by construction; PA26 exists to prove that removing tuning/
# record/coverage/diagnostic plumbing around them is inert for replay.
EXPECTED_REMOVED_MODULES: tuple[str, ...] = (
    "0110_campaign_tune_record_build",
    "0700_coverage_counters",
    "0800_server_shutdown_endpoint",
    "0810_replay_hit_diagnostics",
    "0820_measurement_signature_shapes",
    "0830_split_reduce_telemetry",
    "0900_pool_workspace_metrics",
    "1100_hi70_direct_op_evidence",
)

SERVING_CORE_SOURCE_NAME = "__pa26_serving_core"
SERVING_CORE_PATCH_SET_NAME = "__pa26_serving_core_set"

# GPT hardware-harness review (req_a4974e46d4d643e0): the production replay
# build is intentionally diagnostics-free on BOTH arms (the candidate never
# has 0810 at all; the control build itself carries no
# GGML_HIP_REPLAY_DIAGNOSTICS instrumentation either), so neither arm can
# emit a per-dispatch hit log for runtime comparison. A separate DIAGNOSTIC
# companion pair re-adds 0810 to the candidate ONLY as an ephemeral PA26
# probe (never persisted, exactly like the production candidate) so both
# diagnostic arms share the same observer; it must never substitute for the
# production pair's own process/output/correctness comparison.
DIAGNOSTIC_ADDBACK_MODULE = "0810_replay_hit_diagnostics"
SERVING_CORE_DIAGNOSTIC_SOURCE_NAME = "__pa26_serving_core_diagnostic"
SERVING_CORE_DIAGNOSTIC_PATCH_SET_NAME = "__pa26_serving_core_diagnostic_set"


def build_serving_core_config(cfg: config.Config) -> config.Config:
    """Return an ephemeral copy of ``cfg`` carrying the ad-hoc PA26 candidate
    source/patch-set. Never persisted to ``config/recipes.toml``."""
    if "framework" not in cfg.patch_sets:
        raise ReplayEquivalenceError("cfg carries no 'framework' patch-set")
    if "bigcherry-native" not in cfg.sources:
        raise ReplayEquivalenceError("cfg carries no 'bigcherry-native' source")
    framework = cfg.patch_sets["framework"]
    missing_expected = set(EXPECTED_REMOVED_MODULES) - set(framework.patches)
    if missing_expected:
        raise ReplayEquivalenceError(
            "config/recipes.toml 'framework' patch-set no longer declares "
            f"expected-removed module(s) {sorted(missing_expected)} -- PA26's "
            "hardcoded EXPECTED_REMOVED_MODULES is stale and must be updated"
        )
    candidate_ids = tuple(
        pid for pid in framework.patches if pid not in EXPECTED_REMOVED_MODULES
    )
    candidate_set = config.PatchSet(
        name=SERVING_CORE_PATCH_SET_NAME,
        patches=candidate_ids,
        required_state=framework.required_state,
    )
    native = cfg.sources["bigcherry-native"]
    candidate_source = config.Source(
        name=SERVING_CORE_SOURCE_NAME,
        ref=native.ref,
        overlay=native.overlay,
        patch_sets=(SERVING_CORE_PATCH_SET_NAME, "upstream-fixes"),
        backend=native.backend,
    )
    return dataclasses.replace(
        cfg,
        patch_sets={**cfg.patch_sets, SERVING_CORE_PATCH_SET_NAME: candidate_set},
        sources={**cfg.sources, SERVING_CORE_SOURCE_NAME: candidate_source},
    )


def build_serving_core_diagnostic_config(cfg: config.Config) -> config.Config:
    """Return an ephemeral copy of ``cfg`` carrying the DIAGNOSTIC companion
    candidate: the serving-core composition PLUS ``DIAGNOSTIC_ADDBACK_MODULE``
    (0810) re-added as a PA26-only probe. Never persisted to
    ``config/recipes.toml``. Distinct source/patch-set names from the
    production candidate -- the two are never resolved together."""
    if "framework" not in cfg.patch_sets:
        raise ReplayEquivalenceError("cfg carries no 'framework' patch-set")
    if "bigcherry-native" not in cfg.sources:
        raise ReplayEquivalenceError("cfg carries no 'bigcherry-native' source")
    framework = cfg.patch_sets["framework"]
    removed_except_diagnostic = frozenset(EXPECTED_REMOVED_MODULES) - {
        DIAGNOSTIC_ADDBACK_MODULE
    }
    missing_expected = removed_except_diagnostic - set(framework.patches)
    if missing_expected or DIAGNOSTIC_ADDBACK_MODULE not in framework.patches:
        raise ReplayEquivalenceError(
            "config/recipes.toml 'framework' patch-set no longer declares "
            "the modules PA26's diagnostic companion depends on -- stale "
            "hardcoded module list"
        )
    candidate_ids = tuple(
        pid for pid in framework.patches if pid not in removed_except_diagnostic
    )
    candidate_set = config.PatchSet(
        name=SERVING_CORE_DIAGNOSTIC_PATCH_SET_NAME,
        patches=candidate_ids,
        required_state=framework.required_state,
    )
    native = cfg.sources["bigcherry-native"]
    candidate_source = config.Source(
        name=SERVING_CORE_DIAGNOSTIC_SOURCE_NAME,
        ref=native.ref,
        overlay=native.overlay,
        patch_sets=(SERVING_CORE_DIAGNOSTIC_PATCH_SET_NAME, "upstream-fixes"),
        backend=native.backend,
    )
    return dataclasses.replace(
        cfg,
        patch_sets={
            **cfg.patch_sets,
            SERVING_CORE_DIAGNOSTIC_PATCH_SET_NAME: candidate_set,
        },
        sources={**cfg.sources, SERVING_CORE_DIAGNOSTIC_SOURCE_NAME: candidate_source},
    )


def require_diagnostic_matches_production(
    production_candidate_ids: tuple[str, ...],
    diagnostic_candidate_ids: tuple[str, ...],
) -> None:
    """Fail closed unless the diagnostic candidate is EXACTLY the production
    candidate plus ``DIAGNOSTIC_ADDBACK_MODULE`` -- explicitly binds the
    diagnostic pair's runtime observations back to the production
    composition they are meant to stand in for."""
    expected = tuple(
        pid for pid in diagnostic_candidate_ids if pid != DIAGNOSTIC_ADDBACK_MODULE
    )
    if expected != production_candidate_ids:
        raise ReplayEquivalenceError(
            "diagnostic candidate composition is not the production "
            "candidate plus the diagnostic add-back module: got "
            f"{list(expected)} (after removing {DIAGNOSTIC_ADDBACK_MODULE!r}), "
            f"want {list(production_candidate_ids)}"
        )
    if DIAGNOSTIC_ADDBACK_MODULE not in diagnostic_candidate_ids:
        raise ReplayEquivalenceError(
            f"diagnostic candidate does not carry {DIAGNOSTIC_ADDBACK_MODULE!r}"
        )


@dataclasses.dataclass(frozen=True)
class CompositionDelta:
    control: resolution.SelectorIdentity
    candidate: resolution.SelectorIdentity
    removed: tuple[str, ...]
    added: tuple[str, ...]


def resolve_composition_delta(
    cfg: config.Config,
    catalog: list[patchset.PatchModule],
    *,
    control_source: str = "bigcherry-native",
    candidate_cfg: config.Config | None = None,
) -> CompositionDelta:
    """Resolve control (current aggregate framework) and the PA26 serving-core
    candidate through the SAME normal resolution path, and report the exact
    ordered-module delta between them.

    ``candidate_cfg`` is exposed only so negative-fixture tests can supply a
    deliberately perturbed candidate config; real callers must omit it.
    """
    control = resolution.resolve_canonical_selection(control_source, cfg, catalog)
    resolved_candidate_cfg = candidate_cfg or build_serving_core_config(cfg)
    candidate = resolution.resolve_canonical_selection(
        SERVING_CORE_SOURCE_NAME, resolved_candidate_cfg, catalog
    )
    control_ids = set(control.identity.patch_ids)
    candidate_ids = set(candidate.identity.patch_ids)
    removed = tuple(
        pid for pid in control.identity.patch_ids if pid not in candidate_ids
    )
    added = tuple(
        pid for pid in candidate.identity.patch_ids if pid not in control_ids
    )
    return CompositionDelta(
        control=control.identity,
        candidate=candidate.identity,
        removed=removed,
        added=added,
    )


def require_expected_composition_delta(delta: CompositionDelta) -> None:
    """Fail closed unless the candidate is EXACTLY control minus
    ``EXPECTED_REMOVED_MODULES``, in the SAME order, with the SAME content
    hashes for every surviving module.

    GPT review (req_19d7a4cde3e84bd8) found the original membership+shared-
    hash check did not actually catch a reordered candidate --
    ``SelectorIdentity.__post_init__`` validates internal alignment within
    one identity, not order between two. Comparing the projected
    ``(patch_ids, module_hashes)`` exactly (control with the expected-removed
    modules filtered out) checks membership, order, and content identity in
    one exact-equality assertion.
    """
    if delta.control.source_ref != delta.candidate.source_ref:
        raise ReplayEquivalenceError(
            "control and candidate resolve different source refs: "
            f"{delta.control.source_ref!r} vs {delta.candidate.source_ref!r} "
            "-- PA26 requires both arms from the same resolved source/ref"
        )
    removed = frozenset(EXPECTED_REMOVED_MODULES)
    expected_ids = tuple(
        pid for pid in delta.control.patch_ids if pid not in removed
    )
    expected_hashes = tuple(
        entry for entry in delta.control.module_hashes if entry[0] not in removed
    )
    if delta.candidate.patch_ids != expected_ids:
        raise ReplayEquivalenceError(
            "serving-core candidate's module order/membership does not equal "
            f"control minus the expected PA20 serving boundary: got "
            f"{list(delta.candidate.patch_ids)}, want {list(expected_ids)}"
        )
    if delta.candidate.module_hashes != expected_hashes:
        raise ReplayEquivalenceError(
            "serving-core candidate's module content hashes diverge from "
            "control's for one or more surviving (shared) modules"
        )


@dataclasses.dataclass(frozen=True)
class WinnersCorpus:
    sha256: str
    header: dict[str, Any]
    entries: tuple[dict[str, Any], ...]


def load_winners_corpus(cache_path: Path) -> WinnersCorpus:
    """Load PA26's fixed winner corpus.

    Always current-schema (v5), always ``enforce_schema=True`` --
    ``replay.py`` explicitly documents that ``enforce_schema=False`` is a
    read-only historical-inspection escape hatch, never valid on a
    production/merge/acceptance path (GPT design review). A historical
    schema-1 cache (e.g. the legacy ``dispatch-27b-v5.cache`` reference
    artifact) must be re-exported via ``replay.build()`` under current
    promotion/correctness gates before use here, not read with the escape
    hatch.
    """
    from ..tuning import replay as replay_module

    blob = Path(cache_path).read_bytes()
    sha256 = hashlib.sha256(blob).hexdigest()
    header, entries = replay_module.read_cache(blob, enforce_schema=True)
    if not entries:
        raise ReplayEquivalenceError(
            f"winners corpus {cache_path} contains no entries"
        )
    return WinnersCorpus(sha256=sha256, header=header, entries=tuple(entries))


def build_offline_receipt(
    cfg: config.Config,
    catalog: list[patchset.PatchModule],
    winners_cache_path: Path,
    *,
    bigcherry_revision: str,
    control_source: str = "bigcherry-native",
) -> dict[str, Any]:
    """PA26's offline (no-GPU) receipt: binds control/candidate identities,
    proves the exact composition delta, and binds the fixed winner corpus --
    but makes no claim about actual dispatch/build/runtime equivalence.

    A real hardware run (real two-arm HIP build, real replay binary, real
    resolved dispatch/signature/winner/output comparison) is still required
    before PA26 can close -- see
    ``docs/planning/active/patching-patch-system/PA26.md``.
    """
    delta = resolve_composition_delta(cfg, catalog, control_source=control_source)
    require_expected_composition_delta(delta)
    corpus = load_winners_corpus(winners_cache_path)
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "bigcherry_revision": bigcherry_revision,
        "control_selector": delta.control.to_payload(),
        "candidate_selector": delta.candidate.to_payload(),
        "expected_composition_delta": list(EXPECTED_REMOVED_MODULES),
        "actual_composition_delta": list(delta.removed),
        "winners_sha256": corpus.sha256,
        "winners_header": corpus.header,
        # v5 caches can retain multiple generations/provenance identities
        # (select_newest_winners(), build()) -- this counts wire entries in
        # the frozen corpus, not "winners" in any narrower sense.
        "cache_entry_count": len(corpus.entries),
        # No replay decision, generated state, build, or runtime output has
        # been compared yet (GPT review req_19d7a4cde3e84bd8): this receipt
        # is composition/corpus preflight only, not an equivalence proof.
        "decision_equivalence": {"status": "NOT_EVALUATED", "differences": None},
        "runtime": {"status": "NOT_EVALUATED"},
    }
