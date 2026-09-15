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

from ..core import config
from ..patch import patchset
from . import resolution
from . import replay_equivalence as pa26

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
