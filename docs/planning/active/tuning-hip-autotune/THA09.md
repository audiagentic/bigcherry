---
id: THA09
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:49:14.603876+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P1
---

# Lazy native_select on the dispatch hot path (reopens HI87 under the sub-1% policy)

## Description

Run the residual guard-deferral experiment for lazy native_select on the HIP dispatch hot path. Preserve the already-completed zero-allocation memoized provider/counters; this item must not reimplement that mechanism.

## Steps

1. Reuse the completed memoized native provider and existing dispatch/L1/native/force counters; do not reimplement or replace them. 2. Add an explicit experiment-only mode that defers BOTH `!native.valid` guards (the guard in `ggml_hip_dispatch_try` and the guard in `ggml_hip_dispatch_resolve`) until after the signature/L1 path, while leaving default behavior unchanged. 3. Make the shadow validity check reachable and non-vacuous in that experiment mode: it must be able to observe disagreement rather than being structurally guaranteed by an earlier guard. 4. Preserve force-once semantics, native fallback, and fail-closed invalid handling; prove the L1-hit path never forces. 5. Compare control versus experiment with structural counters, graph/correctness checks, diagnostics-off E2E, and interleaved production-like runs. 6. Ship only if the explicit sub-1% policy passes; otherwise retain the mechanism as non-promoted experiment evidence.

## Detailed Solution & Technical Design

The completed provider/counter mechanism is the baseline and is not recreated here. The residual question is whether moving both validity guards under an explicit experiment selector can remove native_select work without changing behavior. Default mode keeps both guards and existing routing. Experiment mode makes the shadow path reachable, records both validity decisions and disagreement counts, and falls back safely on invalid native selection. No std::function or allocation is permitted on the hot path; counters prove reachability/mechanism but cannot justify promotion without E2E evidence.

## Code Samples & Guidance



## Files

Existing HIP native_select/dispatch-resolve provider and counters; experiment-mode guard selection; shadow-validity telemetry; hot-path/graph/correctness tests; diagnostics-off and interleaved Brutus E2E evidence.

## Validation

Prove both guard sites are exercised in experiment mode and that shadow validity can report both agreement and disagreement under a non-vacuous test fixture or real signature set. Verify default mode is byte/behavior unchanged, force-once and native fallback remain correct, and L1 hits do not force. Compare native_select calls, dispatch/L1 hits/misses, native calls, forces, resolver overhead, graph/correctness, diagnostics-off, and E2E results using interleaved controls. A counter-only reduction is insufficient; require the explicit sub-1% regression policy before promotion.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

The already-completed memoized provider and counters are reused, not reimplemented. Both `!native.valid` guards are named and moved only under an explicit experiment mode. The shadow validity path is reachable/non-vacuous and can detect disagreement. Default behavior remains unchanged; invalid/unsupported/force cases fail safely. Promotion requires correctness, graph, diagnostics-off E2E, and the sub-1% policy; no E2E benefit means no promotion.

## Notes

Supersedes: HI158
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi158

Supersedes: HI158
Inherited constraint: RV133 — do not reimplement completed mechanism; both validity guards must be named and the shadow counter must be reachable/non-vacuous.
Migration: capability-rebaseline-v3-2026-09

Supersedes HI158. Preserve RV133: the original shadow counter was vacuous because the resolver guard made L1 unreachable for invalid native selection, and the dispatch entry point has a second guard. This successor owns only the follow-up experiment; do not reimplement the completed mechanism.

## Change Log

- 2026-09-09T10:49:14.603876+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:05:16.500617+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.867999+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.389969+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:52:57.320952+00:00 (updated-by): Updated: section:description, section:steps, section:acceptance_criteria, section:notes
- chg_20260910_005948_legacy-planning-folders-now-co_1240
- 2026-09-10T00:59:49.044583+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:30:11.111314+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033046_repaired-four-more-tuning-succ_3978
- 2026-09-10T03:30:46.991974+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T04:01:50.125123+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_040226_fixed-the-four-remaining-seman_2499
- 2026-09-10T04:02:26.584992+00:00 (updated-by): Updated: section:ledger-events
