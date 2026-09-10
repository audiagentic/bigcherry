---
id: THA11
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:49:22.968400+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: M
priority: P0
---

# Replay benchmarks must prove tuned kernels actually launched, not merely that a cache loaded

## Description

Prove that replayed tuned kernels actually launched after final candidate validation; cache load or exact-hit counts alone are insufficient. Existing runs are uninterpretable without graceful teardown and launch-point evidence.

## Steps

1. Add final_tuned_launches at the real executor/launch point after can_execute, blacklist, transform, and fallback revalidation.
2. Emit per-winner launch counts and fallback_after_exact.
3. Persist cache_entries_loaded, eligible_dispatches, exact_replay_resolutions, final_tuned_bindings, final_tuned_launches, fallback_after_exact, miss/unavailable/incompatible counts, and per-winner launches in the artifact.
4. Require graceful shutdown so replay reports are emitted; reject kill-9/incomplete receipts.
5. Make analyse.py fail closed unless loaded>0, exact>0, final_tuned_launches>0, sum(per_winner_launches)==final_tuned_launches, and fallback_after_exact is present.
6. Compare predicted E2E gain from actual launch counts with measured E2E movement.

## Detailed Solution & Technical Design

Capability owner: tuning

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

replay executor/launch telemetry; tune/replay artifact schema; tools/bigcherry analysis and tests; maintained replay evidence

## Validation

Replay artifact invariant: cache entries loaded, exact hits, final tuned launches, per-winner sum, fallback-after-exact, and shutdown receipt all present and consistent. Analysis rejects hollow exact-hit evidence. Predicted versus measured E2E comparison is recorded.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

A replay result is admissible only when final tuned launches are directly proven after final validation, per-winner counts reconcile, fallback-after-exact is reported, and graceful teardown produced the complete artifact. Neutral E2E results are interpreted only after predicted gain is computed.

## Notes

Supersedes: HI160
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi160

Supersedes: HI160
Inherited semantic scope: preserve launch-point proof, per-winner/fallback invariant, graceful teardown, and predicted-E2E arithmetic from HI160.
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-09T10:49:22.968400+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:05:31.135884+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.876973+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.403308+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:23:56.968639+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_022438_the-next-five-high-risk-tuning_6580
- 2026-09-10T02:24:38.688942+00:00 (updated-by): Updated: section:ledger-events
