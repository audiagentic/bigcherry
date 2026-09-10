---
id: THA15
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:49:57.014698+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# HI171 — Provenance-safe promoted-winner execution and winner-vs-runner-up counterfactual replay audit

## Description

Perform provenance-safe promoted-winner execution and winner-versus-runner-up counterfactual replay using immutable source ranking decisions and exact runtime/build identity.

## Steps

1. Persist immutable source top-2 ordering, candidate/config/artifact hashes, selection_id, and source-decision digest at promotion time.
2. Resolve replay manifests strictly from immutable provenance; legacy/incomplete decisions are UNREPLAYABLE_PROVENANCE and require fresh tuning.
3. Generate winner and runner-up manifests differing only in candidate binding under invariant workload, build, runtime, environment, and hardware.
4. Run correctness, activation, and exact-launch gates before timing.
5. Collect multi-session interleaved winner/runner-up/native evidence with cache refresh and teardown receipts.
6. Verify MTP behavioral equivalence and report predicted-versus-measured E2E effect; never rerank from current policy or mutable caches.

## Detailed Solution & Technical Design

Capability owner: tuning

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

production ranker/promotion metadata; replay manifest resolver; counterfactual runner; provenance and MTP equivalence evidence

## Validation

Immutable provenance hash checks; strict missing/mismatch rejection; winner/runner-up manifest diff audit; activation and final-launch evidence; multi-session performance matrix; MTP acceptance/output equivalence; cache refresh and clean shutdown.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Every admitted comparison is provenance-bound, candidate-only divergent, activation-verified, correctness-equivalent, and backed by repeatable counterfactual timing. Incomplete historical decisions are rejected rather than reconstructed.

## Notes

Supersedes: HI171
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi171

Supersedes: HI171
Inherited semantic scope: preserve immutable top-2 ordering, selection digest, strict UNREPLAYABLE_PROVENANCE behavior, invariant counterfactual manifests, and MTP equivalence gates.
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-09T10:49:57.014698+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:06:08.560483+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.918682+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T12:09:23.246039+00:00 (updated-by): Updated: section:title
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.484034+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:24:15.859276+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_022438_the-next-five-high-risk-tuning_6580
- 2026-09-10T02:24:38.740975+00:00 (updated-by): Updated: section:ledger-events
