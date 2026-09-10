---
id: PGC02
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-09-09T10:47:53.449711+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Complete qualification and promotion decision for landed N-way internal AllReduce

## Description

Complete qualification and promotion decision for landed N-way internal AllReduce with explicit negative evidence retention.

## Steps

1. Reconcile patch 1244 metadata and SUMMARY with current evidence.
2. Run soak and root/topology/size matrix across supported N-way regimes.
3. Compare against the correct native baseline and record correctness before performance.
4. Retain patch 1245 as negative/dispositioned evidence; do not reopen it.
5. Record K-COLLECTIVE-03/04/05 as conditional/deferred unless evidence changes their scope.
6. Make promotion/disposition fail-closed and provenance-bound.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patch 1244 source/metadata and SUMMARY; N-way AllReduce qualification harness; RCCL/native baseline; root/topology/size soak artifacts; patch 1245 retained negative evidence; K-COLLECTIVE-03/04/05 disposition records.

## Validation

Correctness before performance; N=3 boundary and supported N-way regimes; root/topology/size matrix; soak; native baseline; 1245 negative evidence; conditional K-item dispositions; exact patch/revision provenance.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Patch 1244 has correctness, soak, root/topology/size coverage and a provenance-bound promotion decision; patch 1245 remains retained negative evidence; conditional K items are explicitly dispositioned.

## Notes

Supersedes: GP11
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-gpu-collectives-gp11

Supersedes: GP11
Inherited constraints: RV103 and RV114 — retain N=3 qualification boundary, soak/root/topology/baseline gates, duplicate-claim reconciliation, 1245 negative evidence, and conditional K-item dispositions.
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-09T10:47:53.449711+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:03:53.119748+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.771380+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.231843+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:53:05.364557+00:00 (updated-by): Updated: section:description, section:steps, section:acceptance_criteria, section:notes
- chg_20260910_005948_legacy-planning-folders-now-co_1240
- 2026-09-10T00:59:49.055704+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:38:05.053695+00:00 (updated-by): Updated: section:files, section:validation, section:acceptance_criteria
- chg_20260910_023824_the-gpu-collective-successors_5773
- 2026-09-10T02:38:24.269287+00:00 (updated-by): Updated: section:ledger-events
