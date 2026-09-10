---
id: PKC03
order: 0
plan: patching-kernel-coverage
state: pending
created-at: '2026-09-09T10:51:51.425061+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: P3
---

# Decide whether collective identity needs a cross-backend extension

## Description

Decision-only gate: determine whether collective identity needs a cross-backend extension. Do not implement a collective registry.

## Steps

1. Inspect EC16/EC19, RRVP identity/persistence, and GP evidence.
2. Test whether collective provider, topology, root, threshold, protocol, fallback, and negative evidence are representable without ambiguity.
3. Assign any proven gap to the existing GP, provider, or contract owner.
4. Close this gate when existing machinery is sufficient.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

successor-specs/patching-kernel-coverage-kc03.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: EC16,EC19,GP03,GP07,GP08,GP10.

Active dependencies: Frozen dependencies: none recorded.

Reference handling: Rewrite forward references (6); preserve historical references (6) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

A documented decision establishes whether existing identity/contracts are sufficient; no CPU-only provider acceptance, collective registry, or backend implementation is introduced by this item.

## Notes

Supersedes: KC03
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-kernel-coverage-kc03

Supersedes: KC03
Inherited constraints: RV107, RV112, RV123 — this is a cross-backend identity sufficiency decision only.
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-09T10:51:51.425061+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:18.957220+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.043490+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.677197+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:52:43.052122+00:00 (updated-by): Updated: section:description, section:steps, section:acceptance_criteria, section:notes
- chg_20260910_005948_legacy-planning-folders-now-co_1240
- 2026-09-10T00:59:49.018109+00:00 (updated-by): Updated: section:ledger-events
