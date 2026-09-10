---
id: THTR01
order: 0
plan: tuning-hip-tune-recovery
state: pending
created-at: '2026-09-09T10:51:21.638252+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Durable BehavioralFailureWitness records (not a candidate blacklist)

## Description

Persist scoped BehavioralFailureWitness records for real recovery failures without creating universal candidate/signature blacklists.

## Steps

Define full witness context identity (candidate/signature/source/build/model/profile/MTP args/hardware/runtime/corpus/assignment/failing set/verdict/divergence/schema); implement exact-match skip, same candidate+environment different ensemble downrank, different context diagnostic-only; invalidate on identity changes; require HI143 gate before publishing.

## Detailed Solution & Technical Design

A witness describes one observed failure context, never an intrinsic candidate property. Preserve age as metadata only; content/context identity controls reuse.

## Code Samples & Guidance



## Files

Recovery witness schema/storage, HTR01 integration, cache and classification tests.

## Validation

Synthetic three-level policy tests and publish invariant requiring HI143 PASS.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Exact failures may be skipped, same-candidate alternatives only downranked, different contexts never auto-excluded, and no cache publishes without behavioral PASS.

## Notes

Supersedes: HTR02
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-tune-recovery-htr02

## Change Log

- 2026-09-09T10:51:21.638252+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:07:43.236610+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.011485+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.618193+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:36:25.570459+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033709_repaired-five-recovery-success_3695
- 2026-09-10T03:37:09.509133+00:00 (updated-by): Updated: section:ledger-events
