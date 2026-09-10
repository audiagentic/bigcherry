---
id: RDR02
order: 12
plan: run-docs-reference
state: pending
created-at: '2026-09-09T10:47:27.966718+00:00'
breadth: ''
skill: basic
created-by: capability-rebaseline-v3
work: M
priority: P3
---

# Migrate docs from hardcoded host facts to environment roles

## Description

The environment mechanism exists, but the source explicitly leaves the mechanical multi-file prose/path migration undone.

## Steps

1. Implement the still-valid future scope.
2. Run the stated acceptance and evidence gates.
3. Preserve predecessor provenance and record successor evidence under this ID.

## Detailed Solution & Technical Design

Capability owner: run

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

successor-specs/run-docs-reference-dr01.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: none extracted.

Active dependencies: Frozen dependencies: none recorded.

Reference handling: Rewrite forward references (1); preserve historical references (0) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: DR01
Migration: capability-rebaseline-v3-2026-09
Successor key: run-docs-reference-dr01

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): GO, after RDR01. Design (the 5-step prose/path migration already specified) is sufficient as-is. Add one explicit rule not previously stated: migration applies to MAINTAINED/LIVE docs only, not to immutable historical evidence/archive/campaign artifacts whose literal host/path/IP values are themselves provenance — those may legitimately keep Brutus-identity statements. No backward-compat aliases for old hardcoded paths (project doctrine: migrate up fully). Execution order: ranked #12, last — depends on RDR01 first establishing which files are live vs immutable evidence, so the mechanical pass doesn't touch the wrong set.

CORRECTION from deeper repo-validated dev-gpt review (2026-09-10): still valid after RDR01, but do NOT blindly reuse the original occurrence counts (83x 'brutus', 64x '/mnt/vault', etc.) quoted in DR01 -- those are stale. Recount against current HEAD before scoping the mechanical pass. Execution order stays #12 (unchanged), last.

## Change Log

- 2026-09-09T10:47:27.966718+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:03:28.103636+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.745382+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:57.299621+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:08:10.126436+00:00 (updated-by): Updated: order=12, priority='P3'
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.911837+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.190357+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:19:10.765467+00:00 (updated-by): Updated: section:notes
