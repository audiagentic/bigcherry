---
id: RRBC03
order: 2
plan: run-reusable-build-campaign
state: pending
created-at: '2026-09-09T10:59:22.190994+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P1
---

# Environment-action campaign inputs for non-patch experiments (driver/ICD/ASPM/hipBLASLt)

## Description

The frozen reusable-build-campaign item remains active with implementation/acceptance work outstanding and no terminal disposition.

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

successor-specs/run-reusable-build-campaign-re37.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: RD87.

Active dependencies: Frozen dependencies: RD87

Reference handling: RD88.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: RE37
Migration: capability-rebaseline-v3-2026-09
Successor key: run-reusable-build-campaign-re37

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): MINOR DESIGN REVISION then GO. EnvironmentInput concept is sound but one digest cannot blindly enter both build and run identity — add explicit projection/scope field: build | run | both | evidence_only. Offline tuning files (e.g. hipBLASLt/rocblas-gemm-tune solution caches) should be keyed by logical-role + content/artifact digest, not host-path identity. Mutating actions require requested-action plus pre/post observation and restoration outcome. This item is confirmed as the durable home for HI173's tuning-file evidence. Relationship to RRVP03 clarified: deliberately SEPARATE, not merged — RRBC03/EnvironmentInput = known/declared BEFORE execution (input to reproducibility/run identity); RRVP03/RuntimeStackAttestation = observed AFTER process launch + provider warmup (evidence validating what actually loaded). Do not put RRVP03's fingerprint into EnvironmentInput — a run evidence record should carry both as separate fields (environment_input_digest, expected_resolved_stack_fingerprint, build_stack_fingerprint, actual_runtime_stack_attestation_fingerprint) sharing canonical serialization/hashing primitives but remaining distinct schemas. Execution order: ranked #2 (directly unblocks durable HI173 evidence).

## Change Log

- 2026-09-09T10:59:22.190994+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:16:30.032077+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events



- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.497934+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:29.804131+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:08:03.286883+00:00 (updated-by): Updated: order=2, priority='P1'
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.954888+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.365735+00:00 (updated-by): Updated: section:ledger-events
