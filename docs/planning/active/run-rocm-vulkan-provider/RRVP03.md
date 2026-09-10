---
id: RRVP03
order: 8
plan: run-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T10:59:58.197113+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P2
---

# Runtime stack attestation

## Description

Actual-loaded runtime attestation implementation is pending.

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

successor-specs/run-rocm-vulkan-provider-ro05.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: none extracted.

Active dependencies: Frozen dependencies: RO04.

Reference handling: Rewrite forward references (5); preserve historical references (0) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: RO05
Migration: capability-rebaseline-v3-2026-09
Successor key: run-rocm-vulkan-provider-ro05

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): PROTOCOL FREEZE REQUIRED then implement. Boundary (RV129: expected identity from RRVP02, expected build identity from BRVP01, this item owns actual-LOADED attestation only) is correct; three details must be nailed down before coding: (1) how actual loaded-module identity is determined — use loaded-module evidence/provider APIs directly, never infer from PATH/ROCM_PATH/LD_LIBRARY_PATH/ICD selectors; (2) provider-exercising warmup/finalization semantics for lazily-loaded libraries — an init-only report cannot satisfy attestation for a lazy provider, must refresh after untimed warmup and before first accepted sample; (3) atomic/versioned report lifecycle. Every evidence-accepting stage must hard-fail before accepting samples on expected/actual mismatch — no warn-and-continue. Execution order: ranked #9, after BRVP01.

IMPORTANT CORRECTION from deeper repo-validated dev-gpt review (2026-09-10): do NOT build a parallel evidence-acceptance subsystem. BigCherry already has ExecutionIdentity, ExecutionAttestation, fail-closed comparators, process-bound KFD observation, and AttestedServerSession as the structural "cannot measure without attesting" seam. RRVP03's provider-stack attestation must EXTEND/COMPOSE WITH that existing seam, not create a second one. Also: TRVP01 is already the RO06 successor owning manifest/DB/run-identity persistence -- RRVP03 must stop at attestation protocol/comparison and hand off to TRVP01 for persistence, not implement persistence itself. If HIP provider-report source instrumentation is genuinely needed, note that existing PRVP01/02 are CM1-specific and do NOT own this -- create or split a separate patching-owned item rather than putting patch content inside this run-* item. Execution order shifts to #8 in the revised sequence, still after BRVP01.

## Change Log

- 2026-09-09T10:59:58.197113+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:17:00.063504+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.528882+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:47.119856+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:08:08.151672+00:00 (updated-by): Updated: order=9, priority='P2'
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.968864+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.413059+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:19:03.493536+00:00 (updated-by): Updated: order=8, section:notes
