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

Capture actual loaded HIP/Vulkan runtime stack before accepted samples, compare against expected identity, and persist fail-closed attestations.

## Steps

Freeze backend-neutral attestation schema/comparator; implement actual loaded HIP reporter and Vulkan adapter; collect after provider-exercising initialization/warmup before timed work; reject missing/malformed/substituted/lazy-loaded provider identities; thread reports through record/tune/verifier/correctness/replay receipts and persistence; normalize effective visibility/ICD state.

## Detailed Solution & Technical Design

Report loaded modules, provider fingerprints and effective runtime state, not PATH/ROCM_PATH/LD_LIBRARY_PATH guesses. Keep software identity out of hardware key; compare requested/build/actual and fail before evidence acceptance on mismatch.

## Code Samples & Guidance



## Files

hip-autotune-stack h/cpp; workflow ServerRunner stages; runtime/attestation tests and report transport.

## Validation

Build under one stack/launch another; DSO/ICD substitution, visibility reorder, missing/malformed/lazy provider tests; report refreshed after provider-exercising warmup; actual identity persisted per run.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Production evidence cannot be accepted without deterministic pre-sample actual-runtime attestation matching expected stack; hardware identity remains separate.

## Notes

Supersedes: RO05
Migration: capability-rebaseline-v3-2026-09
Successor key: run-rocm-vulkan-provider-ro05

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): PROTOCOL FREEZE REQUIRED then implement. Boundary (RV129: expected identity from RRVP02, expected build identity from BRVP01, this item owns actual-LOADED attestation only) is correct; three details must be nailed down before coding: (1) how actual loaded-module identity is determined — use loaded-module evidence/provider APIs directly, never infer from PATH/ROCM_PATH/LD_LIBRARY_PATH/ICD selectors; (2) provider-exercising warmup/finalization semantics for lazily-loaded libraries — an init-only report cannot satisfy attestation for a lazy provider, must refresh after untimed warmup and before first accepted sample; (3) atomic/versioned report lifecycle. Every evidence-accepting stage must hard-fail before accepting samples on expected/actual mismatch — no warn-and-continue. Execution order: ranked #9, after BRVP01.

IMPORTANT CORRECTION from deeper repo-validated dev-gpt review (2026-09-10): do NOT build a parallel evidence-acceptance subsystem. BigCherry already has ExecutionIdentity, ExecutionAttestation, fail-closed comparators, process-bound KFD observation, and AttestedServerSession as the structural "cannot measure without attesting" seam. RRVP03's provider-stack attestation must EXTEND/COMPOSE WITH that existing seam, not create a second one. Also: TRVP01 is already the RO06 successor owning manifest/DB/run-identity persistence -- RRVP03 must stop at attestation protocol/comparison and hand off to TRVP01 for persistence, not implement persistence itself. If HIP provider-report source instrumentation is genuinely needed, note that existing PRVP01/02 are CM1-specific and do NOT own this -- create or split a separate patching-owned item rather than putting patch content inside this run-* item. Execution order shifts to #8 in the revised sequence, still after BRVP01.

PAUSED 2026-09-10 (user directive): Vulkan is out of scope for now -- plans may continue to be updated/reviewed, implementation is paused. Design above (extend existing ExecutionAttestation seam, hand off to TRVP01 for persistence) stands for when resumed.

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
- 2026-09-10T00:25:58.470366+00:00 (updated-by): Updated: section:notes
- chg_20260910_002605_paused-all-vulkan-provider-imp_6846
- 2026-09-10T00:26:05.854516+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:27:56.083117+00:00 (updated-by): Updated: section:files, section:validation
- 2026-09-10T03:28:57.809162+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032911_repaired-three-providerrun-su_5934
- 2026-09-10T03:29:12.009995+00:00 (updated-by): Updated: section:ledger-events
