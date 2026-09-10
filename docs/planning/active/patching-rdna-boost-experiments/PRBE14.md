---
id: PRBE14
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:26.498754+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Evaluate MoE top-k weights folded into down projection

## Description

Evaluate MoE top-k weights folded into down projection while preserving the explicit composition boundary between patch 1207 and patch 1205.

## Steps

1. Preserve patch 1207's explicit MoE top-k/down-projection recipe identity and verify destination-channel scale semantics and scale-vector shape.
2. Preserve existing NVFP4 behavior and native fallback.
3. Compare fused and unfused outputs across MUL_MAT_ID expert routing, IDs, and scale cases.
4. Validate graph capture, false-positive fallback, and expert routing correctness.
5. Treat patch 1205 and 1207 as mutually exclusive unless an explicitly declared composed experiment passes PKC02 graph identity/conflict validation.
6. Qualify performance only after correctness and composition gates pass.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patches/1207_rd17_moe_topk_down_projection; MoE MUL_MAT_ID routing/down-projection seam; destination-channel scale fixtures; NVFP4/fallback tests; graph/DAG recipe identity and fused/unfused campaign artifacts.

## Validation

Correctness: destination-channel scale semantics/shape, expert IDs/routing, fused-vs-unfused output equality, NVFP4 preservation, native fallback, graph capture, and false-positive rejection. Performance: balanced fused versus unfused qualification with explicit recipe identity. Composition: 1205/1207 conflict gate through PKC02; no implicit combination.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

All RD17 requirements are carried forward: scale semantics/shape, NVFP4 preservation, fused/unfused comparison, expert routing/fallback, graph capture, false-positive fallback, and explicit 1205/1207 composition disposition. Promotion requires correctness and performance evidence for the declared recipe.

## Notes

Supersedes: RD17
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd17

Supersedes: RD17
Inherited constraint: RV105 — retain the 1205/1207 composition conflict and mutually exclusive recipe identities; link graph/DAG composition decisions to PKC02.
Migration: capability-rebaseline-v3-2026-09

Supersedes: RD17
Inherited semantic scope: RD17 detailed requirements plus RV105 composition restriction; historical evidence remains on completed predecessor.
Migration: capability-rebaseline-v3-2026-09

Supersedes: RD17
Inherited constraint: RV105 — retain 1205/1207 composition conflict and mutually exclusive recipe identities; link graph/DAG composition decisions to PKC02.
semantic-carryforward: concrete files restored 2026-09-10.

## Change Log

- 2026-09-09T10:54:26.498754+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:35.940393+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.192523+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.895469+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:52:25.957169+00:00 (updated-by): Updated: section:description, section:acceptance_criteria, section:notes
- chg_20260910_005948_legacy-planning-folders-now-co_1240
- 2026-09-10T00:59:48.994822+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:10:12.521404+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_021313_the-semantic-audit-is-now-trac_4827
- 2026-09-10T02:13:13.325520+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:48:59.175349+00:00 (updated-by): Updated: section:files, section:notes
- chg_20260910_024925_rdna-successors-prbe1416-now_9529
- 2026-09-10T02:49:25.497189+00:00 (updated-by): Updated: section:ledger-events
