---
id: PRBE26
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:14.091235+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-MMV-001: Decode matvec without Q8_1 activation quantization

## Description

Resolve the MMVQ dequant-float decode candidate with correct activation-shape coverage. Existing ncols=1 evidence is correctness-positive but null; dominant MTP ncols=5/6 remains the open eligibility follow-up.

## Steps

- Reconcile patch 1241 summary versus hardware evidence and record one authoritative activation/disposition.
- Confirm F32xQ dequant path arithmetic and template plumbing for ncols 1..8; preserve forced-candidate and ncols>8 fallback.
- Run test-backend-ops tolerance correctness for n=1 and widened small-ncols, fused-gate/GLU, forced-candidate and non-target cases.
- Use rocprof/resource checks and interleaved paired A/B on production Qwen3.8-27B MTP where ncols=5/6 dominates; do not rely on the noisy first sequential round.
- Promote only on statistically supported E2E gain with quality/non-inferiority guard; otherwise retain the validated null/shape-mismatch disposition.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patch 1241 f32-act MMVQ; ncols eligibility/instantiations; test-backend-ops and GLU fixtures; forced/non-target controls; rocprof/resource evidence; interleaved production A/B and authoritative disposition.

## Validation

ncols 1..8 tolerance; path execution; forced/ncols>8 fallback; GLU; kernel resources; Qwen3.8-27B MTP ncols 5/6 interleaved A/B; quality and TG guard.

## Effort & Risk



## Standards

Shape-aware eligibility; tolerance not bit identity; interleaved evidence; no noisy sequential conclusion.

## Acceptance Criteria

Correctness holds for widened gate and unsupported paths remain native; only a real interleaved production gain promotes. If effect remains null, close as validated null with dominant-shape coverage documented; do not claim the earlier n=1 result covered MTP.

## Notes

Supersedes: RD33
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd33

## Change Log

- 2026-09-09T10:55:14.091235+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:27.752011+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.244503+00:00 (updated-by): Updated: section:ledger-events
- chg_20260909_143622_cleaned-up-the-tooling-registr_2452
- 2026-09-09T14:36:22.965188+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.975621+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:55:15.763944+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025542_rdna-successors-prbe2628-now_5552
- 2026-09-10T02:55:42.877445+00:00 (updated-by): Updated: section:ledger-events
