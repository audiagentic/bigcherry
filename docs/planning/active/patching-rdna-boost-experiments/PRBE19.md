---
id: PRBE19
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:40.683647+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Bake-in rule: port RD21/RD24/RD15 kernel regions from branch-tip (post-fix) state

## Description

Maintain RD25 as a bake-in sequencing rule: dependent ports must use branch-tip post-fix kernel regions; it is not a standalone patch.

## Steps

- When PRBE16 is ported, take calc_nwarps gfx1151 table including ncols_dst extension from branch-tip.
- When PRBE18 is ported, take SSM fused kernels with 2-warp reduction and qi=QI8_0 fix from tip.
- When PRBE13 is ported, take shexp_down_gated_q8_0 from tip; likewise preserve post-fix regions for any declared PRBE20 dependents.
- For each dependent port, validate fused decode against unfused and the named MTP batch-vs-seq scenario; preserve native non-MTP controls.
- Never materialize a standalone RD25 patch from an absent pre-image; record exact branch-tip identity and dependency linkage.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

Branch-tip 9e46e1fd/post-fix kernel regions; dependent PRBE16/18/13/20 packages; mmvq.cu fixtures; batch-vs-seq regression artifacts; source identity/provenance.

## Validation

Dependent region extraction; pre-fix reachability/reproduction or proof absent; fused decode bit identity/equality; MTP batch-vs-seq; native/non-MTP controls; no standalone patch assumption.

## Effort & Risk



## Standards

Correctness bake-in; branch-tip provenance; no standalone patch; dependency-aware promotion.

## Acceptance Criteria

Every dependent port uses post-fix branch-tip state and passes its own correctness gate; no broken pre-fix region is materialized; RD25 remains a sequencing/provenance rule.

## Notes

Supersedes: RD25
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd25

## Change Log

- 2026-09-09T10:54:40.683647+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:57.480362+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.214040+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.929217+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:50:36.902436+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025049_rdna-successors-prbe1719-now_5726
- 2026-09-10T02:50:49.233411+00:00 (updated-by): Updated: section:ledger-events
