---
id: PRBE13
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:22.843182+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Evaluate shared-expert output-chain fusion

## Description

Qualify the shared-expert output-chain fusion only for the exact six-node model pattern, with post-fix source-state, workspace, graph, scalar/layout and Q8 evidence. PRBE13 is the actionable successor to closed RD15.

## Steps

1. Identify exact six-node model/architecture pattern before enabling any rewrite.
2. Require the shared-expert region, including shexp_down_gated_q8_0 provenance, to be materialized from PRBE19's reviewed post-fix source state.
3. Check Q8_0 down projection, F32/scalar gate shape, residual source wiring, matching K/layout, contiguity, workspace scratch ownership and graph-capture lifetime.
4. Treat PRBE05 as conditional only if source audit proves a real Q8/cache dependency; otherwise keep cache and fusion effects separate.
5. Compare fused against unfused reference for numerical equivalence, workspace lifetime and graph capture; run false-positive fallback and call-weighted timing.

## Detailed Solution & Technical Design

This is model-specific graph fusion, not a general expert optimization. The candidate kernel shexp_down_gated_q8_0 is a port candidate from the historical RD15 source lineage, not current production code. Enforce exact model/architecture/shape predicates, scalar gate nelements==1, matching K and residual/output shapes, packed layout/contiguity, and safe workspace/capture ownership. PRBE19 supplies source-state correctness; it is not a separate patch to apply.

## Code Samples & Guidance



## Files

shared-expert graph source; exact six-node matcher; shexp_down_gated_q8_0 candidate region; model/type/scalar/residual fixtures; workspace/capture tests; PRBE19 post-fix source identity; optional PRBE05 evidence; causal performance artifacts.

## Validation

Six-node pattern; types; scalar/residual wiring; numerical equivalence; workspace ownership; graph capture; model applicability; false-positive fallback; balanced call-weighted timing.

## Effort & Risk



## Standards

Exact pattern; workspace ownership; model-gated promotion; no hidden dependency.

## Acceptance Criteria

Only exact model/graph patterns select; fused output and capture match reference; workspace is safe; nonqualifying paths fall back; promotion requires positive causal evidence and declared PRBE05 relationship.

## Notes

Supersedes: RD15
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd15

Supersedes: RD15 (closed historical predecessor). Preserve source identities 31eb8e953... / upstream 2f0d3c56... in provenance. Use PRBE13 and PRBE19 for live work; RD15/RD25 remain historical references only.

## Change Log

- 2026-09-09T10:54:22.843182+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:31.072998+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.188436+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.888350+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:47:47.250628+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024759_three-rdna-fusion-successors-n_3469
- 2026-09-10T02:47:59.479294+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T09:52:19.850215+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:notes
- chg_20260912_095506_cleaned-the-active-rdna-boost_4906
- 2026-09-12T09:55:06.224060+00:00 (updated-by): Updated: section:ledger-events
