---
id: PRBE18
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:45.862958+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Evaluate SSM pre-scan chain fusion (conv + l2_norm pair + gate/beta)

## Description

Port and qualify the exact 16-node SSM pre-scan chain fusion as a Wave-2 candidate, with explicit PRBE05 cache and PRBE19 post-fix source-state dependencies. PRBE18 is the actionable successor to closed RD24.

## Steps

1. Port the three-file candidate as an isolated PRBE18 experiment; do not port superseded RD14/RD16 separately.
2. Require the exact 16-node graph: conv F32, ne[1]==1 decode shape, SiLU, Q/K/V views, shared alpha/beta, matching Q/K epsilon, packed Q8_0-compatible layout, contiguity and correct residual/output wiring.
3. Confirm producer readiness and reject graphs with missing graph edges such as an untracked conv_states read; retain wrong-wiring/no-fuse fallback.
4. Take the fused SSM region from PRBE19's reviewed post-fix source state, using current selector-derived MMVQ geometry rather than hardcoded historical warp assumptions; audit the four q8_1_cache references as evidence for the real PRBE05 dependency.
5. Run fused vs unfused output, graph-capture and causal timing on gfx1100 before composition.

## Detailed Solution & Technical Design

The candidate combines conv+SiLU+Q/K normalization+V+gate/beta pre-scan in one 16-node path. PRBE05 is a real dependency because the source contains four q8_1_cache references; wire it through the stable, bounded cache rather than importing a second cache. PRBE19 is the post-fix source-state rule, not a separately applied patch. Enforce exact tensor layouts, epsilon equality, scalar/shape predicates, graph-edge visibility, fallback and current MMVQ selector-derived launch geometry.

## Code Samples & Guidance



## Files

ggml-cuda.cu/mmvq.cu/mmvq.cuh; isolated PRBE18 patch; external source entry; PRBE05 cache API and evidence; PRBE19 post-fix identity; exact 16-node/wrong-wiring fixtures; SSM campaign.

## Validation

Guard/wrong-wiring; output equality; graph capture; PRBE05 cache and PRBE19 post-fix source-state checks; gfx1100 causal timing; no RD14/RD16 duplicate port.

## Effort & Risk



## Standards

Exact graph pattern; Wave-2 dependency; no duplicate superseded ports; fallback.

## Acceptance Criteria

Exact 16-node pattern is correct and captures; unsupported graphs do not fuse; only a dependency-complete arm may be promoted on positive causal evidence.

## Notes

Supersedes: RD24
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd24

Supersedes: RD24 (closed historical predecessor); RD14 and RD16 are closed/superseded historical designs and are not separate ports. Preserve RD24 source commit 4a4da30e... as provenance. Live dependencies are PRBE05 and PRBE19.

## Change Log

- 2026-09-09T10:54:45.862958+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:53.591511+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.209644+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.922415+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:50:30.902295+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025049_rdna-successors-prbe1719-now_5726
- 2026-09-10T02:50:49.219544+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T09:52:30.160133+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:notes
- chg_20260912_095506_cleaned-the-active-rdna-boost_4906
- 2026-09-12T09:55:06.240435+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T10:09:39.929105+00:00 (updated-by): Updated: section:validation
- chg_20260912_101015_fixed-the-remaining-plan-taxon_2574
- 2026-09-12T10:10:15.548038+00:00 (updated-by): Updated: section:ledger-events
