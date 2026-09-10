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

Port and qualify the 16-node SSM pre-scan chain fusion as a Wave-2 candidate, with RD09/RD25 bake-in dependencies and exact graph guards.

## Steps

- Port the three-file change as an isolated rd24 experiment with provenance and registry entry; do not port superseded RD14/RD16 separately.
- Verify conv F32, ne[1]==1, silu, q/k/v views, shared alpha/beta, Q8_0 guards and wrong-wiring no-fuse.
- Take the fused SSM region from branch-tip/RD25 post-fix state with 2-warp reduction and qi=QI8_0; audit RD09 cache relationship.
- Run an SSM/Mamba model fused vs unfused output/capture/timing on gfx1100 before any composition.
- Retain fallback and reject unsupported shapes; compare only declared dependency arms.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

ggml-cuda.cu/mmvq.cu/mmvq.cuh; patch 12xx rd24; external source entry; RD09 cache and PRBE19 bake-in identity; exact 16-node/fallback fixtures; SSM campaign.

## Validation

Guard/wrong-wiring; output equality; graph capture; RD09/RD25 dependency; gfx1100 causal timing; no RD14/RD16 duplicate port.

## Effort & Risk



## Standards

Exact graph pattern; Wave-2 dependency; no duplicate superseded ports; fallback.

## Acceptance Criteria

Exact 16-node pattern is correct and captures; unsupported graphs do not fuse; only a dependency-complete arm may be promoted on positive causal evidence.

## Notes

Supersedes: RD24
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd24

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
