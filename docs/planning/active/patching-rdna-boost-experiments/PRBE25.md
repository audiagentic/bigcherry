---
id: PRBE25
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:09.148172+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-MOE-003: Compact per-expert MMQ launch grid

## Description

IMPLEMENTED-AS-PATCH (sub-scope of 1237). PRBE25's scope (compact per-expert MMQ launch grid, depends on PRBE24's map) is already implemented inside patch 1237_rd30_moe_mmq_compact_grid as the compact launch path consuming mmq_build_moe_block_map's output, gated `cc == GGML_CUDA_CC_RDNA3` (gfx1100 exact), fail-closed to the legacy rectangular grid. Real informal evidence already gathered (dual-gfx1100 interleaved A/B, Qwen3.6-35B-A3B): +0.73%..+0.90% pp512, consistent/non-overlapping across 3 rounds -- close to but should be re-verified against the item's own >=2% PP gain / <1% dense loss promotion threshold under the FORMAL campaign (informal evidence alone does not satisfy this item's acceptance criteria).

## Steps

1. Confirm PRBE24's map-construction correctness/overhead validation has passed (hard dependency, per this item's own acceptance criteria: 'PRBE25 cannot proceed on obsolete source assumptions' and the parent item's explicit prerequisite ordering).
2. Author a launched/useful-block telemetry extraction (rocprofv3-based) separate from PRBE24's map-only microbenchmark: real launched block count vs. useful (non-empty) block count, empty-block fraction, kernel time.
3. Use the SAME EC13/RD94 hostile-routing distribution matrix as PRBE23/PRBE24 (uniform/all-one/skew/Zipf/tiny/n_expert=256) so results are directly comparable across all three items.
4. Measure E2E PP for Qwen3.6-35B-A3B at pp128/512/1024/4096, with dense/uniform and tiny-batch controls quantifying indirection overhead.
5. Compare the measured +0.73%..+0.90% informal result against the formal campaign's re-measurement; note that the existing 3 rounds are described in patch.py/notes as informal, not campaign-recorded.
6. Promote conditionally only at >=2% PP gain on target routing with <1% dense loss (the informal evidence is currently BELOW this 2% threshold -- flag this explicitly: the item may end up 'not promoted, retain as negative/marginal evidence' rather than promoted, pending the formal re-measurement).

## Detailed Solution & Technical Design

No new source-code design needed -- reuses patch 1237's existing compact-launch path unchanged. PRBE25's own contribution is block-utilization telemetry and E2E PP measurement isolated from PRBE24's map-only concerns, run under the SAME formal campaign as PRBE23's umbrella qualification. Important finding to carry into the promotion decision: informal evidence (+0.73%..+0.90%) is already below the item's own >=2% promotion bar, so a likely outcome is 'do not promote as a default, retain positive-but-below-threshold evidence' rather than a straightforward promotion -- this should be stated plainly rather than glossed over when the formal run completes.

## Code Samples & Guidance

N/A -- no source edits to the compact launch path itself expected. New validation script: patches/1237_rd30_moe_mmq_compact_grid/validation/rd32_launch_efficiency.py (block-utilization telemetry + E2E PP extraction), sibling to PRBE24's rd31_block_map_correctness.py.

## Files

patches/1237_rd30_moe_mmq_compact_grid/patch.py (read-only reference); patches/1237_rd30_moe_mmq_compact_grid/validation/rd32_launch_efficiency.py (new); dense/tiny-batch control fixtures (new).

## Validation

Exact output equality vs. legacy grid (same routing matrix as PRBE23/24); launched/useful block counts and empty fraction; kernel-time delta; pp128/512/1024/4096 E2E on Qwen3.6-35B-A3B with dense/tiny controls; formal re-measurement against the existing +0.73%..+0.90% informal result and the item's own >=2%/<1% promotion threshold.

## Effort & Risk

S-M: no new kernel code; risk is mainly that the formal campaign may confirm the gain sits below the item's own promotion bar, which is a valid and useful (if unexciting) outcome to record rather than a failure of the plan.

## Standards

Dependency-aware promotion; exact output; conditional fallback; current table-driven architecture only.

## Acceptance Criteria

Compact grid is correct and produces >=2% target PP gain with <1% dense loss, or remains deferred; no promotion without PRBE24 and durable candidate evidence.

## Notes

Supersedes: RD32
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd32

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH (sub-scope of 1237, untested). No GPT design request needed -- no new source design. Flagged: existing informal +0.73%..+0.90% pp512 evidence is below this item's own explicit >=2% promotion threshold; formal campaign should settle whether this is a real marginal gain or noise, and the plan should not assume promotion is likely.

## Change Log

- 2026-09-09T10:55:09.148172+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:23.868961+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.240579+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.968825+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:53:57.011832+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025409_moe-mmq-successors-prbe2325-n_6205
- 2026-09-10T02:54:09.770527+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:32:09.396393+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
