---
id: PRBE24
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:04.645348+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-MOE-002: GPU compact MoE MMQ block-map construction

## Description

IMPLEMENTED-AS-PATCH (sub-scope of 1237). PRBE24's scope (GPU compact MoE MMQ block-map construction, prerequisite of PRBE25) is already implemented inside patch 1237_rd30_moe_mmq_compact_grid as the `mmq_build_moe_block_map` prep kernel (block_start[n_experts+1]+block_expert[max_m_blocks] scheme, built from mm_ids_helper's existing expert_bounds[] prefix-sum, no host readback in steady state). PRBE24's real remaining work is a NARROWER validation than PRBE23's umbrella campaign: prove the GPU map is exactly correct against a CPU reference and measure its own overhead (build time, sync, temp bytes, grid limits) independent of the downstream launch-grid benefit measured by PRBE25.

## Steps

1. Confirm patch 1237's patch.py already contains the block-map prep kernel (verified this batch via SUMMARY.md/docstring; implementer should read patch.py directly for the exact kernel signature before writing tests).
2. Author a standalone GPU-map-vs-CPU-reference correctness harness (NOT the full E2E campaign) that runs mmq_build_moe_block_map for uniform/all-one/skew/Zipf/tiny/n_expert=256 routing distributions across token counts 1..4096 and diffs its block_start[]/block_expert[] output against a CPU reference implementation of the same flattening.
3. Microbenchmark the prep kernel alone: build time, host-device sync count (must be zero in steady state -- no host readback), temporary buffer bytes, and behavior at grid/shared-memory limits (must fail closed to the legacy rectangular grid, verify this path is actually exercised by an overflow fixture).
4. Record this as PRBE24's own narrow validation evidence, separate from PRBE23's umbrella campaign and PRBE25's launch-efficiency evidence.
5. Only once this passes does PRBE25's downstream launch-grid benefit become a properly gated measurement (PRBE24 is its prerequisite per the item's own acceptance criteria).

## Detailed Solution & Technical Design

No new source-code design needed -- reuses patch 1237's existing prep kernel unchanged. PRBE24's contribution is a dedicated correctness/overhead microbenchmark script, isolated from the full model E2E campaign, so a map-construction regression can be attributed precisely rather than being buried in an aggregate PP number.

## Code Samples & Guidance

N/A -- no source edits to mmq_build_moe_block_map itself expected. New validation script: patches/1237_rd30_moe_mmq_compact_grid/validation/rd31_block_map_correctness.py (map-vs-CPU-reference parity across distributions + microtiming), following the pattern of the existing patches/1210.../validation/rd26_correctness.py script referenced elsewhere in this plan set.

## Files

patches/1237_rd30_moe_mmq_compact_grid/patch.py (read-only reference, the prep kernel); patches/1237_rd30_moe_mmq_compact_grid/validation/rd31_block_map_correctness.py (new); CPU reference implementation of the block-flattening (new, small, in the same validation module); overflow/fallback fixture (new).

## Validation

GPU map exactly matches CPU reference for tokens 1..4096 across uniform/all-one/skew/Zipf/tiny/n_expert=256 distributions; map-build microtiming, zero host readback in steady state, temp-bytes and grid-limit accounting; overflow fixture proves the legacy-grid fallback actually engages. This is a narrower, faster check than PRBE23's full campaign and should be runnable as an isolated hardware step on Brutus.

## Effort & Risk

S: no new kernel code, a focused correctness/overhead microbenchmark against an already-implemented prep kernel.

## Standards

Needs-redesign; no stale diff port; exact map correctness; no host synchronization in steady state.

## Acceptance Criteria

Current table-driven redesign produces an exact map with acceptable overhead and safe fallback; PRBE25 cannot proceed on obsolete source assumptions.

## Notes

Supersedes: RD31
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd31

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH (sub-scope of 1237, untested). No GPT design request needed -- no new source design, only a narrower validation script than the umbrella campaign.

## Change Log

- 2026-09-09T10:55:04.645348+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:19.522893+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.236456+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.963459+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:53:47.892982+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025409_moe-mmq-successors-prbe2325-n_6205
- 2026-09-10T02:54:09.755635+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:31:49.104330+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
