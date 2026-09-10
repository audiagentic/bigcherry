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

Redesign AMD-MOE-002 GPU compact MoE MMQ block-map construction against the current table-driven MMQ architecture; it is a prerequisite of PRBE25.

## Steps

- Audit current launch_mul_mat_q<type,mmq_x,fallback> and mmq-config tables; do not apply obsolete two-parameter PR #63 anchors.
- Define device prefix/map kernel from expert token counts with no host readback in steady state.
- Validate GPU map exactly against CPU reference for uniform, all-one, skew, Zipf, tiny and n_expert=256 distributions.
- Measure map-build time, synchronization, temporary bytes and grid limits independently; retain legacy fallback on overflow or unsupported stream-K.
- Only after map correctness/overhead is proven, expose the map to PRBE25 and record dependency identity.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

Current mmq-config/launch seam; compact map prep kernel/workspace; CPU reference; routing distribution fixtures; overflow/fallback and map-overhead evidence.

## Validation

Current architecture anchors; exact GPU/CPU map parity; tokens 1..4096; expert distributions; map microtiming; sync/temp bytes; no host readback; fallback.

## Effort & Risk



## Standards

Needs-redesign; no stale diff port; exact map correctness; no host synchronization in steady state.

## Acceptance Criteria

Current table-driven redesign produces an exact map with acceptable overhead and safe fallback; PRBE25 cannot proceed on obsolete source assumptions.

## Notes

Supersedes: RD31
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd31

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
