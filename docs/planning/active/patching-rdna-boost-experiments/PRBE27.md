---
id: PRBE27
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:19.101435+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-MMQ-002: Dedicated RDNA3.5 MMQ device table

## Description

Redesign AMD-MMQ-002 as a current table-driven gfx1151 MMQ configuration investigation; obsolete PR #25 anchors must not be ported.

## Steps

- Audit mmq-config-rdna3-5.cuh and current table-driven selector before authoring code.
- Define gfx1151-only candidate table, dense/MoE Qwen corpus and exact Q4/Q6/Q8 shapes.
- Use gfx1100/gfx1201 build/non-selection controls and retain decode-neutrality check.
- On gfx1151 measure pp128/512/1024/4096, TG and resource stats with PPL/temp-0 parity.
- Promote only with real gfx1151 evidence; otherwise retain hardware-blocked redesign disposition.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

Current mmq-config-rdna3-5.cuh/table selector; gfx1151 candidate rows; architecture guards; build/non-selection tests; gfx1151 correctness/perf/resource artifacts.

## Validation

Current seam audit; gfx1151 Q4/Q6/Q8; PPL/temp-0; PP/TG; resource stats; gfx1100/gfx1201 non-selection; decode neutrality.

## Effort & Risk



## Standards

Needs-redesign; hardware-scoped; no unsupported extrapolation.

## Acceptance Criteria

A current table-driven gfx1151 design is proven and correct with positive hardware evidence, or no implementation is promoted; obsolete PR #25 diff is never applied.

## Notes

Supersedes: RD34
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd34

## Change Log

- 2026-09-09T10:55:19.101435+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:31.554843+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.249087+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.981923+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:55:23.253944+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025542_rdna-successors-prbe2628-now_5552
- 2026-09-10T02:55:42.894059+00:00 (updated-by): Updated: section:ledger-events
