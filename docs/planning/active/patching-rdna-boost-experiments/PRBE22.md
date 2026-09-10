---
id: PRBE22
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:56.585436+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-MMQ-001: RDNA MMQ tile-width reduction for LDS occupancy

## Description

Investigate AMD-MMQ-001 as a table-driven RDNA MMQ tile-width/LDS occupancy redesign; the original upstream diff is invalid against the current MMQ architecture.

## Steps

- Recheck current mmq-config-rdna3-5.cuh and table-driven selection before porting any source; do not apply obsolete PR #32 anchors.
- Define high-cost dense/MoE signatures on Qwen3.6-27B/35B-A3B and architecture-separated gfx1100/gfx1201/gfx1151 controls.
- Sweep mmq_x around native winner and physical M/ubatch 64..4096; record LDS/workgroup/VGPR/waves/occupancy.
- Require backend tensor parity/temp-0 and bit identity where arithmetic order is unchanged before timing.
- Promote only as architecture/signature selector for >3% repeatable kernel or >1% E2E gain with <=1% non-target regression; otherwise retain redesign findings.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

ggml-cuda/mmq-config-rdna3-5.cuh and architecture tables; candidate selector; high-cost signature campaign; parity/occupancy/profiler artifacts; no obsolete PR #32 diff.

## Validation

Current table audit; backend-op parity; temp-0/bit identity; pp128/512/1024/4096; kernel median/MAD; call-weighted ranking; LDS/VGPR/waves/occupancy; architecture/non-target controls.

## Effort & Risk



## Standards

Needs-redesign disposition; architecture-specific evidence; no upstream-diff assumption.

## Acceptance Criteria

A current table-driven design is proven or a new selector is authored from current seams; obsolete anchors are not ported; promotion meets the explicit gain/regression threshold.

## Notes

Supersedes: RD28
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd28

## Change Log

- 2026-09-09T10:54:56.585436+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:11.248783+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.227051+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.949508+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:52:05.324105+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025218_rdna-successors-prbe2022-now_5714
- 2026-09-10T02:52:18.433749+00:00 (updated-by): Updated: section:ledger-events
