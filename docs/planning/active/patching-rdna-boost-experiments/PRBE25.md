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

Redesign and qualify compact per-expert MMQ launch grid after PRBE24 map qualification.

## Steps

- Require PRBE24 exact map and source identity; use same routing-distribution matrix.
- Launch only compact actual-work blocks while preserving expert/tile enumeration and exact output parity.
- Measure launched/useful blocks, empty fraction, kernel time and E2E PP for Qwen3.6-35B-A3B pp128..4096.
- Use dense/uniform and tiny-batch controls to quantify indirection overhead and retain rectangular fallback where it dominates.
- Promote conditionally only at >=2% PP gain on target routing with <1% dense loss; revalidate candidate tuning/replay identity.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

PRBE24 map API; current MMQ grid enumeration; Qwen3.6-35B-A3B campaign; dense/tiny controls; launched/useful block telemetry; E2E PP evidence.

## Validation

Exact output; same routing matrix; block counts/empty fraction; kernel us; pp128/512/1024/4096; dense/tiny controls; PRBE24 dependency and fallback.

## Effort & Risk



## Standards

Dependency-aware promotion; exact output; conditional fallback; current table-driven architecture only.

## Acceptance Criteria

Compact grid is correct and produces >=2% target PP gain with <1% dense loss, or remains deferred; no promotion without PRBE24 and durable candidate evidence.

## Notes

Supersedes: RD32
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd32

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
