---
id: PRBE15
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:30.186606+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Evaluate IMRoPE plus BF16 SET_ROWS fusion extension

## Description

Qualify the IMRoPE plus BF16 SET_ROWS fusion extension only as an incremental arm over unpromoted patch 1004.

## Steps

- Audit patch 1004 readiness and keep it explicitly unpromoted in the resolved experiment identity.
- Compare B+1004 against B+1004+PRBE15; never claim the extension against raw B.
- Validate exact rope/view/set_rows wiring, BF16 outputs, graph capture and false-positive fallback.
- Measure only incremental effect with resource/timing evidence and preserve failure/unsupported paths.
- Promote neither 1004 nor the extension without its own correctness and causal gates.

## Detailed Solution & Technical Design

The dependency is part of the science: patch 1004 is a control capability, not an implicit baseline. Exact fusion pattern, graph lifetime and fallback must be independently visible.

## Code Samples & Guidance



## Files

patch 1004 package/source; IMRoPE/VIEW/SET_ROWS fusion seam; BF16/reference fixtures; graph capture/fallback tests; B+1004 vs B+1004+PRBE15 campaign artifacts.

## Validation

1004 readiness; causal arms; rope/view/set_rows wiring; BF16 correctness; fallback; graph capture; resource/timing evidence.

## Effort & Risk



## Standards

No hidden baseline promotion; causal dependency arm; exact fusion pattern.

## Acceptance Criteria

The exact extension is correct and captured; only the incremental arm meets the performance/evidence gate; 1004 remains explicitly unpromoted unless independently accepted.

## Notes

Supersedes: RD18
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd18

## Change Log

- 2026-09-09T10:54:30.186606+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:39.895272+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.196499+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.902384+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:49:05.374902+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024925_rdna-successors-prbe1416-now_9529
- 2026-09-10T02:49:25.519390+00:00 (updated-by): Updated: section:ledger-events
