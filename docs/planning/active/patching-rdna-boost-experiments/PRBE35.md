---
id: PRBE35
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:53.286868+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-STREAM-005: Protect concurrent-region join node from fusion

## Description

Protect the concurrent-region join node from graph fusion after PRBE34 shared-expert overlap; this is a mandatory prerequisite for graph-opt defaulting.

## Steps

- Require PRBE34 exact shared-expert concurrency identity.
- Guard graph fusion so the concurrent-region join remains present and aux stream rejoin semantics are preserved.
- Run repeated graph capture/replay on MoE decode with graph-opt on, plus graph-opt-off and dense controls.
- Verify no capture abort, output parity and no material ordinary regression.
- Do not enable PRBE36 default-on until this protection passes.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

Graph fusion eligibility around concurrent-region join; PRBE34 scheduler; capture/replay fixtures; MoE/dense controls; trace and output evidence.

## Validation

Repeated capture/replay; no abort; output parity; graph-opt off/dense controls; no material regression; join/rejoin trace.

## Effort & Risk



## Standards

Concurrency join correctness; dependency-aware defaulting; no fusion false positives.

## Acceptance Criteria

Join protection prevents capture failure and preserves output under eligible concurrency; PRBE36 remains blocked until this gate passes.

## Notes

Supersedes: RD43
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd43

## Change Log

- 2026-09-09T10:55:53.286868+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:05.382498+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.285865+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.037669+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:59:39.595087+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_030005_amd-streamfus-successors-prbe_8761
- 2026-09-10T03:00:05.752127+00:00 (updated-by): Updated: section:ledger-events
