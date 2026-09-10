---
id: RQW01
order: 3
plan: run-qualification-workflow
state: pending
created-at: '2026-09-09T10:53:26.474843+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: M
priority: P2
---

# Add standard console telemetry to CLI harness launches

## Description

Standard CLI launch/progress/completion telemetry still requires shared helper integration, tests, and documentation.

## Steps

1. Implement the still-valid future scope.
2. Run the stated acceptance and evidence gates.
3. Preserve predecessor provenance and record successor evidence under this ID.

## Detailed Solution & Technical Design

Capability owner: run

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

successor-specs/run-qualification-workflow-qu02.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: none extracted.

Active dependencies: Frozen dependencies: none recorded.

Reference handling: Rewrite forward references (1); preserve historical references (0) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: QU02
Migration: capability-rebaseline-v3-2026-09
Successor key: run-qualification-workflow-qu02

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): GO, design sufficient as-is. Build one shared telemetry primitive with typed launch/progress/completion events, stderr-only. Default summaries expose argument NAMES/counts/digests, never prompt/credential VALUES. Tests must prove machine-readable stdout is unchanged and secrets/raw prompts are absent from telemetry output. Confirmed reusable by RHA01 and future workflows without becoming a hard acceptance dependency of them. Execution order: ranked #4 (small cross-cutting win, land before more bespoke per-tool output accumulates).

CORRECTION from deeper repo-validated dev-gpt review (2026-09-10): implementation should NOT invent a new subprocess telemetry abstraction. experiment.bundle.run_managed() already records command/env/timing/return-code/artifacts for managed runs -- telemetry should DECORATE that existing seam (and ServerRunner, campaign workers) rather than wrap subprocess calls a second way. Re-scope the implementation step accordingly before coding. Execution order MOVES UP to #3 (was #4) in the revised sequence: RGC01 -> RRBC03 -> RQW01 -> RRBC01 -> RRVP01 -> RRVP02 -> BRVP01 -> RRVP03 -> RRBC02 -> RHA01 -> RDR01 -> RDR02.

## Change Log

- 2026-09-09T10:53:26.474843+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:10:00.315149+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.125469+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:34.165791+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:08:04.650246+00:00 (updated-by): Updated: order=4, priority='P2'
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.936166+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.792367+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:18:51.311580+00:00 (updated-by): Updated: order=3, section:notes
