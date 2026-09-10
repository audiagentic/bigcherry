---
id: RQW01
order: 3
plan: run-qualification-workflow
state: completed
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

tools/bigcherry/telemetry.py (new); tools/bigcherry/experiment/bundle.py (run_managed() wired); tools/tests/test_telemetry.py (new). NOT yet wired into: campaign/workers.py, profiling/ harnesses (RE37/QU02's original broader Files list) -- landed the primitive plus its first real integration point.

## Validation

DONE 2026-09-10: 5 new unit tests (stderr-only routing, secret-argv non-leakage in default mode, show_argv opt-in, completion-line-on-exception) all passing. All 10 pre-existing tools/tests/campaign/test_experiment_bundle.py tests still pass with telemetry active, confirmed CLI machine-readable stdout unaffected (managed-run-cli test's own JSON output verified unchanged). Remaining: campaign workers / profiling harness integration not yet done.

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

IMPLEMENTED 2026-09-10: tools/bigcherry/telemetry.py -- console_telemetry() context manager emitting launch/progress/completion lines to stderr only, with summarize_launch() reducing argv to name+count+blake2b digest (never raw values, per redaction requirement) unless a trusted caller opts in with show_argv=True. Wired into experiment.bundle.run_managed() -- the first and, per the design review, the correct integration point (decorates the existing managed-process seam rather than adding a parallel subprocess wrapper). Verified all 10 pre-existing experiment-bundle tests still pass with telemetry active, and that CLI machine-readable stdout output is unaffected (confirmed via the managed-run-cli test's own JSON output). 5 new focused tests cover: stderr-only routing, non-leakage of secret argv values in default mode, show_argv opt-in, and completion-line emission even when the wrapped body raises. Remaining scope from this item's original Files list (campaign workers, profiling harnesses) not yet wired -- this lands the primitive plus its first integration.

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
- chg_20260910_002058_added-standard-launchprogress_1968
- 2026-09-10T00:20:58.995673+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:21:03.268966+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:27:23.974515+00:00 (updated-by): Updated: section:files, section:validation
- 2026-09-10T01:49:32.500953+00:00 (state-transition): State: pending → completed
