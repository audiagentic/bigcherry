---
id: RHA03
order: 0
plan: run-hip-autotune
state: completed
created-at: '2026-09-09T10:49:39.970110+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: M
priority: P1
---

# Stage advisories: surface prior findings at the moment a result appears

## Description

Run-stage advisory support now attaches machine-readable findings to runtime-matrix results at the maintained CLI boundary. It distinguishes failed/partial matrices, missing child evidence or metrics, non-admitted performance, and diagnostic-only evidence without changing execution verdicts. Build/A-B/recovery/promotion-specific integrations remain separate future scope where those boundaries are owned by their existing workers.

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

tools/bigcherry/campaign/run_advisories.py
tools/bigcherry/cli/runtime.py
tools/tests/campaign/test_run_advisories.py
docs/reference/testing/TEST.md

## Validation

Added run-stage evaluation with explicit success, degraded/non-admitted, failure/partial, missing-evidence and diagnostic-only cases. CLI writes advisories.json and emits findings to stderr while returning the unchanged matrix state. Focused validation: PYTHONPATH=tools python -m pytest tools/tests/campaign/test_run_advisories.py tools/tests/cli/test_runtime_matrix_cli.py tools/tests/campaign/test_runtime_matrix.py -q (15 passed). Existing tuning advisories remain unchanged and historical provenance stays on HI165.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

- A maintained run-result boundary emits machine-readable advisory findings for success, degraded, failure/partial and no-evidence outcomes.
- Findings are conditional and explicit; malformed result input reports UNKNOWN rather than silent assurance.
- Advisory generation cannot alter matrix state, child verdict or performance admission.
- CLI/UI consumers can read advisories.json without parsing human stderr.
- Build/A-B/recovery/promotion ownership is not duplicated; future integrations consume this reporting contract or remain in their existing worker boundary.

## Notes

Supersedes: HI165
Migration: capability-rebaseline-v3-2026-09
Successor key: run-hip-autotune-hi165

Supersedes: HI165
Migration: capability-rebaseline-v3-2026-09
Successor key: run-hip-autotune-hi165

Split review RV159: the run-result/matrix advisory boundary is independently complete here. Remaining build, A/B, recovery and promotion lifecycle integrations are tracked by RHA07 and are not claimed as completed evidence under this item.

## Change Log

- 2026-09-09T10:49:39.970110+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:05:50.699257+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events



- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.899068+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T17:59:55.057809+00:00 (updated-by): Updated: section:description, section:files, section:validation, section:acceptance_criteria
- 2026-09-09T18:00:41.888165+00:00 (updated-by): Updated: section:notes
- 2026-09-09T18:01:00.232078+00:00 (state-transition): State: pending → completed
- chg_20260909_180121_run-results-now-expose-conditi_8235
- 2026-09-09T18:01:21.336855+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.438469+00:00 (updated-by): Updated: section:ledger-events
