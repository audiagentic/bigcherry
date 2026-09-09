---
id: RHA07
order: 0
plan: run-hip-autotune
state: pending
created-at: '2026-09-09T18:00:32.292398+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: M
priority: P1
---

# Extend stage advisories across build, A/B, recovery and promotion boundaries

## Description

RHA03's remaining lifecycle-advisory scope is now split here. The first implementation slice covers the maintained paired A/B server-capture boundary; runtime-matrix advisories remain complete under RHA03.

## Steps

1. Inventory the existing build, ab-benchmark, recovery and promotion receipts/events and choose their canonical result boundaries.
2. Define one stable advisory schema and stage-specific tags for missing evidence, invalid comparisons, recovery exhaustion and promotion limitations.
3. Integrate advisory emission at each existing worker boundary; preserve child verdicts and admission decisions verbatim.
4. Add fixtures/tests for success, degraded, failure and no-evidence outcomes at every integrated stage.
5. Document UI/CLI consumption and close the review RV159.

## Detailed Solution & Technical Design

Capability owner: run reporting. Reuse the existing campaign/tuning workers and advisory document schema; do not build another executor, benchmark harness, recovery engine or promotion policy. Each stage adapter consumes its worker's completed result and writes a sibling advisories.json (or equivalent embedded field) while returning the original result unchanged. Malformed stage receipts produce UNKNOWN findings rather than silence.

## Code Samples & Guidance



## Files

tools/bigcherry/campaign/run_advisories.py
tools/bigcherry/campaign/benchmark.py
tools/tests/campaign/test_run_advisories.py
tools/tests/campaign/test_server_benchmark_capture.py
docs/reference/testing/TEST.md

## Validation

Added evaluate_ab_result() with conditional AB_RUN_FAILURE, AB_NO_EVIDENCE, AB_NOT_ADMITTED, AB_MISSING_COMPARISON and AB_DIAGNOSTIC_EVIDENCE findings. Existing server-comparison capture now writes sibling advisories.json on every persisted result, including partial/failure paths, without changing its return code or performance-admission field. Focused validation: PYTHONPATH=tools python -m pytest tools/tests/campaign/test_run_advisories.py tools/tests/campaign/test_server_benchmark_capture.py tools/tests/campaign/test_runtime_matrix.py tools/tests/cli/test_runtime_matrix_cli.py -q (32 passed, 4 subtests). Remaining: build, recovery and promotion boundary integrations.

## Effort & Risk



## Standards



## Acceptance Criteria

- Build, A/B, recovery and promotion result boundaries emit the shared machine-readable advisory schema.
- Findings distinguish missing evidence and invalid interpretation from actual worker failure.
- Advisory generation cannot alter execution, correctness or performance-admission decisions.
- Tests cover success, degraded, failure and no-evidence fixtures for every integrated stage.
- UI/CLI documentation identifies the stable artifact and tags.

## Notes

Split from RHA03 by review RV159.
Migration: capability-rebaseline-v3-2026-09
RHA03 owns completed run-result/matrix advisories; RHA07 owns only the remaining lifecycle-stage integrations.

Split from RHA03 by review RV159.
Migration: capability-rebaseline-v3-2026-09
RHA03 owns completed run-result/matrix advisories; RHA07 owns only the remaining lifecycle-stage integrations.

Progress: maintained A/B server-capture advisories are implemented; build/recovery/promotion remain.

## Change Log

- 2026-09-09T18:00:32.292398+00:00 (created-by): Created by agent

## Ledger-events


- chg_20260909_180121_run-results-now-expose-conditi_8235
- 2026-09-09T18:01:21.357937+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T18:03:24.244807+00:00 (updated-by): Updated: section:description, section:files, section:validation, section:notes
- chg_20260909_180344_ab-benchmark-artifacts-now-in_6663
- 2026-09-09T18:03:44.352055+00:00 (updated-by): Updated: section:ledger-events
