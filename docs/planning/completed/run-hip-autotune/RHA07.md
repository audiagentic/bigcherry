---
id: RHA07
order: 0
plan: run-hip-autotune
state: completed
created-at: '2026-09-09T18:00:32.292398+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: M
priority: P1
---

# Extend stage advisories across build, A/B, recovery and promotion boundaries

## Description

RHA03's remaining lifecycle-advisory scope is complete. Maintained paired A/B capture, build, recovery and promotion boundaries now emit shared machine-readable advisories without changing worker policy or admission decisions.

## Steps

1. Inventory the existing build, ab-benchmark, recovery and promotion receipts/events and choose their canonical result boundaries.
2. Define one stable advisory schema and stage-specific tags for missing evidence, invalid comparisons, recovery exhaustion and promotion limitations.
3. Integrate advisory emission at each existing worker boundary; preserve child verdicts and admission decisions verbatim.
4. Add fixtures/tests for success, degraded, failure and no-evidence outcomes at every integrated stage.
5. Document UI/CLI consumption and close the review RV159.

## Detailed Solution & Technical Design

Capability owner: run reporting. Reuse the existing campaign/tuning workers and shared advisory document schema; no new executor, benchmark harness, recovery engine or promotion policy. Each stage adapter consumes its completed worker result and writes a sibling advisory artifact while returning the original result unchanged. Malformed stage receipts become explicit findings rather than silence.

## Code Samples & Guidance



## Files

tools/bigcherry/campaign/run_advisories.py
tools/bigcherry/campaign/benchmark.py
tools/bigcherry/cli/build.py
tools/bigcherry/tuning/workflow.py
tools/bigcherry/tuning/tune_promotion.py
tools/tests/campaign/test_run_advisories.py
tools/tests/campaign/test_server_benchmark_capture.py
tools/tests/campaign/test_runtime_matrix.py
tools/tests/cli/test_runtime_matrix_cli.py
tools/tests/tuning/test_workflow.py
tools/tests/tuning/test_recovery.py
tools/tests/tuning/test_tune_promotion.py
docs/reference/testing/TEST.md

## Validation

Run-stage, A/B, build, recovery and promotion advisory emission is implemented and covered. Focused lifecycle validation: PYTHONPATH=tools python -m pytest tools/tests/campaign/test_run_advisories.py tools/tests/tuning/test_workflow.py tools/tests/tuning/test_recovery.py tools/tests/tuning/test_tune_promotion.py -q (118 passed, 19 subtests). RHA06/RHA07 regression set: PYTHONPATH=tools python -m pytest tools/tests/campaign/test_run_advisories.py tools/tests/campaign/test_server_benchmark_capture.py tools/tests/campaign/test_runtime_matrix.py tools/tests/cli/test_runtime_matrix_cli.py tools/tests/campaign/test_campaign_build.py tools/tests/campaign/test_campaign_workers_build.py -q (78 passed, 1 skipped, 6 subtests).

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

Split from RHA03 by review RV159.
Migration: capability-rebaseline-v3-2026-09
RHA03 owns completed run-result/matrix advisories; RHA07 owns only the remaining lifecycle-stage integrations.

Progress: run-result, maintained A/B server-capture and build advisories implemented; recovery/promotion remain.

Split from RHA03 by review RV159. Migration: capability-rebaseline-v3-2026-09. RHA03 owns completed run-result/matrix advisories; RHA07 owns lifecycle-stage integrations. Complete: maintained A/B, build, recovery and promotion boundaries emit the shared advisory artifact. Advisory findings are explanatory and preserve authoritative worker result, exit status and admission policy. UI/CLI consumers can poll the sibling JSON artifacts documented in TEST.md.

## Change Log

- 2026-09-09T18:00:32.292398+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260909_180121_run-results-now-expose-conditi_8235
- 2026-09-09T18:01:21.357937+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T18:03:24.244807+00:00 (updated-by): Updated: section:description, section:files, section:validation, section:notes
- chg_20260909_180344_ab-benchmark-artifacts-now-in_6663
- 2026-09-09T18:03:44.352055+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T18:05:36.278265+00:00 (updated-by): Updated: section:description, section:files, section:validation, section:notes
- chg_20260909_180600_build-runs-now-leave-an-adviso_1305
- 2026-09-09T18:06:00.890915+00:00 (updated-by): Updated: section:ledger-events
- chg_20260909_181117_recovery-and-promotion-stages_3392
- 2026-09-09T18:11:17.067379+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T18:11:35.364488+00:00 (updated-by): Updated: section:description, section:detailed_solution, section:files, section:validation, section:notes
- 2026-09-09T18:11:55.701961+00:00 (state-transition): State: pending → completed
