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

Continue the lifecycle-advisory portion split from RHA03. Attach conditional, machine-readable findings at the existing build, ab-benchmark, recovery and promotion result boundaries without duplicating their execution or admission policy. Run-stage matrix advisories are complete under RHA03.

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
tools/bigcherry/cli/build.py
tools/bigcherry/cli/tuning.py
tools/bigcherry/tuning/recovery.py
tools/bigcherry/tuning/promotion.py
tools/tests/campaign/
tools/tests/tuning/
docs/reference/testing/TEST.md

## Validation

Stage-specific unit fixtures prove conditional findings and unchanged child verdict/admission for build, A/B, recovery and promotion success/degraded/failure/no-evidence receipts. Existing RHA03 run-result tests remain green. Review RV159 is closed after the split is incorporated.

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

## Change Log

- 2026-09-09T18:00:32.292398+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260909_180121_run-results-now-expose-conditi_8235
- 2026-09-09T18:01:21.357937+00:00 (updated-by): Updated: section:ledger-events
