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

Complete the remaining lifecycle advisory integrations at build, A/B, recovery and promotion result boundaries. Evaluators and build advisory wiring already exist; this item is limited to inventory, missing boundary adapters, tests and documentation.

## Steps

1. Inventory existing evaluate_ab_result, evaluate_build_result, evaluate_recovery_result and evaluate_promotion_result plus their canonical receipts.
2. Mark implemented evaluator/build wiring as complete evidence and identify only missing recovery, promotion and A/B boundary calls.
3. Attach one stable machine-readable advisory schema at each missing worker boundary without changing child verdicts or admission decisions.
4. Add success/degraded/failure/no-evidence fixtures for each remaining boundary; malformed receipts produce UNKNOWN findings.
5. Document stable artifact/tags and close RV159.

## Detailed Solution & Technical Design

Reuse tools/bigcherry/campaign/run_advisories.py and existing campaign/tuning workers. Advisory adapters consume completed results and return the original result unchanged; they do not execute, recover, admit or promote. Do not create another executor or recovery engine.

## Code Samples & Guidance



## Files

tools/bigcherry/campaign/run_advisories.py; tools/bigcherry/campaign/benchmark.py; tools/bigcherry/cli/build.py; tools/bigcherry/cli/tuning.py; tools/bigcherry/tuning/recovery.py; tools/bigcherry/tuning/promotion.py; tools/tests/campaign/**; tools/tests/tuning/**; docs/reference/testing/TEST.md

## Validation

Boundary inventory; unit fixtures for all remaining stages and outcomes; unchanged child verdict/admission assertions; existing RHA03 tests green; UI/CLI artifact documentation.

## Effort & Risk



## Standards

Stable advisory schema; UNKNOWN on malformed receipt; no execution/admission mutation; reuse existing workers.

## Acceptance Criteria

All four lifecycle boundaries emit the shared schema, remaining missing integrations are covered by tests, advisories cannot alter worker/admission outcomes, and RHA03 behavior remains green.

## Notes

Split from RHA03 by review RV159.
Migration: capability-rebaseline-v3-2026-09
RHA03 owns completed run-result/matrix advisories; RHA07 owns only the remaining lifecycle-stage integrations.

RHA03 owns completed run-result/matrix advisories. GPT assessment says this is partially implemented and should be refreshed before coding; do not redo existing evaluator work.

## Change Log

- 2026-09-09T18:00:32.292398+00:00 (created-by): Created by agent

## Ledger-events


- chg_20260909_180121_run-results-now-expose-conditi_8235
- 2026-09-09T18:01:21.357937+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T10:31:41.081187+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260912_103200_updated-the-active-buildrunp_8222
- 2026-09-12T10:32:01.033733+00:00 (updated-by): Updated: section:ledger-events
