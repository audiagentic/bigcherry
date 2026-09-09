---
id: RHA08
order: 8
plan: run-hip-autotune
state: pending
created-at: '2026-09-09T18:20:56.533969+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P1
---

# Exercise reusable BC build types through the maintained server-bench matrix

## Description

Run the remaining BC build types through the canonical runtime-matrix/server-bench path and standardize their evidence without conflating diagnostics with production parity. This successor is split from RHA04 review RV160; RHA04 retains the identity-bound production stock/native/replay parity gate.

## Steps

1. Resolve the e2e-build-matrix lanes from config/recipes.toml and current source/build identities.
2. Run control/record/tune/replay/replay-diagnostic where the model/topology and required inputs are available, serially and uncontended, using bench/run_bench.py --bench-type server-bench.
3. Emit the standard resolved/status/events/summary and per-cell raw artifacts; record diagnostics state, cache identity, replay activation and graceful teardown.
4. Classify diagnostic/framework observations separately from production stock/native/replay arms; do not admit diagnostic throughput as production evidence.
5. Add reusable documentation and close only when every available lane has an explicit pass or unavailable classification.

## Detailed Solution & Technical Design

Capability owner: run. Reuse tools/bigcherry/campaign/runtime_matrix.py and the maintained bench_runner/benchmark capture boundary. The item owns build-type coverage and artifact normalization only; it does not create a second executor or alter winner/promotion policy. Each lane must carry immutable build identity, source revision, model/topology, dispatch mode, diagnostics state and cache identity. Missing evidence is an explicit unavailable/degraded result, not a pass.

## Code Samples & Guidance



## Files

config/recipes.toml
tools/bigcherry/campaign/runtime_matrix.py
tools/bigcherry/campaign/bench_runner.py
tools/bigcherry/campaign/benchmark.py
tools/tests/campaign/test_runtime_matrix.py
docs/reference/testing/TEST.md
docs/evidence/<run-id>/

## Validation

Use the maintained server-bench runner, never llama-bench. Validate with the runtime-matrix focused suite and retain a machine-readable summary for every build type. Production claims require execution attestation, correctness/work-equivalence and diagnostics-off timing; diagnostic builds are explanatory only.

## Effort & Risk



## Standards



## Acceptance Criteria

- Every configured BC build type is run or explicitly classified unavailable with a reason.
- Output schema is identical and UI-pollable across build types.
- Diagnostic/framework observations are separate from production performance claims.
- No worker, admission, correctness or promotion policy is changed.
- Evidence and reusable procedure are committed, ledger-recorded and pushed.

## Notes

Split from RHA04 by review RV160. RHA04 owns decision-grade native/replay parity and dual-XTX attestation; RHA08 owns reusable build-type coverage. Migration: capability-rebaseline-v3-2026-09.

## Change Log

- 2026-09-09T18:20:56.533969+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260909_182139_the-plan-now-separates-the-dua_5938
- 2026-09-09T18:21:39.256909+00:00 (updated-by): Updated: section:ledger-events
