---
id: RHA08
order: 8
plan: run-hip-autotune
state: completed
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

Completed reusable BC build-type coverage for the configured 27B dual-XTX matrix through the maintained runtime-matrix/server-bench path. Control, record, tune, replay, and replay-diagnostic all ran serially with explicit identity, progress, diagnostics, cache/inventory, and teardown evidence. Production stock/native/replay parity remains RHA04; diagnostic throughput is not admitted.

## Steps

1. Resolve the e2e-build-matrix lanes from config/recipes.toml and current source/build identities.
2. Run control/record/tune/replay/replay-diagnostic where the model/topology and required inputs are available, serially and uncontended, using bench/run_bench.py --bench-type server-bench.
3. Emit the standard resolved/status/events/summary and per-cell raw artifacts; record diagnostics state, cache identity, replay activation and graceful teardown.
4. Classify diagnostic/framework observations separately from production stock/native/replay arms; do not admit diagnostic throughput as production evidence.
5. Add reusable documentation and close only when every available lane has an explicit pass or unavailable classification.

1. Resolve the e2e-build-matrix lanes from config/recipes.toml and current source/build identities.\n2. Run control/record/tune/replay/replay-diagnostic serially through runtime-matrix server_capture and bench/run_bench.py --bench-type server-bench.\n3. Emit standard resolved/status/events/summary and per-cell raw artifacts; retain diagnostics, cache/inventory and graceful teardown.\n4. Classify diagnostic/framework observations separately from RHA04 production parity.\n5. Completed: all available configured BC observation lanes have explicit verified results; BC native and stock/native production evidence is linked from RHA04.

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

tools/bigcherry/cli/runtime.py
tools/tests/campaign/test_runtime_matrix.py
docs/reference/testing/TEST.md
docs/evidence/2026-09-10-rha08-build-matrix/

## Validation

Use the maintained server-bench runner, never llama-bench. Validate with the runtime-matrix focused suite and retain a machine-readable summary for every build type. Production claims require execution attestation, correctness/work-equivalence and diagnostics-off timing; diagnostic builds are explanatory only.

docs/evidence/2026-09-10-rha08-build-matrix/ contains resolved-matrix.json, status.json, events.jsonl, summary.json, advisories.json, five per-cell raw bundles, and record/tune/hit-log sidecars. All five cells returned 0, clean SIGINT shutdown, and execution_evidence_status=verified against dual gfx1100 physical RCCL locators. Focused adapter/runtime tests pass (22 passed across runtime_matrix and benchmark sanitizer suites). RHA04 supplies the separate diagnostics-off stock/native/replay production matrix; no diagnostic throughput here is a production claim.

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

Split from RHA04 by RV160. The server_capture adapter reuses run_server_arm_capture, ServerRunner, and bench_runner; it is not a second campaign engine. Tune/replay pp512 observations are framework/cache diagnostics only and must not override RHA04 parity evidence. Matrix artifacts are UI-pollable through status.json/events.jsonl/summary.json.

## Change Log

- 2026-09-09T18:20:56.533969+00:00 (created-by): Created by agent

## Ledger-events


- chg_20260909_182139_the-plan-now-separates-the-dua_5938
- 2026-09-09T18:21:39.256909+00:00 (updated-by): Updated: section:ledger-events
- chg_20260909_182254_build-type-campaigns-now-have_3132
- 2026-09-09T18:22:54.616330+00:00 (updated-by): Updated: section:ledger-events
- chg_20260909_192136_runtime-matrix-cells-can-now-l_7636
- 2026-09-09T19:21:36.810066+00:00 (updated-by): Updated: section:ledger-events
- chg_20260909_192259_build-type-matrix-cells-can-no_8518
- 2026-09-09T19:22:59.223538+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T19:34:51.929527+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:notes
- 2026-09-09T19:35:01.457806+00:00 (state-transition): State: pending → completed
- chg_20260909_193515_rha08-is-complete-every-avail_9842
- 2026-09-09T19:35:15.027803+00:00 (updated-by): Updated: section:ledger-events
