---
id: RHA06
order: 0
plan: run-hip-autotune
state: in_progress
created-at: '2026-09-09T10:49:53.272839+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P1
---

# Make campaign workflows unattended across models and GPU topologies

## Description

Run-owned reusable orchestration layer above existing campaign/build and tuning workflows. The pure whole-matrix resolver and serial file-backed runner are complete; this change adds the declarative JSON CLI adapter. Caller-supplied/configured worker commands remain authoritative for tuning and benchmark execution.

## Steps

1. Define one declarative runtime-matrix configuration and entrypoint over existing e2e-build-matrix, tune-campaign, ab-benchmark and ServerRunner machinery.
2. Preflight-expand every requested cell before GPU work: resolve models.toml, environment.toml and BC_* roles, recipes/build identities, runtime profiles, device sets/topology compatibility, and replay/cache evidence. Decide explicitly whether tune cells require pre-existing identity-bound inventory; missing or stale inventory must fail the whole preflight.
3. Materialize immutable resolved cell descriptors containing model, quantization, physical devices, canonical ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES pair, topology/runtime arguments, build and binary identities, cache evidence, and benchmark configuration.
4. Check parent quiescence before the first cell and after every cell. Execute cells serially with explicit visibility and clean teardown; residual process or device state aborts the matrix without killing unrelated work.
5. Re-check the exact config/tooling revision, runtime-profile digest, model identity, build IDs/binary hashes, and cache evidence immediately before each child execution to close the TOCTOU window.
6. Delegate execution to existing tune-campaign and ab-benchmark/ServerRunner paths. Preserve child verdicts verbatim and keep matrix orchestration status distinct from performance admission.
7. Emit write-once per-cell manifests, a run summary, atomic status.json, and sanitized append-only events.jsonl with state, active cell, completed/total counts, timestamps, errors and terminal outcome. Provide launch/status polling through files; do not add a service/API at this stage.
8. Add fail-closed tests, deterministic ordering, no-overlap guarantees, visibility and teardown tests, stable identity digests, child failure propagation, progress atomicity/event sanitization, and representative dual-XTX/27B plus single-GPU/9B smoke configurations.

## Detailed Solution & Technical Design

Capability owner: run.

Placement: add one campaign/runtime_matrix.py layer consuming campaign planners and execution primitives. Campaign build lanes remain responsible for build identity; runtime_matrix owns physical placement and workload orchestration. It must not import a second selector or duplicate patch/admission semantics.

Resolution contract: resolve_matrix(...) performs no GPU execution and returns immutable ResolvedCell descriptors. run_matrix(resolved_cells) executes only validated descriptors serially and produces a RunSummary. A runtime-placement change that does not change the binary must not force a new build identity; build-affecting projections remain with campaign planning.

Visibility contract: use the canonical core.environment.gpu_visibility_pair()/Host.gpu_visibility_env() resolver. Never independently reuse physical ordinals as HIP visibility values when ROCR filtering is active.

Evidence contract: cell orchestration status (executed, infrastructure_failed, skipped) remains separate from child verdict and performance_admitted. The matrix cannot upgrade a child verdict or turn diagnostic capture into production evidence. Per-cell and aggregate artifacts are immutable and identity-bound.

## Code Samples & Guidance



## Files

tools/bigcherry/campaign/runtime_matrix.py
tools/bigcherry/cli/runtime.py
tools/bigcherry/cli/main.py
tools/tests/campaign/test_runtime_matrix.py
tools/tests/cli/test_runtime_matrix_cli.py
docs/reference/testing/TEST.md

## Validation

Implemented and tested resolve_matrix()/run_matrix() plus the declarative runtime-matrix CLI. The adapter loads canonical config/environment.toml and config/models.toml, rejects unknown models before execution, writes immutable resolved-matrix.json, and delegates cells serially without a shell while propagating visibility and BIGCHERRY_RUNTIME_CELL_JSON. Focused validation: PYTHONPATH=tools python -m pytest tools/tests/cli/test_runtime_matrix_cli.py tools/tests/campaign/test_runtime_matrix.py -q (7 passed). Remaining: real maintained server-bench smoke on Brutus and any worker-specific adapter wiring beyond the generic existing-command delegate.

## Effort & Risk

M effort and risk. Existing execution and admission primitives exist; hazards are GPU visibility leakage, contention, identity drift, inventory timing, and diagnostic evidence being confused with production timing. Mitigate with full preflight, immutable descriptors, strict delegation, quiescence and TOCTOU checks.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md
docs/reference/tooling/TOOLING.md
docs/reference/testing/TEST.md
docs/reference/testing/MULTI_GPU_LARGE_MODEL_VALIDATION.md
Fail closed; canonical models.toml, environment.toml/BC_* and recipes.toml only; maintained server-bench only; no duplicate campaign engine; no concurrent GPU workloads; production and diagnostic roles distinct.

## Acceptance Criteria

- A configuration-only JSON matrix can be resolved through the CLI for a registered model, physical GPU set and runtime profile.
- Preflight rejects unknown models and invalid cells before worker launch.
- Existing worker commands are delegated serially with canonical visibility and immutable cell identity.
- resolved-matrix.json, atomic status.json and sanitized append-only events.jsonl are emitted for UI polling.
- Child non-zero exit/failure remains a failed matrix and cannot be upgraded by the adapter.
- Real maintained server-bench smoke remains required before final completion.

## Notes

Supersedes: HI170
Migration: capability-rebaseline-v3-2026-09
Successor key: run-hip-autotune-hi170

This is deliberately a thin runtime-placement layer, not a second campaign engine. Existing build/tune/ab-benchmark workers are injected as delegates; this layer cannot upgrade performance admission or child verdicts. The initial tune-cell inventory decision remains fail-closed until identity-bound inventory is available.

## Change Log

- 2026-09-09T10:49:53.272839+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:06:04.463091+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.913234+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T14:05:55.347465+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:standards, section:acceptance_criteria
- chg_20260909_140610_made-the-reusable-modelgputo_1377
- 2026-09-09T14:06:10.944350+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T15:45:07.241811+00:00 (updated-by): Updated: section:description, section:files, section:validation, section:notes
- chg_20260909_154517_added-reusable-ui-pollable-ru_5957
- 2026-09-09T15:45:18.008716+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T15:57:03.090273+00:00 (state-transition): State: pending → in_progress
- 2026-09-09T17:40:55.195470+00:00 (updated-by): Updated: section:description, section:files, section:validation, section:acceptance_criteria
- chg_20260909_174106_runtime-matrices-can-now-be-se_6137
- 2026-09-09T17:41:06.460978+00:00 (updated-by): Updated: section:ledger-events
