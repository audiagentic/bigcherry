---
id: RCD07
order: 7
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:52:12.716988+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Durable campaign operations, stage split and scheduler-isolation qualification

## Status — 2026-09-28

**Durable operation/rehydration substrate and deterministic graph compiler are implemented. Actual campaign execution is still monolithic; independent stage scheduling remains future work.**

Implemented:

- `tools/bigcherry/jobs/operations.py`:
  - immutable `OperationSpec`, `ArtifactBinding`, `OperationResult`;
  - separate `operation_spec_hash` and dependency-sensitive `execution_hash`;
  - immutable `operation.json` / `running.json` / `result.json`;
  - `running.json` without valid result rehydrates as interrupted, never success;
  - success reuse only after exact identity and output byte/hash verification.
- `tools/bigcherry/jobs/graph.py`:
  - deterministic cycle-checked DAG/topological order;
  - fixed validation graph compiler;
  - resource classes mapped without scheduling anything itself.
- tests cover spec/dependency hash domains, artifact escape/tamper, immutable result, interrupted-running state, missing dependency/cycle/order and resource mapping.

Current compiled graph:

```text
prepare
  -> correctness-activation
  -> timed-performance
  -> reference-ladder
  -> production-lane? 
  -> evidence-finalize
  -> harvest
  -> report
```

Resource policy remains conservative:

```text
prepare              build
correctness           required GPU(s) + host_activity:1
timed/reference/prod  required GPU(s) + host_activity:2
harvest/report        no GPU
```

Slurm remains execution/resource authority; these modules are durability/identity only and must never become a second scheduler.

## Remaining implementation

- compile a real managed series/attempt into stage execution requests and submit dependencies through `Executor`;
- extract current campaign preparation/correctness/performance/ladder boundaries without changing scientific behavior;
- carry exact verified dependency artifact bindings between stages;
- add typed producer prebuild/premeasure results and structured campaign progress events;
- prove restart/resume of one real stage and scientific parity against monolithic execution;
- run `scheduler-isolation-v1` before allowing any build overlap or active non-conflicting production during timed measurement.

## Reuse rule

A stage is reusable only when operation spec hash, execution hash, all dependency bindings, all output bytes/descriptors and required platform/hardware semantics match exactly and the terminal state is `succeeded`. Failed/interrupted/path-existence-only state is never success.

## Scheduler-isolation gate

Keep host-exclusive timed measurement until a predeclared A/A experiment proves a narrower policy. The standing design remains >=32 randomized paired blocks, paired log-ratio primary statistic, condition-effect CI within +/-0.10%, variance-ratio upper <=1.15, and no systematic clock/power/thermal shift. Do not tune bounds after results.

## Change log

- 2026-09-26: initial RCD01 activation/stage-split design.
- 2026-09-28: implemented durable operation identity/rehydration + deterministic resource graph; left executor wiring/campaign extraction/isolation as explicit future work.
