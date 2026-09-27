---
id: RCD04
order: 4
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:52:00.974407+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Implement the BigCherry durable job domain, service, Executor API and CLI

## Objective

Provide one scheduler-neutral application API for humans, agents and future UI/API adapters. BigCherry owns deterministic JobSpec/BatchSpec expansion, scientific series/session/attempt identity, immutable input/hardware binding, durable inbox/events/history, retry/control policy and review surfaces. Executors own process execution; Slurm owns Brutus scheduling/resources.

No custom scheduler daemon/database authority is introduced.

## Implementation status — 2026-09-27

Implemented and checked in:

- `tools/bigcherry/jobs/model.py`: versioned `BatchSpec`, `JobSpec`, `GpuRequirement`, `TargetPolicy`, canonical hashing and persistence rehydration.
- `executor.py`: scheduler-neutral `Executor` protocol, executor-local `ExecutionRequest`, normalized states/control, native vs stable allocation identity.
- `fake.py`: deterministic FakeExecutor with dependencies, hold/release/cancel/status/correlation/events.
- `store.py`: filesystem authority; immutable batch/series/run/attempt/submission records; client idempotency; pending/processing recovery; checksummed monotonic event JSONL with host OS lock.
- `service.py`: deterministic plan, exact series binding, durable submit, async ingest, crash reconciliation, retry/control/status/log/artifact/review surfaces.
- `identity.py`: focal/common/promoted patch identities, validation-package validity, model/corpus/file identities and attempt-start drift recheck.
- `workspace.py`: pinned workspace abstraction with git and deterministic test implementation.
- `runner.py`: validation-campaign argv rendered from typed job/attempt state; allocation-local `--device-map`; no persisted physical ordinal.
- `cli/jobs.py` + package entrypoint routing: `python -m bigcherry jobs ...`.
- `registry.py`: executor target registry for Slurm/Local/Remote.
- systemd renderer/installer in `tools/admin/install_bigcherry_jobs.py`.
- permanent `tools/tests/jobs` and `.github/workflows/jobs-service-validation.yml`.

CI already found and forced fixes for stale Slurm account/history assumptions, duplicate client submission timestamps, processing-receipt crash duplication, and missing production CLI scientific-identity wiring.

Current implementation is intentionally **not RCD04 complete** until the remaining blockers below are green.

## Remaining blockers

1. Freeze the public `JobService`/DTO contract for future HTTP/UI adapter use; CLI remains a consumer, not the API definition.
2. Add structured request audit (`request_id`, actor kind/id/transport) rather than only free-form actor text.
3. Complete queue/series/executor-doctor/hardware/report/evidence/harvest surfaces through RCD08/RCD09/RCD12 without shell scraping.
4. Add event segment/checkpoint/rotation before unbounded long-term operation.
5. Consume typed runner failure results; monolithic native Slurm requeue remains disabled.
6. Brutus stable-ID allocation attestation remains hardware-gated; native GPU IDs alone are insufficient.
7. Remote protocol is mocked but target-local staging/workspace semantics remain RCD11 work.
8. RCD08 must provide verified evidence/harvest state before `review_ready`; process completion alone is deliberately insufficient.

## Domain contract

Scientific intent is portable; `ExecutionRequest` is executor-local. cwd/command/log paths are not portable scientific identity.

Series identity is finalized only after:

1. scientific input content is frozen;
2. target executor/host/platform environment is resolved;
3. RCD12 binds one deterministic exact stable-device cohort.

Series material includes scientific identity hash, platform environment hash and hardware cohort hash. Every session references that exact immutable series record.

## Scientific identity freeze

`ProjectScientificIdentityResolver` freezes:

- focal patch implementation + validation digest + contract bindings;
- common patch identities;
- current `validated-enhancements` IDs/digests;
- model SHA256/size/path;
- producer corpus SHA256/size/path;
- producer-input file identity when file-backed;
- baseline source and producer identity.

Starting new managed work calls `require_execution_package()` for the focal patch. At attempt creation the pinned runner worktree resolves identity again; mismatch blocks and requires a new series. Queued promotions or patch/contract edits cannot silently mutate an existing series.

## Durable filesystem authority

```text
<work>/jobs/
  requests/
  batches/<batch>/batch.json
  series/<series>/series.json
  runs/<run>/intent.json
  runs/<run>/control.json
  runs/<run>/attempts/NNN/
    attempt.json
    submission-intent.json
    submission.json
    executor-start.json
    executor-result.json
    stdout.log
    stderr.log
  inbox/{pending,processing,accepted,rejected}/
  events.jsonl
```

Rules:

- canonical JSON and atomic replacement for mutable controls;
- immutable intent/attempt/submission/result records are not silently overwritten;
- host OS lock serializes event/control/idempotency mutation;
- event stream recovers only a torn/corrupt final record; earlier corruption fails closed;
- retry makes attempt N+1 while preserving run/session identity;
- state is projected from durable records + executor.

## Client idempotency

```text
same key + same canonical request -> exact existing durable batch, no new receipt/event
same key + changed request        -> IdempotencyConflict
new key                           -> new batch request
```

This protects human/agent retries after SSH/API uncertainty independently of executor submission idempotency.

## Inbox/recovery

Submit returns after durable local acceptance. systemd path/timer invokes `jobs ingest --once`.

Recovery order:

1. existing `processing/` receipts before new pending work;
2. existing attempt + submission -> rebind;
3. attempt + submission-intent but no submission -> executor correlation;
4. one match -> bind;
5. multiple -> ambiguity/wake, never duplicate;
6. proven zero -> submit immutable request once;
7. move receipt accepted/rejected after durable outcome.

## Executor contract

```python
class Executor(Protocol):
    name: str
    def submit(self, request: ExecutionRequest) -> ExecutionHandle: ...
    def correlate(self, execution_id: str) -> tuple[ExecutionHandle, ...]: ...
    def status(self, handle: ExecutionHandle) -> ExecutionStatus: ...
    def cancel(self, handle: ExecutionHandle) -> None: ...
    def control(self, handle: ExecutionHandle, action: ExecutorControl) -> None: ...
    def allocation(self, handle: ExecutionHandle) -> Allocation | None: ...
    def events(self, handle: ExecutionHandle, *, after: int | None = None): ...
```

`Allocation.native_gpu_ids` are scheduler/process-local only. `stable_gpu_ids` exist only after an executor/worker has attested RCD12 mapping.

## CLI implemented

```text
jobs plan SPEC
jobs validate SPEC
jobs submit SPEC --idempotency-key KEY [--actor ACTOR]
jobs ingest --once
jobs list
jobs show RUN
jobs status [RUN]
jobs events --after N [--follow] [--wake-only]
jobs logs RUN --stream stdout|stderr --offset N
jobs artifacts RUN
jobs review SERIES
jobs disable|enable|cancel|hold|release RUN
jobs retry RUN --same-commit|--latest
jobs pause|resume
jobs executors list
```

Machine output is JSON; follow events are JSONL. Reference: `docs/reference/jobs/JOBS_CONTROL_PLANE.md`.

## Required tests

Permanent CI must keep proving:

- canonical plan determinism;
- exact client idempotent replay + conflicting reuse rejection;
- processing receipt survives service crash without attempt duplication;
- submission-intent rebind and ambiguous-correlation fail-closed behavior;
- deterministic hardware binding and discovery-order independence;
- scientific identity frozen and attempt-start drift rejected;
- same-series sessions share exact stable cohort;
- retry same-commit/latest changes only attempt identity;
- disable/enable preserves session slot;
- event torn-tail recovery and monotonic sequence;
- no domain import of Slurm implementation;
- Windows/Linux environments remain separate series;
- FakeExecutor full plan -> submit -> ingest -> complete -> status/review;
- CLI construction uses production identity resolver and accepted inventory.

## Acceptance criteria

- deterministic durable submission usable by humans/agents without shell watcher loops;
- restart reconstructs from files + executor correlation;
- no duplicate external execution across known crash windows;
- exact scientific/hardware series identity frozen before sessions run;
- control/observation APIs do not expose raw Slurm schema;
- FakeExecutor covers full lifecycle without hardware;
- future UI/API can call JobService/domain DTOs without parsing CLI text;
- no custom scheduler daemon/database authority.

## Change log

- 2026-09-26: original durable domain/Executor design.
- 2026-09-27: model/store/service/executor/CLI/installer implementation added.
- 2026-09-27: scientific identity freeze + attempt-start recheck implemented.
- 2026-09-27: client idempotency and processing-receipt crash defects found by CI and corrected.
