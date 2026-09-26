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

# Implement the BigCherry job domain, durable record service, Executor API, FakeExecutor and JSON CLI

## Description

Build the thin platform-neutral service layer over an `Executor`. This is the durable BigCherry authority for JobSpec, series/session/attempt identity, frozen scientific inputs, retry legality, event history and normalized status. It is not a scheduler: no custom worker daemon claims resource leases and no SQLite scheduler is introduced. Slurm schedules Brutus; LocalExecutor runs one local process; FakeExecutor drives deterministic tests.

Operators/agents add durable records and return immediately; a systemd path/oneshot ingester processes pending records automatically. Use filesystem durable records and the existing HI48 durability techniques (canonical JSON, fsync + atomic rename, checksummed append-only JSONL) as the initial run store. Slurm/jobcomp is execution history; the BigCherry store is domain history.

## Steps

1. Extract generic durability helpers from `tuning/journal.py` without changing its wire format; add a host-local OS file-lock primitive for multiprocess serialization.
2. Add typed jobs domain models and canonical hashing.
3. Add `Executor` protocol + `FakeExecutor`; leave Slurm/Local implementations in RCD03/RCD11.
4. Add filesystem `RunStore` with immutable intent/attempt/submission/result manifests and one append-only host event stream.
5. Implement durable inbox ingestion: validate/canonicalize -> persist -> enqueue record atomically -> asynchronous systemd oneshot submits/reconciles executions.
6. Implement series creation/batch expansion/session-slot rules and composition freeze.
7. Implement failure/retry taxonomy and state projection from BigCherry records + executor status.
8. Add `bigcherry jobs` JSON CLI: submit, submit-batch, list, show, retry, disable, enable, cancel, hold, release, pause, resume, status, events.
9. Add submission-idempotency/recovery protocol so a client/service crash around `sbatch` cannot blindly duplicate an execution.
10. Add offline tests using only FakeExecutor/temp directories and mocked executor responses.

## Detailed Solution & Technical Design

### Core models

```python
@dataclass(frozen=True)
class GpuRequirement:
    architecture: str
    count: int = 1
    min_vram_bytes: int = 0
    homogeneous_model: bool = True
    require_peer_access: bool = False
    exact_device_ids: tuple[str, ...] = ()

@dataclass(frozen=True)
class ResourceRequest:
    cpu_slots: int
    gpu: GpuRequirement | None
    activity_class: Literal["build", "correctness", "timed-measure", "harvest"]
    memory_bytes: int | None
    timeout_seconds: int

@dataclass(frozen=True)
class ExecutionRequest:
    execution_id: str
    command: tuple[str, ...]
    cwd: str
    env: tuple[tuple[str, str], ...]
    stdout_path: str
    stderr_path: str
    resources: ResourceRequest
    dependencies: tuple[str, ...] = ()

@dataclass(frozen=True)
class ExecutionHandle:
    executor: str
    native_id: str
    execution_id: str

class ExecutorControl(str, Enum):
    HOLD = "hold"
    RELEASE = "release"
```

`JobSpec` contains only named dimensions supported by `validation_campaign`: run/series/patch/producer/architecture capability/session/planned N/contract/model/toolchain/baseline/common patches/producer inputs/corpus/production lane/code ref+policy/composition policy/priority/stall timeout. No `extra_args`.

Resolved series stores:

- planned session count;
- contract hash;
- base revision;
- focal implementation digest;
- ordered common patch IDs+digests;
- exact frozen validated patch IDs+digests;
- model/corpus/producer-input file identities;
- platform environment hash;
- hardware cohort hash once allocation is resolved;
- composition policy (`freeze` default).

`series_id` is deterministic from these scientific fields excluding session ordinal. `run_id` is unique per series/session. `attempt_no` is monotonically increasing within run.

### Executor boundary

```python
class Executor(Protocol):
    def submit(self, request: ExecutionRequest) -> ExecutionHandle: ...
    def status(self, handle: ExecutionHandle) -> ExecutionStatus: ...
    def cancel(self, handle: ExecutionHandle) -> None: ...
    def control(self, handle: ExecutionHandle, action: ExecutorControl) -> None: ...
    def allocation(self, handle: ExecutionHandle) -> Allocation | None: ...
    def events(self, handle: ExecutionHandle, *, after: int | None = None) -> Iterable[ExecutorEvent]: ...
```

`ExecutionStatus` is BigCherry-normalized: `queued|running|held|completed|cancelled|failed|unknown`. Native spellings never leave adapters. Unsupported controls raise a typed `ExecutorCapabilityError`; domain code does not guess an equivalent.

`FakeExecutor` stores requests/handles in memory, honors dependency completion, supports hold/release/cancel, and can inject allocation, native failure and event sequences; it is the main offline scheduler test double.

### RunStore

Root:

```text
<work>/jobs/
  events.jsonl
  events.lock
  inbox/{pending,processing,accepted,rejected}/
  runs/<run_id>/
    intent.json
    attempts/001/
      attempt.json
      submission-intent.json
      submission.json
      result.json
```

Later stage records extend this without changing run identity.

Functions:

```python
class RunStore:
    def create_series(self, spec: SeriesSpec) -> SeriesRecord: ...
    def create_run(self, spec: JobSpec) -> RunRecord: ...
    def begin_attempt(self, run_id: str, resolved: AttemptSpec) -> AttemptRecord: ...
    def record_submission_intent(self, run_id: str, attempt_no: int, request: ExecutionRequest) -> None: ...
    def record_submission(self, run_id: str, attempt_no: int, handle: ExecutionHandle) -> None: ...
    def finish_attempt(self, run_id: str, attempt_no: int, result: AttemptResult) -> None: ...
    def append_event(self, event: JobEvent) -> int: ...
    def read_events(self, *, after: int = 0) -> tuple[JobEvent, ...]: ...
```

`intent.json`/attempt/submission/result are atomic writes. Existing immutable terminal records are never overwritten; retry makes a new attempt directory. `events.jsonl` is the single canonical host event stream. Append is serialized with `HostFileLock(<work>/jobs/events.lock)`: lock -> validate/read tail sequence -> allocate next sequence -> write complete checksummed record -> flush+fsync -> unlock. Storage is required to be local filesystem for this v1 lock/durability claim.

No authoritative mutable `state` field is needed. `jobs show/status` derives state from intent, disable/hold controls, latest attempt record, and executor status.

### Durable service / inbox

`jobs submit` is a record operation, not a long-running remote execution request:

1. parse and validate JobSpec;
2. resolve/freeze submission-time fields that belong to the series, not attempt-start mutable code;
3. atomically create run intent and a canonical inbox record under `pending/` using temp+fsync+rename;
4. append `job.submitted` event;
5. return JSON `{run_id, series_id, state:"queued"}` after durable local acceptance, without waiting for `sbatch`.

Systemd:

```text
bigcherry-jobs-ingest.path    watches pending/
bigcherry-jobs-ingest.service Type=oneshot, runs `bigcherry jobs ingest --once --json`
```

The oneshot takes a host-local ingest lock, atomically renames one/more pending files to `processing/`, creates the attempt when eligible, submits/reconciles it, and moves the inbox receipt to `accepted/` or `rejected/`. A crash leaves durable processing state for the next invocation; it never relies on the SSH/gateway process remaining alive.

A periodic low-rate timer MAY invoke the same `ingest --once` as a lost-filesystem-notification safety net; it is not a custom scheduling loop. Slurm remains the queue/resource scheduler.

### Submission ambiguity / idempotency

Before calling an external executor, persist `submission-intent.json` containing the stable `execution_id`, exact request hash and attempt identity. Slurm submission includes a correlation tag (`--comment=bigcherry:<execution_id>`; stable job name is secondary). After `sbatch --parsable` returns, persist `submission.json` with native ID.

Recovery when intent exists without submission result:

1. query active executor state by correlation identity where supported;
2. inspect configured completion history and attempt-local runner start/result sentinels;
3. if exactly one matching execution is proven, bind its handle and continue;
4. if no execution is proven, resubmit the same immutable request;
5. if state is ambiguous (e.g. executor unavailable/history inconclusive), emit `recovery.ambiguous`, wake, and **do not submit a duplicate** until reconciled.

FakeExecutor exposes deterministic correlation lookup so these cases are testable without Slurm.

### Retry rules

```text
exit 0  -> terminal pipeline completion; read scientific verdict separately
exit 75 -> same attempt commit/config may Slurm-requeue; record restart number
exit 76 -> terminal harness failure; explicit retry creates attempt N+1 and may resolve latest code
exit 77 -> invalid/contract/composition drift; block until explicit correction
other   -> classify through FailureKind; default fail closed to new-attempt/manual review
```

Scientific FAIL is never `FailureKind`.

### Disable/hold/pause

- disabled: orthogonal BigCherry admission control; preserves run/session slot and prevents new attempt submission.
- hold/release: `Executor.control`; Slurm implements native hold/release, Local may report unsupported, Fake implements both.
- global pause/resume: BigCherry submit/ingest admission flag plus executor hold for pending managed executions where supported; it does not kill running attempts.
- cancel: terminal operator action; calls executor cancel then records event/result without deleting history.

## Code Samples & Guidance

Canonical spec JSON must reject NaN, unordered dict ambiguity and unknown fields. File paths in submitted specs are resolved to file identities before series freeze where scientifically relevant.

Batch expansion is pure:

```python
def expand_batch(batch: BatchSpec) -> tuple[JobSpec, ...]: ...
```

and must reject duplicate `(series_id, session)` or `session > planned_n`.

## Files

Planned:

- `tools/bigcherry/core/durable.py`
- `tools/bigcherry/core/host_lock.py`
- `tools/bigcherry/jobs/__init__.py`
- `tools/bigcherry/jobs/model.py`
- `tools/bigcherry/jobs/executor.py`
- `tools/bigcherry/jobs/fake.py`
- `tools/bigcherry/jobs/store.py`
- `tools/bigcherry/jobs/series.py`
- `tools/bigcherry/jobs/batch.py`
- `tools/bigcherry/jobs/failure.py`
- `tools/bigcherry/jobs/events.py`
- `tools/bigcherry/jobs/service.py`
- `tools/bigcherry/cli/jobs.py`
- `config/systemd/bigcherry-jobs-ingest.path`
- `config/systemd/bigcherry-jobs-ingest.service`
- `config/systemd/bigcherry-jobs-ingest.timer` (optional notification-loss safety net)
- `tools/tests/jobs/`

## Validation

Required offline tests:

- `test_job_spec_canonical_hash_order_independent`
- `test_batch_expansion_applies_arch_model_toolchain_overrides`
- `test_series_rejects_session_above_planned_n`
- `test_series_rejects_duplicate_session_slot`
- `test_series_freezes_exact_validated_ids_and_digests`
- `test_scientific_fail_is_completion_not_harness_failure`
- `test_exit75_keeps_attempt_and_commit`
- `test_exit76_new_attempt_preserves_run_id`
- `test_retry_latest_changes_commit_only_between_attempts`
- `test_disable_enable_preserves_session_slot`
- `test_hold_release_calls_executor_control`
- `test_cancel_preserves_attempt_history`
- `test_events_sequence_resume_after_n`
- `test_events_recovers_torn_final_record`
- `test_concurrent_event_append_has_no_duplicate_sequence`
- `test_fake_executor_blocks_dependency_until_completed`
- `test_fake_executor_cancel_hold_release_and_status_projection`
- `test_submit_returns_after_durable_inbox_record`
- `test_client_exit_after_submit_does_not_cancel_execution`
- `test_ingest_recovery_is_idempotent`
- `test_submission_intent_rebinds_matching_native_execution`
- `test_ambiguous_submit_does_not_duplicate_execution`
- `test_domain_modules_do_not_import_slurm`
- `test_windows_and_linux_environment_hashes_do_not_share_series`

Also run the lab planning simulator; production tests should supersede corresponding mock assertions as modules land.

## Effort & Risk

Large but isolated. Primary risks are accidentally building a second scheduler and duplicate submission across a crash boundary. Acceptance forbids queue resource claiming/worker leases/native scheduling in `jobs/service.py`; it only ingests durable records and delegates execution ownership to the Executor.

## Standards

Reuse HI48 durability semantics and existing BigCherry identity/provenance style. JSON stdout only for `--json`; diagnostics stderr; stable enum values.

## Acceptance Criteria

- an SSH/gateway submit may exit immediately after durable acceptance; systemd ingestion still processes the record;
- FakeExecutor can exercise submit→dependency→run→retry/hold/cancel without Slurm;
- Run/series/attempt history survives process restart from files alone;
- N and frozen composition cannot drift silently;
- scientific FAIL cannot trigger harness retry;
- submission crash recovery cannot blindly duplicate an external execution;
- global event sequences remain unique under multiprocess writers;
- CLI machine output is deterministic JSON/JSONL;
- no domain import of Slurm implementation;
- no custom scheduler daemon/database authority.

## Notes

SQLite may be introduced later for derived indexes/search if needed; it must never become a second scheduler or the only copy of scientific history.

## Change Log

- 2026-09-26T00:52:00.974407+00:00 (created-by): Created by agent
- 2026-09-26 (dev-gpt-agent): Resolved store/scheduler ambiguity; specified durable record ingestion, submission recovery, typed domain, Executor/FakeExecutor and offline tests.
