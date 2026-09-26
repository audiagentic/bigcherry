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

# Implement the BigCherry job domain, run store, Executor API, FakeExecutor and JSON CLI

## Description

Build the thin platform-neutral service layer over an `Executor`. This is the durable BigCherry authority for JobSpec, series/session/attempt identity, frozen scientific inputs, retry legality, event history and normalized status. It is not a scheduler: no custom worker daemon claims queue records and no SQLite lease/fencing scheduler is introduced. Slurm schedules Brutus; LocalExecutor schedules one local process; FakeExecutor drives deterministic tests.

Use filesystem durable records and the existing HI48 durability techniques (canonical JSON, fsync + atomic rename, checksummed append-only JSONL) as the initial run store. Slurm/jobcomp is execution history; the BigCherry store is domain history.

## Steps

1. Extract generic durability helpers from `tuning/journal.py` without changing its wire format.
2. Add typed jobs domain models and canonical hashing.
3. Add `Executor` protocol + `FakeExecutor`; leave Slurm/Local implementations in RCD03/RCD11.
4. Add filesystem `RunStore` with immutable intent/attempt manifests and append-only event stream.
5. Implement series creation/batch expansion/session-slot rules and composition freeze.
6. Implement failure/retry taxonomy and state projection from BigCherry records + executor status.
7. Add `bigcherry jobs` JSON CLI: submit, submit-batch, list, show, retry, disable, enable, cancel, hold, release, pause, resume, status, events.
8. Add offline tests using only FakeExecutor/temp directories.

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
    def allocation(self, handle: ExecutionHandle) -> Allocation | None: ...
    def events(self, handle: ExecutionHandle, *, after: int | None = None) -> Iterable[ExecutorEvent]: ...
```

`ExecutionStatus` is BigCherry-normalized: `queued|running|held|completed|cancelled|failed|unknown`. Native spellings never leave adapters.

`FakeExecutor` stores requests/handles in memory, honors dependency completion, can inject allocation, native failure and event sequences; it is the main offline scheduler test double.

### RunStore

Root:

```text
<work>/jobs/runs/<run_id>/
  intent.json
  events.jsonl
  attempts/001/{attempt.json,submission.json,result.json}
```

Later stage records extend this without changing run identity.

Functions:

```python
class RunStore:
    def create_series(self, spec: SeriesSpec) -> SeriesRecord: ...
    def create_run(self, spec: JobSpec) -> RunRecord: ...
    def begin_attempt(self, run_id: str, resolved: AttemptSpec) -> AttemptRecord: ...
    def record_submission(self, run_id: str, attempt_no: int, handle: ExecutionHandle) -> None: ...
    def finish_attempt(self, run_id: str, attempt_no: int, result: AttemptResult) -> None: ...
    def append_event(self, event: JobEvent) -> int: ...
    def read_events(self, *, after: int = 0) -> tuple[JobEvent, ...]: ...
```

`intent.json`/attempt/result are atomic writes. Existing immutable terminal records are never overwritten; retry makes a new attempt directory. `events.jsonl` is sequence/checksum/fsync capable and recoverable from only a torn final line.

No authoritative mutable `state` field is needed. `jobs show/status` derives state from intent, disable/hold controls, latest attempt record, and executor status.

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
- hold/release: maps to executor where supported; otherwise adapter reports unsupported.
- global pause/resume: BigCherry submit admission flag plus executor hold for pending managed executions; it does not kill running attempts.
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
- `test_cancel_preserves_attempt_history`
- `test_events_sequence_resume_after_n`
- `test_events_recovers_torn_final_record`
- `test_fake_executor_blocks_dependency_until_completed`
- `test_fake_executor_cancel_and_status_projection`
- `test_domain_modules_do_not_import_slurm`
- `test_windows_and_linux_environment_hashes_do_not_share_series`

Also run the lab planning simulator; production tests should supersede corresponding mock assertions as modules land.

## Effort & Risk

Large but isolated. Primary risk is accidentally building a second scheduler in the domain store. Acceptance forbids queue claiming/worker leases/native resource scheduling in `jobs/service.py`.

## Standards

Reuse HI48 durability semantics and existing BigCherry identity/provenance style. JSON stdout only for `--json`; diagnostics stderr; stable enum values.

## Acceptance Criteria

- FakeExecutor can exercise submit→dependency→run→retry/cancel without Slurm.
- Run/series/attempt history survives process restart from files alone.
- N and frozen composition cannot drift silently.
- scientific FAIL cannot trigger harness retry.
- CLI machine output is deterministic JSON/JSONL.
- no domain import of Slurm implementation.
- no custom scheduler daemon/database authority.

## Notes

SQLite may be introduced later for derived indexes/search if needed; it must never become a second scheduler or the only copy of scientific history.

## Change Log

- 2026-09-26T00:52:00.974407+00:00 (created-by): Created by agent
- 2026-09-26 (dev-gpt-agent): Resolved store/scheduler ambiguity; specified typed domain, durable files, Executor/FakeExecutor and offline tests.
