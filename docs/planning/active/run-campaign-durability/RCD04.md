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

A scientific series must already be bound to one accepted hardware cohort before any session is submitted. RCD12 therefore resolves the capability request to a deterministic `SeriesGpuBinding` during series creation; hardware cohort is not discovered opportunistically from whichever GPU Slurm later allocates.

## Steps

1. Extract generic durability helpers from `tuning/journal.py` without changing its wire format; add a host-local OS file-lock primitive for multiprocess serialization.
2. Add typed jobs domain models and canonical hashing.
3. Add `Executor` protocol + `FakeExecutor`; leave Slurm/Local implementations in RCD03/RCD11.
4. Add filesystem `RunStore` with immutable series/run/attempt/submission/result manifests and one append-only host event stream.
5. Implement durable inbox ingestion: validate/canonicalize -> persist -> enqueue record atomically -> asynchronous systemd oneshot submits/reconciles executions.
6. Implement batch expansion and **series binding**: freeze scientific inputs + platform environment, call RCD12 `bind_gpu_requirement()` exactly once per series, compute final `series_id`, then create all planned session slots carrying the same binding.
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
    model: str | None = None
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

`JobSpec` contains only named dimensions supported by `validation_campaign`: run/patch/producer/GPU requirement/session/planned N/contract/model/toolchain/baseline/common patches/producer inputs/corpus/production lane/code ref+policy/composition policy/priority/stall timeout. No `extra_args`.

### Series draft -> bound series

Do not compute a final `series_id` from an unresolved architecture-only request.

```python
@dataclass(frozen=True)
class SeriesDraft:
    scientific_request_hash: str
    planned_sessions: int
    contract_hash: str
    base_revision: str
    focal_patch: FrozenPatch
    common_patches: tuple[FrozenPatch, ...]
    validated_patches: tuple[FrozenPatch, ...]
    model_identity: FileIdentity
    corpus_identity: FileIdentity | None
    producer_inputs_hash: str
    platform_environment_hash: str
    gpu_requirement: GpuRequirement | None

@dataclass(frozen=True)
class SeriesRecord:
    series_id: str
    draft_hash: str
    gpu_binding: SeriesGpuBinding | None
    hardware_cohort_hash: str | None
    accepted_inventory_hash: str | None
    planned_sessions: int
    # plus frozen scientific fields above
```

Series creation algorithm:

1. validate/freeze contract/base/focal/common/validated/model/corpus/producer inputs;
2. resolve `platform_environment_hash`;
3. if GPU work is required, load accepted RCD12 inventory and call `bind_gpu_requirement(draft.gpu_requirement, inventory)`;
4. fail before creating session records if capability binding is ambiguous/unsatisfied;
5. compute final `series_id` from scientific draft + platform environment + exact `hardware_cohort_hash`;
6. persist immutable `SeriesRecord` including selected stable IDs and accepted inventory hash;
7. create exactly `planned_sessions` run slots, each referencing that same series/binding.

A later hardware replacement/topology drift never rewrites this record. The existing series becomes blocked; a new series is required. This is necessary because card-to-card variance is not assumed negligible under the small-wins policy.

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

`ExecutionStatus` is BigCherry-normalized: `queued|running|held|completed|cancelled|failed|unknown`. Native spellings never leave adapters. Unsupported controls raise `ExecutorCapabilityError`.

`FakeExecutor` stores requests/handles in memory, honors dependency completion, supports hold/release/cancel, and can inject allocation, native failure and event sequences.

### Allocation verification

Before a GPU campaign process starts, the executor allocation is checked against the immutable series binding:

```python
selected = verify_series_allocation(series.gpu_binding, allocation, accepted_inventory)
```

The verifier must prove every bound stable device is inside the external allocation and the accepted inventory/topology still matches. Slurm allocation may be wider than the selected cohort only for the RCD12 safe-overallocation case. A different allocated card is never silently substituted.

### RunStore

Root:

```text
<work>/jobs/
  events.jsonl
  events.lock
  inbox/{pending,processing,accepted,rejected}/
  series/<series_id>/series.json
  runs/<run_id>/
    intent.json
    attempts/001/
      attempt.json
      submission-intent.json
      submission.json
      result.json
```

Functions:

```python
class RunStore:
    def create_series(self, record: SeriesRecord) -> SeriesRecord: ...
    def create_run(self, spec: JobSpec, series: SeriesRecord) -> RunRecord: ...
    def begin_attempt(self, run_id: str, resolved: AttemptSpec) -> AttemptRecord: ...
    def record_submission_intent(self, run_id: str, attempt_no: int, request: ExecutionRequest) -> None: ...
    def record_submission(self, run_id: str, attempt_no: int, handle: ExecutionHandle) -> None: ...
    def finish_attempt(self, run_id: str, attempt_no: int, result: AttemptResult) -> None: ...
    def append_event(self, event: JobEvent) -> int: ...
    def read_events(self, *, after: int = 0) -> tuple[JobEvent, ...]: ...
```

`series.json`, `intent.json`, attempt/submission/result are atomic writes. Existing immutable terminal records are never overwritten; retry makes a new attempt directory. `events.jsonl` is the single canonical host event stream. Append is serialized with `HostFileLock(<work>/jobs/events.lock)`: lock -> validate/read tail sequence -> allocate next sequence -> write complete checksummed record -> flush+fsync -> unlock. Storage must be local filesystem for this v1 lock/durability claim.

No authoritative mutable `state` field is needed. `jobs show/status` derives state from immutable records, disable/hold controls, latest attempt record and executor status.

### Durable service / inbox

`jobs submit` is a record operation:

1. parse/validate JobSpec/BatchSpec;
2. build `SeriesDraft`, resolve exact RCD12 series hardware binding and final `series_id`;
3. atomically persist immutable series + run intent(s);
4. atomically create canonical inbox receipt under `pending/`;
5. append `job.submitted` event;
6. return JSON after durable local acceptance, without waiting for `sbatch`.

Systemd:

```text
bigcherry-jobs-ingest.path    watches pending/
bigcherry-jobs-ingest.service Type=oneshot, runs `bigcherry jobs ingest --once --json`
```

The oneshot takes a host-local ingest lock, atomically renames pending files to `processing/`, creates an attempt when eligible, submits/reconciles it, and moves receipt to accepted/rejected. A crash leaves durable processing state for the next invocation. A periodic low-rate timer may invoke the same oneshot as filesystem-notification safety net; it is not a scheduler loop.

### Submission ambiguity / idempotency

Before calling an external executor, persist `submission-intent.json` containing stable `execution_id`, exact request hash and attempt identity. Slurm submission carries correlation identity. After native submit returns, persist `submission.json`.

Recovery when intent exists without submission result:

1. query active executor state by correlation identity where supported;
2. inspect configured completion history and attempt-local runner start/result sentinels;
3. exactly one match -> bind handle;
4. proven no execution -> resubmit immutable request;
5. ambiguous -> emit `recovery.ambiguous`, wake, do not duplicate.

### Retry rules

```text
exit 0  -> terminal pipeline completion; read scientific verdict separately
exit 75 -> same attempt commit/config may Slurm-requeue
exit 76 -> terminal harness failure; explicit retry creates attempt N+1 and may resolve latest code
exit 77 -> invalid/contract/composition drift; block until explicit correction
other   -> FailureKind; default fail closed to new-attempt/manual review
```

Scientific FAIL is never `FailureKind` and never creates another session.

### Disable/hold/pause

- disabled: orthogonal BigCherry admission control; preserves run/session slot and prevents new attempt submission;
- hold/release: `Executor.control`; Slurm/Fake implement, Local may report unsupported;
- global pause/resume: BigCherry admission flag plus executor hold for pending managed executions where supported; does not kill running attempts;
- cancel: terminal operator action; calls executor cancel then records history.

## Code Samples & Guidance

Canonical JSON rejects NaN, unordered ambiguity and unknown fields. Scientifically relevant file paths are resolved to content identities before series freeze.

Batch expansion/binding:

```python
def expand_batch(batch: BatchSpec) -> tuple[JobSpec, ...]: ...
def build_series_draft(specs: tuple[JobSpec, ...]) -> SeriesDraft: ...
def bind_series(draft: SeriesDraft, inventory: HardwareInventory) -> SeriesRecord: ...
def instantiate_session_runs(series: SeriesRecord, specs: tuple[JobSpec, ...]) -> tuple[RunRecord, ...]: ...
```

All specs assigned to one series must agree on every series-scoped field. Reject duplicate session numbers, missing planned slots when completeness is evaluated, or `session > planned_n`.

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
- `test_series_binding_is_independent_of_inventory_discovery_order`
- `test_series_sessions_share_identical_selected_device_ids`
- `test_series_binding_ambiguous_mixed_model_requires_model_or_exact_ids`
- `test_series_hardware_disappears_blocks_instead_of_rebinding`
- `test_allocation_must_contain_bound_stable_device_ids`
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

Also run the lab planning simulator; permanent RCD12 tests supersede its simplified candidate-order behavior before implementation acceptance.

## Effort & Risk

Large but isolated. Primary risks are accidentally building a second scheduler, duplicate submission across a crash boundary, and mixing physical cards in a statistical series. Exact pre-series hardware binding deliberately favors scientific comparability over GPU utilization.

## Standards

Reuse HI48 durability semantics and existing BigCherry identity/provenance style. JSON stdout only for `--json`; diagnostics stderr; stable enum values. RCD12 is authoritative for device identity/capability binding.

## Acceptance Criteria

- submit may exit immediately after durable acceptance; systemd ingestion still processes records;
- FakeExecutor exercises submit/dependency/run/retry/hold/cancel without Slurm;
- run/series/attempt history survives process restart from files alone;
- final `series_id` includes an exact deterministic hardware cohort before session submission;
- every run in a series carries the same RCD12 binding and cannot silently move to a different card;
- N/frozen composition cannot drift silently;
- scientific FAIL cannot trigger harness retry;
- submission crash recovery cannot blindly duplicate an external execution;
- global event sequences remain unique under multiprocess writers;
- CLI machine output deterministic JSON/JSONL;
- no domain import of Slurm implementation;
- no custom scheduler daemon/database authority.

## Notes

SQLite may be introduced later for derived indexes/search if needed; it must never become a second scheduler or the only copy of scientific history.

## Change Log

- 2026-09-26T00:52:00.974407+00:00 (created-by): Created by agent
- 2026-09-26 (dev-gpt-agent): Resolved store/scheduler ambiguity; specified durable record ingestion, submission recovery, typed domain, Executor/FakeExecutor and offline tests.
- 2026-09-26 (dev-gpt-agent): Adversarial follow-up: moved hardware cohort binding before final series creation; all sessions now target one deterministic stable-device cohort.
