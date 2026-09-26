---
id: RCD09
order: 9
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:52:20.106874+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Implement normalized observability, durable events, telemetry and agent wake semantics

## Description

Expose the job service as an observable service without adding a custom web daemon. BigCherry normalizes Slurm/Local execution state, durable run-store identity, hardware inventory, production-window state, disk/ccache and GPU telemetry into stable JSON/JSONL. Agents reconnect by event sequence; a gateway/client disconnect never owns process lifetime. Record scientific outcomes normally; wake only on operational conditions requiring action.

Use RCD04's single checksummed append-only host event stream and existing `telemetry.py` as the console integration point. GPU telemetry is evidence/context only until PVPS09 defines a result-independent health replacement rule.

## Steps

1. Define stable `JobEvent` schema, severity (`record|notify|wake`) and append/read/follow API.
2. Add event sinks: `JsonEventSink` + `CompositeSink` beside current console telemetry; never contaminate command JSON stdout.
3. Add `bigcherry jobs events --after N --jsonl [--follow] [--wake-only]`.
4. Add normalized `bigcherry jobs status --json`; adapters feed native state but raw Slurm schema is not public API.
5. Add host/GPU snapshot collector keyed by RCD12 stable `device_id`.
6. Add `bigcherry-observe.timer`/oneshot writer for `status.json`, `status.md`, `bigcherry.prom`; no bespoke HTTP service.
7. Add service-health/window/inventory-drift events from RCD03/RCD11/RCD12.
8. Test event resume/torn-tail/concurrent-writer/status rendering/telemetry redaction entirely offline.

## Detailed Solution & Technical Design

### Event schema

```python
@dataclass(frozen=True)
class JobEvent:
    schema: int
    seq: int
    event_id: str
    ts: str
    run_id: str | None
    series_id: str | None
    stage: str | None
    attempt: int | None
    execution: ExecutionHandle | None
    kind: str
    severity: Literal["record", "notify", "wake"]
    data: Mapping[str, object]
```

`seq` is a host BigCherry-event sequence, monotonic in the single canonical `<work>/jobs/events.jsonl`. `event_id` is globally unique and events are immutable/checksummed. Do not dual-write a second authoritative per-run event journal; run IDs in the host stream provide filtering, while portable run summaries may reference/copy events as derived artifacts later.

Multiprocess append is serialized with `core.host_lock.HostFileLock(<work>/jobs/events.lock)`:

```text
lock
  recover/validate complete tail
  read last durable sequence
  assign seq + 1
  append one complete checksummed JSON line
  flush + fsync
unlock
```

The event/run-store root must be local filesystem for this v1 lock/durability claim. A process crash releases the kernel lock; a torn final line is recovered under lock before the next sequence is allocated. Event persistence occurs before best-effort console/gateway notification. Failure to notify does not lose the event.

### Severity policy

Wake:

- `attempt.failed-harness`
- `attempt.retry-exhausted`
- `attempt.stalled`
- `host.disk-hard`
- `host.gpu-unhealthy`
- `host.inventory-drift`
- `job.contract-drift`
- `job.composition-drift`
- `executor.node-unavailable`
- `harvest.failed`
- `recovery.ambiguous`
- `service.degraded`
- `production.window-overrun`
- `production.restart-failed`
- `production.contamination`

Notify:

- `series.complete`
- optionally planned maintenance/window opened/closed.

Record only:

- submit/queued/held/released;
- attempt/stage start/completion;
- same-commit automatic requeue;
- telemetry;
- scientific PASS/FAIL;
- ladder/noise/production-lane result.

A scientific FAIL is not operational wake by default.

### Telemetry sinks

Current `console_telemetry()` remains human stderr. Add reusable sink protocol:

```python
class TelemetrySink(Protocol):
    def emit(self, kind: str, payload: Mapping[str, object]) -> None: ...

class JsonEventSink:
    def __init__(self, store: RunStore, context: EventContext): ...

class CompositeSink:
    def __init__(self, *sinks: TelemetrySink): ...
```

Do not log raw secrets/model prompts. Preserve current argv digest/redaction defaults.

### Status schema

`jobs status --json` returns BigCherry schema version and at least:

```text
service: executor health, Slurm services when applicable, production-window state
queue: logical runs and execution counts by normalized state/stage
host: root/work free bytes, ccache bytes/limit, load
hardware: accepted inventory hash + drift state
gpus[]: device_id, arch/model, current locator, allocated run/stage, telemetry
jobs[]: run/series/session, state/stage/attempt, identity hashes, execution handle,
        resource requirement/allocation, progress timestamp, result/verdict
```

No raw `squeue --json` object leaks into this schema. Unknown native state becomes normalized `unknown` plus a wake/service-degraded event, not a guessed mapping.

### Snapshot/Prometheus output

`bigcherry-observe.service` runs:

```text
bigcherry jobs status --json -> atomic status.json
bigcherry jobs status --format markdown -> atomic status.md
bigcherry jobs metrics -> atomic bigcherry.prom
```

every 15-30s via timer. Prometheus textfile metrics include counts/state/failures/retries, disk/ccache, GPU allocation/temp/power/clock/VRAM/util where available, event age and planned/valid session counts. Device label is stable ID (or bounded short digest), not slot index.

### GPU telemetry

Define provider-neutral sample:

```python
@dataclass(frozen=True)
class GpuTelemetrySample:
    device_id: str
    ts: str
    utilization_pct: float | None
    vram_used_bytes: int | None
    core_clock_mhz: float | None
    memory_clock_mhz: float | None
    temperature_c: float | None
    power_w: float | None
```

Linux provider can use the same AMD tooling abstraction as RCD12; Windows provider may expose subset. Missing fields are null, never zero.

Telemetry can flag a predeclared host-health rule only after PVPS09; no `-30%` performance-derived retry.

## Code Samples & Guidance

`events --follow` is a tailer of durable events: poll/read appended complete records and resume from last seq after reconnect. It is not an execution watchdog.

JSON command rule: stdout contains exactly the requested JSON/JSONL payload; warnings/logging stderr.

## Files

Planned:

- `tools/bigcherry/jobs/events.py`
- `tools/bigcherry/jobs/status.py`
- `tools/bigcherry/jobs/metrics.py`
- `tools/bigcherry/jobs/telemetry.py`
- `tools/bigcherry/core/host_lock.py`
- `tools/bigcherry/telemetry.py` (sink extension)
- `tools/bigcherry/cli/jobs.py`
- `config/systemd/bigcherry-observe.service`
- `config/systemd/bigcherry-observe.timer`
- `tools/tests/jobs/test_events.py`
- `tools/tests/jobs/test_status.py`
- `tools/tests/jobs/test_metrics.py`

## Validation

Offline:

- append 100 events, reconnect at sequence N, receive exactly N+1...;
- spawn concurrent writer processes; no duplicate/gap sequences and all checksums validate;
- kill a writer while appending; torn-tail recovery under lock preserves complete prefix and next sequence is correct;
- duplicate/corrupt non-tail sequence rejected;
- `--wake-only` filters severity, not event kind hard-coded in CLI;
- scientific FAIL remains record-only;
- raw native Slurm state fixture maps deterministically or `unknown`;
- status remains valid with executor unavailable and marks service degraded;
- telemetry absent fields render null;
- device BDF/index change while stable ID remains does not change status identity;
- secret-looking argv/env values are not emitted;
- Prometheus text output is deterministic and label-safe;
- atomic status files are never partially readable.

Integration with FakeExecutor: submit/dependency/start/fail/retry/complete sequence produces expected event order without hardware/Slurm.

Hardware acceptance: forced harness failure wakes a remote `events --follow`; live AMD telemetry maps to the allocated stable IDs.

## Effort & Risk

Medium. Main risk is making observability another authority or corrupting event sequence under multiprocess writers. Status/metrics are projections only; run-store/executor/evidence remain authoritative.

## Standards

RQW01 console telemetry semantics; HI48 durable sequence/checksum semantics; RCD04 domain schema and host-lock primitive.

## Acceptance Criteria

- agents need no hand-written watcher loops;
- reconnect from `--after` loses no durable event;
- concurrent writers cannot duplicate event sequence;
- normalized status is executor-independent;
- wake policy does not confuse scientific FAIL with harness failure;
- telemetry cannot trigger result-driven retry;
- no custom dashboard daemon is required.

## Notes

AudiAgentic gateway consumes this protocol over SSH and remembers sequence; it is not an executor owner.

## Change Log

- 2026-09-26T00:52:20.106874+00:00 (created-by): Created by agent
- 2026-09-26 (dev-gpt-agent): Specified multiprocess-safe durable event/status/telemetry projections and gateway-safe wake semantics.
