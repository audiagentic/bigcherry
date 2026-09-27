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

## Status

**Core operator/agent observability is implemented and offline-tested; live Brutus hardware telemetry/gateway acceptance remains pending.**

Implemented:

- durable checksummed global event stream in `jobs/store.py`;
- reconnectable `jobs events --after N [--follow] [--wake-only]`;
- normalized `jobs status` and `jobs list/queue/show`;
- byte-offset `jobs logs ... [--follow]`;
- artifact lookup;
- versioned rebuildable status snapshot in `jobs/status.py`;
- Prometheus text projection and Markdown status;
- `tools/admin/snapshot_bigcherry_jobs.py` atomic snapshot writer;
- installer-generated `bigcherry-jobs-observe.service/.timer` every 15s;
- executor inventory inspection via `jobs executors list/show/doctor`;
- series/evidence/review/report surfaces from RCD08.

No HTTP/dashboard daemon was added. Future UI/API adapters consume the same `JobService`/snapshot/event schemas rather than parsing Slurm or CLI human text.

## Durable event model

`<work>/jobs/events.jsonl` is the canonical host BigCherry event stream.

Properties already implemented:

- monotonically increasing host sequence;
- immutable/checksummed records;
- multiprocess mutation serialized by host kernel file lock;
- final torn record recoverable;
- corruption before the tail fails closed;
- `run_id` filtering without a second authoritative per-run journal;
- severity `record|notify|wake` persisted with the event;
- submission, recovery, control and evidence-harvest events emitted after durable mutations.

`events --follow` tails durable records and reconnects from the last sequence. It is not process ownership or a scheduler watchdog.

## Status/API projection

`jobs status` without a run ID returns schema `bigcherry.jobs.status.v1` generated from authoritative run/event/executor state:

```text
generated_ns
last_event_seq
paused
queue.pending_receipts / processing_receipts
state_counts
jobs[]
series[]
executors[]
```

Each `series[]` includes planned/completed sessions, execution completeness, review readiness and evidence commit. The projection is disposable/rebuildable; deleting `status.json` loses no authority.

`jobs status RUN` returns the normalized per-run state. Native scheduler state is adapter input only; public users do not consume raw `squeue --json` schemas.

## Logs and artifacts

`jobs logs RUN --stream stdout|stderr --offset N --limit N` returns a versioned byte-window response with `next_offset` and EOF. `--follow` repeatedly advances that durable offset and exits after terminal state + EOF.

This avoids SSH path knowledge and gives a future HTTP/WebSocket adapter a stable range/tail primitive.

`jobs artifacts RUN` exposes known attempt outputs without making filesystem discovery part of the public API.

## Derived snapshots / monitoring

`tools/admin/snapshot_bigcherry_jobs.py` writes atomically:

```text
<jobs>/status/status.json
<jobs>/status/status.md
<jobs>/status/bigcherry.prom
```

The installer generates:

```text
bigcherry-jobs-observe.service
bigcherry-jobs-observe.timer
```

with a 15-second cadence. These files are projections only.

Current Prometheus projection includes:

- admission pause state;
- last durable event sequence;
- pending/processing durable receipts;
- run count by normalized state;
- review-ready series count.

Additional disk/ccache/GPU/process metrics may be added behind the same schema without changing scheduler authority.

## Wake semantics

Scientific FAIL is data, not an operational incident.

Wake remains reserved for conditions such as:

- ambiguous recovery/duplicate correlation;
- executor/service unavailability;
- scientific identity/contract/composition drift;
- inventory drift;
- disk hard limit;
- GPU/production contamination;
- production window failure/overrun;
- harvest verification failure;
- retry exhaustion/stall.

The CLI filters persisted severity (`--wake-only`); it does not infer severity from event names.

## Hardware/production telemetry boundary

Pure production-coexistence logic is implemented under `jobs/production.py` and mocked:

- exact UUID claims;
- conservative architecture claims;
- malformed/missing claim => fail closed;
- dynamic target conflict rather than static GPU class;
- config/target-use drift contamination;
- crash-safe measurement-window state machine.

What remains hardware-gated is the provider that supplies live stable-ID process/VRAM/temperature/power/clock/utilization observations on Brutus and the policy qualification deciding which telemetry may invalidate/retry a measurement. No result-derived performance outlier rule is introduced here; PVPS09 remains authoritative for that decision.

## Offline validation already implemented

The jobs-service suite proves:

- torn event-tail recovery;
- idempotent submission and crash-window recovery;
- normalized executor states;
- events/status/log/artifact interfaces;
- large stdout/stderr streaming without memory capture;
- reconnectable offset semantics;
- deterministic status/Markdown/Prometheus projection;
- systemd unit rendering;
- production coexistence/window pure state machine;
- hardware inventory/cohort mapping;
- evidence harvest/report visibility.

Earlier durability falsifiers additionally cover concurrent HI151 maintenance/admission races and durable submission recovery under process restart.

## Remaining acceptance

Before RCD09 is complete:

1. map live AMD telemetry/process attribution to accepted stable IDs on Brutus;
2. emit/observe inventory drift and production contamination wake events from real incidents;
3. prove an agent/gateway disconnect + reconnect from event sequence loses no durable notification;
4. force a harness failure and verify wake delivery while a scientific FAIL stays record/notify only;
5. verify status/metrics remain useful while Slurm is temporarily unavailable;
6. define/qualify event rotation/checkpointing before long-term file size requires it (not needed for initial cutover if bounded soak confirms acceptable size).

## Acceptance criteria

- no hand-written watcher loop required for submit/manage/observe/review;
- durable event reconnect is lossless from sequence N;
- status and logs are executor-independent public contracts;
- status/metrics are rebuildable projections, never authority;
- wake policy separates operational failure from scientific FAIL;
- live hardware/process telemetry uses stable IDs and cannot silently trigger result-driven retries;
- offline suite green and Brutus/gateway acceptance complete.

## Change log

- 2026-09-26: initial design.
- 2026-09-26: specified multiprocess-safe events/status/telemetry and wake semantics.
- 2026-09-27: implemented durable follow/status/log projections, Prometheus/Markdown snapshot timer, expanded human/agent CLI and mocked production coexistence; live GPU/gateway acceptance remains.
