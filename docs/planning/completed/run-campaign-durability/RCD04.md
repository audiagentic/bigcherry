---
id: RCD04
order: 4
plan: run-campaign-durability
state: superseded
created-at: '2026-09-26T00:52:00.974407+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Implement the BigCherry durable job domain, service, Executor API and CLI

## Description



## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

Software criteria are met when jobs CI remains green for deterministic planning, idempotency, crash recovery, exact scientific/hardware identity, retry/control, harvest/report and normalized observability. Production cutover additionally requires RCD10/RCD11/RCD12 host acceptance.

## Notes

Consolidation disposition (2026-11-20): superseded into active RCD02 after RV4246. RCD02 Steps/Validation/Acceptance now carry the outstanding software hardening, host-specific acceptance, and future evolution requirements from this item. This is a planning-tracker consolidation only; it does not claim completion, tests, hardware acceptance, or production cutover. Preserve this record and its historical evidence/ledger references.

## Status — 2026-09-28

**Core v1 service is implemented and offline-CI green. Remaining items are hardening/extensions, not blockers to the scheduler-neutral domain model.**

Implemented under `tools/bigcherry/jobs/`:

- versioned portable batch/job/GPU/target model and canonical hashing;
- `Executor` protocol plus Slurm, Local, Remote and Fake adapters;
- filesystem authority for requests/batches/series/runs/attempts, durable inbox and checksummed event JSONL;
- client idempotency and submit-intent -> correlate/rebind crash recovery;
- exact scientific identity + accepted hardware cohort freeze and attempt-start recheck;
- pinned workspaces, typed runner, stable-ID allocation attestation and monitored execution;
- queue/list/show/status/events/logs/artifacts/series/review controls;
- executor doctor, hardware CLI, evidence, verified harvest, report and metrics;
- systemd ingest/observe units; no custom scheduling daemon/database authority;
- review readiness requires planned sessions plus committed verified evidence.

Public application authority is `JobService`; CLI is a consumer. Slurm-native IDs/schema remain adapter metadata only.

## Durable lifecycle

```text
client idempotency key
  -> durable request/batch/series/session run
  -> immutable attempt + submission-intent
  -> executor correlation/submit
  -> attempt-local start/result/logs/attestation
  -> verified harvest
  -> report/review-ready
```

Retries create attempt N+1. Monolithic v1 does not native-requeue.

## Remaining hardening / extension work

- structured actor/transport audit beyond the current actor string;
- event segmentation/checkpoint/rotation for very long-lived installations;
- retention/cleanup policy for harvested worktrees and non-evidence logs;
- target-local source/model/corpus staging before RemoteExecutor is enabled for Windows scientific work;
- dev/Windows hardware acceptance.

These do not justify a second scheduler, database, or Slurm leakage into domain DTOs.

## Change log

- 2026-09-27: durable service/executor/CLI/installer implementation established.
- 2026-09-28: reconciled implemented harvest/report/hardware/doctor surfaces and narrowed remaining work to hardening/remote/hardware acceptance.

## Change Log

- 2026-11-20T02:52:39.807095+00:00 (updated-by): Updated: section:notes
- 2026-11-20T02:52:47.500370+00:00 (state-transition): State: pending → superseded
