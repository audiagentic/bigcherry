---
id: RCD10
order: 10
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:52:23.945064+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Accept v1, migrate active qualification work and retire the lab queue

## Objective

Retire the shell queue only after the managed service passes capability-derived real-hardware acceptance and soak. Acceptance is based on current accepted inventory and scientific requirements, never hard-coded GPU slots.

## Implementation status — 2026-09-27

Implemented/offline-tested:

- `tools/bigcherry/jobs/acceptance.py`: deterministic acceptance cases derived from BatchSpecs + accepted inventory; unsupported/missing capabilities remain explicit instead of being dropped.
- `python -m bigcherry jobs acceptance SPEC...`: emits `bigcherry.jobs.acceptance-matrix.v1`, exits 2 until every requested case is supported.
- `tools/bigcherry/jobs/migration.py`: fail-closed converter for the exact historical `run_campaign.sh` wrapper shape.
- `python -m bigcherry jobs migrate-legacy ...`: requires explicit planned N plus original `BC_MODEL`/`BC_HIP_PATH`; legacy GPU ordinal/run-name are metadata only.
- unknown one-off legacy flags refuse conversion rather than becoming arbitrary passthrough arguments.
- permanent tests cover capability support/absence/deduplication, slot-independent migration, unknown-flag refusal and no planned-N inference.
- operational procedures: `docs/reference/jobs/ACCEPTANCE.md` and `MIGRATION.md`.

Still real-host gated:

1. generate the matrix from the actual pending qualification BatchSpecs;
2. complete Brutus RCD03/RCD11/RCD12 GPU/production/cgroup gates;
3. force the incident/recovery matrix on Brutus;
4. complete at least one full predeclared series through submit -> execute -> committed harvest -> report with no shell watcher;
5. soak alternate required capability classes;
6. disable then archive old queue/watch/switch entrypoints only after the soak passes.

## Acceptance matrix semantics

A case includes target executor/platform, patch, architecture, model, producer, HIP/toolchain path, GPU count/min-VRAM/peer requirement, production-lane flag and common-patch composition. Device index/BDF/render node is not case identity.

Missing executor inventory, host/platform mismatch, absent architecture, insufficient VRAM, ambiguous homogeneous model selection or unavailable peer cohort yields `supported=false` + reason. Absent hardware is never simulated as accepted.

## Migration rules

For each pending shell-queue item:

- preserve legacy logs read-only;
- translate the human source to a canonical BatchSpec;
- re-resolve current contracts/composition/model/toolchain/hardware;
- declare planned sessions explicitly;
- submit as new managed work with a new idempotency key;
- never import `CAMPAIGN_EXIT=` or old shell logs as a completed service attempt/evidence record.

Unsupported legacy arguments require an explicit domain-model change or manual JobSpec conversion. JobSpec remains free of arbitrary shell passthrough.

## Required Brutus acceptance

At minimum cover every active architecture/toolchain/model class, single/multi-GPU peer shapes in use, producer/input/corpus variations and production-lane behavior. Force harness retry, submission-intent recovery, client disconnect, hold/release/cancel, branch/scientific drift, hardware drift, dirty harvest destination, controller restart, resource/disk preflight failure and production contamination/window recovery.

Scientific FAIL remains a normal completed experiment. No optional stopping or result-driven retry is introduced.

## Retirement/rollback

Retirement candidates after soak:

```text
tools/lab/plan-qualification/queue.sh
tools/lab/plan-qualification/run_campaign.sh
tools/lab/plan-qualification/make-serial-2.sh
hand-written watcher/switch scripts
```

`summarize.py`/`noise.py` may retire once the RCD08 report path has been compared on real equivalent evidence. Rollback means pause new managed submissions and stop managed scheduler execution before using the direct/manual campaign path; never run two schedulers against the same GPUs.

## Acceptance criteria

- offline jobs suite and real Noble scheduler reference remain green;
- every currently required Brutus capability is real-accepted or explicitly unsupported;
- incident/recovery behavior matches the runbook;
- one full planned series completes without shell queue/watcher intervention;
- committed evidence/report parity is confirmed;
- active operational docs no longer instruct agents to use the old queue before archival.

## Change log

- 2026-09-26: capability-derived acceptance/incident/soak design.
- 2026-09-27: acceptance-matrix and fail-closed legacy migration tooling implemented and offline-tested; hardware soak/retirement remains pending.
