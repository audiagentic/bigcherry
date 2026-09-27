---
id: RCD02
order: 2
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:51:52.243021+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Freeze the job-service architecture and implementation boundaries

## Status — 2026-09-28

**Architecture frozen; software implementation exists. Remaining gates are host/hardware acceptance, not platform-choice objections.**

Normative design: `docs/design/JOBS_ORCHESTRATOR.md`.

## Frozen decisions

1. Brutus uses host-installed single-node Slurm behind the scheduler-neutral BigCherry `Executor` API.
2. Slurm owns native execution/resource scheduling; BigCherry owns scientific identity, attempts, retry policy, hardware cohort, production coexistence, evidence, harvest, report and durable events.
3. V1 services are `munge + slurmctld + slurmd`; no slurmdbd/MariaDB/slurmrestd/account/sacct requirement.
4. Generic partitions: `bc-build`, `bc-measure`; licenses `host_activity:2,build_slot:1`; timed measurement remains host-exclusive.
5. GPU GRES is architecture-typed. Stable device identity is BigCherry inventory/series state, never slot/ordinal/GRES type.
6. Hardware inventory is discovered/accepted runtime state; `environment.local.toml` is policy/paths only.
7. Production may use any GPU. Conflict is derived from runtime production potential/observed device claims; no static production-GPU partition exists.
8. Initial cgroup policy is `ConstrainDevices=no`; `yes` requires the Brutus ROCm `/dev/kfd`/render-node qualification matrix.
9. **Monolithic v1 never uses native Slurm requeue.** Exit 75 is retryable-environment classification; every retry is a new BigCherry attempt/new native job. `retry --same-commit` or `--latest` chooses commit policy explicitly. Mechanical Slurm requeue remains CI-tested only for future restart-safe stages.
10. Linux/Windows and distinct physical stable-device cohorts are distinct scientific series.

## Implemented boundaries

Production modules now provide:

- durable `BatchSpec`/`JobSpec`/series/run/attempt identity;
- platform-neutral Slurm/Local/Remote/Fake executors;
- exact pre-series stable-GPU cohort binding and attempt-start allocation attestation;
- client idempotency + submission-intent/correlation crash recovery;
- pinned detached workspaces and scientific-identity recheck;
- streamed managed process evidence and monitored runner execution;
- hardware discovery/accepted inventory/GRES rendering;
- verified harvest/report/review-ready state;
- status/events/logs/metrics/operator CLI;
- production claim/window/watchdog domain model;
- RCD07 durable operation identity/rehydration substrate without creating a second scheduler.

Real Noble CI has exercised MUNGE/slurmctld/slurmd/sbatch/squeue/scontrol/scancel, dependencies, holds, licenses, controller restart, jobcomp and cgroup process/task/accounting with `ConstrainDevices=no`. Current production policy deliberately does not enable the mechanically tested native requeue path.

## Remaining acceptance gates

These are not architecture blockers:

- Brutus AMD stable-ID discovery across reboot/driver changes and generated `gres.conf` + `slurmd -G`;
- ROCm visibility/peer/tensor-split and optional `ConstrainDevices=yes` matrix;
- live llama-swap production claim/window integration and scheduler-noise qualification;
- representative disk-pressure/process-tree/large-log acceptance;
- Windows HIP stable identity + target-local staging before remote scientific campaigns;
- one complete real planned series through submit -> execute -> harvest -> report.

## Closure rule

RCD02 itself is design-complete. Keep the plan item open only as an umbrella until Brutus cutover acceptance is recorded; do not reopen superseded slurmdbd/static-GPU/native-requeue designs.

## Change log

- 2026-09-26: architecture frozen and real Noble validation established.
- 2026-09-28: reconciled final v1 implementation; removed stale buffering/native-requeue assumptions and separated hardware acceptance from architecture.
