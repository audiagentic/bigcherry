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

## Description



## Steps

Consolidated residual obligations from RCD04/RCD05/RCD06 (2026-11-20; software scope only, no completion claim):
- Retain exact attempt commit resolution, detached pinned workspaces, scientific identity recheck, stable-ID allocation attestation, durable executor start/attestation/result evidence, file-backed managed process streaming, process-group TERM/grace/KILL, CPU/output liveness detection, typed infrastructure/configuration incidents, and scientific FAIL must never trigger result-driven retries. Retry creates a new BigCherry attempt/native job; no canonical checkout mutation or stash.
- Define/qualify dev-specific absolute free-space reserves for root/work/tmp and a nonzero stall threshold. Keep retention/cleanup fail-closed: never delete active/unharvested evidence; only eligible harvested/failed roots. Validate real disk pressure, process-tree cancellation, large server logs and gitignored vendor/environment workspace inputs; preserve Windows-local process-tree acceptance.
- Preserve the verified isolated attempt evidence destination, exact-path harvest and record/patch/validation/contract identity checks; dirty/staged canonical destinations fail closed. Series composition is frozen and re-resolved at attempt start; mismatches block execution.
- For future extracted producer stages, define typed ProducerPreflightResult(ok,retryable,invalid) with producer-owned examples (MTP row/source-parser and tensor topology/cap limits); generic service code must not special-case producers. Add stable phase/lane/round campaign event callbacks for progress only, never result authority. Any prepare-only/execute-only split must build on RCD07 operation identity/rehydration primitives and preserve direct CLI compatibility; do not add parallel resume authority.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

RCD04/RCD05/RCD06 software guarantees remain subject to jobs CI and owning module tests; no new implementation or implementation-completion claim is made by consolidation. Keep host validation pending: qualified dev disk reserves/stall threshold, real disk-pressure/process-tree/large-log and ignored-input acceptance, Windows process-tree acceptance, plus RCD03/RCD10/RCD11/RCD12 real-host cutover. Future producer-stage work must test typed preflight classification, producer-specific checks, stable progress callbacks as non-authoritative, and prepare/execute compatibility atop RCD07. Evidence harvest must prove exact identity, committed verified evidence, and dirty/staged destination refusal.

## Effort & Risk



## Standards



## Acceptance Criteria

RCD02 remains an active umbrella until dev cutover acceptance is recorded. RCD04/RCD05/RCD06 residuals are explicitly tracked here: pinned exact-attempt execution and scientific-identity fail-closed behavior; stable-ID allocation attestation and durable evidence; bounded file-backed output and process-tree termination; qualified host-specific disk/stall policy and safe retention; isolated verified harvest and explicit commits; retryable environment/configuration classification distinct from scientific FAIL; producer-owned typed preflight and non-authoritative stage progress callbacks; any prepare/execute split reuses RCD07 durable operation identity and preserves direct CLI compatibility. RCD03/RCD10/RCD11/RCD12 host/platform acceptance remains open; this item does not claim the job service or production cutover is accepted.

## Notes

Consolidated from active sources RCD04, RCD05 and RCD06 on 2026-11-20 following RV4246. Their duplicate implementation trackers may be superseded after this active owner records the inherited residual gates. Source history remains in each superseded item and in its original evidence/ledger records. No tests, hardware runs, cleanup, deletion, or implementation acceptance are implied by this plan consolidation.

## Status — 2026-09-28

**Architecture frozen; software implementation exists. Remaining gates are host/hardware acceptance, not platform-choice objections.**

Normative design: `docs/design/JOBS_ORCHESTRATOR.md`.

## Frozen decisions

1. dev uses host-installed single-node Slurm behind the scheduler-neutral BigCherry `Executor` API.
2. Slurm owns native execution/resource scheduling; BigCherry owns scientific identity, attempts, retry policy, hardware cohort, production coexistence, evidence, harvest, report and durable events.
3. V1 services are `munge + slurmctld + slurmd`; no slurmdbd/MariaDB/slurmrestd/account/sacct requirement.
4. Generic partitions: `bc-build`, `bc-measure`; licenses `host_activity:2,build_slot:1`; timed measurement remains host-exclusive.
5. GPU GRES is architecture-typed. Stable device identity is BigCherry inventory/series state, never slot/ordinal/GRES type.
6. Hardware inventory is discovered/accepted runtime state; `environment.local.toml` is policy/paths only.
7. Production may use any GPU. Conflict is derived from runtime production potential/observed device claims; no static production-GPU partition exists.
8. Initial cgroup policy is `ConstrainDevices=no`; `yes` requires the dev ROCm `/dev/kfd`/render-node qualification matrix.
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

- dev AMD stable-ID discovery across reboot/driver changes and generated `gres.conf` + `slurmd -G`;
- ROCm visibility/peer/tensor-split and optional `ConstrainDevices=yes` matrix;
- live llama-swap production claim/window integration and scheduler-noise qualification;
- representative disk-pressure/process-tree/large-log acceptance;
- Windows HIP stable identity + target-local staging before remote scientific campaigns;
- one complete real planned series through submit -> execute -> harvest -> report.

## Closure rule

RCD02 itself is design-complete. Keep the plan item open only as an umbrella until dev cutover acceptance is recorded; do not reopen superseded slurmdbd/static-GPU/native-requeue designs.

## Change log

- 2026-09-26: architecture frozen and real Noble validation established.
- 2026-09-28: reconciled final v1 implementation; removed stale buffering/native-requeue assumptions and separated hardware acceptance from architecture.

## Reviews

- RV4246

## Change Log

- 2026-11-20T02:51:50.455044+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes
