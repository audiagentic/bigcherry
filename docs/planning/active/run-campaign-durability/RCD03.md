---
id: RCD03
order: 3
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:51:56.813138+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Install and qualify the Brutus Slurm execution substrate

## Objective

Use Noble Slurm as Brutus execution/resource authority while BigCherry remains scientific/domain authority. v1 is MUNGE + slurmctld + slurmd, without MariaDB/slurmdbd/slurmrestd.

## Implementation status — 2026-09-27

Implemented/offline or real-Noble validated:

- `SlurmExecutor`: `sbatch --parsable`, `squeue --json`, exact execution-ID correlation, hold/release/cancel, normalized states, no-account default and no `sacct` dependency;
- production `slurm.conf`/`cgroup.conf` templates with architecture GRES, license admission and no monolithic native requeue;
- accepted-inventory GRES renderer and Linux AMD-SMI discovery/observed/accepted inventory flow;
- `tools/admin/render_bigcherry_slurm.py` exact accepted-inventory rendering;
- `tools/admin/install_brutus_slurm.py` repeatable dry-run/apply installer: accepted inventory precondition, minimal `slurm-wlm+munge+jq`, MUNGE provisioning/round-trip, rendered `/etc/slurm`, `slurmd -G`, service start and `scontrol ping`;
- installer action plan/preflight tests;
- real Ubuntu 24.04 / Slurm 23.11.4 CI proving controller/daemon startup, no-account execution, JSON queue, dependency, hold/release/cancel, license priority, jobcomp and cgroup fallback.

Retry rule: production v1 does **not** configure `RequeueExit`. Every campaign retry is BigCherry attempt+1/new native job; `--same-commit` may pin identical code. Real Noble CI may continue exercising native requeue only as a scheduler capability reference.

Still Brutus hardware/production acceptance:

1. capture real AMD-SMI provider output and prove stable identity persistence;
2. accept real peer topology and render/apply real host GRES;
3. pass `slurmd -G` and exact Slurm visibility -> stable-ID attestation;
4. run ROCm `/dev/kfd`/render/HIP/peer/tensor-split cgroup matrix;
5. retain `ConstrainDevices=no` if any supported device-fencing cell fails;
6. pass RCD11 live llama-swap/exclusive-window gates and RCD10 full-series soak.

## Production scheduler policy

```text
bc-build:   build_slot:1 + host_activity:1
bc-measure: host_activity:2 + gpu:<architecture>:<reserved-count>
```

One generic build partition and one generic measurement partition. No per-card or static production-GPU partition. Timed scientific measurement owns whole GPUs; v1 has no scientific GPU sharding.

Core policy:

```ini
SchedulerType=sched/backfill
SchedulerParameters=bf_licenses,bf_interval=2,sched_interval=2
PriorityType=priority/basic
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
Licenses=host_activity:2,build_slot:1
AccountingStorageType=accounting_storage/none
JobCompType=jobcomp/filetxt
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
JobAcctGatherType=jobacct_gather/cgroup
```

Initial device policy:

```ini
CgroupPlugin=autodetect
ConstrainCores=yes
ConstrainDevices=no
```

Enable `ConstrainDevices=yes` only after the complete Brutus matrix succeeds.

## GRES and submission rules

GRES `Type` is architecture only. Stable ID remains BigCherry scientific identity; BDF/render/ordinal remain current locators. Material hardware drift drains before GRES mutation and `slurmd -G` is mandatory after rendering changes.

A proper-subset frozen cohort reserves the complete accepted architecture pool and is narrowed only inside the exclusive allocation after stable-ID attestation.

Before `sbatch`, BigCherry persists immutable `submission-intent.json` containing stable `execution_id`; the same ID is placed in the Slurm job comment. Recovery is one-match rebind / proven-zero immutable submit / multiple-match wake+block. Short-job terminal authority is preserved by attempt-local start/result sentinels and BigCherry records.

## Installation/acceptance

Preferred procedure is `docs/reference/jobs/SLURM_BRUTUS.md`. Cutover acceptance is `docs/reference/jobs/ACCEPTANCE.md`.

Required final gates:

- real Noble scheduler workflow green;
- Brutus accepted inventory/GRES + `slurmd -G` green;
- exact stable-ID allocation attestation before scientific work;
- device cgroup qualified or explicit fallback recorded;
- no database/accounting-service dependency;
- production coexistence/window acceptance;
- one full managed scientific series through committed harvest/report.

## Change log

- 2026-09-26: initial design and real Noble validation.
- 2026-09-27: production Slurm adapter/no-account/no-sacct behavior implemented.
- 2026-09-27: removed production native requeue for monolithic campaigns.
- 2026-09-27: repeatable Brutus installer plus Linux AMD discovered/accepted inventory path implemented; remaining status is real-host qualification.
