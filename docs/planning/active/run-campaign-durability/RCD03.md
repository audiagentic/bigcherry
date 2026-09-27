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

Use Noble Slurm as Brutus execution/resource authority while BigCherry remains scientific/domain authority. v1 service topology is **MUNGE + slurmctld + slurmd**, with no MariaDB/slurmdbd/slurmrestd requirement.

## Implementation status — 2026-09-27

Implemented and checked in:

- `tools/bigcherry/jobs/slurm.py`: scheduler-only adapter, `sbatch --parsable`, `squeue --json`, hold/release/cancel, execution correlation by exact `--comment=bigcherry:<execution_id>`, normalized state.
- no-account Slurm is the default; `--account` is opt-in only.
- production adapter does not depend on `sacct`; current status is `squeue --json`, with BigCherry attempt sentinels/domain records providing terminal authority.
- `config/slurm/slurm.conf.example`, `cgroup.conf.example`, `gres.conf.example`.
- RCD12 architecture-typed GRES renderer now exists in `tools/bigcherry/hardware/slurm.py`.
- real Ubuntu 24.04 / Slurm 23.11.4 CI has already proved controller/daemon startup, no-account execution, JSON queue state, dependencies, hold/release/cancel, license priority, jobcomp and cgroup fallback.
- real CI also proved Slurm native requeue mechanically works.

**Retry safety correction:** production v1 no longer configures `RequeueExit=75` for monolithic campaigns. Current BigCherry campaign/workspace paths are not uniformly restart-idempotent, so all campaign retries create a **new BigCherry attempt/new Slurm job**; `--same-commit` can deliberately pin the same commit. Native requeue remains a tested Slurm capability and may later be enabled per RCD07 stage only after that stage is proven restart-safe.

Still Brutus-only:

- actual AMD inventory provider/stable IDs;
- generated host `gres.conf` + `slurmd -G`;
- ROCm `/dev/kfd`/render-node behavior and `ConstrainDevices=yes` matrix;
- exact Slurm allocation -> stable-ID attestation;
- production coexistence/noise gates.

## Required production policy

```ini
AuthType=auth/munge
AuthInfo=cred_expire=30
CredType=cred/munge
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
PartitionName=bc-build   Nodes=brutus PriorityTier=10  State=UP MaxTime=00:45:00
PartitionName=bc-measure Nodes=brutus PriorityTier=100 State=UP MaxTime=00:45:00
```

Initial cgroup device policy:

```ini
CgroupPlugin=autodetect
ConstrainCores=yes
ConstrainDevices=no
```

Only switch to `ConstrainDevices=yes` after the full Brutus ROCm matrix succeeds.

## Resource ownership

```text
build / prepare:
  bc-build
  build_slot:1,host_activity:1

timed scientific execution:
  bc-measure
  host_activity:2
  --gres gpu:<architecture>:<reserved-count>
```

Timed scientific execution owns whole GPUs. Slurm GPU sharding is not part of v1 scientific measurement.

One generic build partition and one generic measure partition are used. No per-card partitions and no production-GPU partition state.

## GRES rules

RCD12 accepted inventory renders explicit resources:

```text
Name=gpu Type=gfx1100 File=/dev/dri/renderD128 Flags=amd_gpu_env
Name=gpu Type=gfx1100 File=/dev/dri/renderD129 Flags=amd_gpu_env
Name=gpu Type=gfx1201 File=/dev/dri/renderD130 Flags=amd_gpu_env
```

Rules:

1. GRES `Type` is architecture only.
2. slot/index/BDF/UUID is never encoded in the GRES type.
3. accepted hardware drift drains before GRES mutation.
4. `slurmd -G` is mandatory after generated GRES changes.
5. architecture/count allocation is reconciled to the series-frozen stable-ID cohort before scientific measurement.
6. a proper-subset cohort reserves all accepted GPUs of that architecture, then narrows within the exclusive allocation.

## Submission contract

Before scheduler submission BigCherry persists `submission-intent.json`. `SlurmExecutor` submits an argv vector equivalent to:

```text
sbatch --parsable
  --job-name bc:<execution_id>
  --comment bigcherry:<execution_id>
  --partition <bc-build|bc-measure>
  --licenses <resolved>
  --cpus-per-task N
  --time finite
  [--gres gpu:<arch>:<count>]
  --output <attempt>/stdout.log
  --error <attempt>/stderr.log
  --export ALL
  <attempt>/launch.sh
```

No account is required by default. Domain code never imports Slurm types/syntax.

Crash recovery:

1. exact stable `execution_id` is persisted before `sbatch`;
2. exact same ID is in Slurm job comment;
3. restart correlates active Slurm jobs by comment;
4. attempt-local `executor-start.json`/`executor-result.json` and BigCherry records cover short jobs that vanish from `squeue`;
5. one match -> rebind; multiple -> wake/block; proven zero -> submit immutable request once.

## Brutus installation/qualification

1. install `slurm-wlm`, `munge`, `jq`;
2. configure/start MUNGE and require round-trip;
3. discover and explicitly accept RCD12 inventory;
4. render node/GRES/cgroup config from accepted inventory;
5. check `slurmd -C` host resources;
6. check `slurmd -G` GRES;
7. enable/start `slurmctld` and `slurmd`;
8. require `scontrol ping`, `sinfo -Nel`, `squeue --json`, `scontrol show lic`;
9. run CPU scheduler/license/dependency/control smoke;
10. run GPU/stable-ID attestation smoke under `ConstrainDevices=no`;
11. execute `/dev/kfd`/render-node/HIP/peer/tensor-split matrix;
12. enable `ConstrainDevices=yes` only if every supported cell passes;
13. run RCD11 production coexistence and RCD10 full-series acceptance.

Concrete runbook: `docs/reference/jobs/SLURM_BRUTUS.md`.

## Tests

Permanent offline adapter tests must cover:

- exact argv rendering;
- no account default;
- architecture/count GRES only;
- dependency native-ID resolution;
- state normalization;
- hold/release/cancel;
- exact comment correlation;
- unknown native state fails to `unknown`, never guessed;
- domain package contains no Slurm imports outside adapter/registry.

Real Noble workflow remains the scheduler compatibility gate. It may continue testing native requeue as a mechanism even though production monolithic v1 does not enable it.

## Acceptance criteria

- real Noble workflow green;
- Brutus `slurmd -G` green for accepted discovered inventory;
- exact stable-ID allocation attestation before scientific GPU work;
- cgroup device filtering either passes or recorded `ConstrainDevices=no` fallback remains active;
- Slurm adapter command/recovery tests green;
- no slurmdbd/MariaDB/sacct dependency;
- production monolithic campaign cannot native-requeue until restart safety is explicitly proven.

## Change log

- 2026-09-26: initial design and real Noble validation.
- 2026-09-27: production Slurm adapter implemented; no-account/no-sacct behavior encoded and tested.
- 2026-09-27: removed production `RequeueExit=75` for monolithic campaigns; retries are new BigCherry attempts/jobs until RCD07 stage restart-safety exists.
