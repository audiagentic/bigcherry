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

## Description

Install the minimal single-node Slurm execution substrate used by `SlurmExecutor`: **MUNGE + `slurmctld` + `slurmd`**. v1 does **not** require MariaDB, slurmdbd or slurmrestd. BigCherry remains the durable domain authority; Slurm provides queue/process/resource ownership plus controller state and `jobcomp/filetxt` completion history.

This is now empirically validated, not a paper design. GitHub Actions run `36219538263` on Ubuntu 24.04.5 installed Noble `slurm-wlm 23.11.4-1.2ubuntu5`, started real MUNGE/slurmctld/slurmd, scheduled real jobs without accounts/slurmdbd, ran the current BigCherry process harness inside a Slurm allocation, proved license priority/dependencies/requeue/cancel/controller recovery/jobcomp, then restarted using the production cgroup plugins with `ConstrainDevices=no` and completed a job. AMD device filtering remains Brutus-only.

## Dependencies

- RCD02 architecture freeze.
- RCD12 accepted hardware inventory + deterministic architecture-typed GRES render.
- RCD04 Executor protocol for production adapter implementation.
- RCD11 production coexistence is orthogonal; Slurm does not own llama-swap.

## Tested v1 service topology

```text
MUNGE
  |
slurmctld  <---- sbatch/squeue/scontrol/scancel
  |
slurmd

BigCherry run store + jobcomp/filetxt = durable/history evidence
```

No LAN Slurm REST/API. Agents use BigCherry over SSH. Native Slurm IDs are execution metadata only.

## Exact tested scheduler policy

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
JobCompLoc=/var/log/slurm/bigcherry-jobcomp.log
RequeueExit=75

ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
JobAcctGatherType=jobacct_gather/cgroup
JobAcctGatherFrequency=30

PartitionName=bc-build   Nodes=brutus PriorityTier=10  State=UP MaxTime=00:45:00
PartitionName=bc-measure Nodes=brutus PriorityTier=100 State=UP MaxTime=00:45:00
```

### Why these settings

- `priority/basic`: one host does not need fair-share/account association machinery.
- `PriorityTier`: measurement outranks future build work.
- `bf_licenses`: backfill accounts for licenses rather than allowing build churn to starve measurement.
- `host_activity:2`: build consumes one token; timed measurement consumes both.
- `build_slot:1`: one build initially; increase only after isolation qualification.
- `AuthInfo=cred_expire=30`: Noble 23.11 uses credential expiry as automatic requeue eligibility delay. Default 120s was measured and rejected as operationally slow. `SchedulerParameters=requeue_delay` is not a Noble 23.11 option.
- `AccountingStorageType=none`: real jobs execute without an association DB. `Reason=InvalidAccount` can appear transiently in no-association mode and is therefore diagnostic text, not a BigCherry permanent-failure classifier.
- `jobcomp/filetxt`: lightweight Slurm-native completion trail; BigCherry store remains authoritative for attempts/results/events.

## Resource classes

```text
build / v1.5 prepare:
  partition=bc-build
  licenses=build_slot:1,host_activity:1
  finite time

monolithic v1 / timed execute:
  partition=bc-measure
  licenses=host_activity:2
  architecture GRES when GPU required
  finite time

v1.5 execute:
  dependency=afterok:<prepare-job>
```

Run `36219538263` produced actual order:

```text
build1 -> measure -> build2
```

where build1 was confirmed RUNNING before measure/build2 submission. This is the starvation/progress acceptance test.

## GRES model

RCD12 generates architecture-only types from accepted discovery:

```text
Name=gpu Type=gfx1100 File=/dev/dri/renderD128 Flags=amd_gpu_env
Name=gpu Type=gfx1100 File=/dev/dri/renderD129 Flags=amd_gpu_env
Name=gpu Type=gfx1201 File=/dev/dri/renderD130 Flags=amd_gpu_env
```

Node summary:

```text
Gres=gpu:gfx1100:2,gpu:gfx1201:1,...
```

Rules:

1. no per-slot/per-UUID `Type`;
2. `File=` is regenerated from current accepted inventory;
3. `Flags=amd_gpu_env` owns Slurm `ROCR_VISIBLE_DEVICES`;
4. BigCherry maps allocated resources back to stable RCD12 IDs;
5. if a series-bound cohort is a proper subset of one architecture, reserve all accepted GPUs of that architecture, then narrow to the pre-bound stable IDs inside the exclusive allocation;
6. inventory drift drains the node before GRES mutation.

`slurmd -G` is mandatory on Brutus before resuming the node after every generated GRES change. Do not prescribe `slurmctld -t`: Noble 23.11.4 has no such config-test option. Controller startup + `scontrol show config` validates scheduler config; `slurmd -G` validates GRES.

## cgroup policy

Initial Brutus mode:

```ini
# slurm.conf
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
JobAcctGatherType=jobacct_gather/cgroup
JobAcctGatherFrequency=30

# cgroup.conf
CgroupPlugin=autodetect
ConstrainCores=yes
ConstrainDevices=no
ConstrainRAMSpace=no
ConstrainSwapSpace=no
```

Real Noble CI proved this fallback stack launches jobs in Slurm cgroups. Only switch to `ConstrainDevices=yes` after Brutus proves, for every accepted ROCm/toolchain/cohort:

- `/dev/kfd` usable;
- allocated render nodes usable;
- unallocated render nodes inaccessible;
- Slurm visibility resolves exactly to bound stable IDs;
- no conflicting HIP/CUDA selector;
- 100 timeout-bounded HIP init/property cycles;
- llama-bench/server smoke;
- peer/`-sm tensor` + 4096-context preflight where required.

Failure of any supported cell keeps `ConstrainDevices=no`; GRES then provides scheduling isolation, not a device security boundary.

## SlurmExecutor contract

Planned module:

```python
class SlurmExecutor(Executor):
    def __init__(self, *, runner: CommandRunner, policy: SlurmPolicy): ...
    def submit(self, request: ExecutionRequest) -> ExecutionHandle: ...
    def status(self, handle: ExecutionHandle) -> ExecutionStatus: ...
    def cancel(self, handle: ExecutionHandle) -> None: ...
    def control(self, handle: ExecutionHandle, action: ExecutorControl) -> None: ...
    def allocation(self, handle: ExecutionHandle) -> Allocation | None: ...
    def events(self, handle: ExecutionHandle, *, after: int | None = None) -> Iterable[ExecutorEvent]: ...
```

Exact submission shape:

```text
sbatch --parsable
  --job-name bc:<run_id>:a<attempt>
  --comment bigcherry:<execution_id>
  --partition <bc-build|bc-measure>
  --licenses <...>
  --gres gpu:<arch>:<reserved-count>        # GPU jobs only
  --time <finite>
  --output <attempt>/slurm-%j.out
  --error <attempt>/slurm-%j.err
  --export ALL,BIGCHERRY_RUN_ID=...,BIGCHERRY_ATTEMPT=...,BIGCHERRY_ATTEMPT_ROOT=...
  <attempt>/launch.sh
```

No `--account` in v1. Production code builds argv arrays, never shell command strings.

Status/control interfaces:

```text
squeue --json                 current queue/running state
scontrol show job -o <id>     targeted diagnostic/allocation detail
scontrol hold/release <id>
scancel <id>
jobcomp/filetxt               lightweight terminal history
```

Do not depend on `sacct` in v1 because no accounting DB is configured. Pending `reason` is diagnostic; BigCherry derives execution state from Slurm state/resource behavior plus bounded timeouts, never from one reason string alone.

## Submission crash/idempotency

Before `sbatch`:

```text
persist submission-intent.json
execution_id is stable
--comment=bigcherry:<execution_id>
```

After acceptance persist returned native job ID. On restart:

- one correlated execution -> bind it;
- zero correlated executions + acceptance disproved -> submit once;
- multiple correlated executions/uncertain acceptance -> block + wake; never blind duplicate.

The file-backed service recovery smoke already proves these state-machine rules offline. Production Slurm recovery combines `squeue --json`, BigCherry start/result sentinels and jobcomp because a very short completed job can leave `squeue` before handle persistence.

## Retry semantics proven on Noble

```text
0   execution completed; scientific verdict is separate
75  RequeueExit: same native job, same attempt, same commit
76  terminate; BigCherry creates attempt+1/new job and may resolve newer commit
77  invalid/drift; block
```

Real run observed the same job restart with:

```text
SLURM_RESTART_COUNT: 0 -> 1
Restarts=1
```

Scientific FAIL remains exit 0 and cannot trigger scheduler requeue.

## Real CI evidence

GitHub Actions run `36219538263` / head `4b53e9df3cb89b6cb62feb68c381f178d571f4dd` passed both jobs.

Real Noble/Slurm assertions:

```text
Ubuntu 24.04.5
slurm-wlm 23.11.4-1.2ubuntu5
MUNGE round-trip
slurmctld + slurmd node registration
squeue --json
no-account job completion
real current-branch BigCherry process harness inside Slurm job
hold/release
running-build license blocking
build1 -> measure -> build2 priority/progress
afterok dependency
RequeueExit=75; SLURM_RESTART_COUNT 0 -> 1
cancel
slurmctld restart retaining queued + running ownership
jobcomp/filetxt completion history
proctrack/task/jobacct cgroup stack with ConstrainDevices=no
```

Separate BigCherry CI job on the same run passed planning/process/failure/recovery/race/submission-event tests.

## Files

Tracked:

- `config/slurm/slurm.conf.example`
- `config/slurm/cgroup.conf.example`
- `config/slurm/gres.conf.example`
- `docs/reference/jobs/SLURM_BRUTUS.md`
- `.github/workflows/rcd-slurm-validation.yml`
- canonical lab validation under `tools/lab/run-campaign-durability/`

Planned production:

- `tools/bigcherry/jobs/slurm.py`
- `tools/bigcherry/jobs/slurm_config.py`
- `tools/tests/jobs/test_slurm_executor.py`
- `tools/tests/jobs/test_slurm_config.py`

Generated host state:

- `/etc/slurm/slurm.conf`
- `/etc/slurm/gres.conf`
- `/etc/slurm/cgroup.conf`
- `/var/spool/slurmctld`
- `/var/spool/slurmd`
- `/var/log/slurm/bigcherry-jobcomp.log`

## Brutus install/acceptance sequence

1. install `slurm-wlm munge jq` only;
2. configure/start MUNGE; require round-trip;
3. discover/accept RCD12 hardware inventory;
4. render `slurm.conf`, `gres.conf`, `cgroup.conf` from accepted state;
5. `slurmd -C` host CPU/memory sanity;
6. `slurmd -G` generated GRES validation;
7. start/enable `slurmctld`, then `slurmd`;
8. require `scontrol ping`, `sinfo -Nel`, `squeue --json`, `scontrol show lic`;
9. run CPU scheduler smoke equivalent to CI;
10. run AMD cgroup/GRES matrix while `ConstrainDevices=no` is known fallback;
11. switch to `ConstrainDevices=yes` only on full matrix pass;
12. run RCD11 production coexistence and scheduler-isolation gates;
13. record version/config/inventory/cgroup hashes before RCD10 cutover.

## Acceptance Criteria

- real Noble CI remains green;
- Brutus generated GRES passes `slurmd -G`;
- allocation maps to stable RCD12 identities;
- AMD cgroup mode either passes or recorded fallback stays active;
- production Slurm adapter has command/render/parser/idempotency tests;
- no BigCherry domain module imports Slurm adapter/types;
- no slurmdbd/MariaDB dependency in v1.

## Change Log

- 2026-09-26T00:51:56.813138+00:00: created.
- 2026-09-26: dynamic inventory/GRES design added.
- 2026-09-26: real Noble validation added; several assumptions falsified during iteration.
- 2026-09-26: final real Noble run proved minimal no-DB stack, BigCherry-under-Slurm, license priority, requeue, controller recovery and cgroup fallback; removed slurmdbd/MariaDB from v1.
