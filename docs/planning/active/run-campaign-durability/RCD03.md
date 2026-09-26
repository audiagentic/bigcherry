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

Install and qualify the single-node Slurm service used by `SlurmExecutor`: MUNGE, loopback-only MariaDB, `slurmdbd`, `slurmctld`, and `slurmd`. No `slurmrestd` in v1. The accounting layer is minimal scheduler state, not a second BigCherry authority: BigCherry still owns run/series/attempt identity, commit pinning, retry legality, evidence and reporting.

This requirement is empirically derived. The real Ubuntu 24.04/Slurm 23.11.4 CI smoke successfully started MUNGE/slurmctld/slurmd, registered the node, exposed `squeue --json`, and completed hold/release, then reproduced `PENDING Reason=InvalidAccount` on later submissions when no accounting association existed. v1 therefore provisions exactly one local cluster/account and explicit user association through slurmdbd rather than relying on no-accounting behavior.

GPU inventory/GRES come from RCD12 discovered state. Production coexistence remains RCD11 BigCherry policy, not static Slurm GPU partitions.

## Steps

1. Maintain tracked config templates plus `docs/reference/jobs/SLURM_BRUTUS.md`; rendered host files and DB secrets remain untracked.
2. Install Noble packages `slurm-wlm slurmdbd munge mariadb-server`.
3. Bind MariaDB to loopback only; create `slurm_acct_db` + least-purpose `slurm` DB user.
4. Configure/start MUNGE, slurmdbd, cluster `bigcherry`, account `bigcherry`, service-user association, then slurmctld/slurmd in that order.
5. Configure `bc-build`/`bc-measure`, `bf_licenses`, priority tiers, licenses, `AccountingStorageEnforce=associations`, `RequeueExit=75`, jobcomp and cgroup accounting.
6. Generate architecture-typed `gres.conf` and node `Gres=` from accepted RCD12 inventory; `slurmd -G` is mandatory before node resume.
7. Implement `SlurmExecutor` behind RCD04 `Executor`; command runner is injectable and Slurm syntax stays adapter-local.
8. Run real CPU/service validation in GitHub `ubuntu-24.04` on every RCD/config change.
9. On Brutus run the AMD GRES/cgroup/ROCm falsification matrix; fallback to `ConstrainDevices=no` if any supported cell fails.
10. Record Slurm/config/accounting/inventory/cgroup hashes/modes in normalized status and acceptance evidence.

## Detailed Solution & Technical Design

### Service topology

```text
MariaDB 127.0.0.1:3306
  ^
  | accounting_storage/mysql
slurmdbd 127.0.0.1:6819
  ^
  | accounting_storage/slurmdbd
slurmctld
  ^       ^
  |       |
sbatch   slurmd
```

Only SSH is agent-facing. MariaDB/slurmdbd are never LAN APIs. MUNGE authenticates Slurm components. BigCherry uses CLI adapter calls (`sbatch`, `squeue --json`, `scontrol`, `scancel`, `sacct`) and never connects to MariaDB directly.

### Accounting bootstrap

Tracked `/etc/slurm/slurmdbd.conf` template:

```ini
AuthType=auth/munge
DbdHost=@NODE_NAME@
DbdAddr=127.0.0.1
DbdPort=6819
SlurmUser=slurm
StorageType=accounting_storage/mysql
StorageHost=127.0.0.1
StoragePort=3306
StorageLoc=slurm_acct_db
StorageUser=slurm
StoragePass=@HOST_SECRET@
```

Host bootstrap:

```bash
sacctmgr -i add cluster bigcherry
sacctmgr -i add account bigcherry Cluster=bigcherry Description=BigCherry Organization=BigCherry
sacctmgr -i add user bigcherry Account=bigcherry DefaultAccount=bigcherry Cluster=bigcherry
```

Every managed `sbatch` explicitly includes `--account=bigcherry`. Human users get explicit associations only if they submit directly.

### Scheduler policy

```ini
SchedulerType=sched/backfill
SchedulerParameters=bf_licenses
PriorityType=priority/multifactor
Licenses=host_activity:2,build_slot:1

AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=127.0.0.1
AccountingStoragePort=6819
AccountingStorageEnforce=associations

JobCompType=jobcomp/filetxt
JobCompLoc=/var/log/slurm/bigcherry-jobcomp.log
JobAcctGatherType=jobacct_gather/cgroup
JobAcctGatherFrequency=30

ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
RequeueExit=75

PartitionName=bc-build   Nodes=brutus PriorityTier=10  State=UP MaxTime=00:45:00
PartitionName=bc-measure Nodes=brutus PriorityTier=100 State=UP MaxTime=00:45:00
```

`host_activity=2` is the host-quiet token. v1 monolithic measure consumes both; build consumes one. `bf_licenses` + higher measurement tier are tested against real Slurm, not assumed.

### GRES

RCD12 generates one line per current render node:

```text
Name=gpu Type=gfx1100 File=/dev/dri/renderD128 Flags=amd_gpu_env
Name=gpu Type=gfx1100 File=/dev/dri/renderD129 Flags=amd_gpu_env
Name=gpu Type=gfx1201 File=/dev/dri/renderD130 Flags=amd_gpu_env
```

Node summary groups architecture counts only:

```text
Gres=gpu:gfx1100:2,gpu:gfx1201:1
```

Never encode ordinal/stable UUID in `Type`. Slurm owns allocation and `ROCR_VISIBLE_DEVICES`; BigCherry maps allocation back to RCD12 stable IDs and preserves visibility. A series-bound proper subset reserves all accepted GPUs of that architecture in v1, then narrows to the pre-bound stable IDs inside the exclusive allocation.

### SlurmExecutor contract

```python
# tools/bigcherry/jobs/slurm.py
class SlurmExecutor(Executor):
    def __init__(self, *, runner: CommandRunner, policy: SlurmPolicy): ...
    def submit(self, request: ExecutionRequest) -> ExecutionHandle: ...
    def status(self, handle: ExecutionHandle) -> ExecutionStatus: ...
    def cancel(self, handle: ExecutionHandle) -> None: ...
    def control(self, handle: ExecutionHandle, action: ExecutorControl) -> None: ...
    def allocation(self, handle: ExecutionHandle) -> Allocation | None: ...
    def events(self, handle: ExecutionHandle, *, after: int | None = None) -> Iterable[ExecutorEvent]: ...
```

Submission argv includes at least:

```text
sbatch --parsable
  --account bigcherry
  --job-name bc:<run_id>:a<attempt>
  --comment bigcherry:<execution_id>
  --partition <bc-build|bc-measure>
  --licenses ...
  --gres gpu:<arch>:<reserved_count>     # when GPU stage
  --time <finite>
  --output <attempt>/slurm-%j.out
  --error <attempt>/slurm-%j.err
  --export ALL,BIGCHERRY_RUN_ID=...,BIGCHERRY_ATTEMPT=...,BIGCHERRY_ATTEMPT_ROOT=...
  <attempt>/launch.sh
```

`submission-intent.json` is durable before `sbatch`. The returned job ID is opaque execution metadata. Recovery searches by `--comment=bigcherry:<execution_id>` plus sentinels/accounting; ambiguous acceptance never causes blind duplicate submission.

Machine parsers consume `squeue --json`/structured fields; never human column output. `sacct` is historical executor evidence, not scientific authority.

### cgroups

Candidate:

```ini
CgroupPlugin=autodetect
ConstrainDevices=yes
ConstrainCores=yes
```

Brutus acceptance per toolchain/cohort:

- `/dev/kfd` works;
- allocated render nodes work, unallocated nodes blocked;
- Slurm visibility maps exactly to stable IDs;
- no conflicting HIP/CUDA visibility selectors;
- 100 timeout-bounded HIP init/property loops;
- llama-bench/server smoke;
- peer/`-sm tensor` + 4096-context producer preflight where required.

Fallback: `ConstrainDevices=no`, while retaining GRES scheduling, Slurm visibility and BigCherry attestation. Record fallback explicitly; do not call it device isolation.

## Code Samples & Guidance

Deterministic adapter/config functions:

```python
def render_slurm_gres(inventory: HardwareInventory) -> str: ...
def render_slurm_node_gres(inventory: HardwareInventory) -> str: ...
def build_sbatch_argv(request: ExecutionRequest, policy: SlurmPolicy) -> tuple[str, ...]: ...
def parse_squeue_json(payload: str) -> tuple[ExecutionStatus, ...]: ...
def parse_slurm_allocation(env: Mapping[str, str], inventory: HardwareInventory) -> Allocation: ...
```

No shell command construction in production adapter code.

## Files

Tracked/implemented during plan validation:

- `config/slurm/slurm.conf.example`
- `config/slurm/slurmdbd.conf.example`
- `config/slurm/cgroup.conf.example`
- `config/slurm/gres.conf.example`
- `docs/reference/jobs/SLURM_BRUTUS.md`
- `.github/workflows/rcd-slurm-validation.yml`
- `tools/lab/run-campaign-durability/slurm_noble_smoke.sh`

Planned production:

- `tools/bigcherry/jobs/slurm.py`
- `tools/bigcherry/jobs/slurm_config.py`
- `tools/tests/jobs/test_slurm_executor.py`
- `tools/tests/jobs/test_slurm_config.py`

Generated host state:

- `/etc/slurm/slurm.conf`
- `/etc/slurm/slurmdbd.conf` mode 0600
- `/etc/slurm/gres.conf`
- `/etc/slurm/cgroup.conf`
- MariaDB `slurm_acct_db`
- `/var/log/slurm/bigcherry-jobcomp.log`

## Validation

### Already executed on real GitHub Ubuntu 24.04

The CI installs the actual Noble packages, not mocks. Observed/passed so far:

- exact `slurm-wlm 23.11.4-1.2ubuntu5` installation;
- MUNGE round-trip;
- real `slurmctld` + `slurmd` launch;
- node registration into `bc-build`/`bc-measure`;
- `squeue --json` schema availability;
- real hold/release job;
- no-accounting configuration reproduced `InvalidAccount`, driving the slurmdbd correction.

Current CI additionally must pass with the corrected accounting stack:

- loopback MariaDB + real `slurmdbd`;
- cluster/account/user association;
- build/license/measurement ordering;
- `afterok` dependency;
- `RequeueExit=75` and `SLURM_RESTART_COUNT=0,1`;
- cancellation;
- jobcomp and `sacct` history.

### Offline permanent tests before implementation completion

- golden GRES/node renders for synthetic inventories;
- exact `sbatch` argv snapshots including `--account=bigcherry`;
- `squeue --json` fixture mappings including unknown states;
- allocation -> RCD12 stable-ID mapping and stale-inventory rejection;
- import guard: domain modules do not import Slurm adapter;
- submission ambiguity/recovery correlation fixtures.

### Brutus-only gates

- `slurmd -G` against generated real GPU config;
- single/multi-GPU stable-ID attestation;
- complete ROCm/cgroup matrix or explicit fallback;
- production gate/window integration;
- controller restart/recovery with real queued/running work;
- dynamic hardware drift drain/reconfigure/resume.

## Effort & Risk

Medium-high. Real CI has already falsified one simplifying assumption, validating the approach. Remaining high-risk area is AMD ROCm/cgroup behavior and dynamic production coexistence, both Brutus-only.

## Standards

RCD02 architecture; RCD12 inventory; tested Ubuntu Noble Slurm 23.11.4 behavior; `docs/reference/jobs/SLURM_BRUTUS.md` is the installation procedure.

## Acceptance Criteria

- Real Noble CI passes complete MUNGE/MariaDB/slurmdbd/slurmctld/slurmd smoke.
- Explicit BigCherry account association eliminates `InvalidAccount`.
- Real license/priority/dependency/requeue/cancel tests pass.
- Generated AMD GRES passes `slurmd -G` on Brutus.
- Slurm allocation maps to stable RCD12 device identities.
- cgroup mode is empirically qualified or fallback selected.
- No BigCherry domain module depends on Slurm types.

## Notes

GitHub CI validates real Slurm service/scheduler semantics without GPUs. It does not qualify AMD GRES/cgroups/ROCm or production interference; those remain Brutus gates.

## Change Log

- 2026-09-26T00:51:56.813138+00:00 (created-by): Created by agent
- 2026-09-26 (dev-gpt-agent): Replaced static four-card assumptions with dynamic RCD12 inventory and generic partitions.
- 2026-09-26 (dev-gpt-agent): Real Noble 23.11.4 CI added; no-accounting design falsified by `InvalidAccount`, so v1 now includes minimal loopback MariaDB/slurmdbd associations.
