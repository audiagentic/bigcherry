---
id: RCD03
order: 3
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:51:56.813138+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Install and qualify the Brutus Slurm execution substrate

## Description

Install the minimal single-node Slurm service used by `SlurmExecutor`: MUNGE, `slurmctld`, `slurmd`; no slurmdbd/MariaDB/slurmrestd in v1. Slurm is an execution/resource substrate only. GPU inventory and generated architecture-typed GRES come from RCD12 discovered hardware state, not hand-maintained slot configuration. Production coexistence remains a BigCherry gate (RCD11), not a Slurm partition model.

This item must produce reproducible config templates/generators and offline rendering tests before owner/root installation. Hardware qualification is explicit and cannot be simulated away.

## Steps

1. Add tracked Slurm config templates/default policy; keep rendered host files under `/etc/slurm` generated from accepted RCD12 inventory.
2. Install `slurm-wlm`/MUNGE on Brutus; create `bigcherry` Slurm account/user mapping only as required by local auth.
3. Configure one build and one measurement partition, license-aware backfill, basic completion log, cgroup accounting/containment.
4. Generate `gres.conf` + node `Gres=` from RCD12 inventory; run `slurmd -G` before applying.
5. Add `SlurmExecutor` command rendering/parsing behind the generic RCD04 interface; subprocess calls are injectable for offline tests.
6. Run cgroup/ROCm falsification matrix. If any supported configuration fails, set `ConstrainDevices=no`; retain scheduling GRES + `ROCR_VISIBLE_DEVICES` + BigCherry attestation.
7. Record installed Slurm version/config digest and selected cgroup mode in host policy/status.

## Detailed Solution & Technical Design

### Slurm policy

Tracked template semantics:

```ini
SchedulerType=sched/backfill
SchedulerParameters=bf_licenses
PriorityType=priority/multifactor
Licenses=host_activity:2,build_slot:1

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

No static production-GPU partition and no per-card partition. Every job has finite time. `bf_licenses` + higher `PriorityTier` ensures a queued measurement can reserve/plan `host_activity:2` and is not indefinitely displaced by new build work.

Initial v1 monolithic campaign requests `bc-measure`, `host_activity:2`, required architecture GRES. v1.5 prepare requests `bc-build`/`build_slot:1,host_activity:1`; execute requests `bc-measure`/`host_activity:2`.

### GRES

RCD12 returns explicit architecture-typed lines:

```text
Name=gpu Type=gfx1100 File=/dev/dri/renderD128 Flags=amd_gpu_env
Name=gpu Type=gfx1100 File=/dev/dri/renderD129 Flags=amd_gpu_env
Name=gpu Type=gfx1201 File=/dev/dri/renderD130 Flags=amd_gpu_env
```

and node summary:

```text
Gres=gpu:gfx1100:2,gpu:gfx1201:1
```

No type contains a slot/index. `File=` is an operational binding generated after discovery. `Flags=amd_gpu_env` supplies Slurm-managed `ROCR_VISIBLE_DEVICES`. Under Slurm BigCherry must preserve this value and must not write host-global physical indices into `HIP_VISIBLE_DEVICES`.

For a constraint requiring an exact/subset card among identical architecture types, RCD12 may request all GPUs of that architecture, then select UUID(s) inside the allocation; correctness over utilization.

### SlurmExecutor

Planned module:

```python
# tools/bigcherry/jobs/slurm.py
class SlurmExecutor(Executor):
    def __init__(self, *, runner: CommandRunner, policy: SlurmPolicy): ...
    def submit(self, request: ExecutionRequest) -> ExecutionHandle: ...
    def status(self, handle: ExecutionHandle) -> ExecutionStatus: ...
    def cancel(self, handle: ExecutionHandle) -> None: ...
    def allocation(self, handle: ExecutionHandle) -> Allocation | None: ...
    def events(self, handle: ExecutionHandle, *, after: int | None = None) -> Iterable[ExecutorEvent]: ...
```

`CommandRunner` is injected so unit tests feed saved `sbatch --parsable`, `squeue --json`, `scontrol`, and environment fixtures without Slurm installed. Domain packages may not import this module.

Submission wrapper writes an immutable BigCherry attempt manifest before `sbatch`. Slurm native job ID is returned as opaque `ExecutionHandle.native_id`, never incorporated into scientific identity.

### cgroups

Start with:

```ini
CgroupPlugin=autodetect
ConstrainDevices=yes
ConstrainCores=yes
```

Hardware matrix per installed toolchain and each discovered capability cohort must prove:

- `/dev/kfd` usable;
- allocated render nodes usable and unallocated nodes inaccessible;
- Slurm `ROCR_VISIBLE_DEVICES` resolves to exactly the allocated stable devices;
- `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` do not conflict;
- 100 bounded HIP init/enumeration loops without hangs/errors;
- llama-bench and llama-server attestation succeed;
- required peer/`-sm tensor` topology succeeds with its producer preflight.

Fallback is an explicit recorded host mode: `ConstrainDevices=no`. This weakens isolation, not scheduling identity.

## Code Samples & Guidance

Offline renderer must be deterministic:

```python
def render_slurm_gres(inventory: HardwareInventory) -> str: ...
def render_slurm_node_gres(inventory: HardwareInventory) -> str: ...
def build_sbatch_argv(request: ExecutionRequest, policy: SlurmPolicy) -> tuple[str, ...]: ...
def parse_squeue_json(payload: str) -> tuple[ExecutionStatus, ...]: ...
```

Do not shell-construct command strings; argv only. Do not parse human-column output.

## Files

Planned:

- `tools/bigcherry/jobs/slurm.py`
- `tools/bigcherry/jobs/slurm_config.py`
- `tools/tests/jobs/test_slurm_executor.py`
- `tools/tests/jobs/test_slurm_config.py`
- `config/slurm/slurm.conf.example`
- `config/slurm/cgroup.conf.example`
- `docs/reference/jobs/SLURM_BRUTUS.md`

Generated/untracked host state:

- `/etc/slurm/slurm.conf`
- `/etc/slurm/gres.conf`
- `/etc/slurm/cgroup.conf`
- `/var/log/slurm/bigcherry-jobcomp.log`

## Validation

Offline, required before sudo/install:

- golden render tests for 0/1/N devices and mixed architectures;
- no generated GRES type contains device ordinal/BDF;
- `sbatch` argv snapshots for build, monolithic measure, dual-GPU and dependency jobs;
- `squeue --json` parser fixtures: pending/running/completed/held/cancelled/unknown;
- `SLURM_JOB_GPUS`/allocation mapping fixture resolves through RCD12 inventory and rejects unknown/stale inventory hash;
- starvation policy config contains `bf_licenses`, higher measurement `PriorityTier`, finite MaxTime;
- domain import guard proves `bigcherry.jobs.model/series/service` do not import `bigcherry.jobs.slurm`.

Hardware gates:

- `slurmd -G` clean against generated config;
- single and multi-GPU allocation attestation;
- complete cgroup matrix or recorded fallback;
- kill/restart `slurmctld` does not lose accepted queued/running execution ownership;
- `RequeueExit=75` increments `SLURM_RESTART_COUNT`.

## Effort & Risk

Medium. Highest risk is ROCm behavior under cgroup device filtering, explicitly gated. Slurm install/config errors are operational and reversible; keep old lab queue until RCD10 acceptance.

## Standards

RCD02 architecture; RCD12 inventory; Ubuntu Noble Slurm 23.11 behavior fixed by the design review. No slurmdbd in v1.

## Acceptance Criteria

- Minimal services running: MUNGE/slurmctld/slurmd only.
- Generated architecture GRES accepted by `slurmd -G`.
- Slurm allocation maps to stable RCD12 device identities.
- License/priority configuration guarantees measurement progress.
- cgroup mode is empirically qualified or fallback explicitly selected.
- Offline executor/config tests pass.
- No BigCherry domain module depends on Slurm types.

## Notes

Owner/root installation is required and is not claimed by offline tests.

## Change Log

- 2026-09-26T00:51:56.813138+00:00 (created-by): Created by agent
- 2026-09-26 (dev-gpt-agent): Replaced static four-card/sacct assumptions with dynamic RCD12 inventory, generic partitions and testable Slurm adapter.
