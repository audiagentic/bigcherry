# BigCherry job service — normative design

Status: **architecture agreed; implementation/hardware acceptance pending** (2026-09-26).

This file supersedes the earlier round-by-round text. RCD02-RCD12 implement this design. `docs/reference/jobs/SLURM_BRUTUS.md` is the concrete host-install runbook.

## 1. Goal and boundary

Replace `tools/lab/plan-qualification/{queue.sh,run_campaign.sh,make-serial-2.sh}` and hand-written watcher/switch loops with a service path where operators/agents submit durable job records and execution proceeds independently of the submitting SSH/gateway request.

**Buy scheduling; keep scientific policy in BigCherry.**

| Slurm/slurmdbd owns | BigCherry owns |
| --- | --- |
| queued/running/held/cancelled execution state | run/series/session/attempt identity |
| CPU/GPU allocation, dependencies, priorities, licenses | planned N, contract/composition freeze |
| scheduler process lifetime and same-commit requeue | commit resolution/pinning and retry legality |
| minimal account associations/executor accounting | model/corpus/producer/toolchain identity |
| cgroup process/device containment when qualified | stable hardware identity/cohort binding |
| basic execution history | production coexistence policy |
| | scientific verdict/evidence/harvest/report/events |

Rejected as primary: custom SQLite scheduler daemon, pueue, HTCondor migration solely for one Windows host, Prefect/Dagster layered over another resource allocator, Nomad.

## 2. Brutus service topology

Ubuntu 24.04 / distro Slurm 23.11.4 target:

```text
MariaDB 127.0.0.1:3306   # Slurm association/accounting DB only
  ^
slurmdbd 127.0.0.1:6819
  ^
slurmctld ---- slurmd
  ^
  |
BigCherry SlurmExecutor (sbatch/squeue/scontrol/scancel/sacct CLI)
```

Also: `munge`, BigCherry systemd ingestion/observation/window helpers. No `slurmrestd` in v1.

### Why MariaDB/slurmdbd is v1-required

Real GitHub Ubuntu 24.04 validation installed Slurm 23.11.4, started MUNGE/slurmctld/slurmd, registered the node and exercised `squeue --json` and hold/release. Without an association store, subsequent jobs reproduced `PENDING Reason=InvalidAccount`. Therefore the earlier "jobcomp only, no slurmdbd" design is rejected.

Use exactly one local cluster/account for managed work:

```text
cluster: bigcherry
account: bigcherry
service user default account: bigcherry
```

Every managed `sbatch` passes `--account=bigcherry`. MariaDB binds loopback only. BigCherry never queries the database directly.

Tracked templates:

```text
config/slurm/slurm.conf.example
config/slurm/slurmdbd.conf.example
config/slurm/cgroup.conf.example
config/slurm/gres.conf.example
```

## 3. Executor abstraction

BigCherry domain modules never import Slurm types/syntax.

```python
@dataclass(frozen=True)
class SchedulerGpuRequest:
    architecture: str
    reserved_count: int

@dataclass(frozen=True)
class ResourceRequest:
    cpu_slots: int
    gpu: SchedulerGpuRequest | None
    activity_class: Literal["build", "correctness", "timed-measure", "harvest"]
    memory_bytes: int | None
    timeout_seconds: int

@dataclass(frozen=True)
class ExecutionRequest:
    execution_id: str
    command: tuple[str, ...]
    cwd: str
    env: tuple[tuple[str, str], ...]
    stdout_path: str
    stderr_path: str
    resources: ResourceRequest
    dependencies: tuple[str, ...] = ()

@dataclass(frozen=True)
class ExecutionHandle:
    executor: str
    native_id: str
    execution_id: str

class Executor(Protocol):
    def submit(self, request: ExecutionRequest) -> ExecutionHandle: ...
    def status(self, handle: ExecutionHandle) -> ExecutionStatus: ...
    def cancel(self, handle: ExecutionHandle) -> None: ...
    def control(self, handle: ExecutionHandle, action: ExecutorControl) -> None: ...
    def allocation(self, handle: ExecutionHandle) -> Allocation | None: ...
    def events(self, handle: ExecutionHandle, *, after: int | None = None) -> Iterable[ExecutorEvent]: ...
```

Important split: scientific `GpuRequirement` (arch/count/min VRAM/model/peer/exact IDs) is resolved by RCD12/series binding **before** execution. The Executor receives only the resulting scheduler reservation (`architecture + reserved_count`). Slurm must not implement scientific capability policy.

Adapters:

- `SlurmExecutor`: Brutus; only adapter that knows `sbatch`, `squeue`, `scontrol`, GRES, licenses, partitions, `SLURM_*`.
- `LocalExecutor`: Windows workstation/direct emergency execution.
- `FakeExecutor`: deterministic offline service tests.

## 4. Hardware discovery and series binding

Hardware is discovered state, not static `environment.local.toml` truth.

`DeviceRecord` contains at least:

```text
stable device_id + identity_source
arch
model
VRAM
current PCI BDF
current /dev/dri/renderD*
NUMA / relevant topology
AMD driver
```

Stable ID preference: AMD UUID -> RSMI unique ID -> serial -> explicit weak hardware epoch. Ordinal/BDF/render node are locators only.

Persist:

```text
/var/lib/bigcherry/hardware/observed.json
/var/lib/bigcherry/hardware/accepted.json
```

Material inventory drift drains Brutus and wakes the operator before new work. Add/remove/replacement/topology change requires explicit acceptance and regenerated GRES config.

### Capability request

```python
@dataclass(frozen=True)
class GpuRequirement:
    architecture: str
    count: int = 1
    min_vram_bytes: int = 0
    homogeneous_model: bool = True
    model: str | None = None
    require_peer_access: bool = False
    exact_device_ids: tuple[str, ...] = ()
```

### Deterministic series binding

A scientific series binds one exact hardware cohort **before any session is submitted**:

1. sort accepted candidates by stable `device_id`, never discovery order;
2. filter arch/VRAM/model;
3. reject ambiguous homogeneous model groups;
4. validate exact IDs;
5. for peer work choose canonical valid stable-ID tuple;
6. compute `hardware_cohort_hash` from selected stable IDs + relevant topology;
7. persist selected IDs and accepted inventory hash in immutable series record;
8. every session/attempt in that series uses that same binding.

Same-model replacement card or relevant topology move starts a new hardware cohort/series unless equivalence was pre-qualified. Windows-HIP gfx1100 and Linux-ROCm gfx1100 are separate platform environments/series.

### Architecture-only GRES and exact cohort safety

Slurm GRES types are architecture only:

```text
Name=gpu Type=gfx1100 File=/dev/dri/renderD128 Flags=amd_gpu_env
Name=gpu Type=gfx1100 File=/dev/dri/renderD129 Flags=amd_gpu_env
Name=gpu Type=gfx1201 File=/dev/dri/renderD130 Flags=amd_gpu_env
```

Never `gfx1100_0`.

Architecture/count GRES cannot promise which same-arch card is returned. Therefore if a bound series cohort is a **proper subset** of accepted GPUs of that architecture, v1 reserves **all GPUs of that architecture**, verifies the external allocation contains the pre-bound stable IDs, then narrows visibility to those IDs inside the already-exclusive allocation. Correctness/scientific comparability beats utilization.

## 5. Slurm scheduler policy

```ini
SchedulerType=sched/backfill
SchedulerParameters=bf_licenses
PriorityType=priority/multifactor
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory

Licenses=host_activity:2,build_slot:1

AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=127.0.0.1
AccountingStoragePort=6819
AccountingStorageEnforce=associations

RequeueExit=75

PartitionName=bc-build   Nodes=brutus PriorityTier=10  State=UP MaxTime=00:45:00
PartitionName=bc-measure Nodes=brutus PriorityTier=100 State=UP MaxTime=00:45:00
```

One generic build partition and one generic measurement partition. No static per-card or production-GPU partitions.

Resource policy:

```text
build:            build_slot:1 + host_activity:1
monolithic v1:    host_activity:2 + required GRES
v1.5 execute:     host_activity:2 + required GRES
```

`bf_licenses` + higher measurement `PriorityTier` must be real-service tested so a stream of builds cannot indefinitely delay a pending measurement. Every job has finite time.

## 6. Submission contract and crash boundary

Before external submission BigCherry durably writes:

```text
submission-intent.json
  execution_id
  request hash
  run_id / attempt
  exact pinned commit
```

Slurm argv shape:

```bash
sbatch --parsable \
  --account bigcherry \
  --job-name 'bc:<run_id>:a<attempt>' \
  --comment 'bigcherry:<execution_id>' \
  --partition '<bc-build|bc-measure>' \
  --licenses '<resolved>' \
  [--gres 'gpu:<arch>:<reserved-count>'] \
  --time '<finite>' \
  --output '<attempt>/slurm-%j.out' \
  --error '<attempt>/slurm-%j.err' \
  --export 'ALL,BIGCHERRY_RUN_ID=...,BIGCHERRY_ATTEMPT=...,BIGCHERRY_ATTEMPT_ROOT=...' \
  '<attempt>/launch.sh'
```

Then persist `submission.json` with returned native handle.

Recovery of intent-without-handle:

1. query Slurm by correlation comment/name and active state;
2. inspect `sacct`/jobcomp and attempt-local start/result sentinels;
3. exactly one match -> bind it;
4. proven no execution -> resubmit immutable request;
5. ambiguous -> wake `recovery.ambiguous`; **never blind duplicate**.

Gateway/SSH lifetime is irrelevant after durable local submission.

## 7. Commit pinning and worktrees

`run_id` is scientific session identity; `attempt_no` is execution/harness attempt.

At new attempt creation:

1. resolve requested code ref immediately before attempt start;
2. create detached per-attempt BigCherry worktree at exact SHA;
3. link canonical gitignored `vendor/llama.cpp` and export absolute `BIGCHERRY_ENVIRONMENT`;
4. persist SHA before spawn;
5. all stages in attempt use that worktree.

Branch changes after spawn never alter a running attempt. A harness fix uses `jobs retry --latest`: same `run_id`, `attempt+1`, new commit/new Slurm job. Same-commit transient requeue remains same attempt/native job.

Per-attempt worktree is required in v1 because current evidence persistence writes under `REPO_ROOT`; external evidence sink later allows shared/read-only code worktrees.

Real temporary-Git CI verifies branch advance does not mutate an existing detached attempt worktree.

## 8. Retry/failure semantics

```text
exit 0   execution completed; scientific verdict may PASS or FAIL
exit 75  predeclared same-commit transient -> Slurm RequeueExit, same attempt
exit 76  harness/code/config correction required -> terminal; explicit new attempt
exit 77  invalid input / contract / composition drift -> block
```

Record `SLURM_RESTART_COUNT` on requeue. Scientific FAIL, small effect, or a one-off `-30%` result is not a retry predicate. PVPS09 must define any future result-independent health replacement rule before automatic remeasurement.

Failure taxonomy remains separate from scientific verdict: harness/parser/preflight/build/disk/process/stall/host failures can be retried only by predeclared policy; real correctness/performance FAIL is pipeline completion.

## 9. Production coexistence

llama-swap may use any GPU/card combination and may change over time. Slurm cannot see processes outside Slurm; production conflict is a BigCherry cross-world gate.

Before measurement derive:

```python
@dataclass(frozen=True)
class ProductionSnapshot:
    config_hash: str
    potential_devices: frozenset[str] | None   # None == all/unknown/fail closed
    running_devices: frozenset[str]
    observed_devices: frozenset[str]
```

Sources: llama-swap configuration, `/running`, backend command/env selectors, live process environments, AMD process/VRAM attribution. Production models should declare a stable-device claim; missing/contradictory claim => potential `all`.

After Slurm allocation:

```text
if target stable IDs intersect production potential set:
    exclusive production window
else:
    production may remain loaded, subject to qualified idle/noise attestation
```

### Exclusive window

Root-owned `bigcherry-measure-window.service`:

- drain in-flight production requests;
- stop/unload llama-swap/backend;
- prove target/host quiet;
- run measurement while reload is blocked;
- watchdog config/process/inventory drift;
- idempotent cleanup/restart production;
- hard `RuntimeMaxSec` and boot recovery.

Agent `bigcherry` user may only `systemctl start|stop` that exact unit via narrow sudoers/polkit rule; no general root/scontrol permission.

Non-conflicting production remains up only under the currently qualified idle/noise policy. Active non-conflicting inference coexistence is not allowed until `scheduler-isolation-v1` proves it harmless.

## 10. Tree maintenance and other direct users

HI151 `core/tree_activity.py` remains repository maintenance fencing. Validation found and fixed a real admission gap.

Race-safe protocol:

```text
runner:      check maintenance -> publish lease -> recheck maintenance
maintenance: publish maintenance lock -> scan live leases
```

Thus pin-bump/maintenance and a supported long-running campaign cannot both be admitted even when they race.

Ad-hoc build/bench activity that can perturb timed measurement must use supported BigCherry wrappers/activity policy. A privileged raw shell can always bypass a cooperative scheduler; that is an operator-policy violation and cannot be solved by Slurm.

## 11. Phasing

### v1 — queue replacement

One monolithic Slurm job per existing `validation_campaign` run. It holds measurement resources throughout build+producer+ladder+production lane. Lower throughput/longer production exclusion is accepted only for initial cutover.

Required before retirement: RCD06 M1 service safety (external evidence target/frozen composition/typed preflight/progress as defined there), dynamic production gate, hardware identity, monitoring, real Brutus acceptance.

### v1.5 — cheap downtime reduction

Add:

```text
validation_campaign --prepare-only
validation_campaign --execute-only --prepared <manifest>
```

Prepare materializes/builds stock/base/control/subject/producer artifacts and emits an immutable prepared manifest. Execute re-verifies attempt commit, contract/composition/source/build identities before correctness/performance/ladder/production/evidence.

Slurm `afterok:<prepare>` dependency; production remains available during build.

### v2+

Activate RCD01 durable operation/result semantics, then full build/correctness/measure DAG, harvest/report and only later overlap timed measurement with builds after isolation qualification.

## 12. Existing component consolidation

| Component | Decision |
| --- | --- |
| `core/tree_activity.py` HI151 | keep/extend; race-safe lease/maintenance admission |
| `experiment/bundle.run_managed()` | extend to streamed files, heartbeat/event sink; current `capture_output=True` is not acceptable for 1.5 GB logs |
| `telemetry.py` RQW01 | keep; add JSON/composite sinks |
| `tuning/journal.py` HI48 | extract reusable canonical/atomic/checksummed durability primitives without changing existing wire format |
| campaign/build resource locks | local fallback/cache integrity only; Slurm owns managed GPU/build/quiet resources |
| RCD01 | activate operation/result durability; do not recreate scheduler/distributed lock engine |
| AudiAgentic gateway | client/provider adapter only; never execution authority |

## 13. Observability

Authorities:

```text
squeue --json     current Slurm state
sacct/slurmdbd    native historical executor state
jobcomp/filetxt   lightweight independent Slurm completion trail
BigCherry store   domain/scientific history/events/artifacts
journald          service diagnostics
```

Normalized `bigcherry jobs status --json` never exposes raw Slurm schema as public API. Include service health, queue/stage counts, disk/ccache, accepted inventory/drift, stable-ID GPU allocation/telemetry, run/series/session/stage/attempt/identity/progress/result.

Host event stream is immutable checksummed JSONL with monotonic sequence; agents reconnect with `events --after N --jsonl [--follow] [--wake-only]`.

Wake: harness failure/retry exhaustion/stall/disk hard/GPU unhealthy/inventory drift/contract or composition drift/node unavailable/harvest failure/recovery ambiguity/service degradation/production contamination/window overrun/restart failure.

Record only: submit/queue/start/completion, telemetry, scientific PASS/FAIL, ladder/noise, same-commit automatic requeue. Series complete = notify.

## 14. Logs/disk/cache

- work/build/tmp/attempt logs on `/mnt/data`;
- global ccache under work root, `CCACHE_BASEDIR` common root, `CCACHE_NOHASHDIR=1`, bounded ~100 GiB policy;
- root and work filesystem preflight + runtime hard-reserve watchdog;
- current `run_managed(capture_output=True)` must be replaced with streaming before production job service; never buffer server logs in memory;
- completed server logs may be truncated/rotated according to evidence policy; full required structured artifacts must be preserved separately;
- disk-pressure abort is harness/environment failure, never scientific FAIL.

## 15. Real validation already in repository

### Planning falsifier

```bash
PYTHONPATH=tools python tools/lab/run-campaign-durability/mock_pipeline.py --self-test
# 32 checks
```

### Real BigCherry process smoke

`real_bigcherry_process_smoke.py` imports current production modules and uses real subprocesses. It verifies `run_managed`, module CLI, child return codes 0/76/127, stdout/stderr, 8 MiB output, HI48 torn-tail recovery, and actual `validation_campaign.main()` producer-dispatch parsing with only hardware/build producer body mocked.

### Real recovery smoke

`real_recovery_smoke.py` uses production HI151 plus real Git repositories/worktrees to verify maintenance/lease exclusion, crashed stale lease pruning, detached attempt pinning across branch advance, dirty attempt isolation and worktree cleanup.

### Real Noble Slurm CI

`.github/workflows/rcd-slurm-validation.yml` installs actual Ubuntu Noble Slurm/MUNGE/MariaDB/slurmdbd and exercises real daemons/commands. It is required to prove:

```text
MUNGE
slurmdbd association
slurmctld/slurmd registration
squeue --json
hold/release/cancel
afterok
license priority/progress
RequeueExit=75 / SLURM_RESTART_COUNT
jobcomp + sacct
```

Failures in this test are design/config findings, not waived as mocks.

## 16. Brutus-only acceptance gates

GitHub CPU/service CI cannot establish:

- actual AMD stable UUID/RSMI ID availability;
- generated real `gres.conf` accepted by `slurmd -G`;
- `/dev/kfd` + render node behavior under `ConstrainDevices=yes`;
- HIP enumeration/order with cgroups for both toolchains;
- peer access and `-sm tensor` topology/4096-context preflight;
- llama-swap process attribution/window behavior;
- real disk/ccache/root-pressure behavior under builds;
- production-loaded-idle/active noise equivalence;
- monolithic evidence parity on real GPUs.

These remain explicit RCD10 hardware acceptance requirements. If cgroup device filtering fails any supported cell, set `ConstrainDevices=no` and record that GRES is scheduling/isolation-by-cooperation, not a security boundary.

## 17. Security

- dedicated `bigcherry` Unix account for managed submission;
- SSH Ed25519, no password/root SSH, LAN firewall;
- MariaDB loopback only;
- slurmdbd local only;
- no slurmrestd;
- root measurement-window helper callable only through exact unit start/stop permission;
- no secrets in JobSpec/events/argv telemetry; host DB password lives in root-readable untracked config.

## 18. Retirement gate

Do not retire shell queue until:

1. real Noble CI passes;
2. RCD03/RCD04/RCD05/RCD06-M1/RCD09/RCD11/RCD12 implementation is complete;
3. all currently required Brutus capability classes pass real hardware acceptance or are explicitly unsupported;
4. forced incident matrix passes;
5. one complete planned series runs without shell watcher intervention;
6. evidence/report parity is confirmed;
7. rollback path is documented/tested.

Retire queue/switch/watch wrappers first. Keep `summarize.py`/`noise.py` until RCD08 parity.

## 19. Remaining empirical gates

No unresolved platform-choice objection remains. Open **acceptance** gates are the Brutus-only items above plus completion of real Noble Slurm CI and production implementation modules. Those are not assumed passed by planning mocks.
