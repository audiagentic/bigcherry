# BigCherry job service — normative design

Status: **architecture agreed; real CPU/service validation green; Brutus GPU/production acceptance pending** (2026-09-26).

This file supersedes the earlier round-by-round text. RCD02-RCD12 implement this design. `docs/reference/jobs/SLURM_BRUTUS.md` is the concrete Brutus install/qualification runbook.

## 1. Goal and authority boundary

Replace `tools/lab/plan-qualification/{queue.sh,run_campaign.sh,make-serial-2.sh}` and hand-written watcher/switch loops with a durable service path where operators/agents submit job records and execution proceeds independently of the submitting SSH/gateway request.

**Buy scheduling; keep scientific policy in BigCherry.**

| Slurm owns | BigCherry owns |
| --- | --- |
| queued/running/held/cancelled native execution | run/series/session/attempt identity |
| CPU/GPU allocation, dependencies, partition tier, licenses | planned N, contract/composition freeze |
| scheduler process lifetime and same-commit requeue | commit resolution/pinning and retry legality |
| controller state and `jobcomp/filetxt` completion trail | model/corpus/producer/toolchain identity |
| cgroup process/device containment when qualified | stable hardware identity/cohort binding |
| | production coexistence policy |
| | scientific verdict/evidence/harvest/report/events |

Rejected as primary: custom SQLite scheduler daemon, pueue, HTCondor migration solely for one Windows host, Prefect/Dagster layered over another allocator, Nomad.

## 2. Brutus service topology

Ubuntu 24.04 / Noble Slurm 23.11.4 target:

```text
munge
  |
slurmctld ---- slurmd
  ^             |
  |             +-- managed child process / cgroup
  |
BigCherry SlurmExecutor
  sbatch / squeue --json / scontrol / scancel
```

BigCherry also installs systemd ingestion/observation/production-window helpers. No `slurmrestd`, MariaDB, or `slurmdbd` in v1.

### Empirical basis

GitHub Actions real-service validation on Ubuntu 24.04.5 installs distro `slurm-wlm 23.11.4-1.2ubuntu5` and proves the minimal stack works without accounts or an accounting DB. Latest green reference run: `36219793101`, validation head `ffdc08d00b76f0b8775f50a2cc6e48e1a4d20e01`.

The real run proved:

```text
MUNGE round-trip
slurmctld + slurmd registration
squeue --json
no-account job completion
real current-branch BigCherry process harness executed inside a Slurm job
hold/release
license-aware priority progress: build1 -> measure -> build2
afterok dependency
RequeueExit=75 with same job + SLURM_RESTART_COUNT 0 -> 1
cancel
controller restart retaining queued + running ownership
jobcomp/filetxt completion history
proctrack/task/jobacct cgroup stack with ConstrainDevices=no
```

Earlier experiments that appeared to require account associations were falsified by later minimal-stack runs; `Reason=InvalidAccount`/other pending reasons are not treated as permanent BigCherry failure classifiers.

## 3. Executor abstraction

BigCherry domain modules never import Slurm syntax or types.

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

Scientific `GpuRequirement` is resolved by RCD12/series binding **before** execution. Executor receives only the scheduler reservation (`architecture + reserved_count`). Slurm never implements scientific capability policy.

Adapters:

- `SlurmExecutor`: Brutus; only adapter that knows Slurm CLI, GRES, licenses, partitions, `SLURM_*`.
- `LocalExecutor`: Windows workstation/direct emergency execution.
- `FakeExecutor`: deterministic offline service tests.

## 4. Durable domain and service model

BigCherry is the durable domain authority. v1 uses filesystem records, not a custom scheduler DB.

```text
<work>/jobs/
  events.jsonl
  events.lock
  inbox/{pending,processing,accepted,rejected}/
  series/<series_id>/series.json
  runs/<run_id>/
    intent.json
    attempts/001/
      attempt.json
      submission-intent.json
      submission.json
      result.json
      stdout.log
      stderr.log
```

Rules:

1. canonical JSON + fsync + atomic rename for immutable manifests;
2. checksummed append-only event stream with host file lock and monotonic sequence;
3. retry creates a new attempt directory; terminal immutable records are never rewritten;
4. current status is projected from records + executor status; mutable `state` is not scientific authority;
5. SQLite may later be a derived index only, never scheduler or sole scientific history.

Submission path:

```text
jobs submit
  -> validate/canonicalize/freeze
  -> bind series hardware cohort
  -> durable run + inbox record
  -> return to caller
  -> systemd path/oneshot ingest
  -> attempt creation
  -> executor submission/reconciliation
```

A low-rate timer may invoke the same oneshot as a missed-path-event safety net; no custom scheduler loop is introduced.

## 5. Hardware discovery and scientific series binding

Hardware is discovered state, not hand-maintained `environment.local.toml` truth.

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

Scientific capability request:

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

A scientific series binds one exact hardware cohort **before any session is submitted**:

1. sort accepted candidates by stable `device_id`, never discovery order;
2. filter arch/VRAM/model;
3. reject ambiguous homogeneous model groups;
4. validate exact IDs;
5. for peer work choose canonical valid stable-ID tuple;
6. compute `hardware_cohort_hash` from selected stable IDs + relevant topology;
7. persist selected IDs and accepted inventory hash in immutable series record;
8. every session/attempt in that series uses the same binding.

Same-model replacement card or relevant topology move starts a new hardware cohort/series unless equivalence was pre-qualified. Windows-HIP gfx1100 and Linux-ROCm gfx1100 are separate platform environments/series.

### Architecture-only GRES and exact cohort safety

Slurm GRES types are architecture only:

```text
Name=gpu Type=gfx1100 File=/dev/dri/renderD128 Flags=amd_gpu_env
Name=gpu Type=gfx1100 File=/dev/dri/renderD129 Flags=amd_gpu_env
Name=gpu Type=gfx1201 File=/dev/dri/renderD130 Flags=amd_gpu_env
```

Never encode slot/UUID in `Type`.

Architecture/count GRES cannot promise which same-arch card is returned. Therefore if a bound series cohort is a **proper subset** of accepted GPUs of that architecture, v1 reserves **all GPUs of that architecture**, verifies the external allocation contains the pre-bound stable IDs, then narrows visibility to those IDs inside the already-exclusive allocation.

## 6. Slurm scheduler policy

Tracked `config/slurm/slurm.conf.example` is the policy source. Core settings:

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

One generic build partition and one generic measurement partition. No static per-card or production-GPU partitions.

Resource policy:

```text
build / v1.5 prepare:  build_slot:1 + host_activity:1
monolithic v1:        host_activity:2 + required GRES
v1.5 execute:         host_activity:2 + required GRES
```

`bf_licenses` + higher `PriorityTier` are accepted only because real Noble CI proved a running build followed by pending measure and later build executes `build1 -> measure -> build2`. All jobs have finite time.

`AuthInfo=cred_expire=30` is the v1 requeue delay policy; change only after Brutus launch/requeue validation.

## 7. Slurm submission contract and crash boundary

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

No `--account` in v1. Then persist `submission.json` with returned native handle.

Recovery of intent-without-handle:

1. query active Slurm state by stable correlation identity;
2. inspect `jobcomp/filetxt` plus attempt-local runner start/result sentinels;
3. exactly one match -> bind it;
4. proven no execution -> resubmit immutable request once;
5. ambiguous -> wake `recovery.ambiguous`; **never blind duplicate**.

A short completed job can disappear from `squeue` before handle persistence, so active queue state alone is insufficient.

The repository CI `service_recovery_smoke.py` currently passes 29 checks for durable submission/event recovery semantics.

## 8. Commit pinning and worktrees

`run_id` is scientific session identity; `attempt_no` is execution/harness attempt.

At new attempt creation:

1. resolve requested code ref immediately before attempt start;
2. create detached per-attempt BigCherry worktree at exact SHA;
3. link canonical gitignored `vendor/llama.cpp` and export absolute `BIGCHERRY_ENVIRONMENT`;
4. persist SHA before spawn;
5. all work in that attempt uses that worktree.

Branch changes after spawn never alter a running attempt. Harness fix uses `jobs retry --latest`: same `run_id`, `attempt+1`, new commit/new Slurm job. Same-commit transient requeue remains same attempt/native job.

Per-attempt worktree remains required while current evidence persistence writes under `REPO_ROOT`; external evidence sink later allows shared/read-only code worktrees.

## 9. Retry/failure semantics

```text
exit 0   execution completed; scientific verdict may PASS or FAIL
exit 75  predeclared same-commit transient -> Slurm RequeueExit, same attempt
exit 76  harness/code/config correction required -> terminal; explicit new attempt
exit 77  invalid input / contract / composition drift -> block
```

Record `SLURM_RESTART_COUNT` on requeue. Real Noble CI proves same native job restarts with counter `0 -> 1` under exit 75.

Scientific FAIL, small effect, or one-off performance outlier is not a retry predicate. PVPS09 must define any future result-independent health replacement rule before automatic remeasurement.

## 10. Production coexistence

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

Root-owned `bigcherry-measure-window.service`:

- drain in-flight production requests;
- stop/unload llama-swap/backend when conflicting;
- prove target/host quiet;
- block production reload while conflicting measurement runs;
- watchdog config/process/inventory drift;
- idempotent cleanup/restart production;
- hard `RuntimeMaxSec` and boot recovery.

Agent account may only start/stop that exact unit through narrow sudoers/polkit permission; no general root/scontrol permission.

Active non-conflicting inference coexistence remains disabled until `scheduler-isolation-v1` proves it harmless.

## 11. Tree maintenance and unsupported direct users

HI151 `core/tree_activity.py` remains repository maintenance fencing. Race-safe protocol:

```text
runner:      check maintenance -> publish lease -> recheck maintenance
maintenance: publish maintenance lock -> scan live leases
```

CI currently passes:

```text
real_recovery_smoke.py                 18 checks
tree_activity_race_smoke.py           100 checks over 50 race iterations
```

Ad-hoc build/bench activity that can perturb timed measurement must use supported BigCherry wrappers/activity policy. A privileged raw shell can always bypass a cooperative scheduler; that is an operator-policy violation, not a solvable Slurm property.

## 12. Phasing

### v1 — queue replacement

One monolithic Slurm job per existing `validation_campaign` run. It holds measurement resources throughout build+producer+ladder+production lane. This is deliberately conservative and can extend production exclusion.

Required before queue retirement: RCD06 M1 service safety, dynamic production gate, hardware identity, monitoring, Brutus GPU acceptance.

### v1.5 — downtime reduction

Add:

```text
validation_campaign --prepare-only
validation_campaign --execute-only --prepared <manifest>
```

Prepare materializes/builds stock/base/control/subject/producer artifacts and emits immutable prepared manifest. Execute re-verifies attempt commit, contract/composition/source/build identities before correctness/performance/ladder/production/evidence.

Slurm dependency: `afterok:<prepare>`. Production remains available during build.

### v2+

Activate RCD01 operation/result durability, then full build/correctness/measure DAG, harvest/report and only later any qualified overlap of timed measurement with builds.

## 13. Existing component consolidation

| Component | Decision |
| --- | --- |
| `core/tree_activity.py` HI151 | keep/extend; race-safe lease/maintenance admission |
| `experiment/bundle.run_managed()` | extend to streamed files, heartbeat/event sink; `capture_output=True` is not acceptable for ~1.5 GB logs |
| `telemetry.py` RQW01 | keep; add JSON/composite sinks |
| `tuning/journal.py` HI48 | extract reusable canonical/atomic/checksummed durability primitives without changing existing wire format |
| campaign/build resource locks | local fallback/cache integrity only; Slurm owns managed GPU/build/quiet resources |
| RCD01 | activate operation/result durability; do not recreate scheduler/distributed lock engine |
| AudiAgentic gateway | client/provider adapter only; never execution authority |

## 14. Observability

Authorities:

```text
squeue --json     current Slurm queue/running state
jobcomp/filetxt   native terminal execution trail
BigCherry store   domain/scientific history/events/artifacts
journald          service diagnostics
```

No `sacct` dependency in v1 because no accounting DB is configured.

Normalized `bigcherry jobs status --json` never exposes raw Slurm schema as public API. Include service health, queue/stage counts, disk/ccache, accepted inventory/drift, stable-ID GPU allocation/telemetry, run/series/session/attempt/identity/progress/result.

Host event stream is immutable checksummed JSONL with monotonic sequence; agents reconnect with `events --after N --jsonl [--follow] [--wake-only]`.

Wake: harness failure/retry exhaustion/stall/disk hard/GPU unhealthy/inventory drift/contract or composition drift/node unavailable/harvest failure/recovery ambiguity/service degradation/production contamination/window overrun/restart failure.

Record only: submit/queue/start/completion, telemetry, scientific PASS/FAIL, ladder/noise, same-commit automatic requeue. Series complete = notify.

## 15. Logs/disk/cache

- work/build/tmp/attempt logs on `/mnt/data`;
- global ccache under work root, `CCACHE_BASEDIR` common root, `CCACHE_NOHASHDIR=1`, bounded ~100 GiB policy;
- root and work filesystem preflight + runtime hard-reserve watchdog;
- current `run_managed(capture_output=True)` must be replaced with streaming before production job service; CI proves 8 MiB correctness but explicitly keeps large-output buffering as a known gap;
- completed server logs may be truncated/rotated according to evidence policy; required structured artifacts remain separate;
- disk-pressure abort is harness/environment failure, never scientific FAIL.

## 16. Validation already executed

Repository validation workflow: `.github/workflows/rcd-slurm-validation.yml`.

Latest green reference: GitHub Actions run `36219793101`, validation head `ffdc08d00b76f0b8775f50a2cc6e48e1a4d20e01`.

### Planning/domain falsifier

```text
mock_pipeline.py --self-test: 32 checks
```

Covers capability resolution, deterministic hardware cohort selection, production claims/conflicts, retry semantics and FakeExecutor dependency behavior.

### Real BigCherry process/CLI

```text
real_bigcherry_process_smoke.py: 25 checks
real_bundle_failure_smoke.py:   12 checks
```

Uses current production modules and real subprocesses. Verifies managed process success/failure/launch-failure, stdout/stderr, 8 MiB output, HI48 torn-tail recovery, real `validation_campaign.main()` producer-dispatch parsing, interruption/failure injection and durable bundle behavior.

### Real recovery/concurrency

```text
real_recovery_smoke.py:       18 checks
tree_activity_race_smoke.py: 100 checks / 50 iterations
service_recovery_smoke.py:    29 checks
```

Verifies HI151 exclusion/crash cleanup, real detached git worktree pinning, concurrent admission racing, submission-intent crash windows, ambiguous recovery and event sequencing.

### Real Noble Slurm

Actual Noble daemons/commands prove all items listed in §2. The Slurm job also executes the **current branch BigCherry process harness**, so scheduler/process integration is not only mocked.

Failures encountered while developing this CI were treated as design findings: unsupported `slurmctld -t`, overly specific pending-reason parsing, and unnecessary slurmdbd/account assumptions were removed rather than waived.

## 17. Brutus-only acceptance gates

GitHub CPU/service CI cannot establish:

- actual AMD stable UUID/RSMI ID availability;
- generated real `gres.conf` accepted by `slurmd -G`;
- `/dev/kfd` + render node behavior under `ConstrainDevices=yes`;
- HIP enumeration/order with cgroups for both toolchains;
- peer access and `-sm tensor` topology/4096-context preflight;
- llama-swap process attribution/window behavior;
- real disk/ccache/root-pressure behavior under builds;
- production-loaded-idle/active noise equivalence;
- monolithic `validation_campaign` evidence parity on real GPUs.

Initial Brutus cgroup mode is the already-proven fallback:

```text
ConstrainDevices=no
```

Only enable `ConstrainDevices=yes` after the full ROCm matrix passes. If it fails any supported cell, keep the fallback and record that GRES is scheduler/cooperative isolation, not a device security boundary.

## 18. Security

- dedicated `bigcherry` Unix account for managed submission;
- SSH Ed25519, no password/root SSH, LAN firewall;
- no MariaDB/slurmdbd/slurmrestd in v1;
- root measurement-window helper callable only through exact unit start/stop permission;
- no secrets in JobSpec/events/argv telemetry;
- generated host config and accepted hardware inventory are operator-controlled state.

## 19. Retirement gate

Do not retire shell queue until:

1. real Noble CI remains green;
2. RCD03/RCD04/RCD05/RCD06-M1/RCD09/RCD11/RCD12 implementation is complete;
3. required Brutus capability classes pass real hardware acceptance or are explicitly unsupported;
4. forced incident matrix passes;
5. one complete planned series runs without shell watcher intervention;
6. evidence/report parity is confirmed;
7. rollback path is documented/tested.

Retire queue/switch/watch wrappers first. Keep `summarize.py`/`noise.py` until RCD08 parity.

## 20. Remaining open work

No unresolved platform-choice objection remains. Real Noble/CPU/service validation is green. Remaining work is implementation of production `tools/bigcherry/jobs`/hardware/production-gate modules plus Brutus-only AMD/ROCm/llama-swap/noise/evidence acceptance. These are explicit acceptance gates, not assumptions hidden behind mocks.
