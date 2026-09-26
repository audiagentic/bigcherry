# BigCherry job service — agreed design

Status: **agreed, no open objections** (2026-09-26). Adversarial review with
dev-gpt-agent session `ses_4019cfc56e774dd2`: first pass `req_c70aecafc33e4b59`,
deep review `req_fb219ca02c184706`, build-vs-buy `req_27c6e52d2ffa4904`, round 2
`req_714f0d248f364ce6` (O1-O8), round 3 `req_b2e5be0c112047b0` (O9-O11).
Replaces the lab queue in `tools/lab/plan-qualification/`. Plan items:
`docs/planning/active/run-campaign-durability/RCD02`-`RCD11`.

Remaining gates are empirical acceptance tests, not design questions:
ROCm under cgroup device constraints (O3 matrix) and production-loaded-idle
isolation on GPU2/GPU3 (scheduler-isolation-v1).

## 1. Architecture

Adopt free, open-source **Slurm** (GPL, Ubuntu Noble 23.11.4) as the durable
queue, process lifetime, GPU and host-resource scheduler on Brutus. BigCherry
builds only a thin domain layer behind a **platform-neutral Executor**; no
BigCherry domain module imports Slurm types, job IDs, GRES syntax, partitions or
`squeue`.

| Slurm owns | BigCherry owns |
|---|---|
| queued/running/held/cancelled state, dependencies, priority | run_id / series_id / session ordinal, JobSpec + batch expansion |
| CPU slots, GPU allocation (static GRES), licenses | contract hash, planned N, frozen validated patch IDs + digests |
| process containment (cgroups, if qualified), requeue mechanics | commit selection/pinning per attempt, toolchain logical name -> path/digest |
| basic completion history (`jobcomp/filetxt`), `squeue --json` | model/corpus identity, producer preflights, failure classification |
| | legality of retries (no optional stopping), evidence harvest + git commit, reports, events |

Rejected: custom SQLite scheduler daemon (rebuilds what Slurm provides); pueue
(no GPU/device model, not a system service by design); HTCondor as primary (viable
AMD HIP discovery incl. Windows, but more pool complexity than one Linux box
needs); Prefect/Dagster (still need a GPU allocator); Nomad (Business Source
License, NVIDIA-only official GPU plugin); Docker for Slurm (see BRVP02 for
toolchain images).

### Executor interface

```python
@dataclass(frozen=True)
class ResourceRequest:
    cpu_slots: int
    gpu_devices: tuple[str, ...]      # logical BigCherry device IDs
    activity_class: str               # build | correctness | timed-measure
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
    executor: str                     # "slurm" | "local" | "fake"
    native_id: str
    execution_id: str

@dataclass(frozen=True)
class AllocatedDevice:
    logical_id: str
    architecture: str
    physical_id: str
    locator: str | None

@dataclass(frozen=True)
class Allocation:
    devices: tuple[AllocatedDevice, ...]
    env: tuple[tuple[str, str], ...]
```

Adapters: `SlurmExecutor` (Brutus; the only code that knows sbatch/squeue/
scancel/GRES/licenses/partitions/`SLURM_JOB_GPUS`/`ROCR_VISIBLE_DEVICES`),
`LocalExecutor` (Windows workstation, tests, emergencies; host device mapping +
ResourceLock/tree leases), `FakeExecutor` (offline tests).

## 2. Brutus services (v1)

`munge`, `slurmctld`, `slurmd`, plus BigCherry systemd units. **No MariaDB/slurmdbd
in v1**: `JobCompType=jobcomp/filetxt` + the BigCherry run store; add slurmdbd only
for TRES/gpuutil accounting, QOS/fairshare or multi-user policy. No `slurmrestd`.

```ini
SchedulerType=sched/backfill
SchedulerParameters=bf_licenses          # license-aware backfill: no starvation
PriorityType=priority/multifactor
Licenses=host_activity:2,build_slot:1
JobCompType=jobcomp/filetxt
JobCompLoc=/var/log/slurm/bigcherry-jobcomp.log
JobAcctGatherType=jobacct_gather/cgroup
JobAcctGatherFrequency=30
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
RequeueExit=75

PartitionName=bc-build         Nodes=brutus PriorityTier=10  State=UP   MaxTime=00:45:00
PartitionName=bc-measure-other Nodes=brutus PriorityTier=100 State=UP   MaxTime=00:45:00   # gfx1201/gfx1030
PartitionName=bc-measure-xtx   Nodes=brutus PriorityTier=100 State=DOWN MaxTime=00:45:00   # gfx1100, window only
```

`gres.conf`: static per-device GRES generated from verified PCI-BDF ->
`/dev/dri/renderD*` mapping (from the untracked environment.local inventory), e.g.
`Name=gpu Type=gfx1100_0 File=/dev/dri/renderD128 Flags=amd_gpu_env`.
`Flags=amd_gpu_env` sets only `ROCR_VISIBLE_DEVICES`. Under Slurm, BigCherry reads
`SLURM_JOB_GPUS` (global IDs), asserts them against the inventory, preserves
Slurm's `ROCR_VISIBLE_DEVICES` and leaves `HIP_VISIBLE_DEVICES` unset.
`bigcherry jobs doctor --gpu-map` verifies index/arch/PCI/renderD/GRES.

`cgroup.conf`: `CgroupPlugin=autodetect`, `ConstrainDevices=yes`, `ConstrainCores=yes`
— **provisional**: hardware falsification matrix per toolchain over
{GPU0}, {GPU1}, {GPU2}, {GPU3}, {GPU0+GPU1}: `/dev/kfd` usable, allocated render
nodes usable, unallocated blocked, visibility env exact, `hipGetDeviceCount` exact,
properties readable, rocminfo/amd-smi terminate, 100 init cycles without hang or
EACCES, llama-bench + llama-server attestation succeed, dual `-sm tensor` +
4096-ctx preflight succeed, direct use of an unallocated GPU fails (hard timeouts
everywhere). Fallback on any failure: `ConstrainDevices=no` + GRES scheduling +
`ROCR_VISIBLE_DEVICES` + BigCherry attestation.

Resource classes: build `bc-build --licenses=build_slot:1,host_activity:1`;
timed measurement `--licenses=host_activity:2` + GRES. Build + correctness may
overlap; build + timed measurement, correctness + timed, and two timed
measurements cannot. All jobs carry finite `--time`.

## 3. Production coexistence (llama-swap on gfx1100 GPU0/1)

| Job target | Production policy |
|---|---|
| gfx1100 GPU0/1 (incl. dual-XTX) | exclusive measurement window: drain, stop/unload production, run queued gfx1100 jobs contiguously (nightly/maintenance), restore |
| gfx1201 GPU2, gfx1030 GPU3 | production stays loaded; campaign starts after a retryable production-idle attestation |
| contract requiring host-global isolation | exclusive window regardless of GPU |

Idle attestation (GPU2/3): llama-swap `/running`; every backend `/slots`
`is_processing=false`; `/metrics` `requests_processing==0` and
`requests_deferred==0`; stable idle >= 10 s; GPU2/3 idle and host-noise limits;
then start immediately.

Measurement window: root-owned `bigcherry-measure-window.service`
(`Type=simple`, `ExecStart=... hold`, `ExecStop`/`ExecStopPost=... exit`,
`RuntimeMaxSec=4h`, `KillMode=control-group`, `Restart=no`). `hold`: flock, drain
`bc-measure-xtx`, wait zero measurement jobs, drain production, stop/unload
backend, prove GPU0/1 + host quiet, `bc-measure-xtx` UP, stay foreground. `exit`
(idempotent): drain/DOWN partition, clear marker/lock, start llama-swap, verify
`/health`. Boot recovery unit `bigcherry-measure-window-recover.service`
(`Before=llama-swap.service`). Alerts: window over expected duration, llama-swap
restart/health failure, cleanup failure, unit failed. Production unit gets
`ExecStartPre=bc-assert-no-measure-window`. Ad-hoc agent work goes through
`bigcherry host-run --class build|bench -- ...` (shared activity lock; refuses
during a window).

Privilege: `/etc/sudoers.d/bigcherry-measure-window` allows the `bigcherry` user
exactly `systemctl start|stop bigcherry-measure-window.service`; nothing else.
GPU2/3 jobs need no privilege.

## 4. Identity, attempts, retries

- Series key: patch + arch + `execution_environment_hash` + contract hash; freezes
  planned N, base revision, focal/common/validated patch IDs + implementation
  digests (`composition_policy = freeze` default). A 5th session in a 4-session
  series is refused; a new series needs a new contract version.
- `execution_environment_hash` = hash(OS family/version, kernel or Windows build,
  runtime family (rocm-linux | hip-sdk-windows) and version, HIP runtime,
  compiler, AMD driver, device model/PCI identity). **Windows HIP gfx1100 and
  Linux ROCm gfx1100 are separate series**; Windows gives portability evidence,
  never fills a Linux session.
- Commit frozen **per attempt** (resolved at attempt creation, pinned runner
  worktree, `BIGCHERRY_ATTEMPT_ROOT`); no stage resolves the branch itself.
- Exit protocol: `0` pipeline completed (including scientific FAIL);
  `75` transient same-commit failure -> Slurm `RequeueExit` (`SLURM_RESTART_COUNT`);
  `76` harness failure needing a code/config fix -> terminate, wake, then
  `bigcherry jobs retry RUN --latest` = new attempt, new Slurm job, same run_id;
  `77` invalid input / contract or composition drift.
- A reused build from an earlier attempt is allowed only when the new attempt's
  build OperationSpec and execution hash match and artifacts re-verify (RCD01).
- No result-driven re-measure (a -30% round is not a retry predicate) until PVPS09.

## 5. Phasing

| Version | Content |
|---|---|
| v1 (this week) | one Slurm job per validation run (monolithic `validation_campaign`), GPU class policy above, retire queue.sh/run_campaign.sh |
| v1.5 | `validation_campaign --prepare-only` (materialize + build all trees, write `prepared-campaign.json`, no verdicts) and `--execute-only --prepared <manifest>` (verify commit/contract/composition/source/binary hashes, run producer, ladder, production lane, persist evidence); build job in `bc-build` (production keeps serving), execute job `--dependency=afterok` |
| v2 | external evidence sink + RCD01 activation (OperationSpec, running/result records, rehydration) |
| v3 | extracted build/correctness/measure operations as a stage DAG |
| v4 | build overlap with timed measurement, only after scheduler-isolation-v1 passes |

scheduler-isolation-v1 (pre-declared, non-patch evidence): identical binaries A/A;
conditions idle vs one representative HIP compile, and for GPU2/GPU3 also gfx1100
production unloaded vs loaded-idle vs active inference; >= 32 randomized paired
blocks per condition; accept iff condition effect 95% CI within +-0.10%,
variance-ratio upper bound <= 1.15, no clock/power/thermal shift. Until loaded-idle
qualifies, GPU2/3 require production idle (not stopped).

## 6. Consolidation

| Component | Action |
|---|---|
| `core/tree_activity.py` (HI151) | keep/extend: repository maintenance fencing; new stages refuse while the maintenance lock exists; `jobs pause` = hold pending jobs, wait running, take the lock |
| `experiment/bundle.run_managed()` | extend into the per-operation executor: streamed stdout/stderr files (never buffered), event sink, heartbeats, stall policy |
| `telemetry.py` (RQW01) | keep; add `JsonEventSink` / `CompositeSink` |
| `tuning/journal.py` (HI48) | extract `core/durable.py` (canonical JSON, atomic write, checksummed JSONL, torn-tail recovery); tuning and jobs share it |
| campaign `ResourceLock` | `resource_policy` = local or external; under Slurm GPU/build/quiet claims are external (no double scheduler) |
| RCD01 | activate the durable operation/result semantics only; scheduler ownership, distributed locks and generic workflow engine de-scoped (Slurm) |
| AudiAgentic gateway | standalone service with a gateway-neutral event envelope; the gateway is a client adapter (SSH submit / events --after), never the execution authority |

## 7. Observability and agent interface

Authorities: `squeue --json` (current scheduler state), `jobcomp/filetxt` (completed
Slurm jobs), BigCherry run store (domain identity, attempts, failure class,
verdicts, artifact hashes, telemetry, events), journald (service health).

Run layout: `/mnt/data/bigcherry-jobs/runs/<run_id>/{intent.json, submission.json,
events.jsonl, stages/<stage>/attempt-NNN/{operation.json, running.json, result.json,
stdout.log, stderr.log}, telemetry/{gpu,host}.jsonl, evidence/, reports/}`.

CLI (all `--json`, JSON-only stdout, diagnostics on stderr): `bigcherry jobs
submit|submit-batch|list|show|retry|disable|enable|cancel|hold|release|pause|resume|
status|events --after <seq> --jsonl --follow|doctor|harvest|report`. Windows
Claude Code uses SSH as transport (`ssh bigcherry@brutus "... jobs submit - --json"`).
Optional inbox: `bigcherry-jobs-ingest.path` watching `/mnt/data/bigcherry-jobs/inbox`
(validate -> processing/ -> persist intent -> submit -> accepted/ or rejected/).
`bigcherry-observe.timer` writes `status.json`, `status.md`, `bigcherry.prom` every
15-30 s (node_exporter textfile collector if present; no custom web service yet).

`jobs status --json` is BigCherry-normalized (never raw Slurm JSON): service health,
queue counts by state/stage, host disk + ccache, per-GPU allocation + telemetry,
per-job identity/stage/attempt/Slurm state/progress/result.

Event envelope: `{schema, seq, event_id, ts, run_id, series_id, stage, attempt,
slurm_job_id, kind, severity, data}`.

| Wake | Record only |
|---|---|
| harness non-zero exit, retry exhausted, stall, disk hard threshold, GPU unhealthy/reset/missing, contract/composition drift, Slurm node DOWN/DRAIN, harvest/git failure, RCD01 recovery ambiguity, service degraded, measurement window overrun or production restart failure | submitted/queued, stage started/completed, telemetry, scientific PASS, scientific FAIL, ladder/noise results; series complete = notify |

Security (LAN test server): dedicated `bigcherry` Unix account, Ed25519 SSH key,
no password or root SSH, SSH restricted to the LAN, no MariaDB/Slurm ports exposed,
MUNGE local; individual accounts for multiple humans.

## 8. Tests

Offline (FakeExecutor): batch expansion/overrides, canonical hashes, series N+1
refusal, retry = same run_id + new attempt, scientific FAIL != harness failure,
disable/enable preserving slots, commit frozen per attempt, contract/composition
drift blocks, freeze vs restart-on-promotion, GPU allocation parsing
(`SLURM_JOB_GPUS` -> inventory), env hash separation (Windows vs Linux), exit-code
protocol, idle-attestation parsing, measurement-window state machine, events
append-only + resume after seq, domain layer imports no Slurm module.

Hardware acceptance before retiring the lab scripts: gfx1100 job (in a window),
gfx1201 alternate-ROCm job (idle attestation), gfx1030 alternate-model job, dual
gfx1100 job; forced harness failure + `retry --latest`, client disconnect,
hold/release, disk guard, promotion between sessions, window overrun auto-close.

## 9. Round 4 amendments (supersede conflicting text above)

Owner objections 2026-09-26: production may use any GPU; hardware on Brutus changes.
GPT `req_e22e2f08be3d49e4`: resolved, **no open design objections**.

**Production on any GPU (O12).**
- One measurement partition `bc-measure` (PriorityTier=100); `bc-measure-xtx` /
  `bc-measure-other` and every GPU-class production rule are deleted.
- Per dispatch, after Slurm allocates GPUs and before validation starts, BigCherry
  builds a `ProductionSnapshot{config_hash, potential_devices | "all",
  running_devices, observed_devices}` from the llama-swap config, `/running`, each
  model's cmd/env selectors, live backend process environments and AMD-SMI process/VRAM
  attribution. Production models declare `env: BIGCHERRY_GPU_CLAIM:
  "uuid:<id>[,...]" | "arch:<gfx>,count=N" | "all"`; missing, unparsable or
  contradictory claims mean `all` (fail closed).
- Allocated GPUs intersect potential production GPUs -> exclusive window (drain, stop
  llama-swap, verify no production GPU processes, measure, restart). The root window
  service reads a validated `/run/bigcherry/measure-window-request.json`
  (`slurm_job_id`, `target_device_ids`, `inventory_hash`, `production_config_hash`); it
  no longer toggles partitions. Otherwise production stays up behind the idle
  attestation.
- Contamination watchdog every ~2 s during measurement: config hash unchanged, potential
  set still disjoint, no unexpected process on target GPUs, telemetry sane. Any violation
  terminates the measurement, discards the sample and classifies a retryable
  environment contamination (wake).

**Dynamic hardware (O13).**
- `tools/bigcherry/hardware/{model,linux_amd,windows_hip,inventory,topology}.py`;
  `DeviceRecord{device_id, identity_source (amd_uuid | rsmi_unique_id | serial |
  hip_uuid | luid | weak), arch, model, vram_bytes, pci_bdf, render_node, numa_node,
  driver_version}`. BDF, render node and index are observations, never identity.
- `bigcherry-hardware-discovery.service` (Before=slurmd) discovers GPUs, writes
  `/var/lib/bigcherry/hardware/observed.json`, compares `accepted.json`, generates an
  **architecture-typed** `gres.conf` (`Type=gfx1100`, not `gfx1100_0`) and node
  `Gres=gpu:gfx1100:2,...`; material drift starts the node DRAINED and wakes the
  operator. Runtime drift (udev/timer) drains immediately; reconfigure only with no
  active jobs; operator acknowledges the new inventory hash, then RESUME.
  AutoDetect=rsmi is not the authority; `slurmd -G` remains a validation step.
- Jobs request `GpuRequirement{architecture, count, min_vram_bytes,
  homogeneous_model, require_peer_access, exact_device_ids}`. Rare subset/exact
  constraints over-allocate all GPUs of that architecture and select UUIDs inside the
  allocation (`ROCR_VISIBLE_DEVICES=<UUIDs>`); peer pairs come from the discovered
  topology and are re-attested.
- Series key: patch, contract hash, architecture, `platform_environment_hash` (OS,
  kernel/Windows build, ROCm/HIP SDK, HIP runtime, compiler, driver) and
  `hardware_cohort_hash` (sorted stable device IDs, arch, model, VRAM, PCIe/NUMA/peer
  topology fingerprint). A replacement card of the same model, or a card moved to another
  slot, starts a new series unless equivalence was pre-qualified. `weak` identity never
  continues a series automatically; the operator declares a new hardware epoch.
- `environment.local.toml` keeps host policy (toolchains, models, allowed architectures,
  device aliases); it is no longer GPU inventory truth.
- Windows LocalExecutor discovers via a small HIP probe (count, properties, UUID/LUID,
  PCI fields, VRAM, driver/runtime versions, peer matrix); the ordinal is launch-local
  only.

Remaining gates (implementation/acceptance, not design): real UUID availability on these
RDNA cards, generated-GRES/reconfigure tests, ROCm cgroup falsification, production-claim
parser and process attestation, the cross-GPU isolation experiment.
