# BigCherry job service — normative design

Status: **control-plane implementation active; offline/service and real Noble Slurm validation in place; Brutus GPU/production acceptance pending** (2026-09-27).

RCD02–RCD12 implement this design. `docs/reference/jobs/JOBS_CONTROL_PLANE.md` documents the implemented human/agent interface; `docs/reference/jobs/SLURM_BRUTUS.md` is the Brutus install/qualification runbook.

## 1. Goal

Replace `tools/lab/plan-qualification/{queue.sh,run_campaign.sh,make-serial-2.sh}` plus watcher/switch loops with a repeatable durable path:

```text
human / agent / future UI/API
              |
           JobService
              |
 deterministic plan + scientific freeze
              |
 target placement + exact hardware cohort
              |
 durable batch/series/run/attempt store
              |
          Executor
        /    |      \
     Slurm  Local   RemoteWorker
      |      |          |
   Brutus  direct   Windows/future hosts
```

Submitting clients may disconnect after durable acceptance. No shell watcher is execution authority.

## 2. Authority boundaries

**Buy scheduling; keep scientific policy in BigCherry.**

Slurm owns on Brutus:

- native queue/running/held/cancelled execution;
- CPU/GPU reservation;
- partition tier, dependencies, licenses;
- process/cgroup ownership;
- controller state and lightweight job completion trail.

BigCherry owns:

- BatchSpec/JobSpec/series/session/run/attempt identity;
- planned session count;
- exact code commit per attempt;
- frozen focal/common/promoted patch identity;
- contract/validation/model/corpus/producer identity;
- platform environment + stable hardware cohort;
- executor selection/placement policy;
- retry legality;
- durable events/log/artifact references;
- production coexistence;
- scientific verdict/evidence/harvest/review.

Rejected as primary authority: custom scheduler daemon/SQLite queue, slurmdbd/MariaDB requirement, HTCondor migration solely for Windows, orchestration frameworks layered over Slurm.

## 3. Implemented control plane

Production package now exists under `tools/bigcherry/jobs/`:

```text
model.py       versioned scheduler-neutral requests
identity.py    scientific content freezing/recheck
store.py       filesystem durability + inbox/events
service.py     application API/control/recovery
executor.py    platform-neutral execution protocol
fake.py        deterministic tests
slurm.py       Slurm-only adapter
local.py       direct Linux/Windows process adapter
remote.py      restricted remote transport
worker.py      target-side restricted worker
registry.py    configured execution targets
workspace.py   exact-commit workspaces
runner.py      attempt-local validation_campaign launch
```

Hardware domain currently provides models/capability binding, drift classification, allocation verification and GRES rendering under `tools/bigcherry/hardware/`. Provider discovery remains hardware-gated.

`python -m bigcherry jobs ...` routes to the same `JobService` application layer that future UI/API adapters must use.

## 4. Public domain model

Portable scientific request is separate from host-local launch details.

```python
BatchSpec / JobSpec
  patch / architecture / model / producer
  planned sessions
  baseline/common/producer inputs/corpus
  code ref
  GPU capabilities
  target executor/host/platform policy

ExecutionRequest
  execution_id
  target-local command/cwd/env/log paths
  resolved scheduler resources
```

Do not put arbitrary passthrough command args in JobSpec.

Client submission requires an idempotency key:

```text
same key + same canonical request -> existing batch, no new events/receipts
same key + changed request        -> conflict
new key                           -> new logical request
```

## 5. Scientific identity and placement

Final series identity is computed only after placement/accepted target inventory is known.

Freeze before run creation:

- focal patch implementation/validation identity and contract binding;
- common patches;
- current `validated-enhancements` composition IDs/digests;
- model SHA256/size;
- producer corpus SHA256/size;
- file-backed producer inputs;
- producer/baseline identity;
- platform environment hash;
- exact deterministic GPU stable-ID cohort/hash.

At attempt start the exact pinned BigCherry worktree re-resolves scientific identity. Any mismatch blocks the attempt; the existing series is never mutated.

Different Windows/Linux runtime environments or different physical stable-device cohorts are different series.

## 6. Durable store and ingestion

Filesystem authority:

```text
<work>/jobs/
  requests/
  batches/<batch>/batch.json
  series/<series>/series.json
  runs/<run>/intent.json
  runs/<run>/control.json
  runs/<run>/attempts/NNN/
    attempt.json
    submission-intent.json
    submission.json
    executor-start.json
    executor-result.json
    stdout.log
    stderr.log
  inbox/{pending,processing,accepted,rejected}/
  events.jsonl
```

Properties:

- canonical JSON;
- atomic writes;
- host kernel file locks for multiprocess mutations;
- checksummed monotonically sequenced event JSONL;
- final torn event record may be recovered; prior corruption fails closed;
- retries create new attempt directories;
- projected state derives from records + executor.

Systemd path unit watches durable `pending/`; a low-rate timer is notification-loss protection only. There is no custom polling scheduler daemon.

Crash recovery prioritizes existing `processing/` receipts. Stable `execution_id` plus `submission-intent.json` allows correlation/rebind without duplicate submission.

## 7. Executor abstraction

```python
class Executor(Protocol):
    name: str
    def submit(request) -> ExecutionHandle: ...
    def correlate(execution_id) -> tuple[ExecutionHandle, ...]: ...
    def status(handle) -> ExecutionStatus: ...
    def cancel(handle) -> None: ...
    def control(handle, action) -> None: ...
    def allocation(handle) -> Allocation | None: ...
    def events(handle, *, after=None): ...
```

Normalized `Allocation` separates:

```text
native_gpu_ids    scheduler/launch-local identifiers; never scientific identity
stable_gpu_ids    optional attested RCD12 identities
```

Adapters:

- `SlurmExecutor`: Brutus.
- `LocalExecutor`: direct Linux/Windows/testing/emergency.
- `RemoteExecutor`: restricted worker transport for future Windows/non-Slurm hosts.
- `FakeExecutor`: deterministic tests.

Remote scientific use remains disabled until target-local staging/workspace semantics and Windows HIP discovery pass RCD11 acceptance. Controller absolute paths are not portable.

## 8. Hardware and GPU resource model

Hardware is discovered/accepted state, not `environment.local.toml` truth.

Scientific capability request:

```python
GpuRequirement(
    architecture,
    count,
    min_vram_bytes,
    homogeneous_model,
    model,
    require_peer_access,
    exact_device_ids,
)
```

Binding rules:

1. candidate order by stable device ID;
2. filter capabilities;
3. ambiguous same-architecture model groups fail;
4. peer pair chosen deterministically;
5. exact selected stable IDs persisted before sessions execute;
6. replacement/topology drift never silently rebinds a series.

Slurm GRES types are architecture-only:

```text
Name=gpu Type=gfx1100 File=/dev/dri/renderD128 Flags=amd_gpu_env
```

If the bound cohort is a proper subset of one architecture, reserve the complete accepted architecture pool and narrow only inside the exclusive allocation.

`verify_series_allocation()` requires attested stable IDs; native ordinals alone fail.

Timed scientific measurement uses whole-GPU exclusivity. Slurm GPU sharding is not part of v1 measurement.

## 9. Slurm v1

Brutus topology:

```text
munge
  |
slurmctld ---- slurmd
  ^             |
  |             +-- managed job/process cgroup
  |
SlurmExecutor
```

No slurmdbd/MariaDB/slurmrestd in v1.

Policy:

```text
bc-build:   build_slot:1 + host_activity:1
bc-measure: host_activity:2 + required architecture GRES
```

Higher measure `PriorityTier` + `bf_licenses` was validated with real Noble Slurm.

Initial cgroup device policy is `ConstrainDevices=no`. Enable device fencing only after Brutus proves ROCm `/dev/kfd`/render-node/toolchain/peer behavior.

### Retry correction

Real Noble CI proves Slurm native requeue works, but **production monolithic v1 does not configure `RequeueExit`**. Current campaign/workspace paths are not uniformly restart-idempotent.

Every campaign retry creates attempt N+1/new native job. `retry --same-commit` can use the previous exact commit; `retry --latest` may resolve a newer commit. RCD07 may enable native requeue for an individual durable stage only after restart-safety tests for that stage pass.

## 10. Attempt execution

At new attempt:

1. resolve requested code ref -> exact commit;
2. create detached workspace;
3. recheck frozen scientific identity;
4. write attempt + submission intent;
5. render typed executor-local launch;
6. submit/correlate;
7. persist native handle;
8. runner writes start/result sentinels.

`validation_campaign --device-map` is derived from allocation-local positions only. Stable GPU identity is recorded separately.

Slurm jobs use external resource ownership. Local/remote-worker direct paths use OS-held locks keyed by exact stable device IDs; process death releases the locks.

Before production cutover, `experiment.bundle.run_managed()` must be converted from memory capture to large-log-safe streaming and process-group cancellation.

## 11. Production coexistence

llama-swap may use any GPU. No static production GPU partition exists.

RCD11 must derive a fail-closed production snapshot from config, claims, `/running`, backend argv/env and AMD process/VRAM observation.

```text
allocated stable IDs intersect production potential -> exclusive production window
ambiguous production ownership                    -> exclusive window
proven disjoint                                    -> idle/noise policy after qualification
```

Supported production/ad-hoc starts must participate in a preventive host activity gate so new work cannot enter between preflight and timed launch. Watchdog detects config/process/inventory contamination during timed measurement.

Exclusive window is a narrow root-owned systemd helper with declarative request file, deadline, idempotent cleanup, boot recovery and exact start/stop permission only.

## 12. Remote/Windows and additional hosts

Windows is not scheduled by Slurm. Extension path:

```text
central JobService -> RemoteExecutor -> SSH JSON worker -> target LocalExecutor
```

Remote requests use stable execution identity and correlation. Before production use add target-local source/model/corpus staging, target-local paths/logs/cache and Windows HIP UUID/LUID discovery. Evidence from Windows remains a separate platform series from Linux ROCm.

Additional Linux machines can later either join the Slurm cluster or use the same remote-worker pattern. If Slurm becomes multi-node, current cluster-wide `host_activity`/`build_slot` licenses must become host-scoped resources so unrelated nodes do not serialize each other.

## 13. Human/agent CLI

Implemented core surface:

```text
jobs plan / validate / submit
jobs ingest
jobs list / show / status
jobs events / logs / artifacts / review
jobs disable / enable / hold / release / cancel
jobs retry --same-commit|--latest
jobs pause / resume
jobs executors list
```

Future RCD08/RCD09/RCD12 additions: series/queue views, executor/hardware doctor, verified evidence/report/harvest and richer service health/metrics.

Future HTTP/UI must call JobService/application DTOs directly; it must not parse CLI text or raw Slurm schemas.

## 14. Observability

Authorities:

```text
squeue --json      current Brutus native execution
jobcomp/filetxt    lightweight Slurm terminal trail
BigCherry store    domain/scientific history/events/log/artifact refs
journald           service diagnostics
```

No `sacct` dependency in v1.

Events use reconnectable monotonic sequence (`--after N`). Before long-term operation add event segment/checkpoint/rotation while preserving cursor semantics.

`review_ready` is stricter than process completion: all sessions can be execution-complete while review remains blocked until RCD08 verified evidence/harvest is present.

## 15. Validation

Already exercised:

- planning/domain capability/cohort falsifier;
- real BigCherry CLI/process bundle/failure/recovery tests;
- HI151 concurrent lease/maintenance races;
- durable submission-event crash-window tests;
- permanent `tools/tests/jobs` for the implemented production control-plane modules;
- real Ubuntu 24.04 / Slurm 23.11.4 service workflow, including BigCherry process execution inside Slurm.

New control-plane CI is `.github/workflows/jobs-service-validation.yml`; every implementation change must keep it green.

Hardware-only gates remain:

- AMD stable identity provider;
- generated real GRES + `slurmd -G`;
- stable-ID allocation attestation;
- ROCm cgroup/device matrix;
- peer/tensor split;
- llama-swap coexistence/window recovery;
- real disk/ccache pressure;
- scheduler isolation/noise;
- full real-GPU validation/evidence parity.

## 16. Retirement gate

Do not retire shell queue until:

1. jobs-service CI and real Noble Slurm CI are green;
2. RCD03/04/05/06-M1/09/11/12 production paths are complete;
3. Brutus hardware acceptance passes or an explicit supported fallback is recorded;
4. forced incident matrix passes;
5. one complete planned multi-session series runs with no watcher intervention;
6. evidence/report parity is confirmed;
7. rollback path is tested.

Retire queue/switch/watcher wrappers first. Keep legacy summarization/noise tools until RCD08 parity.

## 17. Current open implementation work

No platform-choice objection remains. Remaining blockers are concrete implementation/acceptance work:

- large-log streaming/monitoring/retention;
- Linux AMD and Windows HIP discovery providers;
- observed/accepted hardware CLI/services;
- Slurm stable allocation attestation on Brutus;
- production claim/window/watchdog;
- target-local remote staging;
- verified evidence/report/harvest control-plane integration;
- long-term event rotation/indexing;
- Brutus GPU/noise/evidence acceptance.
