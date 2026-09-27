# BigCherry job service — normative design

Status: **software/control-plane implementation complete enough for cutover qualification; offline jobs CI and real Noble Slurm reference are green; Brutus GPU/production and Windows remote hardware acceptance remain pending** (2026-09-27).

Operator reference: `docs/reference/jobs/JOBS_CONTROL_PLANE.md`. Brutus runbook: `docs/reference/jobs/SLURM_BRUTUS.md`. Cutover/migration: `docs/reference/jobs/{ACCEPTANCE,MIGRATION}.md`.

## 1. Goal

Replace `tools/lab/plan-qualification/{queue.sh,run_campaign.sh,make-serial-2.sh}` plus watcher/switch loops with:

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

Client lifetime never owns execution.

## 2. Authority boundaries

**Buy scheduling; keep scientific policy in BigCherry.**

Slurm owns Brutus queue/execution/resource/process lifetime. BigCherry owns BatchSpec/JobSpec, planned-N, series/session/run/attempt identity, exact code/content/contract/composition/model/toolchain/hardware identity, retry legality, production coexistence policy, durable events/log/artifact references, evidence harvest and scientific report/review state.

Rejected as primary authority: a custom scheduler daemon/SQLite queue, mandatory slurmdbd/MariaDB, slurmrestd v1, HTCondor solely for Windows, or another orchestration framework layered above Slurm.

## 3. Scheduler-neutral implementation

`tools/bigcherry/jobs/` contains versioned request models, scientific identity, durable store/inbox/events, application service, Executor protocol, Slurm/Local/Remote/Fake adapters, workspaces/runner, evidence harvest/git transaction, reports, status/metrics, production-coexistence policy, acceptance matrix and legacy migration.

`tools/bigcherry/hardware/` contains provider-neutral inventory/cohort models, observed/accepted state, binding/drift/allocation verification, runtime stable-ID mapping, architecture-only Slurm GRES rendering and Linux AMD-SMI discovery.

Only adapters know native scheduler/transport syntax. Scientific/domain code never keys identity on Slurm job ID, GPU slot/BDF/render node/ordinal, partition name or native environment variable.

## 4. Public domain model

Portable scientific request:

```text
BatchSpec / JobSpec
  focal patch / architectures / model / producer
  predeclared planned sessions
  baseline/common/producer inputs/corpus
  code ref / HIP path
  GPU capability requirement
  target executor/host/platform policy
```

Executor-local request:

```text
ExecutionRequest
  stable execution_id
  target-local argv/cwd/env/log paths
  normalized resources/dependencies
```

No arbitrary passthrough shell arguments in JobSpec.

Submission idempotency:

```text
same key + same canonical request -> existing batch
same key + changed request        -> hard conflict
new key                           -> new logical request
```

## 5. Scientific identity and placement

Final series identity is computed only after accepted target inventory and exact cohort binding are known.

`bigcherry.scientific-identity.v2` freezes at least:

- focal/common/validated-enhancement implementation + validation digests;
- resolved experiment-contract IDs/hashes;
- recipes/experiment-contract registry content identity;
- model, corpus and file-backed producer-input hashes/sizes;
- producer/baseline identity;
- platform environment hash;
- exact stable physical GPU cohort/hash.

Attempt start resolves the requested BigCherry ref to one commit, creates a detached workspace, and re-resolves identity from that workspace. Drift blocks the attempt; a series is never mutated to match newer definitions.

Different physical stable-device cohorts or Windows/Linux runtime environments are different scientific series.

## 6. Hardware model

Hardware truth is discovered/accepted state, not `environment.local.toml`.

`GpuRequirement` expresses architecture/count/min-VRAM/homogeneous-model/optional model/peer/exact-stable-ID constraints. Binding is deterministic by stable device ID and fails on ambiguous same-architecture model groups.

Linux stable identity preference:

```text
AMD/HIP UUID -> confirmed hardware serial -> explicit weak hardware_epoch
```

BDF/render/ordinal are locators only. Replacement or relevant topology change starts a new cohort. Locator-only remapping requires scheduler reconciliation but need not change cohort when stable identity/topology are unchanged.

Slurm GRES type is architecture only. If the frozen cohort is a proper subset of one architecture, reserve the full accepted architecture pool and narrow to the frozen IDs inside the already-exclusive allocation.

`Allocation.native_gpu_ids` is scheduler/launch-local; `Allocation.stable_gpu_ids` is scientific attestation. Native ordinals alone cannot verify a series allocation.

## 7. Durable store and recovery

Filesystem records under `<work>/jobs/` are BigCherry run authority. State includes request/batch/series/run/control/attempt documents, submission intent/handle, start/result sentinels, file-backed logs, durable inbox receipts, checksummed monotonic `events.jsonl`, verified evidence/report state and rebuildable status projections.

Host file locks serialize multiprocess mutation. Canonical JSON and atomic writes are used. A torn final event line can be recovered; prior corruption fails closed.

Submission persists immutable `submission-intent.json` before calling the executor. Stable `execution_id` supports recovery:

```text
recorded handle            -> rebind
intent + exactly one match -> rebind
intent + proven zero match -> submit immutable request once
multiple matches           -> block/wake; never duplicate blindly
```

Retries create a new attempt directory and native job.

## 8. Slurm v1

Stack:

```text
munge + slurmctld + slurmd
```

No slurmdbd/MariaDB/slurmrestd. Completion trail uses `jobcomp/filetxt`.

Resource policy:

```text
bc-build:   build_slot:1 + host_activity:1
bc-measure: host_activity:2 + architecture GRES
```

Higher measurement `PriorityTier` plus `bf_licenses` was validated on real Noble Slurm. Production cgroup baseline is `ConstrainDevices=no`; enable device fencing only after the full Brutus ROCm matrix proves it.

**Production monolithic v1 has no `RequeueExit`.** Every retry is BigCherry `attempt+1` / a new Slurm job. `--same-commit` may reuse exact code; `--latest` may resolve a newer commit. Native requeue may only be enabled later for an individually proven restart-idempotent stage.

## 9. Attempt execution

New attempt:

1. resolve code ref -> exact commit;
2. detached workspace;
3. scientific identity recheck;
4. persist attempt + submission intent;
5. render target-local execution/resource request;
6. submit/correlate/persist native handle;
7. runner verifies allocation and maps selected stable IDs to allocation-local positions;
8. run campaign with file-backed streaming logs;
9. persist terminal result/classification.

Scientific FAIL is a normal experiment completion. Harness/environment failure is separate retry/incident state.

`experiment.bundle.run_managed()` streams stdout/stderr directly to files and owns process-group cancellation; large campaign logs are never accumulated with `capture_output=True`.

## 10. Production coexistence

llama-swap may use any GPU. No static production GPU partition exists.

Pure policy is implemented: declarative model GPU claims (`all`, exact UUID set, conservative architecture/count), fail-closed ambiguity, dynamic conflict after exact target allocation, and continuous contamination detection.

```text
target intersects production potential OR ownership ambiguous -> exclusive window
disjoint -> loaded-idle only after isolation qualification
```

Until loaded-idle isolation is qualified, timed gating measurement must quiesce production rather than assuming disjoint GPUs eliminate host-level interference.

Still required on Brutus: live llama-swap config/`/running`/argv/env/process/VRAM adapter plus narrow root-owned window admission/recovery service. Production drift during measurement invalidates/discards the sample; it is not scientific regression evidence.

## 11. Local/remote/Windows

LocalExecutor is durable/reconnectable and uses stable-ID locks/mapping. RemoteExecutor + restricted JSON worker implement stable execution correlation and normalized status/control/allocation/events.

Remote scientific use remains disabled until target-local source/model/corpus/artifact staging and Windows HIP UUID/LUID discovery/re-attestation are implemented and hardware-accepted. Controller absolute paths are never portable authority. Windows evidence remains a separate platform series from Linux ROCm.

## 12. Evidence and reporting

Successful attempt worktrees produce normal patch evidence. `jobs harvest` identifies only records added relative to each frozen attempt commit, verifies record/focal/validation/contract/architecture identity, takes HI151 maintenance fencing, refuses staged or dirty canonical destination state, merges via the existing append-only patch evidence API, stages only exact destinations and optionally commits them.

No stash/reset/`git add -A`/implicit lifecycle promotion.

Verified harvest records exact record digests + evidence commit SHA. `jobs report` reads only those committed verified records and uses the existing session-bootstrap estimator/policy. Planned-N remains fixed; no materiality threshold is invented.

## 13. Observability / UI boundary

Durable events use monotonic sequence + severity `record|notify|wake`; clients reconnect by sequence. Logs reconnect by byte offset.

`jobs status` emits a versioned scheduler-neutral snapshot. A systemd observe timer atomically rebuilds `status.json`, `status.md` and Prometheus text. These are projections, not authority. Future HTTP/UI adapters consume JobService DTOs/events/log cursors directly, never raw Slurm output or parsed CLI presentation text.

## 14. Acceptance and migration

`jobs acceptance SPEC...` derives cutover cases from active BatchSpecs and accepted inventory. Unsupported capability remains explicit and blocks `ready`; no absent GPU is simulated.

`jobs migrate-legacy` converts only the exact historical `run_campaign.sh` shape plus known JobSpec-equivalent options. It requires original `BC_MODEL`/`BC_HIP_PATH` and explicit planned N. Legacy GPU index/run name are metadata only. Unknown one-off flags fail closed.

Queue retirement requires real Brutus gates, forced incident matrix, one full planned series through committed harvest/report, representative alternate capability soak and rollback readiness. See `docs/reference/jobs/ACCEPTANCE.md`.

## 15. Remaining non-software acceptance boundary

Offline jobs-service code/tests and real Noble scheduler reference are green. The project must not claim production cutover complete until real-host evidence covers:

- Brutus AMD stable-ID persistence and peer topology;
- generated real GRES + `slurmd -G` and allocation->stable-ID attestation;
- ROCm cgroup matrix or explicit `ConstrainDevices=no` fallback;
- live llama-swap snapshot/exclusive-window admission/recovery;
- scheduler-isolation/noise qualification;
- full planned managed series + evidence/report parity;
- Windows HIP discovery and remote staging before any Windows scientific executor is enabled.
