---
id: RCD02
order: 2
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:51:52.243021+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Freeze the job-service architecture and implementation boundaries

## Description

Make `docs/design/JOBS_ORCHESTRATOR.md` normative and freeze executable boundaries for RCD03-RCD12. Brutus uses host-installed single-node Slurm behind a platform-neutral BigCherry `Executor`. Slurm owns execution/resource scheduling; BigCherry owns scientific/domain identity, series policy, commit pinning, retry legality, evidence/reporting, production coexistence and stable hardware identity. No custom scheduler daemon/database is execution authority.

The freeze is falsification-driven. CI now imports current BigCherry production modules, launches real child processes, exercises real git/HI151 recovery, runs crash/idempotency simulators, and installs/starts actual Ubuntu Noble Slurm 23.11.4. Findings update the design rather than being waived.

## Frozen decisions

1. **V1 Slurm services:** `munge + slurmctld + slurmd`. No MariaDB, `slurmdbd`, `slurmrestd`, `--account`, or `sacct` dependency.
2. **Native terminal trail:** `jobcomp/filetxt`; BigCherry run/event store remains domain/scientific authority.
3. **Partitions/resources:** generic `bc-build` and `bc-measure`; `host_activity:2,build_slot:1`; `bf_licenses`; measurement higher `PriorityTier`; finite job time limits.
4. **GPU GRES:** architecture typed only; RCD12 discovered inventory renders current counts/render nodes. No slot/ordinal in GRES type or scientific identity.
5. **Series hardware:** bind one deterministic exact stable-device cohort before any session submission. A proper subset of same-arch devices reserves the whole accepted architecture pool in v1, then narrows inside the exclusive allocation.
6. **Production:** runtime BigCherry production snapshot/gate; no static production-GPU partition.
7. **Attempt identity:** commit frozen per attempt; exit 75 requeues same attempt/commit; exit 76 creates explicit new attempt and may pick up a new commit; scientific FAIL exits 0.
8. **Cgroups:** initial qualified mode `ConstrainDevices=no`; only enable device filtering after Brutus ROCm matrix passes.
9. **Rollout:** v1 monolithic campaign; v1.5 prepare/execute; RCD01/full DAG later; timed overlap only after isolation qualification.
10. **Cross-platform:** domain depends only on `Executor`; Windows LocalExecutor is separate platform environment and must honor the pre-bound series cohort.

## Ownership

```text
Slurm:
  queued/running/held/cancelled native state
  CPU/GPU allocation
  dependencies
  priority/backfill/licenses
  process containment/accounting plugins when qualified
  same-commit requeue
  controller state + jobcomp trail

BigCherry:
  JobSpec/series/run/attempt
  planned N + contract/composition freeze
  deterministic hardware-cohort binding
  commit/toolchain/model/corpus identity
  retry legality/classification
  production coexistence/window
  evidence/harvest/report/events/status
```

Slurm native IDs are execution metadata, never BigCherry scientific identity.

## Executor boundary

```python
class Executor(Protocol):
    def submit(self, request: ExecutionRequest) -> ExecutionHandle: ...
    def status(self, handle: ExecutionHandle) -> ExecutionStatus: ...
    def cancel(self, handle: ExecutionHandle) -> None: ...
    def control(self, handle: ExecutionHandle, action: ExecutorControl) -> None: ...
    def allocation(self, handle: ExecutionHandle) -> Allocation | None: ...
    def events(self, handle: ExecutionHandle, *, after: int | None = None) -> Iterable[ExecutorEvent]: ...
```

Adapters:

```text
SlurmExecutor   Brutus managed execution
LocalExecutor   Windows/Linux direct/test/emergency execution
FakeExecutor    deterministic service tests
```

Only `tools/bigcherry/jobs/slurm.py` may know `sbatch`, `squeue --json`, `scontrol`, `scancel`, GRES/licenses/partitions or `SLURM_*`. Domain modules may not import Slurm types.

## Durable service boundary

V1 filesystem authority:

```text
<work>/jobs/
  events.jsonl
  events.lock
  inbox/{pending,processing,accepted,rejected}/
  series/<series_id>/series.json
  runs/<run_id>/
    intent.json
    attempts/NNN/
      attempt.json
      submission-intent.json
      submission.json
      result.json
      stdout.log
      stderr.log
```

Requirements:

- canonical JSON + fsync + atomic rename;
- checksummed append-only event stream under a host-level multiprocess lock;
- immutable terminal attempt records;
- status is a projection of domain records + executor status;
- durable submission intent exists before external submit;
- intent-without-handle recovery correlates native state/jobcomp/sentinels and never blindly duplicates;
- SQLite, if later added, is a derived index only.

## Dependency/order freeze

- RCD04 and RCD12 may implement in parallel; final series creation consumes RCD12 deterministic binding.
- RCD03 consumes RCD12 inventory/GRES rendering.
- RCD05 consumes RCD03/RCD04/RCD12 and existing HI151/HI47 seams.
- RCD06 may proceed in parallel; M1 is needed for v1 safety.
- RCD11 consumes frozen RCD04/RCD12 identity and never rebinds hardware.
- RCD09 consumes RCD04/RCD05 durable event/run-store primitives.
- RCD10 gates shell-queue retirement after RCD03/RCD04/RCD05/RCD06-M1/RCD09/RCD11/RCD12 plus Brutus acceptance.
- RCD07/RCD08 are later durability/throughput/harvest work unless parity requires them earlier.

## Executable validation already performed

Canonical workflow: `.github/workflows/rcd-slurm-validation.yml` on `ubuntu-24.04`.

Latest green reference at design freeze: Actions run `36219793101`, validation head `ffdc08d00b76f0b8775f50a2cc6e48e1a4d20e01`.

### Planning/domain

```text
mock_pipeline.py --self-test: 32 checks PASS
```

Covers capability resolution, deterministic exact cohort binding, whole-architecture reservation, production claims/conflicts, platform identity, retry semantics and FakeExecutor dependencies.

### Current BigCherry modules/processes

```text
real_bigcherry_process_smoke.py: 25 PASS
real_bundle_failure_smoke.py:    12 PASS
```

Uses current production modules and real subprocesses. Covers managed success/nonzero/launch failure, stdout/stderr, 8 MiB output, HI48 torn-tail recovery, real `validation_campaign.main()` argument/producer dispatch, fail-closed CLI cases and managed-process failure injection.

Known implementation gap: `experiment.bundle.run_managed()` still buffers child output via `capture_output=True`; RCD05 must stream logs before production use with ~1.5 GB server logs.

### Recovery/concurrency

```text
real_recovery_smoke.py:        18 PASS
tree_activity_race_smoke.py:  100 PASS / 50 races
service_recovery_smoke.py:     29 PASS
```

Covers HI151 exclusion/crashed leases, detached git worktree pinning, concurrent maintenance-vs-lease admission, durable submission crash windows, zero/one/multiple correlation outcomes and event recovery.

### Real Noble Slurm

Actual Ubuntu 24.04.5 distro Slurm `23.11.4-1.2ubuntu5` + MUNGE was installed and real `slurmctld/slurmd/sbatch/squeue/scontrol/scancel` were exercised. Green run proved:

```text
MUNGE round-trip
minimal no-db/no-account controller/node
squeue --json
no-account job completion
current-branch BigCherry process harness executed inside a real Slurm job
hold/release
license/priority progress: build1 -> measure -> build2
afterok dependency
RequeueExit=75 + same job + SLURM_RESTART_COUNT 0 -> 1
cancel
controller restart preserving queued/running ownership
jobcomp/filetxt terminal history
cgroup process/task/jobacct plugins with ConstrainDevices=no
```

CI development also falsified assumptions that are now forbidden in the plan:

- Noble 23.11 has no `slurmctld -t` config-test flag;
- pending `Reason=` strings are diagnostics, not stable failure classifiers;
- slurmdbd/account associations are unnecessary for accepted v1.

## Brutus-only gates

Do not claim these from CI:

- AMD UUID/RSMI stable identity availability;
- generated real `gres.conf` accepted by `slurmd -G`;
- `/dev/kfd`/render-node behavior with `ConstrainDevices=yes`;
- HIP/ROCr enumeration/order for supported toolchains under cgroups;
- peer access/`-sm tensor`/4096-context producer preflights;
- llama-swap claim/process attribution/window behavior;
- build/disk/ccache pressure behavior;
- production loaded-idle/active noise equivalence;
- full real-GPU `validation_campaign` evidence parity.

## Files/contracts

Validated/planning assets:

```text
.github/workflows/rcd-slurm-validation.yml
config/slurm/slurm.conf.example
config/slurm/cgroup.conf.example
config/slurm/gres.conf.example
docs/design/JOBS_ORCHESTRATOR.md
docs/reference/jobs/SLURM_BRUTUS.md
tools/lab/run-campaign-durability/*
```

Production modules remain RCD03-RCD12 work under `tools/bigcherry/**`; permanent tests go under `tools/tests/**`.

## Acceptance Criteria

RCD02 closes when:

- RCD03-RCD12 contain no superseded static-GPU/slurmdbd/account assumptions;
- exact deterministic series cohort is frozen before submission;
- LocalExecutor honors that frozen cohort;
- generic multiprocess event/run-store serialization lock is explicitly owned by RCD04/RCD09, distinct from HI151 tree-maintenance locking;
- all retained lab validation files are registered in the repository tool-disposition registry;
- full RCD validation workflow is green after the final plan/config cleanup;
- no unresolved platform-choice/design objection remains.

Hardware gates above remain implementation/acceptance work, not RCD02 architecture blockers.

## Notes

Decision: **minimal Noble Slurm + thin BigCherry durable domain service + platform-neutral Executor**. Empirical real-service validation supersedes the earlier slurmdbd/accounting design.

## Change Log

- 2026-09-26T00:51:52.243021+00:00 (created-by): Created by agent
- 2026-09-26 (dev-gpt-agent): Fully specified architecture/dependencies/offline falsification.
- 2026-09-26 (dev-gpt-agent): Added deterministic pre-series hardware binding and whole-architecture subset reservation.
- 2026-09-26 (dev-gpt-agent): Added real BigCherry process/recovery/Noble Slurm validation.
- 2026-09-26 (dev-gpt-agent): Reconciled architecture to green real Noble minimal no-db/no-account stack and current validation pack.
