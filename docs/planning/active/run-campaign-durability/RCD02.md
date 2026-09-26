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

Make `docs/design/JOBS_ORCHESTRATOR.md` the normative architecture for the run-campaign service and convert the agreed decisions into executable boundaries. Adopt host-installed single-node Slurm on Brutus behind a platform-neutral BigCherry `Executor`; retain BigCherry authority for scientific identity, series policy, commit pinning, retry legality, evidence/reporting, production coexistence and hardware identity. Reject a custom scheduler database/daemon as execution authority.

The design freeze is falsification-driven. Real GitHub Ubuntu 24.04 tests now execute current BigCherry modules/CLI/child processes and install/start actual Noble Slurm 23.11.4 services. Any contradiction found by those tests updates the design instead of being waived. The first real Slurm pass already falsified the earlier no-accounting assumption: later jobs pended `InvalidAccount`, so v1 now includes minimal loopback MariaDB/slurmdbd associations.

## Steps

1. Treat `docs/design/JOBS_ORCHESTRATOR.md` as normative; RCD02-RCD12 implement it.
2. Freeze ownership:
   - Slurm: execution queue/state, CPU/GPU allocation, dependencies, priority, licenses, process containment when qualified, same-commit requeue.
   - slurmdbd: minimal cluster/account/user associations plus executor accounting only; never BigCherry scientific authority.
   - BigCherry: JobSpec/series/run/attempt, contract/composition freeze, commit pinning, capability/hardware binding, production gate, retry classification, evidence/harvest/report/events.
3. Freeze adapter boundary: domain modules depend only on `Executor`; only `jobs/slurm.py` may know `sbatch/squeue/scontrol/scancel/sacct`, GRES/licenses/partitions or `SLURM_*`.
4. Freeze rollout:
   - v1: monolithic `validation_campaign` Slurm job with RCD06 M1 safety seams; retire shell queue only after hardware acceptance/soak.
   - v1.5: prepare/execute split with verified prepared manifest.
   - v2: RCD01 durable operation/result protocol over external evidence.
   - v3: full stage DAG.
   - v4: timed overlap only after `scheduler-isolation-v1`.
5. Freeze dependencies:
   - RCD04 and RCD12 implement/test mostly in parallel; RCD04 final series creation consumes deterministic RCD12 `SeriesGpuBinding`.
   - RCD03 consumes RCD12 inventory/GRES.
   - RCD05 consumes RCD03/RCD04/RCD12.
   - RCD06 can proceed in parallel with RCD03-RCD05.
   - RCD11 consumes RCD04/RCD12 identity.
   - RCD09 consumes RCD04/RCD05 durable event/run-store primitives.
   - RCD10 gates v1 retirement after RCD03/RCD04/RCD05/RCD06-M1/RCD09/RCD11/RCD12.
   - RCD07/RCD08 are post-v1 durability/stage/harvest work unless needed for parity.
6. Maintain executable validation under `tools/lab/run-campaign-durability/`; production code remains under `tools/bigcherry/**`.
7. Register all retained RCD lab files in `TOOL_DISPOSITION.md` before closing RCD02.

## Detailed Solution & Technical Design

### Platform decision

Brutus v1 services:

```text
mariadb (loopback only)
munge
slurmdbd (loopback association/accounting service)
slurmctld
slurmd
```

No `slurmrestd`. One generic `bc-build` and one `bc-measure` partition; production GPU conflicts are dynamic BigCherry policy, not static partitions/reservations.

Minimal Slurm accounting is required because the real Noble 23.11.4 smoke reproduced `InvalidAccount` without associations. Every managed submission explicitly uses account `bigcherry`.

### Executor boundary

```python
class Executor(Protocol):
    def submit(self, request: ExecutionRequest) -> ExecutionHandle: ...
    def status(self, handle: ExecutionHandle) -> ExecutionStatus: ...
    def cancel(self, handle: ExecutionHandle) -> None: ...
    def control(self, handle: ExecutionHandle, action: ExecutorControl) -> None: ...
    def allocation(self, handle: ExecutionHandle) -> Allocation | None: ...
    def events(self, handle: ExecutionHandle, *, after: int | None = None) -> Iterable[ExecutorEvent]: ...
```

Adapters: `SlurmExecutor`, `LocalExecutor`, `FakeExecutor`. Windows HIP and Linux ROCm attempts never share a series environment.

### Identity invariants

- `run_id` is one planned scientific session and survives harness retries.
- new-code/config retry increments `attempt_no`; same-commit exit-75 Slurm requeue does not.
- BigCherry commit is frozen per attempt; no stage independently resolves a branch.
- series freezes planned N, contract/base/focal/common/validated implementation identity, platform environment and one deterministic exact hardware cohort before any session is submitted.
- every session uses the same stable-device cohort.
- architecture/count GRES cannot promise which identical card is returned; a series-bound proper subset reserves the entire accepted architecture pool in v1 then narrows inside that exclusive allocation.
- slot/index/BDF/render node are observations, not scientific identity.
- same-model replacement or relevant topology change creates a new cohort/series unless equivalence was pre-qualified.
- exit 0 includes scientific PASS/FAIL; 75 same-commit transient; 76 new attempt; 77 invalid/drift.
- anomalous performance is never itself an automatic retry before PVPS09.

### Tree-maintenance invariant

HI151 is part of the service admission boundary. Validation found that its old API documented maintenance exclusion but `Lease.__enter__()` did not enforce it. The implementation now uses a complementary two-phase handshake:

```text
runner:      check maintenance -> publish lease -> recheck maintenance
maintenance: publish maintenance lock -> scan live leases
```

Thus a campaign and pin-bump cannot both be admitted through the supported API, including the check/create race.

## Executable validation

### Planning model

```bash
PYTHONPATH=tools python tools/lab/run-campaign-durability/mock_pipeline.py --self-test
# {"checks": 32, "ok": true}
```

Covers capability resolution, exact cohort binding, whole-architecture reservation, production conflict fail-closed behavior, environment/cohort identity, retry actions and FakeExecutor ordering.

### Real BigCherry process validation

`real_bigcherry_process_smoke.py` imports current production modules and launches real subprocesses. GitHub Ubuntu 24.04 currently passes 25 checks including:

- `experiment.bundle.run_managed()` success/nonzero/launch failure;
- separate `python -m bigcherry.experiment.bundle` process;
- durable stdout/stderr and an 8 MiB output regression;
- HI48 durable journal + torn-tail recovery;
- real `validation_campaign.main()` producer selector/input/common-patch/device-map/production-lane parsing while mocking only the hardware/build producer body;
- fail-closed producer-selector mismatch and missing mandatory runtime inputs.

`real_recovery_smoke.py` additionally exercises current HI151 code and real Git worktrees: maintenance/lease exclusion, crash-stale lease pruning, detached commit pinning, branch advance isolation, runner dirtiness and cleanup.

### Real Noble Slurm validation

`.github/workflows/rcd-slurm-validation.yml` runs on `ubuntu-24.04` and installs actual distro packages. Required final pass:

```text
Slurm 23.11.4
MUNGE round trip
loopback MariaDB + slurmdbd
bigcherry cluster/account/user association
slurmctld/slurmd node registration
squeue --json
hold/release/cancel
afterok dependency
license/priority progress
RequeueExit=75 + SLURM_RESTART_COUNT
jobcomp + sacct history
```

This validates real CPU/service/scheduler behavior, not AMD hardware.

### Brutus-only gates

- generated real AMD `slurmd -G`;
- RCD12 stable IDs/UUID availability and GRES mapping;
- ROCm cgroup device matrix or explicit `ConstrainDevices=no` fallback;
- peer/`-sm tensor`, 4096-context and producer-specific preflights;
- llama-swap conflict/idle/window watchdog;
- scheduler-isolation/noise measurements;
- monolithic campaign evidence parity.

## Files

Validation/config now tracked:

```text
.github/workflows/rcd-slurm-validation.yml
config/slurm/slurm.conf.example
config/slurm/slurmdbd.conf.example
config/slurm/cgroup.conf.example
config/slurm/gres.conf.example
docs/reference/jobs/SLURM_BRUTUS.md
tools/lab/run-campaign-durability/mock_pipeline.py
tools/lab/run-campaign-durability/real_bigcherry_process_smoke.py
tools/lab/run-campaign-durability/real_recovery_smoke.py
tools/lab/run-campaign-durability/slurm_noble_smoke.sh
```

Production packages remain the responsibility of RCD03-RCD12.

## Validation

Required static review before RCD02 closes:

- no physical GPU index is scientific identity;
- no production policy hard-codes current slots;
- no per-slot GRES type;
- no custom SQLite scheduler authority;
- no mixed-commit attempt;
- no session silently changes stable GPU cohort;
- no architecture-only allocation substitutes a different card for a bound subset;
- no scientific result drives automatic retry;
- all hardware claims are explicit Brutus gates;
- Slurm v1 documents/uses the tested association layer;
- all RCD lab files are registered in `TOOL_DISPOSITION.md`.

## Effort & Risk

Medium. The validation harness has already found/fixed three design defects: allocation-time same-arch card substitution, HI151 maintenance admission, and Noble 23.11 no-accounting `InvalidAccount`. Remaining uncertainty is concentrated in real AMD/production behavior.

## Standards

- `docs/design/JOBS_ORCHESTRATOR.md`.
- `docs/reference/jobs/SLURM_BRUTUS.md`.
- Existing BigCherry provenance/identity rules.
- permanent code under `tools/bigcherry`, permanent tests under `tools/tests`, RCD validation under `tools/lab` with registry disposition.

## Acceptance Criteria

- RCD03-RCD12 are implementation-ready with explicit hardware-only gates.
- 32-check planning model passes.
- real BigCherry process/recovery smokes pass.
- real Noble Slurm accounting/service/scheduler smoke passes.
- final series identity binds one exact deterministic hardware cohort before submission.
- proper subsets use whole-architecture reservation in v1.
- static contradiction scan passes.
- RCD lab tooling is registered.
- no unresolved platform choice remains.

## Notes

Decision: Slurm + minimal local slurmdbd association layer + thin BigCherry domain service + platform-neutral Executor. Real validation results supersede earlier no-accounting/static assumptions.

## Change Log

- 2026-09-26T00:51:52.243021+00:00 (created-by): Created by agent
- 2026-09-26 (dev-gpt-agent): Fully specified architecture/dependencies/offline falsification.
- 2026-09-26 (dev-gpt-agent): Added deterministic pre-series hardware binding and whole-architecture subset reservation.
- 2026-09-26 (dev-gpt-agent): Added real BigCherry-process, recovery and Noble Slurm CI; real tests corrected HI151 admission and required minimal slurmdbd associations.
