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
work: S
---

# Freeze the job-service architecture and implementation boundaries

## Description

Make `docs/design/JOBS_ORCHESTRATOR.md` the normative architecture for the run-campaign job service and convert its agreed decisions into executable plan boundaries. Adopt host-installed single-node Slurm on Brutus for durable queue/process/resource scheduling behind a platform-neutral BigCherry `Executor`; retain BigCherry authority for scientific identity, series policy, commit pinning, retry legality, evidence, reporting, production coexistence and hardware identity. Reject a custom scheduler database/daemon as an execution authority.

This item is a design freeze, not a Slurm installation task. Later RCD items may refine implementation details only when falsification finds a contradiction; they must not silently reintroduce physical-index identity, static production-GPU assumptions, per-slot GRES types, mixed-commit attempts, allocation-time card selection, or result-driven retries.

## Steps

1. Treat `docs/design/JOBS_ORCHESTRATOR.md` as normative; RCD02-RCD12 are implementation plans derived from it.
2. Freeze ownership boundaries:
   - Slurm: queued/running/held/cancelled execution state, CPU/GPU allocation, dependencies, priority, licenses, containment when qualified, same-commit requeue.
   - BigCherry: JobSpec/series/run/attempt identity, contract/composition freeze, commit pinning, capability resolution, production gate, retry classification, evidence/harvest/report/events.
3. Freeze executor abstraction: domain modules depend only on `Executor`; only `jobs/slurm.py` may know `sbatch`, `squeue`, GRES, licenses, `SLURM_*` or Slurm state spellings.
4. Freeze rollout:
   - v1: monolithic `validation_campaign` Slurm job with RCD06 M1 service-safety seams (external evidence, frozen composition, preflights/progress); retire shell queue only after hardware acceptance.
   - v1.5: prepare/execute split with verified prepared manifest.
   - v2: RCD01 durable operation/result protocol over already-externalized evidence.
   - v3: full stage DAG.
   - v4: timed-measure overlap only after `scheduler-isolation-v1`.
5. Freeze execution order/dependencies:
   - RCD04 and RCD12 can implement/test most code offline in parallel, but RCD04 final series creation consumes RCD12 deterministic `SeriesGpuBinding`.
   - RCD03 consumes RCD12 inventory/GRES generation.
   - RCD05 consumes RCD03/RCD04/RCD12.
   - RCD06 can proceed in parallel with RCD03-RCD05.
   - RCD11 consumes RCD04/RCD12 stable device identity.
   - RCD09 consumes RCD04/RCD05 event/run-store primitives.
   - RCD10 gates v1 retirement after RCD03/RCD04/RCD05/RCD06 milestone M1/RCD09/RCD11/RCD12; RCD06 prepare/execute M2 is v1.5 and not a v1 cutover blocker.
   - RCD07/RCD08 are post-v1 durability/stage/harvest work.
6. Maintain a plan-only falsification harness under `tools/lab/run-campaign-durability/`; promote code to `tools/bigcherry/**` only when implementing an RCD item.
7. Register retained lab files in `docs/reference/tooling/TOOL_DISPOSITION.md` before RCD02 completion; classify the harness/README as RCD-owned `TRANSITIONAL` planning tooling, not production/evidence authority.

## Detailed Solution & Technical Design

### Platform decision

Use Slurm 23.11.x on Ubuntu 24.04 with MUNGE, `slurmctld`, `slurmd`, `jobcomp/filetxt`; no MariaDB/slurmdbd/slurmrestd in v1. Use one generic `bc-measure` partition and one `bc-build` partition. Dynamic production conflicts are BigCherry pre-dispatch/runtime policy, not static partitions/reservations.

`Executor` must remain platform-neutral:

```python
class Executor(Protocol):
    def submit(self, request: ExecutionRequest) -> ExecutionHandle: ...
    def status(self, handle: ExecutionHandle) -> ExecutionStatus: ...
    def cancel(self, handle: ExecutionHandle) -> None: ...
    def control(self, handle: ExecutionHandle, action: ExecutorControl) -> None: ...
    def allocation(self, handle: ExecutionHandle) -> Allocation | None: ...
    def events(self, handle: ExecutionHandle, *, after: int | None = None) -> Iterable[ExecutorEvent]: ...
```

Adapters: `SlurmExecutor`, `LocalExecutor`, `FakeExecutor`. A Windows HIP attempt is a distinct execution environment/series from Linux ROCm.

### Identity invariants

- `run_id` identifies one planned scientific session and survives harness retries.
- `attempt_no` increases when a retry uses a new commit/config execution attempt.
- BigCherry commit is frozen per attempt; stages never resolve branches independently.
- final series identity freezes planned N, contract/base/focal/common/validated implementation identity, platform environment, and one deterministic exact RCD12 hardware cohort selected from accepted stable device IDs before any session is submitted.
- every session in a series uses that same stable-device cohort; Slurm may allocate a wider set only for safe all-of-architecture reservation and BigCherry then narrows to the pre-bound IDs.
- slots/ordinals/BDF/render node are operational observations, never scientific card identity.
- same-model replacement card starts a new hardware cohort; topology change starts a new cohort unless equivalence was prequalified.
- exit 0 includes scientific PASS or FAIL; 75 is same-commit transient requeue; 76 requests new attempt; 77 blocks as invalid/drift.
- no anomalous result is itself a retry trigger before PVPS09.

### Non-hardware falsification

`tools/lab/run-campaign-durability/mock_pipeline.py --self-test` is the executable planning model. It covers capability resolution, peer/exact-device constraints, deterministic series hardware binding, production conflict fail-closed behavior, environment/cohort identity, retry actions and FakeExecutor dependency ordering. It is not production code and cannot satisfy any hardware acceptance criterion.

Permanent tests must still replace the lab model before production acceptance.

## Code Samples & Guidance

Production packages planned by later items:

```text
tools/bigcherry/jobs/
tools/bigcherry/hardware/
tools/tests/jobs/
tools/tests/hardware/
```

Plan-only validation stays:

```text
tools/lab/run-campaign-durability/
```

Never make `docs/design/JOBS_ORCHESTRATOR.md` executable configuration. Runtime policy is represented by typed code/config and persisted resolved manifests.

## Files

- `docs/design/JOBS_ORCHESTRATOR.md` — normative architecture.
- `docs/planning/active/run-campaign-durability/RCD02.md` ... `RCD12.md` — implementation plans.
- `tools/lab/run-campaign-durability/mock_pipeline.py` — planning simulator.
- `tools/lab/run-campaign-durability/README.md` — validation scope.
- `docs/reference/tooling/TOOL_DISPOSITION.md` — required lab-tool classification.

## Validation

Completed during planning:

```bash
PYTHONPATH=tools python tools/lab/run-campaign-durability/mock_pipeline.py --self-test
# {"checks": 30, "ok": true}
```

Required static review before closing RCD02:

- no RCD plan treats physical GPU index as identity;
- no RCD plan hard-codes production to current GPU slots;
- no per-slot GRES type such as `gfx1100_0`;
- no custom SQLite scheduler is execution authority;
- no stage in one attempt may select a different BigCherry commit;
- no series may silently select a different stable GPU cohort between sessions;
- no scientific FAIL/non-material result is auto-retried;
- every hardware-dependent claim is an explicit acceptance gate;
- both RCD lab files are classified in the current TOOL_DISPOSITION registry.

## Effort & Risk

Low implementation effort; high leverage. Primary risks are later plans drifting from the normative design and seemingly harmless capability resolution moving a series between physical cards. Mitigate by capability-based executor-neutral policy plus exact pre-series cohort binding.

## Standards

- `docs/design/JOBS_ORCHESTRATOR.md`.
- Existing BigCherry provenance/identity rules.
- `AGENTS.md` / `docs/reference/tooling/TOOLING.md`: permanent code under `tools/bigcherry`, plan-specific falsification under `tools/lab`, tests under `tools/tests`, and every retained lab tool registered in TOOL_DISPOSITION.

## Acceptance Criteria

- Architecture/ownership/rollout/dependency graph above is explicit.
- RCD03-RCD12 contain implementation-ready modules/signatures/tests and cite hardware-only gates separately.
- Planning simulator reports 30 checks passing.
- Final series identity is bound to one deterministic hardware cohort before session submission.
- Static contradiction scan above passes.
- Lab harness/README are classified in TOOL_DISPOSITION.
- No unresolved platform choice remains.

## Notes

Decision: Slurm + thin BigCherry domain layer + platform-neutral Executor. The Round-4 dynamic production/hardware design supersedes older GPU-class/static-inventory text.

## Change Log

- 2026-09-26T00:51:52.243021+00:00 (created-by): Created by agent
- 2026-09-26 (dev-gpt-agent): Fully specified final architecture, dependency order, invariants and offline falsification.
- 2026-09-26 (dev-gpt-agent): Adversarial follow-up added deterministic pre-series hardware binding and lab-tool disposition acceptance gate.
- 2026-09-26 (dev-gpt-agent): Planning harness extended to 30 checks; validation contract updated.
