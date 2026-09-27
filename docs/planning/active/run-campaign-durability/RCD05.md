---
id: RCD05
order: 5
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:52:05.135167+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Implement attempt resolution, pinned workspaces and monitored execution

## Objective

Every attempt resolves one exact BigCherry commit, runs from an isolated workspace, renders the existing `validation_campaign` command from typed state, and produces durable process/result/log evidence without mutating the canonical checkout. Executor/scheduler state is execution metadata; scientific run/series identity remains stable across harness retries.

## Implementation status — 2026-09-27

Implemented:

- `tools/bigcherry/jobs/workspace.py`: production git workspace manager plus deterministic passthrough test manager.
- `JobService` creates immutable attempt records with exact resolved commit, project root, scientific identity and frozen hardware binding.
- attempt-start scientific identity is re-resolved against the pinned workspace and compared to the frozen series identity; drift blocks rather than mutates the series.
- `tools/bigcherry/jobs/runner.py` renders `python -m bigcherry.patch.validation_campaign` from persisted typed job state.
- managed `--device-map` contains allocation-local positions only (`0,1,...`), never persisted host physical indices.
- Local/direct mode uses host OS device locks keyed by frozen stable device IDs; Slurm mode uses `BIGCHERRY_RESOURCE_POLICY=external` and does not double-lock scheduler-owned GPUs.
- runner writes `executor-start.json` and `executor-result.json` sentinels for crash/status reconciliation.
- same-commit retry creates attempt N+1 with the prior commit; latest retry resolves a new attempt commit.
- production v1 Slurm template no longer uses native monolithic `RequeueExit`; native requeue remains test-only until a stage is explicitly restart-safe.

Still incomplete:

1. `experiment.bundle.run_managed()` still needs true streaming output rather than memory capture for ~1.5 GB server logs.
2. runner needs disk/root/work reserve monitoring and typed disk-pressure termination.
3. heartbeat/progress/stall monitoring is not production-complete.
4. attempt launch needs full accepted-inventory/allocation stable-ID attestation before GPU child spawn.
5. process-group TERM -> bounded grace -> KILL handling needs permanent tests across Linux and Windows-local paths.
6. retention/cleanup policy is not implemented.
7. canonical gitignored vendor/environment linking must be verified in the production git workspace path on Brutus.

## Attempt semantics

```text
run_id       scientific session identity
attempt_no   BigCherry harness/execution attempt
commit       immutable after attempt submission
```

Retry rules:

```text
retry --same-commit -> attempt+1, exact previous commit
retry --latest      -> attempt+1, resolve configured code ref again
scientific FAIL     -> execution completed; never a harness retry trigger
```

A new attempt must recheck all series-frozen scientific identities. If focal/common/promoted patch bytes, validation/contract identity, model/corpus identity or other frozen scientific material differs, the attempt is rejected and a new series is required.

## Workspace layout

```text
<jobs>/runs/<run>/attempts/NNN/
  attempt.json
  submission-intent.json
  submission.json
  executor-start.json
  executor-result.json
  stdout.log
  stderr.log
  launch.sh                 # Slurm only

<jobs>/worktrees/<run>/<attempt>/
  detached BigCherry worktree at exact commit
```

Never stash or dirty the canonical checkout to start managed work.

## Campaign rendering

The runner renders only named supported flags:

```text
--patch
--baseline-source
--amdgpu-targets
--device-map <arch>=<allocation-local positions>
--model
--hip-path
--workdir
--worktree-root
--build-root
--validation-producer
--common-patches
--producer-input
--producer-corpus
--production-lane
```

No arbitrary `extra_args` escape hatch exists in JobSpec.

`BIGCHERRY_SELECTED_DEVICE_IDS` carries the series-bound stable identities separately from the launch-local `--device-map`.

## Resource locking

- Slurm-managed execution: Slurm GRES/licenses/partitions own GPU/build/host resources; campaign-local GPU/resource locks must operate in external mode.
- LocalExecutor/remote worker: OS-held `HostFileLock` instances keyed by stable device ID serialize exact bound devices. Process death releases kernel locks.
- HI151 `tree_activity` remains repository-maintenance fencing only; do not reuse it as GPU/event locks.

## Required monitoring

Add production `monitor.py` with:

- child/process-group liveness;
- stdout/stderr byte progress;
- structured phase/event progress;
- process CPU ticks;
- selected artifact growth;
- root/work/tmp free-space thresholds;
- configurable stall interval.

A quiet compile with CPU progress is not a stall. Disk/stall termination is infrastructure failure and cannot become scientific FAIL.

Host policy example:

```toml
[jobs.disk]
root_min_free_gib = 50
work_min_free_gib = 200
poll_seconds = 15
server_log_retain_mib = 20
```

## Streaming requirement

Before production cutover, replace child `capture_output=True` in `experiment.bundle.run_managed()` with direct file-backed streaming/process-group execution while preserving:

1. intent durable before spawn;
2. stdout/stderr opened before spawn;
3. terminal result/artifact hashes atomic after child exit;
4. launch failure and interruption explicitly recorded;
5. Python RSS does not scale with child output size.

Permanent test must generate/sparsely stream representative ~1.5 GB output without buffering it in Python memory.

## Retention

Planned policy:

- active/unharvested evidence: never auto-delete;
- successful+harvested runner/worktree: eligible after 24h;
- failed/stalled: eligible after 7d;
- disk pressure removes only eligible roots oldest first;
- evidence-required files are never destructively truncated;
- large non-evidence logs may be compacted only after original byte count/full digest are recorded.

## Tests required for completion

- branch moves before attempt -> latest resolves new commit;
- branch moves after submission -> attempt remains pinned;
- scientific identity drift blocks before spawn;
- deterministic campaign argv/no passthrough;
- allocation-local device map never contains stable ID/BDF/host ordinal;
- LocalExecutor maps only frozen stable cohort;
- Slurm allocation attestation rejects missing/substituted stable device;
- intent/start/result crash windows recover without duplicate attempt;
- streamed large output bounded memory;
- process-group cancellation;
- disk preflight + mid-run hard threshold;
- CPU-progress compile not falsely stalled;
- all-channel inactivity is stalled;
- HI151 maintenance races remain zero-overlap;
- cleanup never removes unharvested evidence.

## Brutus-only gates

- real detached workspace with canonical vendor/environment inputs;
- real disk pressure behavior during build;
- process-tree cancellation of validation campaign/server children;
- representative huge-server-log behavior;
- exact Slurm allocation/stable-ID preflight before HIP initialization.

## Acceptance criteria

- exact attempt commit immutable after submission;
- no canonical checkout stash/mutation required;
- scientific drift blocks instead of silently changing series;
- managed output streaming is safe for large logs;
- typed infrastructure incidents are durable;
- GPU identity is stable-ID based while launch selectors remain local;
- Slurm/local lock domains are correct and non-duplicative;
- all offline tests green.

## Change log

- 2026-09-26: initial pinned runner/worktree/monitor design.
- 2026-09-27: workspace/runner/attempt implementation added; scientific identity recheck and allocation-local device mapping implemented.
- 2026-09-27: native Slurm requeue removed from monolithic production policy until restart-safe stage boundaries exist.
