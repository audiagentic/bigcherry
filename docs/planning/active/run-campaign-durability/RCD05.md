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

# Implement attempt resolution, pinned runner worktrees and monitored execution

## Description

Implement the per-attempt execution boundary used by every `Executor`. Resolve mutable code refs only when creating a new attempt, pin one exact BigCherry commit for that attempt, create an isolated runner worktree, resolve host/toolchain/model/composition identities, render the existing validation-campaign argv, stream logs, monitor disk/progress, and classify infrastructure failures without interpreting scientific results.

This item extends existing seams rather than replacing them: HI151 remains the repository-maintenance fence; `experiment.bundle.run_managed()` remains the managed child-process seam; `validation_campaign` remains the scientific campaign entry point.

## Steps

1. Add attempt resolver: mutable ref -> exact commit; persist commit/toolchain/model/composition/file identities in `attempt.json`.
2. Create detached per-attempt runner worktree at that commit; use explicit canonical paths for gitignored vendor/config inputs.
3. Render `python -m bigcherry.patch.validation_campaign ...` from typed JobSpec/AttemptSpec; no arbitrary passthrough argv.
4. Map executor allocation to launch environment:
   - Slurm: preserve scheduler-provided `ROCR_VISIBLE_DEVICES`, resolve allocation to RCD12 stable IDs, normally leave `HIP_VISIBLE_DEVICES` unset;
   - LocalExecutor: resolve stable IDs to launch-local selectors immediately before spawn.
5. Change `run_managed()` from `capture_output=True` to direct/streamed stdout/stderr files while preserving durable intent-before-spawn and terminal artifact hashing.
6. Add root/work disk guards, heartbeat/progress fingerprint, process-group cancellation, bounded log-retention policy and typed infrastructure failures.
7. Use current HI151 `Lease` around each long-running attempt. Do **not** add a second lock protocol: HI151 already uses the complementary publish/recheck handshake for Lease vs MaintenanceLock.
8. Implement result/retention cleanup; never delete unharvested or sole evidence.
9. Promote the current RCD real-process/recovery/race smokes into permanent `tools/tests/**` coverage when implementing this item.

## Detailed Solution & Technical Design

### Attempt identity

```python
@dataclass(frozen=True)
class AttemptSpec:
    run_id: str
    attempt_no: int
    bigcherry_commit: str
    runner_root: Path
    toolchain: ToolchainIdentity
    model: FileIdentity
    corpus: FileIdentity | None
    platform_environment_hash: str
    inventory_hash: str
    frozen_composition_hash: str
    command: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
```

`retry --latest`:

1. read immutable run/series intent;
2. resolve `code_ref` in canonical checkout;
3. verify contract/focal/common/frozen-composition identities still satisfy series rules;
4. create attempt N+1 at the resolved commit;
5. create detached runner worktree;
6. immediately before first submission, if policy is `latest-at-start` and the mutable ref moved, discard the unsubmitted attempt/worktree and resolve again;
7. after submission, commit is immutable for every stage/requeue of that attempt.

Exit 75 reuses this attempt and commit. Exit 76 creates a new attempt under the same `run_id` and may resolve a new commit.

### Runner layout

```text
<work>/jobs/runs/<run>/attempts/NNN/
  runner/
  work/
  attempt.json
  submission-intent.json
  submission.json
  stdout.log
  stderr.log
  result.json
```

Creation/environment:

```text
git worktree add --detach <runner> <commit>
<runner>/vendor/llama.cpp -> canonical gitignored vendor clone
BIGCHERRY_ENVIRONMENT=<canonical>/config/environment.local.toml
PYTHONPATH=<runner>/tools
TMPDIR=<work>/jobs/tmp
CCACHE_DIR=<work>/ccache
CCACHE_BASEDIR=<work>
CCACHE_NOHASHDIR=1
CCACHE_MAXSIZE=100G
CCACHE_COMPILERCHECK=content
```

Never stash or mutate the canonical checkout to start a job.

### Campaign argv

```python
def build_validation_campaign_argv(job: JobSpec, attempt: AttemptSpec) -> tuple[str, ...]: ...
```

Render only current named flags: patch, baseline source, common patches, architecture, model, HIP path, work/build/worktree roots, validation producer, producer inputs/corpus, production lane, and transitional device-map compatibility. RCD06 adds frozen/prepared/external-evidence seams; this layer does not duplicate campaign parsing.

The CI smoke already exercises the real `validation_campaign.main()` producer-dispatch seam with mocked hardware body and proves selector mismatch/missing mandatory runtime inputs fail closed.

### Managed child process

Preserve the current HI47 invariant: `experiment.json` intent is atomically durable before child spawn. Replace memory-buffered execution:

```python
subprocess.run(..., capture_output=True)
```

with a process-group-aware streaming implementation:

```text
open stdout.log/stderr.log before spawn
spawn child in its own process group/session
stream directly to files
emit heartbeat/progress events independently
on cancel/disk/stall: TERM group -> grace -> KILL group
fsync/close logs
hash terminal artifacts
atomically write terminal result
```

The RCD CI currently proves intent-before-spawn, success, exit 76 preservation, launch failure 127, KeyboardInterrupt -> interrupted/130, artifact tamper rejection and an 8 MiB real output. The 8 MiB case also confirms the present implementation still buffers output, so streaming remains a required implementation change before 1.5 GB server logs are safe.

### Monitoring

Progress fingerprint:

- latest structured phase/event sequence;
- stdout/stderr byte sizes;
- selected artifact mtimes/sizes;
- process-tree CPU ticks;
- child/process-group liveness.

A stall requires all channels unchanged for the configured interval. Quiet compilation with CPU progress is not a stall.

Host-configured disk policy, not constants:

```toml
[jobs.disk]
root_min_free_gib = 50
work_min_free_gib = 200
poll_seconds = 15
server_log_retain_mib = 20
```

Hard breach terminates the process group, persists `disk-pressure`, retains the failed attempt. A completed oversized non-evidence server log may be tail-compacted only after original byte count/full hash are recorded. Evidence-required files are never destructively truncated.

### HI151 maintenance fencing

Current production HI151 is authoritative:

```text
Lease admission:
  if maintenance exists -> refuse
  publish lease
  recheck maintenance
  if maintenance appeared -> withdraw lease + refuse

Maintenance admission:
  mkdir maintenance marker first
  scan live leases
  if any live/remote-unknown lease -> remove marker + refuse
  otherwise publish owner and proceed
```

This complementary ordering closes the check/create race without an extra `protocol.lock`. Dead local PID leases may be explicitly pruned; remote-host leases remain fail-closed/live.

Planning CI evidence on 2026-09-26:

```text
real_recovery_smoke.py: 18 checks PASS
  lock->lease refusal
  lease->lock refusal
  crashed child stale lease detection/prune
  pinned git worktree across branch advances
  runner dirtiness isolated from control checkout

tree_activity_race_smoke.py: 50 simultaneous races / 100 assertions PASS
  entered=50 blocked=50 overlap=0
```

Permanent implementation tests must retain a concurrent stress test; an additional host lock is only justified if a future platform/storage implementation invalidates the current atomic-directory/file assumptions.

### Retention

- active/unharvested: never automatically remove;
- successful+harvested runner: eligible after 24h;
- failed/stalled runner: eligible after 7d;
- cleanup uses `git worktree remove --force` then `git worktree prune`;
- disk pressure cleans only eligible roots oldest-first.

## Planned Files

- `tools/bigcherry/jobs/attempt.py`
- `tools/bigcherry/jobs/runner.py`
- `tools/bigcherry/jobs/monitor.py`
- `tools/bigcherry/jobs/retention.py`
- `tools/bigcherry/experiment/bundle.py`
- `tools/bigcherry/core/tree_activity.py` only if a demonstrated defect remains
- `tools/tests/jobs/test_attempt.py`
- `tools/tests/jobs/test_runner.py`
- `tools/tests/jobs/test_monitor.py`
- `tools/tests/core/test_tree_activity.py`

No `core/host_lock.py` is required by the validated design.

## Validation

Offline/permanent tests:

- mutable branch moves before submission -> unsubmitted attempt is re-resolved;
- branch moves after submission -> running attempt remains pinned;
- new attempt can use new commit while preserving `run_id`;
- runner vendor/config inputs resolve explicitly;
- campaign argv deterministic/no unknown passthrough;
- Slurm launch env never rewrites scheduler visibility with host physical indices;
- intent durable before spawn;
- streamed large output does not scale process RSS with log size;
- child success/nonzero/launch failure/interruption/tamper all persist correctly;
- disk guard before spawn and mid-run;
- CPU-progress compiler is not false-stalled; all-channel inactivity is stalled;
- 1.5 GB log behavior tested with sparse/generated stream without retaining 1.5 GB in Python memory;
- concurrent HI151 lease/maintenance stress has zero overlaps;
- crashed local lease prunable only after PID death; remote unknown remains blocking;
- cleanup never removes unharvested evidence.

Brutus-only:

- queued branch-update incident;
- real disk-pressure threshold exercise;
- process-group cancellation of real campaign tree;
- representative server-log retention.

## Acceptance Criteria

- attempt commit immutable after first submission;
- canonical checkout remains clean/no stash required;
- managed output is streamed, not captured in memory;
- infrastructure incidents have durable typed results;
- HI151 maintenance/job admission retains zero-overlap regression coverage;
- no physical GPU ordinal is scientific identity;
- all offline tests pass.

## Notes

Per-attempt runner worktrees remain required while campaign evidence writes repo-relative state. RCD06 external evidence output later removes that coupling.

## Change Log

- 2026-09-26T00:52:05.135167+00:00 (created-by): Created by agent
- 2026-09-26 (dev-gpt-agent): Specified pinned attempt/worktree/monitoring lifecycle and physical-device-independent execution.
- 2026-09-26 (dev-gpt-agent): Replaced proposed redundant host-lock protocol with empirically validated HI151 two-phase admission; recorded real process/recovery/race CI gates.
