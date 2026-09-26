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

Implement the per-attempt execution boundary used by all executors. Resolve mutable code refs only when creating a new attempt, pin that exact BigCherry commit for the attempt, create an isolated runner worktree, supply gitignored vendor/config inputs explicitly, resolve toolchain/model/file identities, monitor disk/log/progress, and classify infrastructure failures without interpreting scientific results.

GPU resources are capability requests resolved by RCD12/Executor allocations. This layer never treats host physical index as identity and does not globally override Slurm visibility.

## Steps

1. Add attempt resolver: mutable ref -> exact commit; freeze commit/toolchain/model/composition/file identities in `attempt.json`.
2. Add detached runner-worktree creation/cleanup; link canonical gitignored `vendor/llama.cpp`; export canonical host environment file.
3. Render the existing `python -m bigcherry.patch.validation_campaign ...` invocation from typed JobSpec/AttemptSpec; no arbitrary args.
4. Implement executor-specific allocation environment handling:
   - Slurm: preserve scheduler `ROCR_VISIBLE_DEVICES`, map allocation to stable RCD12 IDs, normally leave `HIP_VISIBLE_DEVICES` unset.
   - Local: RCD11 resolves launch-local device selector.
5. Implement preflight/continuous root+work disk guards, streaming logs, stall progress fingerprint and bounded log retention.
6. Integrate `core.tree_activity.Lease`; fix the start race so `Lease.__enter__` refuses while `MaintenanceLock` is held.
7. Implement attempt result/failure classification and retention cleanup.
8. Test entirely with temporary git repos, fake process handles and FakeExecutor before hardware.

## Detailed Solution & Technical Design

### Attempt resolution

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

Algorithm for `retry --latest`:

1. read immutable run/series intent;
2. resolve `code_ref` in canonical checkout;
3. verify contract/focal/common/frozen-composition identities still match the series rules;
4. create attempt N+1 with resolved commit;
5. create worktree at that commit;
6. re-resolve mutable branch immediately before submission; if it moved and policy is `latest-at-start`, discard unsubmitted worktree/spec and resolve again;
7. once submitted, never change commit.

A branch update after submission affects only future attempts. Same-commit Slurm requeue (`75`) reuses the same attempt manifest.

### Runner worktree

Layout:

```text
<work>/jobs/runs/<run>/attempts/NNN/
  runner/
  work/
  stdout.log
  stderr.log
  attempt.json
  submission.json
  result.json
```

Creation:

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

Do not copy environment.local into runner. Do not modify/stash canonical checkout to run an attempt.

### Campaign argv

```python
def build_validation_campaign_argv(job: JobSpec, attempt: AttemptSpec) -> tuple[str, ...]: ...
```

It renders named current flags only: patch, baseline source, common patches, architecture, model, HIP path, work/build/worktree roots, validation producer, producer inputs/corpus, production lane, device-map compatibility where still required. RCD06 later replaces physical-map compatibility with allocation-aware/frozen/prepared seams.

### Monitoring

`run_managed()` is extended to stream stdout/stderr directly to files; never `capture_output` potentially gigabyte logs.

Progress fingerprint contains:

- latest structured phase/event sequence when available;
- stdout/stderr byte sizes;
- selected artifact mtimes/sizes;
- process-tree CPU ticks;
- child process existence.

Stall requires all channels unchanged for configured threshold. Quiet compiler periods with CPU progress are not stalls.

Disk policy is host config, not code constants:

```toml
[jobs.disk]
root_min_free_gib = 50
work_min_free_gib = 200
poll_seconds = 15
server_log_retain_mib = 20
```

A hard breach terminates process group, writes failure `disk-pressure`, retains failed attempt. Completed oversized server logs may be tail/truncated only after byte count + full-content hash are recorded; evidence-required logs are never destructively truncated.

### Tree activity

Modify `Lease.__enter__` to atomically/fail-closed check `maintenance.lock` immediately before lease publication. Maintenance still refuses while live leases exist. Offline concurrency test must prove either lease or maintenance wins, never both.

### Retention

- active/unharvested: unlimited;
- successful+harvested runner: eligible after 24h;
- failed/stalled runner: eligible after 7d;
- remove via `git worktree remove --force` then `git worktree prune`;
- disk pressure cleans only eligible roots oldest-first; never deletes sole evidence.

## Code Samples & Guidance

Planned signatures:

```python
def resolve_attempt(job: JobSpec, series: SeriesRecord, *, repo: Path, work_root: Path) -> AttemptSpec: ...
def ensure_runner_worktree(attempt: AttemptSpec, *, canonical_repo: Path) -> Path: ...
def build_validation_campaign_argv(job: JobSpec, attempt: AttemptSpec) -> tuple[str, ...]: ...
def progress_fingerprint(ctx: MonitorContext) -> ProgressFingerprint: ...
def classify_process_result(ctx: AttemptContext, returncode: int) -> AttemptResult: ...
def cleanup_attempts(policy: RetentionPolicy, *, now: datetime) -> tuple[Path, ...]: ...
```

Failure classification based on direct observations (disk guard, timeout, preflight status, process exit), not broad regex guesses where avoidable.

## Files

Planned:

- `tools/bigcherry/jobs/attempt.py`
- `tools/bigcherry/jobs/runner.py`
- `tools/bigcherry/jobs/monitor.py`
- `tools/bigcherry/jobs/retention.py`
- `tools/bigcherry/experiment/bundle.py`
- `tools/bigcherry/core/tree_activity.py`
- `tools/tests/jobs/test_attempt.py`
- `tools/tests/jobs/test_runner.py`
- `tools/tests/jobs/test_monitor.py`
- `tools/tests/core/test_tree_activity.py`

## Validation

Offline:

- temporary git branch moves between initial resolution and submission -> re-resolved before submission;
- branch moves after submission -> running attempt unchanged;
- new-attempt retry may use new commit while same run_id remains;
- runner vendor symlink points to canonical clone;
- host config path is explicit and runner checkout may omit it;
- campaign argv deterministic/no unknown passthrough;
- Slurm env preservation test never rewrites scheduler `ROCR_VISIBLE_DEVICES` with physical indices;
- root/work disk guards classify before spawn and mid-run;
- compiler-like quiet output + CPU activity does not stall;
- all channels idle does stall;
- 1.5GB-size simulation uses sparse/mock file metadata and verifies retention policy without allocating 1.5GB;
- tree lease vs maintenance race falsification;
- cleanup never removes unharvested attempts.

Hardware:

- queued branch update scenario on Brutus;
- real disk-pressure guard with safe temporary threshold;
- process-group cancellation;
- server-log retention on representative campaign.

## Effort & Risk

Medium. Main risks: Git worktree lifecycle under crashes, false stall detection, and losing evidence during cleanup. Fail closed: ambiguous attempt remains retained/failed-harness.

## Standards

Existing build code already handles stale foreign CMake cache/configure-request identity; do not duplicate that logic here.

## Acceptance Criteria

- attempt commit immutable after submission;
- runner requires no dirty/stashed canonical checkout;
- disk/log/stall incident classes are handled durably;
- tree maintenance and job lease cannot overlap;
- no physical GPU ordinal is scientific identity;
- all offline tests pass.

## Notes

Per-attempt runner worktrees remain necessary while current campaign evidence writes through repo-relative paths. RCD06 external evidence output later permits safer shared/read-only runners where otherwise valid.

## Change Log

- 2026-09-26T00:52:05.135167+00:00 (created-by): Created by agent
- 2026-09-26 (dev-gpt-agent): Specified pinned attempt/worktree/monitoring lifecycle and removed physical-device assumptions.
