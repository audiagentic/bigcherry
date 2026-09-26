# BigCherry job orchestrator — design (draft)

Status: design, not implemented. Source: dev-gpt-agent session `ses_4019cfc56e774dd2`
(`req_c70aecafc33e4b59` first pass, `req_fb219ca02c184706` deep-review revision,
2026-09-26), requested to replace the lab queue in `tools/lab/plan-qualification/`.
Open question pending in the same session: build on an existing scheduler (Slurm
single-node / pueue) versus the custom daemon below, and consolidation with
existing components (see "Open decisions").

## Final decisions (revision)

| Area | Decision |
|---|---|
| Runtime | One Brutus-local daemon, one active campaign in the MVP |
| Durable state | SQLite WAL; append-only `events` table (triggers forbid UPDATE/DELETE) |
| Identity | `run_id` stable; retries create `attempt_no + 1`; failed attempts never overwritten |
| Code | Resolve the branch commit immediately before spawn (re-check, then pin); running attempts never change code |
| Series | Freeze N, contract hash, base revision, focal/common implementation digests and the validated-set composition |
| Promotion mid-series | Default `freeze`: the campaign is given the exact frozen validated patch IDs + digests and fails before build on drift |
| Evidence | MVP: per-attempt detached runner worktree (campaign writes evidence into its repo); Phase 3 adds an external evidence sink so worktrees can be shared read-only |
| GPUs | Jobs declare physical device indices, resolved through `Host.gpu_visibility_env()`; no raw `VIS=` |
| Measurement | Timed performance stages are host-exclusive until isolation is qualified |
| Retry | Harness/environment/host failures only; a scientific FAIL is `done-fail` and never auto-retried |
| Outliers | No result-driven re-measure before PVPS09 defines a result-independent rule |
| Parallelism | Not in the queue replacement; stage split first |
| Agent API | `--json` on every command, `watch --jsonl --after <seq>` with stable event sequence numbers |

## Adversarial review of the first pass (incident-driven)

- Commit resolved at submission would run known-broken code minutes later -> resolve at attempt start.
- Recording only a validated-set hash lets later sessions silently measure a different control -> freeze IDs + digests and force them into the campaign.
- Shared per-commit runner worktree collides with `write_record()` writing into the repo -> per-attempt worktree first.
- Runner worktrees lack the gitignored `vendor/llama.cpp` and the untracked host config -> symlink the canonical vendor clone; export `BIGCHERRY_ENVIRONMENT`.
- Stale CMake cache after a promotion is already handled by `campaign/build.py` -> the scheduler must not duplicate it.
- Mirrored JSONL journal and lease/fencing are over-engineering for a serial single host -> SQLite events table + daemon lock.
- Post-run log truncation cannot stop a 1.5 GB/arm log filling a disk mid-job -> preflight both filesystems + continuous disk watchdog + post-run truncation; later bounded server logging.
- "No log growth" misfires on quiet compiles -> stall only when stdout, artifacts, process-tree CPU ticks and phase markers are all idle for T.
- MTP full-vocab rows and `-sm tensor` topology are producer semantics -> typed producer preflights (`ok` / `retryable` / `invalid`), not scheduler special cases.
- A -30% round is not a legal retry trigger -> telemetry only until PVPS09.
- A gateway restart is irrelevant: submission writes SQLite; the Brutus daemon owns the process.

## Throughput (70 jobs, 10-15 min build + 5-10 min measure each)

| Schedule | Wall time |
|---|---|
| Strict serial | 17.5-29.2 h (midpoint 23.3 h) |
| 2 builds, measurement host-exclusive | midpoint 16.0 h |
| 1 build overlapping 1 timed measurement (needs qualification) | midpoint 14.6 h (-37.5%) |

Qualification order: (1) builds during untimed correctness/activation (safe now);
(2) one build overlapping one timed measurement, qualified by a non-patch A/A
experiment `scheduler-isolation-v1` (>= 32 paired blocks per condition, randomized,
production affinity/nice/ionice; accept only if the condition effect's 95% CI is
within [-0.10%, +0.10%], variance ratio upper bound <= 1.15, no clock/power/thermal
shift; bound fixed before results); (3) concurrent GPU measurements only if still needed.

## Worktrees and caches

`/mnt/data/bigcherry-jobs/{state/jobs.sqlite3, attempts/<run>/<attempt>/{runner,run,stdout.log,attempt.json}, worktrees, builds, tmp, ccache}`.
Runner worktree: `git worktree add --detach`, `vendor/llama.cpp` symlinked, `BIGCHERRY_ENVIRONMENT` exported,
`CCACHE_COMPILERCHECK=content` for multi-ROCm safety. Build/source caches stay keyed by
llama.cpp composition identity, never by BigCherry commit. Retention: harvested done
attempts 24 h, failed-harness/stalled 7 days, active/unharvested unlimited;
`git worktree remove --force` + `prune`, oldest first under disk pressure.

## Frozen series

Store base revision, focal/common/validated patch IDs + implementation digests, contract
hash, producer identity, planned N. `validated_set_hash` and `control_composition_hash`
are sha256 of canonical JSON. `validated_enhancement_patches(..., frozen=...)` uses exactly
the frozen set and verifies digests; orchestrator passes `--frozen-validated-composition <json>`.
A newer harness commit may join a series only if contract, base revision and focal/common
digests are unchanged. `restart-on-promotion` supersedes remaining sessions; a new series is
created explicitly.

## MVP (Phase 1+2): replace queue.sh / run_campaign.sh

Files: `tools/bigcherry/jobs/{model,db,failure,resolve,monitor,worker,service}.py`,
`tools/bigcherry/cli/jobs.py`. No stage parallelism, harvest automation, telemetry
re-measure or multi-worker scheduling in this phase.

- `JobSpec` (frozen dataclass): run/series/patch/producer/arch/devices/session/sessions_planned/
  contract_hash/model/toolchain, baseline, common_patches, producer_inputs, producer_corpus,
  production_lane, code_ref/code_policy/pinned_commit, composition_policy, priority,
  stall_timeout_s. No generic `extra_args`: new campaign dimensions become schema fields.
- `ResolvedAttempt`: commit, runner root, hip path, toolchain digest, validated-set and
  control-composition hashes, argv, env.
- `FailureKind`: same-run retryable = harness-bug, preflight-harness, attestation,
  build-config, compiler-crash, disk-pressure, oom-runtime, process-crash, stall, host-gpu;
  done-invalid = invalid-input; block-series = contract-drift, composition-drift.
  A scientific FAIL is not a FailureKind.
- Job states: queued, running, done-pass, done-fail, done-invalid, failed-harness, stalled,
  disabled, superseded, cancelled; build/measure are `attempt.phase`.
- SQLite tables: series, jobs (UNIQUE(series_id, session_no)), attempts
  (UNIQUE(job_id, attempt_no)), events (append-only triggers), settings.
- Start algorithm: claim (BEGIN IMMEDIATE) -> resolve HEAD -> validate contract/source hashes
  -> resolve toolchain -> disk guards -> runner worktree -> GPU visibility -> re-resolve HEAD
  (repeat if moved) -> record attempt -> spawn -> monitor -> classify -> truncate logs -> persist.
- Disk policy (host-local `[jobs]`): root >= 50 GiB, work >= 200 GiB free, server logs retain
  20 MiB, poll 15 s; hard breach -> SIGTERM group, SIGKILL, failed-harness/disk-pressure.
- Named offline tests (43) are listed in the source response: series slots and N+1 rejection,
  attempt retry identity, done-fail vs failed-harness, commit resolution/re-check, drift
  blocking, freeze/restart policies, disable/enable/cancel, per-job toolchain/model/inputs,
  GPU visibility (single, dual, wrong arch, duplicate), runner worktree, disk guards, log
  truncation, stall channels, daemon restart recovery, client exit independence, append-only
  events, watch resume, MTP missing rows (retryable vs invalid), no outlier auto-retry.
- Hardware acceptance before retiring the shell scripts: one gfx1100 job, one gfx1201
  alternate-ROCm job, one gfx1030 alternate-model job, one dual-gfx1100 job; forced harness
  failure + same-run retry, client disconnect, disable/re-enable, disk guard, promotion
  between sessions.

## Later phases

3. Campaign seams: external evidence sink, frozen validated composition input, producer
   preflight API (`run_preflight(ctx, phase="prebuild"|"premeasure")`), structured progress events.
4. Stage split + resources: build slots, GPU claims, host timed-measurement lock; build vs
   untimed correctness allowed, build vs timed measurement forbidden until qualified.
5. Harvest/report: verify identity, copy explicit evidence paths, `git add` explicit paths,
   `git diff --cached --check`, commit, record SHA; never stash, never `add -A`;
   review-ready only with committed evidence.
6. PVPS09 telemetry: record clocks/temp/power/utilization/VRAM; replacement only under a
   predeclared result-independent health predicate.

## Agent interface

Every command supports `--json` (stdout JSON only, diagnostics on stderr, RFC3339, stable
enums, non-zero exit only for API failure). `bigcherry jobs watch --jsonl --after <seq>
[--wake-only]`. Default wake events: attempt.failed-harness, attempt.stalled,
job.contract-drift, job.composition-drift, host.disk-hard, host.gpu-unhealthy,
harvest.failed, daemon.recovery-ambiguous, series.complete. Not wake: started, phase
changes, done-pass, done-fail, ladder results.

## Open decisions (asked in the same GPT session)

- Build on Slurm (single node: GPU gres, licenses for host-exclusive measurement, arrays,
  dependencies, requeue, sacct) or pueue (groups, dependencies, `status --json`, callbacks)
  instead of a custom daemon; keep a thin BigCherry layer for series/contract identity,
  failure classification, harvest and reports.
- Consolidate: `core/tree_activity.py` leases, `experiment/bundle.run_managed` +
  `telemetry.py`, `tuning/journal.py`, build/resource locks, and the dormant RCD01 durable
  campaign-state design (its activation trigger is now arguably met).
- Relationship to the AudiAgentic gateway (standalone, provider, or gateway-compatible events).
