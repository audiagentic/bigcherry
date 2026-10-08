# Plan-item patch qualification campaigns

Plan item: RDNA/nasone plan implementation (PRBE*/PNRO*)
Status: active legacy launcher; being superseded by the jobs control plane
Owner: plan implementation loop
Question state: reusable launcher

## Question

Does a plan item's patch pass its validation.toml checks (activation,
correctness, performance, controls) on real hardware at the current pin?

## Inputs

`run_campaign.sh <patch-id> <producer|-> <arch> <device> <run-name> [args]`
with `BC_HIP_PATH` and `BC_MODEL` set in the environment.

`queue.sh <jobs-file>` accepts normal campaign rows plus:

`PROFILE <patch> <arch> <device> <prefill|decode> <run-name> [args...]`

Optional leading `MODEL=`, `HIP=` and `VIS=` tokens are supported.

`PREFLIGHT <run-name> <binary> <model> <marker-regex> [server args...]` proves a
patch marker fires on the target model (traced short completion on GPUs 0,1, under
the same host/GPU locks). A campaign row carrying `REQUIRES=<run-name>` is blocked
unless that preflight exited 0, so a patch that does not fire never spends timed
sessions. The implementation is owned here by `preflight-fire.sh` and `activation-check.sh`.

## Scheduling policy

The legacy queue now matches the evidence policy conservatively:

- shared content-addressed worktrees/build roots and a 100G ccache are reused
  across patches/sessions for the same architecture+toolchain;
- PROFILE/kernel-trace diagnostics use a two-phase handoff: CMake/Ninja
  preparation is serialized only for a shared architecture+toolchain build
  root, writes a SHA-256-bound prepared manifest, then releases that exclusive
  build lock;
- `queue.sh` completes the PROFILE preparation fan-out first, then launches the
  successfully prepared trace fan-out; same-build traces hold shared build-key
  locks, so they may overlap on different GPUs while a concurrent preparer is
  prevented from changing their executable or shared build tree;
- every GPU has an exclusive `flock`, so two jobs can never touch one card;
- PROFILE preparation and trace jobs take a shared host-activity lock;
- monolithic validation campaigns take the exclusive host-activity lock, so
  no profile, build/test campaign, or second performance campaign can overlap
  their timed evidence;
- writer intent prevents a continuous stream of new profiles from starving a
  pending performance campaign and stale intent is crash-repaired;
- shell wrappers preserve the child exit status as well as writing
  `PROFILE_EXIT=` / `CAMPAIGN_EXIT=` restart markers;
- independent queue jobs continue after a failure, but `queue.sh` exits nonzero
  if any launched job failed.

The queue order is therefore `PROFILE prepare -> PROFILE trace -> campaign`.
Multiple queue processes remain safe because host/per-GPU/build-key locks are
cross-process.

The remaining conservative boundary is the monolithic validation campaign:
build, correctness and timed measurement still live in one process, so it holds
the host-exclusive gate for longer than the timed lane strictly needs. The
current `bigcherry.patch.validation_campaign` has no prepare/execute-only seam;
RCD06/RCD07 and the jobs stage scheduler are the path to parallel deterministic
build/functional stages while retaining host-exclusive timed measurement only.

## Outputs

`$BIGCHERRY_WORK_ROOT/runs/<run-name>/` campaign workdir (resolved by
`work-root.sh`: the environment variable, else `BIGCHERRY_WORK_ROOT` in the
untracked `config/environment.local.toml` `[env]` table, else `work/`; on the
build server point it at a large scratch volume). Campaign evidence is appended
to `patches/<id>/evidence/validation.json`.

PROFILE runs also write `prepared-profile.json`, binding the exact control and
subject binaries used by the GPU phase. A binary/selector change invalidates the
handoff instead of silently profiling a changed build.

## Runtime

GPU required: yes. Real compilation: yes. Mutates canonical BigCherry state:
appends patch evidence only.

## Safety

Use the queue/launch scripts rather than invoking profilers or campaigns
ad-hoc; the lock protocol is part of evidence validity. Existing finished logs
are skipped, so queues are restartable.

## Disposition

Keep only while the current plan-qualification queue is active; jobs/RCD07 is
the long-term replacement.
