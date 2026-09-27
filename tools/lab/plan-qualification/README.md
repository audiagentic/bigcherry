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

`PROFILE <patch> <arch> <device> <prefill|decode> <run-name> [args]`

Optional leading `MODEL=`, `HIP=` and `VIS=` tokens are supported.

## Scheduling policy

The legacy queue now matches the evidence policy conservatively:

- shared content-addressed worktrees/build roots and a 100G ccache are reused
  across patches/sessions for the same architecture+toolchain;
- PROFILE/kernel-trace diagnostics run concurrently on different GPUs;
- every GPU has an exclusive `flock`, so two jobs can never touch one card;
- PROFILE jobs take a shared host-activity lock;
- monolithic validation campaigns take the exclusive host-activity lock, so
  no profile, build/test campaign, or second performance campaign can overlap
  their timed evidence;
- writer intent prevents a continuous stream of new profiles from starving a
  pending performance campaign and stale intent is crash-repaired.

`queue.sh` therefore runs all PROFILE rows first in parallel, waits for them,
then runs campaign rows serially. Multiple queue processes remain safe because
the host/per-GPU locks are cross-process.

This is intentionally conservative: the legacy campaign contains build,
correctness and timed measurement in one process, so its exclusive window is
larger than necessary. RCD07/jobs stage scheduling is the path to parallel
build/correctness/trace stages while retaining host-exclusive timed stages.

## Outputs

`$BIGCHERRY_WORK_ROOT/runs/<run-name>/` campaign workdir (resolved by
`work-root.sh`: the environment variable, else `BIGCHERRY_WORK_ROOT` in the
untracked `config/environment.local.toml` `[env]` table, else `work/`; on the
build server point it at a large scratch volume). Campaign evidence is appended
to `patches/<id>/evidence/validation.json`.

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
