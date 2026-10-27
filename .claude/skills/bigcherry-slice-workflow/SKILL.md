---
name: bigcherry-slice-workflow
description: The one way to make a change in BigCherry - slice worktree, PR, auto-merge on ci-pass, finish and prune - plus the rules for waiting on CI or the build server without polling loops and for keeping clean workspace-local. Use for every change, by every session and agent.
---

# BigCherry Slice Workflow

Purpose

Every change goes through one standard path so worktrees live in one place, every PR merges, and nothing is left
behind. Background agents start cold: give them this skill by name instead of re-describing the procedure.

## 1. Start

```
cd <primary checkout>            # this PC: H:/development/projects/bigcherry ; build server: ${BC_WORK_ROOT}/wt-main
git pull --ff-only origin main
PYTHONPATH=tools python -m bigcherry slice start <type>/<topic>     # e.g. feat/rad23-n-helpers
cd worktrees/<type>/<topic>
```

- `slice start` branches from current origin/main and creates the worktree at `<checkout>/worktrees/<branch>`.
  Never create ad-hoc `../wt-*` folders. One slice = one branch = one PR.
- Hardware work on the build server: `bigcherry slice lab` (a detached slice worktree there), not a hand-made clone.
- Never `git stash` (shared multi-agent checkout); never commit to main.

## 2. Work, commit, PR

- Commit and push from the slice; `gh pr create --fill --base main` (conventional title: feat/fix/docs/ci/chore...).
- Ledger: `record_change_event` writes into the primary checkout's `docs/releases/CURRENT_RELEASE_LEDGER.ndjson`.
  Move the new line into the slice's copy and commit it with the PR, then `git restore` the primary checkout's file -
  an uncommitted ledger line there blocks `git pull`.
- Planning: the ag-planning MCP tools also write into the primary checkout; move a created/edited item file into the
  slice before committing. Plan groups need a build-/patching-/run-/tuning- prefix.

## 3. Merge

```
gh pr merge <n> --auto --squash
```

- `main` requires two checks: `conventional-title` and `ci-pass`. `ci-pass` (pr-ci.yml) passes only when every
  area job that the PR's changes trigger passed or was skipped; GitHub merges as soon as it is green.
- Do not poll `gh pr checks` in loops and do not merge by hand past a failing check. If you must wait in a script,
  use a background task with a hard timeout.
- Typical time to ci-pass: docs-only ~1.5 min; full (compile-only is the longest) ~5 min; radiance build ~2 min warm,
  up to 30 min cold (cold time varies ~1.4x with the hosted runner CPU, not with the image).

## 4. Finish and prune

```
PYTHONPATH=tools python -m bigcherry slice finish <branch>    # after the PR merged: removes worktree + branch
PYTHONPATH=tools python -m bigcherry slice prune [--apply]    # before ending a session: merged branches, orphans
```

GitHub deletes merged PR branches automatically; local branches and worktrees are yours to finish.

## 5. Waiting on long work (build server runs, CI)

- Run long commands as background tasks with a time limit; never block the conversation.
- Never wait with `pgrep -f <pattern>` inside an `ssh ... 'while pgrep -f ...'` - the pattern matches the waiting
  shell's own command line and the loop never ends. Use a PID file (`echo $! > run.pid`, `kill -0 $(cat run.pid)`)
  or an exact process name (`pgrep -x`).
- Do not run builds, test suites or scans on the build server while a GPU A/B or KLD run is in progress; close the
  browser there during measurements. The helpers' lane threads each hold a CPU core while polling (RAD20).

## Lab worktrees (WSL07)

A lab script that needs a second tree (e.g. a main baseline arm) makes a per-run worktree under
`$BC_WORK_ROOT/worktrees/<name>-$(basename $OUT)` with `git worktree add -q --detach "$WT" origin/main || exit 1` and
removes it with `trap 'git worktree remove --force "$WT" 2>/dev/null' EXIT`; `bigcherry check` (lab-worktrees) fails any
other `git worktree add` in tools/lab. `slice status --lab` lists the lab host's worktrees as dirty / unique /
removable (clean and contained in an origin branch after a fetch); `slice prune --lab [--apply]` removes only the
removable ones. Run it before ending a session that used the lab.

## 6. Clean is local

`bigcherry clean` cleans the workspace it runs in. Do not run host-wide inventories or scans of legacy trees (owner,
2026-10-24). Shared builds and caches stay under the work root (`${BC_WORK_ROOT}`; the cache, builds, runs, sources and
worktrees are fixed subfolders of it).

## Self-check before ending

- `git worktree list` shows only the primary checkout and slices with open PRs.
- `gh pr list --state open` shows only PRs queued with --auto or waiting on review.
- No background task or remote loop of yours is still running.
