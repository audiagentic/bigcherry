---
id: WSL07
order: 0
plan: run-workspace-lifecycle
state: pending
created-at: '2026-10-27T02:33:31.124111+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: S
---

# Lab worktree hygiene: lint ad-hoc worktrees in lab scripts; slice status/prune see lab-host worktrees

## Description

Ad-hoc worktrees accumulated on the lab host: lab scripts created their own (rad20-burst-ab.sh made a persistent main-baseline next to the primary checkout), sessions reused a mislabelled worktree for other branches, and `slice status`/`prune` only look at the local checkout, so stale lab-host worktrees were invisible (three of mine plus a stale entry from another session, 2026-10-27).

## Steps

1. Lint (part of `bigcherry check`): a tools/lab script that runs `git worktree add` must target $BC_WORK_ROOT/worktrees/ and remove it on exit (trap); flag any other worktree creation.
2. `bigcherry slice status --lab` / `slice prune --lab`: list (and with --apply remove) the configured lab host's worktrees, marking clean ones whose commits are on origin as removable; never remove one with local changes or unique commits.
3. Document the per-run baseline worktree pattern (rad20-burst-ab.sh) in the slice workflow skill.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

tools/bigcherry/cli (slice), lint/check module, tests, .claude/skills/bigcherry-slice-workflow

## Validation

Unit tests for the lint (positive/negative script fixtures) and for the lab worktree classifier (clean+on-origin, dirty, unique commits); dry run against the lab host lists the expected set.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Follows PA47 (slice tooling). Owner rules: worktrees via bigcherry slice only; clean is workspace-local (the --lab view covers this workspace's lab host, not a host-wide sweep).

## Change Log

- 2026-10-27T02:33:31.124111+00:00 (created-by): Created by agent
