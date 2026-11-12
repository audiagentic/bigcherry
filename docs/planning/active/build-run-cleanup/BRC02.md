---
id: BRC02
order: 0
plan: build-run-cleanup
state: pending
created-at: '2026-11-12T22:25:58.279269+00:00'
breadth: ''
skill: intermediate
created-by: bigcherry-1
priority: P2
work: M
---

# Migrate slice worktree management onto audiagentic worktree commands

## Description

bigcherry carries its own slice tooling (tools/bigcherry/cli/slice.py, plus worktree_record.py from PR #928). audiagentic already ships generic worktree management (DW01: audiagentic worktree start|list|prune|sync|push|ship|merge|finish|tidy, central owner records, merge-proof eligibility, quarantine before removal). The user decided on 2026-11-13 to migrate bigcherry onto it, with the capability kept generic in audiagentic and nothing bigcherry-specific in its core. PR #928 (WORKTREE.md plus registry) is a stopgap and is removed at cutover.

## Steps

1. Wait for audiagentic's gap item (required purpose, evidence-pending hold in finish, hook for unmanaged trees, append-only audit trail, archive tags for experiment branches, adopt command) and its release version; record the item id here.
2. Move the shared runtime to a release that carries the worktree command set and the gap work (needs 0.97 or later; bigcherry is on 0.88.1).
3. Settings: worktrees.root under the primary (worktrees/<branch>), branch prefix, protected trees, pre-push checks.
4. Replace slice start/finish/status/prune with a thin bigcherry wrapper that calls audiagentic start/merge/finish and keeps only the bigcherry steps (ledger and planning moves into the slice, lab slices).
5. Adopt or finish the existing stray trees (shown as unmanaged by reconcile) once.
6. Remove tools/bigcherry/cli/worktree_record.py, the WORKTREE.md and registry logic from #928, and the superseded slice code, migrating every caller in the same change (no legacy path).
7. Update the bigcherry-slice-workflow skill and AGENTS.md.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

tools/bigcherry/cli/slice.py, tools/bigcherry/cli/worktree_record.py, tools/bigcherry/cli/main.py, tools/bigcherry/release/patch_promote.py, tools/tests/cli/test_slice_commands.py, .claude/skills/bigcherry-slice-workflow/SKILL.md

## Validation

After cutover every worktree is created through the audiagentic command with a recorded purpose; reconcile reports no unmanaged worktrees; slice-related tests pass against the wrapper; no code path creates a worktree any other way; the slice skill matches the commands.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Blocked on audiagentic's gap plan item (their user must approve it) and a runtime release at or above 0.97 that carries it. Do not start the cutover before then.

## Change Log

- 2026-11-12T22:25:58.279269+00:00 (created-by): Created by bigcherry-1
