---
id: WSL06
order: 6
plan: run-workspace-lifecycle
state: pending
created-at: '2026-10-19T00:00:00+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Adopt and clear the legacy work roots on dev

## Description

One-off migration, run once the tools above exist: inventory runs/ (4,635 entries), the old build trees, worktrees and the vault work folders; wrap what is referenced by evidence or plans in manifests
(`run adopt`), harvest what is worth keeping, and list the rest with sizes for the owner. Items that belong to other sessions or are undocumented (legacy-cache, test-clone, rccl-analysis, pvps02-*,
rd04-*, rd08-*, runs-kl, framework-validation-*) are listed for the owner and not removed by default. Outcome: a work root where every directory is explained by a manifest, and free space back.

## Validation

After adoption `bigcherry clean` in dry-run mode accounts for every byte under the work root (no unexplained directories); the owner signs off the list of items that are not ours before any is removed.
