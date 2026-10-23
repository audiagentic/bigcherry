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

## Execution gate review (2026-10-23)

- **WSL06-A, read-only discovery PR:** classify every legacy directory by actual path owner, manifest/provenance, use/reference graph, rebuildability, last active process/lease, exact occupied bytes and suggested disposition. Cover `runs/`, existing worktrees, vault work folders, build trees and legacy caches. Do not equate absence of references in an index with deletion safety.
- **WSL06-B, adoption PR:** adopt only objects with verifiable original identity; mark inferred provenance distinctly and require owner sign-off to release unknown assets. Check duplicate/alias paths by device/inode and realpath; avoid double counting symlinked roots. Keep a machine-readable inventory with clear unowned/protected reasons.
- **WSL06-C, owner-authorized cleanup PR:** use WSL03 quarantine mechanism and WSL04 rebuildability gate, never `rm -rf` on ambiguous items. Preserve worktree Git pointers, stored kernels, raw experiment data and release/evidence records unless the authority and mirrors have been verified. Never purge `legacy-cache`, `wt-pr8`, or any foreign directory based only on stale metadata.
- Acceptance: inventory reconciles used bytes and unknown objects, remains reproducible across repeated runs, and explicitly documents which paths were not reclaimed and why. No cleanup of physical-host data until dry runs, explicit owner approval, and verification of off-host copies when relevant.

## Validation

After adoption `bigcherry clean` in dry-run mode accounts for every byte under the work root (no unexplained directories); the owner signs off the list of items that are not ours before any is removed.
