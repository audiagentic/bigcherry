---
id: WSL09
order: 0
plan: run-workspace-lifecycle
state: pending
created-at: '2026-11-16T06:37:50.072802+00:00'
breadth: ''
skill: intermediate
created-by: claude
work: M
priority: P2
---

# Build cache GC: candidates and clean --builds --apply (follow-up to WSL04)

## Description

`bigcherry clean --builds` (WSL04) is a read-only inventory that protects every build and offers no apply. the work root builds directory grew to 438G (128 recipe directories, 568 build directories) with no way to prune it through the tool; on 2026-11-16 about 284G of builds older than 7 days with no live process were removed by hand.

Make build GC a first-class, safe operation: define candidacy, add `--apply`, retire the "no GC authorization" blanket.

## Steps

1. Candidate rule for a build directory: metadata sidecar present and older than N days (default 7, from the Policy like runs), no live process (exe, cwd, maps) or run manifest referencing it, not the newest K builds per recipe, not the build a current pin/main or a held run depends on; a build with no metadata older than N days is also a candidate (failed partial).
2. `clean --builds` reports candidates and protected with the reason for each; `--apply` deletes candidates only, dry-run by default, taking the same lock as builds.
3. Source cache (`sources/`) keeps its read-only inventory until separately designed.
4. Add a record of what was removed (path, size, age) under the work root.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Tests with a synthetic work root: aged unreferenced build is a candidate; a build with a live-process reference, a newest-K build or a held build is protected; --apply removes only candidates; dry run changes nothing. Run once on the real work root and compare to the manual 2026-11-16 deletion list in the 2026-11-16 deletion list kept in the task scratch folder.

## Effort & Risk

Small-medium. Risk is deleting a build still needed; mitigated by the process and manifest reference checks and the dry-run default.

## Standards



## Acceptance Criteria



## Notes

Progress (2026-11-17): implemented and merged in #1017 (tools/bigcherry/core/build_gc.py, docs/reference/build/BUILD_CACHE_GC.md): candidacy, finite policy, references from processes, run manifests, build-plan locks and installed release receipts, fail-closed inspection, quarantine-rename deletion, journal under cleanup-records. Six independent reviews; every finding has a regression test. Remaining for completion: the read-only run on the real work root protects all 145 builds because a runner lease or unclassified entry exists under the runner root and 57 builds have untrusted metadata (rad42-bar-push slices); until those are classified or retired no --apply can run on the real root. Do not weaken the guards: classify the entries (or retire the unmanaged run manifests), then run --apply once and compare with the manual 2026-11-16 deletion list.

Progress (2026-11-17): implemented and merged in #1017, #1020, #1021 (tools/bigcherry/core/build_gc.py, docs/reference/build/BUILD_CACHE_GC.md): candidacy, finite policy, references from processes, run manifests, build-plan locks and installed release receipts, fail-closed inspection, quarantine-rename deletion, journal under cleanup-records; process scan scoped to the work root owner. Six independent reviews; every finding has a regression test. Direction change (owner): old runs are not adopted wholesale. A run is either migrated to docs/evidence as small result files and deleted, or deleted; nothing stays unmanaged. The legacy-run adoption (#1026) stays as the mechanism for any run that must remain. Work root cleanup done 2026-11-17 (#1027, #1030, #1034): 5,752 entries removed, runs/ from 337 to 23 entries, evidence on main. Real-root read-only result after the cleanup: 80 builds, all protected, every one by a single unadopted flat run (runs/histogram-period-default) plus age and newest-K rules; 66 unclassified roots are non-content-addressed recipe directories (rad42-bar-push 60, rsc11-s13-switch 3, hl02-attributes 1). Remaining for completion: migrate and delete histogram-period-default, classify or retire the 64 non-content-addressed recipe directories, rerun the read-only listing and show the candidates, then run --apply once and compare with the manual 2026-11-16 deletion list. Do not weaken the guards.

## Change Log

- 2026-11-16T06:37:50.072802+00:00 (created-by): Created by claude
- 2026-11-16T19:22:30.159084+00:00 (updated-by): Updated: section:notes
- 2026-11-17T03:19:15.618493+00:00 (updated-by): Updated: section:notes
