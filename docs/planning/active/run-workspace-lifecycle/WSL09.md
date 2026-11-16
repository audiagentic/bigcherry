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

## Change Log

- 2026-11-16T06:37:50.072802+00:00 (created-by): Created by claude
