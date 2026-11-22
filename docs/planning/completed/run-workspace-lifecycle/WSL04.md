---
id: WSL04
order: 4
plan: run-workspace-lifecycle
state: superseded
created-at: '2026-10-19T00:00:00+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Build and source cache GC keyed by plan id

## Description

Builds (554 directories) and materialised sources (618) are not run directories, but they are the largest reclaimable pool and are managed the same way: keep every plan id referenced by a pin release
record, a recipe, a live lane or a harvested run's manifest, plus the newest K per lane; remove the rest, oldest first, with the same dry-run-first command and cleanup record. A build in use (lease or
running process) is never removed. ccache and tune caches get a size cap with least-recently-used eviction.

## Steps

1. Reference index: which plan ids are named where (releases/, recipes, manifests, evidence); build and source age and size listing (`bigcherry clean --builds`).
2. Eviction policy and tests; ccache cap.
3. Rebuildability check: a removed build is rebuilt by its lane from the recorded plan id (verified once).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

CONSOLIDATION 2026-11-22 NATIVE-OWNERS: The older build/source/cache GC plan and the implemented build-GC follow-up should be one policy and one cleanup command. Canonical remaining owner: WSL09. WSL09 expressly says it is the follow-up to WSL04 and shipped the one build_gc implementation (#1017/#1020/#1021); WSL04 is the older conceptual duplicate and has no independent implemented cleanup API. Exact remaining acceptance and stop/fallback requirements are in the active receiver. This lifecycle state retires duplicate planning ownership only; original source/evidence/negative results are retained and not declared complete.

## Reviews

- RV4509

## Change Log

- 2026-11-22T10:37:02.681193+00:00 (updated-by): Updated: section:notes
- 2026-11-22T10:44:13.335572+00:00 (state-transition): State: pending → superseded
