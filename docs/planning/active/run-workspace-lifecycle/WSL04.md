---
id: WSL04
order: 4
plan: run-workspace-lifecycle
state: pending
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
