---
id: PRBE33
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:45.334008+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-STREAM-003: Dedicated scratch for concurrent branches

## Description

Qualify dedicated scratch for concurrent graph branches after AMD-STREAM-001, with safety before memory optimization.

## Steps

- Require RD39/AMD-STREAM-001 graph-concurrency prerequisite and identify shared-expert diamond/independent branches.
- Separate temporary allocation lifetimes so concurrently executing branches cannot alias; preserve sequential allocator behavior.
- Stress varying shapes and repeated concurrent graph runs against sequential control; check bit-identical outputs and memory corruption.
- Measure metadata/VRAM overhead and allocation cost only after safety passes.
- Keep optimizations scoped to proven concurrent branches and fail safely otherwise.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

Graph optimizer/concurrent buffer allocator; branch lifetime/scratch ownership; shared-expert diamond fixtures; sequential/concurrent stress and memory diagnostics.

## Validation

Concurrent vs sequential output identity; varying shapes; repeated runs; corruption/race detection; memory overhead/allocation cost; fallback.

## Effort & Risk



## Standards

Correctness prerequisite; isolate branch lifetimes; optimize memory only after safety.

## Acceptance Criteria

No scratch aliasing or output corruption under stress; memory cost is bounded; optimization is not promoted before safety proof.

## Notes

Supersedes: RD41
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd41

## Change Log

- 2026-09-09T10:55:45.334008+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:54.282206+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.276223+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.024905+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:58:24.631748+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025843_amd-stream-successors-prbe323_9820
- 2026-09-10T02:58:43.103972+00:00 (updated-by): Updated: section:ledger-events
