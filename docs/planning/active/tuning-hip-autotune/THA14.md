---
id: THA14
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:49:36.383115+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# Resolve bindings at graph-node lifetime -- the dispatch lookup may not need to run at all

## Description

Resolve bindings at graph-node lifetime where invariance is proven, eliminating repeated native selection/signature/cache lookup without unsafe tensor-pointer memoization.

## Steps

1. Confirm graph-node invariance across type, extents, strides, fusion semantics, device, replay generation, and cache reload boundaries.
2. Store resolved Binding in graph/node execution metadata with explicit invalidation.
3. Prototype one family and compare against the current per-dispatch path.
4. Consume canonical prepared facts for native selection and signature construction to prevent selector/key drift.
5. Keep tensor-pointer caching out unless invalidation covers shape/type, buffer identity/usage, allocation size, and view_src.
6. Widen only after correctness, graph capture/replay, invalidation, and E2E evidence pass.

## Detailed Solution & Technical Design

Capability owner: tuning

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

graph execution metadata; binding resolver/native selector; invalidation tests; profiling and E2E evidence

## Validation

Graph reuse and replay-generation invalidation tests; binding equality and fallback tests; graph/non-graph correctness; no stale binding across buffer/view/device changes; host dispatch overhead and E2E comparison against current path.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

A graph-invariant binding is resolved once with explicit invalidation, never reused across changed semantics, and demonstrates correctness plus measurable host-path benefit before widening. Unsafe pointer-lifetime memoization is not admissible.

## Notes

Supersedes: HI164
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi164

Supersedes: HI164
Inherited semantic scope: preserve graph-lifetime invariance conditions, explicit invalidation, canonical prepared facts, and anti-pointer-cache constraints.
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-09T10:49:36.383115+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:05:45.501143+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.892896+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.426902+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:24:09.113698+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_022438_the-next-five-high-risk-tuning_6580
- 2026-09-10T02:24:38.727991+00:00 (updated-by): Updated: section:ledger-events
