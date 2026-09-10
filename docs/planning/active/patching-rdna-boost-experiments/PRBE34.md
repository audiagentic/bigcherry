---
id: PRBE34
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:49.488869+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# AMD-STREAM-004: Overlap MoE shared expert on auxiliary stream

## Description

Evaluate overlapping MoE shared-expert compute on an auxiliary stream after per-stream handles and branch scratch prerequisites.

## Steps

- Require PRBE32 and PRBE33 identities; verify gated shared-expert diamond has independent branches and safe join.
- Implement scheduling only for exact Qwen MoE decode patterns; preserve graph-opt off and unsupported fallbacks.
- Validate byte-identical output, join ordering, branch overlap and no races with profiler/trace evidence.
- Compare tg128/tg512 at context 0/64K with graph opt on/off; use MoE without shared expert and dense controls.
- Promote only when a workload predicate guarantees independence and E2E gain exceeds 1%; retain negative evidence otherwise.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

Graph optimizer shared-expert scheduler; PRBE32 handles; PRBE33 scratch; Qwen MoE gated/shared-expert fixtures; profiler/trace and TG campaign.

## Validation

Decode Q8/Q4; prefill controls; graph on/off; context 0/64K; byte parity; profiler overlap/join; concurrency/utilization/power; E2E TG.

## Effort & Risk



## Standards

Dependency-aware scheduling; workload predicate; join correctness; no unconditional overlap.

## Acceptance Criteria

Exact eligible workload overlaps safely with byte-identical output and >1% E2E TG gain; all other paths retain existing behavior; prerequisites are explicit.

## Notes

Supersedes: RD42
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd42

## Change Log

- 2026-09-09T10:55:49.488869+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:58.630166+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.280175+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.031786+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:58:31.227383+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025843_amd-stream-successors-prbe323_9820
- 2026-09-10T02:58:43.116897+00:00 (updated-by): Updated: section:ledger-events
