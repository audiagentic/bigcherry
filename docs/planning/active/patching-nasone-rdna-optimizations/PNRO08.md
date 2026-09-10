---
id: PNRO08
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:46.423935+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# GPU-resident LRU cache for host-offloaded MoE expert weights

## Description

Evaluate a bounded GPU-resident LRU cache for host-offloaded MoE expert weights, decode-only and explicitly separate from fully resident models.

## Steps

- Freeze 6ed7fb04... and enumerate CLI/context, routing callback, cache ownership, device tables, CPU skip table, GPU cached mul_mat_id chain, upload worker and decode-boundary publication.
- Prove exact decomposition: CPU skips cached rows, GPU remaps uncached IDs to zero slot, and summed outputs equal stock for every routing ID.
- Start with a deterministic static mapping and synthetic routing trace; separate LRU policy from graph correctness before asynchronous replacement.
- Model mapping generations and publish only after full expert-slice upload; test eviction, active requests, teardown, repeated context creation and callback lifetime.
- Instrument hits/misses/inserts/evictions/bytes avoided; sweep slots/throttle across no-locality and phase-changing traces; measure host/PCIe traffic, decode TPS, VRAM and warmup.
- Keep prefill and fully GPU-resident models as strict non-selection controls; promote only when net gain exceeds maintenance and VRAM cost.

## Detailed Solution & Technical Design

Device companion tensors hold K cached slices plus an all-zero dummy; host/device ID maps must remain coherent across asynchronous worker publication. This is a cross-backend graph decomposition with bounded resources and decode-only scope.

## Code Samples & Guidance



## Files

Future package/patch after implementation review; common args; ggml CPU mul_mat_id; llama context/graph; cache support; async upload worker; generation/mapping diagnostics; targeted correctness/perf tests.

## Validation

Exact stock-vs-cache outputs for every ID; deterministic and dynamic LRU stress; no torn/stale slot publication; eviction/teardown/thread lifetime; prefill and resident non-selection; host traffic/VRAM cost; warmed and cold decode performance.

## Effort & Risk



## Standards

Correctness first; bounded resources; source SHA; no global callback leaks; decode-only until separately expanded.

## Acceptance Criteria

Cache path is selected only for host-offloaded decode; exact outputs and safe publication hold under stress; measured traffic reduction and end-to-end gain outweigh maintenance/VRAM budget; otherwise reject without using third-party hit-rate claims.

## Notes

Supersedes: NRO09
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro09

## Change Log

- 2026-09-09T10:52:46.423935+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:09:08.508331+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.086059+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.732296+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:42:53.658407+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024304_three-nasone-successors-now-pr_2691
- 2026-09-10T02:43:04.884669+00:00 (updated-by): Updated: section:ledger-events
