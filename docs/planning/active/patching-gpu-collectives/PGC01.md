---
id: PGC01
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-09-09T10:47:49.461438+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Device-3 / PCI-atomics-incapable topology: safe AND performant fallback investigation (go/no-go against META)

## Description

Investigate safe and performant fallbacks for device-3/PCI-atomics-incapable topologies. A candidate ships only if it beats META/layer-split on the exact topology; a measured negative result is valid closure.

## Steps

- Probe RCCL 2.27.7 structurally and then through GP07's hardened qualification, measuring real latency/bandwidth rather than treating structural admission as success.
- Prototype the hierarchical safe-subset bridge at GP10 base level: measure {0,3} as a single-GPU target separately from {0,1,3}/{0,1,2,3} multi-GPU subsets.
- Measure general pinned-host N-way only as the last-resort candidate, with correctness and memory accounting.
- Promote to a BigCherry patch only after base-level evidence shows it can beat META/layer-split; qualify exact topology and preserve native/META fallbacks.
- If no candidate clears the bar, close each topology with real numbers and retain META/layer-split as authoritative.

## Detailed Solution & Technical Design

Execution order is RCCL probe, hierarchical bridge, pinned-host fallback; expected performance priority is bridge first, pinned-host last. Device 3 must never join an RCCL communicator when the topology is unsafe. For {0,3}, bridge to one GPU; do not assume the multi-GPU subset design transfers.

## Code Samples & Guidance



## Files

GP07 RCCL qualification; GP10 base harness; topology-specific bridge/pinned-host prototypes; META/layer-split controls; per-topology campaign evidence and negative disposition.

## Validation

Structural metadata and real probes; {0,3}/{0,1,3}/{0,1,2,3}; correctness; latency/bandwidth; memory; candidate vs META/layer-split; exact topology/revision provenance.

## Effort & Risk



## Standards

Fail closed on AtomicOps admission; measure before patching; no slower-but-safe production fallback; preserve negative evidence.

## Acceptance Criteria

A fallback is promotable only when real hardware numbers beat META/layer-split for its exact topology with correctness intact. Otherwise record a topology-specific negative result and retain META/layer-split; safety alone is insufficient.

## Notes

Supersedes: GP09
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-gpu-collectives-gp09

## Change Log

- 2026-09-09T10:47:49.461438+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:03:48.855653+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.767507+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.226021+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:37:38.846667+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023824_the-gpu-collective-successors_5773
- 2026-09-10T02:38:24.253696+00:00 (updated-by): Updated: section:ledger-events
