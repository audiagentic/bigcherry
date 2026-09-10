---
id: PRBE32
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:40.618469+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-STREAM-002: Per-(device,stream) BLAS handles

## Description

Qualify per-(device,stream) BLAS handles after the AMD-STREAM-001 prerequisite, preserving ordinary single-stream behavior.

## Steps

- Require and identify the RD39/AMD-STREAM-001 context prerequisite.
- Implement independent handle/stream/workspace state per device+stream in the HIP BLAS compatibility context.
- Exercise two auxiliary streams with repeated concurrent GEMMs and compare deterministic outputs/race/error behavior to single-stream control.
- Measure single-stream overhead and actual overlap; do not promote if ordinary path slows materially.
- Record handle lifetime/destruction and fail-safe behavior under repeated context creation.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

HIP/cuBLAS compatibility backend context; per-device/stream handle registry; stream/workspace lifetime tests; concurrent GEMM fixtures; overlap/overhead evidence.

## Validation

Single-stream control; 1/2 aux streams; thousands of iterations; output parity; race/error detection; handle lifecycle; overlap and ordinary overhead.

## Effort & Risk



## Standards

Concurrency correctness before performance; explicit handle ownership; no ordinary-path penalty.

## Acceptance Criteria

Concurrent GEMMs are correct and independent with no leaks/races; any promotion requires useful overlap without material single-stream regression; otherwise retain as prerequisite evidence.

## Notes

Supersedes: RD40
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd40

## Change Log

- 2026-09-09T10:55:40.618469+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:50.778200+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.272314+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.017739+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:58:17.807856+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025843_amd-stream-successors-prbe323_9820
- 2026-09-10T02:58:43.089488+00:00 (updated-by): Updated: section:ledger-events
