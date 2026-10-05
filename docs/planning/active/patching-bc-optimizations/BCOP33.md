---
id: BCOP33
order: 33
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T21:47:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Standardize AMD roofline and per-op telemetry

## Description

Create one reusable AMD performance-observation contract so BigCherry optimizations can distinguish compute, memory, transfer, synchronization and launch bottlenecks. Reuse existing telemetry/profiling infrastructure and upstream instrumentation; do not create a second benchmark framework.

## Steps

1. Define a normalized per-op record: op/kernel identity, architecture/device, shape, quant/type, bytes read/written, estimated operations, kernel duration, launch/sync duration, achieved bandwidth, occupancy/utilization when trustworthy, and graph/request correlation.
2. Add request-level aggregation linking per-op evidence to PP/TG wall time, GPU idle gaps, H2D/D2H traffic, allocator peaks and useful-token work.
3. Support gfx1100 and gfx1201 with explicit unavailable/unknown fields rather than fabricated counters.
4. Integrate existing BigCherry benchmark/evidence provenance and upstream profiler outputs. Preserve raw source evidence alongside normalized fields.
5. Add derived roofline classification only when hardware peak/bandwidth inputs are measured or versioned; classify compute-bound, memory-bound, transfer-bound, launch/sync-bound or indeterminate.
6. Export the normalized observation schema to RPL01 and HIP-autotune. Consumers must not independently reinterpret raw counters into conflicting cost models.

## Validation

- Synthetic compute-heavy and bandwidth-heavy controls classify as expected.
- Known BigCherry kernel regressions/wins retain the same end-to-end direction after normalization.
- Missing counters fail to `unknown`, never zero.
- Telemetry overhead is measured and disabled/low-overhead production modes are explicit.

## Acceptance Criteria

- One schema covers HIP kernel, transfer and request-level timing evidence across gfx1100/gfx1201.
- Existing benchmark/provenance machinery is reused.
- RPL01 and HIP-autotune can consume the same records without duplicate parsers.
- Telemetry does not become a new dispatch policy owner.
