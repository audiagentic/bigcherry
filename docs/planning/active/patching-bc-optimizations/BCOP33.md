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
work: S
---

# Consolidate AMD performance telemetry inputs

## Description

Action/disposition ledger for a common AMD performance-observation contract. This item does not own a new profiler or benchmark framework. Existing benchmark/telemetry sources remain authoritative; RPL01 and HIP-autotune are consumers.

## Actions

1. Inventory the fields already emitted by BigCherry benchmark/provenance tooling, HIP/rocprof tooling and relevant upstream profiling support.
2. Define only the missing normalization needed to correlate op/kernel, device/architecture, shape/type, duration, transfer/sync, memory traffic and request-level PP/TG.
3. Represent unavailable counters as `unknown`; never synthesize occupancy, FLOPs or bandwidth values.
4. Prove the normalization on one known compute-sensitive and one known bandwidth/transfer-sensitive BigCherry result before adding instrumentation.
5. If existing outputs are sufficient, close as `schema-only`. If a concrete instrumentation gap remains, add it to the existing telemetry/benchmark owner rather than creating a BCOP subsystem.
6. Record terminal disposition: `schema-only`, `existing-owner-extension`, or `rejected-no-decision-value`.

## Gate

Any retained telemetry must have measured overhead and preserve raw provenance. BCOP33 cannot own dispatch policy; RPL01/HIP-autotune may consume the normalized evidence but must not create duplicate raw-counter parsers where an adapter suffices.

## Related

RPL01; BCOP30/32; HIP-autotune; existing benchmark/evidence tooling.
