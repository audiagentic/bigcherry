---
id: BCOP30
order: 30
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-06T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Qualify the cross-capability runtime placement cost model

## Description

Action/disposition record for the 2026-10-06 system-level placement audit. RPL01 is the authoritative technical owner. The audit found mature local ownership for expert residency, auxiliary transport, RCCL/topology admission, kernel crossover and llama.cpp scheduling, but no single owner for comparing their combined compute/transfer/VRAM/synchronization cost across BigCherry's asymmetric no-P2P topology.

The resulting action is deliberately NOT to build another runtime scheduler. First prove that a read-only offline cost model can predict already-measured configuration winners and produce useful experiment recommendations.

## Actions

1. Implement RPL01 Phase A as schema/adapters over existing PHA/MET/QFP evidence; do not create another topology probe or telemetry collector.
2. Implement the deterministic decomposed Phase-B scorer for placements already expressible by current llama.cpp/BigCherry configuration.
3. Run hardware-free tests for fail-closed topology, VRAM capacity, physical bandwidth plausibility, phase isolation and deterministic ranking.
4. Hold out existing ABBA results and require >=80% pairwise winner prediction before advisory promotion.
5. If the model passes, use it only to rank experiments and emit existing flags/placement JSON. Do not mutate runtime configuration.
6. Dynamic actuation remains blocked until recommendations demonstrate >=5% end-to-end improvement in at least two materially different workload regimes and the winning transition cannot be expressed by existing configuration.
7. If the simple model cannot predict existing winners, narrow/close RPL01 rather than creating a more complex scheduler.

## Consolidation / ownership

- RPL01: cross-capability cost observations and recommendation only.
- PHA03+: topology/transport/RCCL admission remains authoritative.
- MET01: expert residency solver remains authoritative.
- MET05/1328: auxiliary 6900 transport remains authoritative.
- HIP-autotune: kernel/shape crossover remains authoritative.
- llama.cpp fit/backend scheduler: execution, allocation and inter-backend copies remain authoritative.

No second scheduler, allocator, expert solver, transport layer, dispatch table or telemetry framework is authorized by this item.

## Completion / disposition

Close BCOP30 when RPL01 reaches one of:

- **advisory-promoted:** hold-out ranking >=80%, all topology/capacity gates pass, and recommendations map to existing runtime controls;
- **narrowed:** model is useful only for one domain, with ownership moved into that existing domain plan;
- **rejected:** model fails to predict existing measured winners and adds no decision value; remove disposable prototype tooling;
- **actuation-gated:** advisory model proves >=5% benefit across two workload regimes and a separate ownership review identifies an existing execution seam to extend.

## Acceptance Criteria

- Existing evidence owners are reused rather than copied.
- Hardware-free discriminator exists before new hardware campaign work.
- No runtime scheduler is introduced during observation/scoring qualification.
- Every recommendation exposes its cost decomposition and provenance.
- Failed prototype code is removed rather than becoming another permanent planning surface.
