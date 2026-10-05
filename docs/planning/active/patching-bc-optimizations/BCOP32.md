---
id: BCOP32
order: 32
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T21:46:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Qualify topology-aware heterogeneous multi-GPU scheduling

## Description

Develop the evidence and candidate model required to place work across unequal RDNA devices without assuming symmetric compute, VRAM, PCIe bandwidth or peer access. Production target includes 2x gfx1100 XTX plus gfx1201 R9700, with optional gfx1100 6900XT on the chipset path. RPL01 is the global decision owner; this item owns heterogeneous scheduling qualification and candidate mechanisms only.

## Steps

1. Consume canonical topology/admission evidence from PHA03 rather than rediscovering PCIe topology. Record per-device compute class, usable VRAM, host-device bandwidth, pairwise transfer capability, root-complex path and P2P/RCCL admission.
2. Measure per-device service rates for dense, attention, MoE and MTP workloads. Do not use VRAM size or nominal FLOPS as a proxy for execution speed.
3. Evaluate existing layer split/tensor split/scheduler-copy mechanisms first. Establish when simple weighted layer placement is sufficient.
4. Model heterogeneous critical path: per-device compute, host copies, cross-device/collective cost, synchronization and idle bubbles. Feed observations/candidates to RPL01.
5. Prototype only the smallest missing scheduling primitive if existing llama.cpp configuration cannot express a measured winning placement. Keep scheduler-core changes upstreamable and policy-free.
6. Test failure/fallback semantics: unavailable auxiliary GPU, changed device order, insufficient VRAM, RCCL rejection and no-P2P topology must fail closed or select an already-qualified fallback.

## Validation

- Single-device controls for each GPU plus 2xXTX, 2xXTX+R9700 and optional 6900 topology.
- PP512/2048 and TG128/512, MTP on/off, representative dense and MoE models.
- Report device utilization, idle gaps, transfer/sync time, peak VRAM and end-to-end throughput.
- Compare predicted critical path with observed wall time before any adaptive runtime policy.

## Acceptance Criteria

- Heterogeneous placements are derived from measured device/topology costs, not equal-layer heuristics alone.
- RPL01 remains the sole cross-capability placement decision owner.
- PHA03 remains the topology/RCCL admission owner.
- New scheduler machinery is permitted only when a >=5% measured opportunity cannot be expressed by existing controls.
- No-P2P and chipset-routed devices are explicitly represented rather than treated as symmetric peers.
