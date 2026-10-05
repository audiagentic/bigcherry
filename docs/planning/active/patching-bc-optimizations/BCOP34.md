---
id: BCOP34
order: 34
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T21:48:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Evaluate no-P2P MoE expert parallelism

## Description

Determine whether routed MoE work can be profitably partitioned across BigCherry's multi-GPU topology when direct GPU peer access is unavailable. The design must account for host-mediated activation/routing traffic and must not duplicate MET01 residency policy, MET02 sparse expert execution, MET05 auxiliary transport or PHA03 collective admission.

## Steps

1. Establish a communication lower bound from actual routed token counts, activation width/type, expert fanout and required gather/reduction semantics.
2. Compare candidate decompositions: whole-layer placement, expert-group ownership, auxiliary expert service, and host-mediated dispatch/gather. Reuse MET01 placement data and MET02 grouped/range semantics.
3. Measure host-device and device-host transfer latency/bandwidth for the production topology; include root-complex contention and concurrent transfers.
4. Implement a trace/replay simulator before runtime transport changes. Given captured routing, predict per-device expert work, bytes moved, overlap and critical path.
5. Reject expert parallelism where communication/synchronization exceeds saved compute. Prefer persistent/local expert residency where MET01 already captures the win.
6. If a viable region exists, prototype the smallest transport path through existing scheduler/backend-copy or MET05-owned auxiliary seams. Do not add a second expert router.
7. Feed candidate cost/result evidence to RPL01 for system-level placement selection.

## Validation

- Routing traces from code, prose/chat, retrieval/long-context and MTP workloads.
- gfx1100/gfx1201 production topology with P2P explicitly disabled/unavailable.
- Correctness parity, exact work/byte accounting and ABBA end-to-end PP/TG.
- Include skewed and near-uniform expert distributions.

## Acceptance Criteria

- A communication lower-bound model predicts observed transport cost within a useful error bound before promotion.
- Any promoted no-P2P expert-parallel path improves end-to-end throughput >=5% at equal model/VRAM budget.
- MET01 remains residency owner; MET02 remains sparse expert execution owner; MET05 remains auxiliary transport owner.
- No duplicate routing, cache, placement or collective-admission subsystem is introduced.
