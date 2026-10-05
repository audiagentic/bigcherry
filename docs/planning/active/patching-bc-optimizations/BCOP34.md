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
work: S
---

# Gate no-P2P MoE expert parallelism

## Description

Action/disposition ledger for deciding whether expert-parallel execution is viable on BigCherry's no-P2P topology. MET01 owns residency, MET02 sparse/grouped expert execution, MET05 auxiliary transport, PHA03 collective/topology admission and RPL01 cross-capability placement. BCOP34 owns none of those mechanisms.

## Actions

1. Use existing routing evidence plus measured transport costs to calculate the minimum activation/gather bytes and synchronization required for candidate expert partitioning.
2. Compare that lower bound with the compute time that could actually be removed. If the bound cannot support a >=5% end-to-end opportunity, reject without implementing transport.
3. If viable, use a trace/replay mock before GPU code to test skew, balance, overlap and critical path.
4. Route any proven implementation to MET01/02/05 or the existing backend-copy seam according to ownership; do not add a second router/cache/placement store.
5. Record terminal disposition: `residency-sufficient`, `transport-prototype-justified`, `wait-upstream`, or `rejected-communication`.

## Gate

Promotion requires exact work/byte accounting, correctness parity and >=5% end-to-end improvement at equal model/VRAM budget on the real no-P2P topology. Failed prototypes are removed.

## Related

MET01; MET02; MET05; PHA03; RPL01; BCOP32.
