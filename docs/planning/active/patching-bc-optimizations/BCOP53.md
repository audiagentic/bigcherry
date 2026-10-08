---
id: BCOP53
order: 53
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-08T09:10:31+11:00'
created-by: agent
priority: P2
---

# PRBE62 Vulkan graphics-queue qualification disposition

## Discovery / change

PRBE62's proposed AMD-proprietary-only per-MoE-graph dual-queue patch is premature. Pinned b11474 already exposes a process-scoped graphics-family opt-in; it also changes async transfer-queue preference. The old `eAmdProprietary` predicate misses Linux AMDVLK's `eAmdOpenSource` driver ID, and the suggested patch number 1263 collides with PRBE41. Upstream #20599 deliberately made graphics queue opt-in after regressions. External R9700/AMDVLK evidence shows MoE +4.7% but dense -8.1% at fixed rm_kq=1; **not BigCherry results**.

## Authoritative owners / already-acting work

- PRBE62: only bounded driver/queue qualification and terminal disposition.
- Upstream Vulkan backend: queue family, pool, transfer and synchronization.
- PRBE61: rm_kq geometry; PRBE55: MMVQ routing. Keep fixed in this A/B.
- Current MTP, QFP37, MET05 and MoE-cache owners retain their active paths; this ledger does not change them.
- No existing PRBE62 implementation or hardware campaign was found. No duplicate selector/allocator/scheduler is authorized.

## Unresolved action and terminal gate

1. Gate 0: identify a real AMDVLK/proprietary driver, separate queue families and an actual flag-induced compute-family change; otherwise close `no applicable lane`.
2. Gate 1: same-pin, same-quant interleaved process-level flag A/B, MoE and dense/pp controls, route and async-transfer attribution, synchronization and multi-request correctness.
3. Close `no change` if CI95-low MoE E2E <3%, route unchanged or any correctness failure. Close `process opt-in` if >=3% MoE passes but dense/prefill suffers; use existing upstream launch flag, not a global default. Only propose a separate future dual-queue implementation if a mixed-model single process demonstrably needs it and >=5% measured mixed E2E opportunity remains after transfer-path isolation.

## Dependencies / blockers

Driver availability, queue-family proof and transfer safety (upstream #25195; proposed #25196 is unmerged). External benchmark was on PCIe 5.0 x16 R9700, not BigCherry's heterogeneous/no-P2P setup. AMDVLK is discontinued; record ICD/version. No build, hardware test or benchmark ran in this audit; pinned-source static checks passed. Full technical design and controls are in PRBE62.
