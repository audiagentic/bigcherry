---
id: QFP47
order: 47
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-08T22:20:00+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Prefill-only scratch lease: reclaim eligible transient VRAM to qualify larger physical microbatches

## Description

QFP17 owns QSA scratch reduction and the ub512/1024/2048 experiment; QFP45 owns reference-stack batching comparison; MSM02/03 own per-device Meta arenas/mirrored attention allocation; QFP24 owns pinned input staging. Their solutions should be exhausted before introducing a new allocator. Strata's temporary prefill VRAM borrowing suggests one residual mechanism: grant extra *phase-scoped* scratch from a demonstrably evictable, non-critical cache/pool and restore it before decode. It is not safe to evict persistent model weights, f16 KV, active MTP buffers or a graph-captured pointer.

This item owns only the lifecycle/fit proof for optional reclaimable memory. It does not create a generic allocator, modify GGML tensor strides, resize a live KV buffer, or supersede QFP17's QSA query tiling. Priority is conditional on a positive actual reclaimable-byte census.

## Steps

1. Gate 0: on b11474 production 245760 ctx f16 KV, run 1339/1329 memory reports and QFP17 QSA chunking at ub512/1024/2048, per rank. Identify the actual failing allocation and a named reclaimable buffer that is idle for all of prefill and needed again for decode. If none exists, close: do not invent free VRAM by trimming live weights or caches.
2. Use a static plan-only fit simulation for each rank: available = budget - nonreclaimable allocations - peak prefill scratch - driver/graph safety margin. Borrowable bytes must be positive on **every** rank that needs the larger ubatch. Record reclaim/restore cost and context-dependent fit, not only t/s.
3. Define a bounded lease protocol for a specifically named evictable owner: IDLE -> BORROWED_PREFILL -> RESTORING -> IDLE; restore on successful flush, error, cancellation, context shift, graph-cache rebind and request teardown. Hold events until all child devices stop accessing leased storage. Do not reuse a slot on generation mismatch.
4. Prefer prefill/decode graph-build boundary reallocation or a pre-reserved disjoint scratch pool; never mutate an address referenced by an in-flight GPU graph. If the same pointer cannot be safely restored, use the existing allocator/fallback unchanged.
5. Provide a same-binary default-off ablation: QFP17 chunk controls unchanged, ub512/1024/2048 isolated, memory high-water per rank, events, graph replay, host overhead, prefill, TTFT, decode and target greedy identity. Check repeated requests, two slots and final MTP flush.
6. Coordinate only the eventual integration with MET12's pinned expert slots. Independent plan and fit/mock work can run in parallel with QFP42, PGC15 and QFP41.

## Detailed Solution & Technical Design

Do not add a pool whose only benefit is moving prefill allocations: borrowing is eligible ONLY after a named owner proves its previous data can be reconstructed exactly, is not needed by current compute and can be restored before decode. GPU idle time is a separate measurement; a larger batch that triggers PCIe paging or RCCL skew is a regression, not a successful fit.

Mock contract at tools/tests/prefill/test_prefill_contract_model.py covers negative decode-phase borrow, oversize, concurrent lease denial and multi-device restoration. Backend mechanics and memory safety still require actual HIP/ggml tests.

## Files

Future narrow owner-specific cache/pool patch only if Gate 0 passes; current read-only diagnostics 1329/1339 and QFP17 tools; no production patch added here.

## Validation

A real allocator audit (fragmentation, per-device residency, capture ownership), hardware oom/fit and restoration tests at full ctx 245760, exact greedy/f32 probes, memory high-water plus safety reserve, multi-request lifecycle and default-off equivalence. Mock tests are ordering examples, not VRAM fit proof.

## Acceptance Criteria

- At least one safe named pool supplies enough *measured* reclaimable bytes for a larger ubatch; no live Meta/graph/weight/KV pointer is repurposed.
- Every successful or failed prefill transition restores the declared decode state and buffer contents before any decode consumer.
- Default-off and unsupported backend paths are byte/behavior identical to QFP17 controls.
- Repeatable net prefill gain at the chosen context/quant and no material decode or TTFT regression; otherwise stop after Gate 0.

## Change Log

- 2026-10-08: Created as gated Strata-style transient-memory idea, subordinate to QFP17's already measured QSA batch solution.
