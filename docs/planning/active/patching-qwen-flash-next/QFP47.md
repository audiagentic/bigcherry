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

## 2026-10-14 implementation audit — b11552 / BCOP203

**Disposition: BLOCKED_NO_NAMED_RECLAIMABLE_OWNER.** No implementation or hardware experiment is admitted until an actual, idle, decode-only VRAM owner is identified on each short physical rank. None was identified in the inspected 1329/1331/1340/1358 paths; this does not prove none exists elsewhere. The original steps below are superseded wherever they imply a ready lease or standalone 1339 package.

### Exact code and memory ownership

- Former 1339_meta_memory_report is now **inside** engines/llamacpp/patches/1340_meta_per_device_arena (patch.py, README, SUMMARY). BIGCHERRY_META_MEM reports per-device static and arena sizes. Pinned ggml/src/ggml-backend-meta.cpp functions ggml_backend_meta_buffer_type_alloc_buffer_n (static model/KV) and ggml_backend_meta_buffer_type_alloc_buffer (compute) have different owners. BigCherry 1340 reserves a grow-only physical arena per simple device in ggml_backend_meta_reserve_graph and binds graph-shape plans in ggml_backend_meta_alloc_graph. **The reserved compute arena is graph-referenced storage, not an idle decode-only pool.** Releasing it would invalidate existing bindings without an independently proven remap/rebuild boundary.
- 1329_alloc_top_trace::bc_alloc_top_trace ranks logical graph tensors and sums sizes, **not** physical simultaneous residency. 1331_alloc_peak_live::bc_peak_snapshot tracks logical peak live bytes, explicitly **not** fragmentation or backend reserved chunks. Pair both with 1340 BIGCHERRY_META_MEM physical per-device sizes and runtime physical GPU free/peak/graph allocations. 1358_meta_split_cache_local_evict owns **CPU-side Meta split-state metadata**, not GPU VRAM. Neither logical tensor sizes nor split-state cache entries are loan capacity.
- Historical 1329 fit probes at ~240K f16 KV require about **+1.3 GiB per XTX** and **+2.0 GiB on R9700** for ub1024 versus ub512; these are old-pin deficits, not borrowable supply. Already-promoted 1340+1341 compute arenas (1020.9 MiB common versus 880.8/880.8/378.8 MiB physical) are already in the control baseline. Do not double-count them. The gfx1030 drafter and other ranks cannot lend target-rank VRAM over unavailable P2P.

### Bounded Gate 0 (read-only; terminal if no owner)

1. On the exact production pin, model/GGUF, 245760 context and f16 KV, run existing BIGCHERRY_ALLOC_TOP (1329), BIGCHERRY_ALLOC_PEAK (1331) and BIGCHERRY_META_MEM (1340) at ub512/1024/2048. Record the first failing allocation, stable physical device ID, static/KV and graph-reserved buffers, driver/fragmentation margin, runtime physical free/peak and graph UID. These diagnostics are distinct; do not subtract logical Meta totals from physical capacity.
2. Per GPU g compute deficit_g = max(0, additional_ubatch_peak_g - free_after_graph_driver_margin_g). Admit only a **named decode-only owner** with measured reclaimable_g >= deficit_g **on that same GPU**, not a weight/KV/active MTP/QSA/captured-graph buffer, and with exact release/restore method, bytes and time. Reject zero/negative/unknown values, aliases, host metadata, cross-rank aggregation and graph-bound pointers. If no owner passes, close QFP47 as **NO_ELIGIBLE_POOL**, without a new patch or hardware queue.
3. Only for a proven owner: use its existing lifetime primitive, not a new allocator. Host-check IDLE -> BORROWED_PREFILL -> RESTORING -> IDLE with positive finite sizes, monotonic generation, all-child completion, empty-event terminal handling, error/cancel/context-shift/graph-rebind/two-slot cleanup and exact content restoration before decode. If graph pointer/value identity cannot be preserved, reject.
4. Only after the ownership proof, one default-off, same-binary owner-local ablation may run on gfx1100/gfx1201 against QFP17 best ubatch/QSA chunk controls at 8K/24K/98K and 240K-fit. Require multi-request, multi-ubatch, two-slot, MTP final flush, logits/KLD/greedy, graph replay, measured per-rank VRAM/PCIe and restore cost, >=4 independent sessions and >=10 ABBA pairs/session. Promote only if ub1024+ newly fits safely, CI95-low >=3% prefill/TTFT and <=1% decode/control regression; else **NO_NET_GAIN**.

### Consolidation, host discriminator and external mechanisms

QFP47 owns only admission/disposition. QFP17 owns QSA chunk/ubatch; MSM02/03 and 1340 own Meta arenas; QFP24/1326 own pinned host inputs; MET12 owns expert residency. No new allocator, scheduler, cache, placement table or telemetry. Do not edit active Radiance/RAD03/RAD04 work.

The existing tools/tests/prefill/test_prefill_contract_model.py ScratchLease is a **model only**: borrow accepts zero/negative byte requests; restore with no outstanding events remains stuck in restoring; there is no generation identity. A separate source-shaped host discriminator ran **10/10** assertions (after correcting one fixture's equality input) covering these limits, per-rank fit, graph/live ownership and restoration. This is **not** a production GPU race or measured VRAM gain. No repository pytest, HIP build, patch application or GPU benchmark ran.

Upstream [llama.cpp #29838](https://github.com/ggml-org/llama.cpp/pull/29838) is open WIP: its inspected diff adds Meta get_alloc_size_n using per-simple-device tensor views and summed sizes, **accounting only**, not a safe release/loan primitive. Wait; do not port the unfinished API. [vLLM CuMemAllocator](https://github.com/vllm-project/vllm/blob/main/vllm/device_allocator/cumem.py) uses allocation tags, sleep/unmap and wake/remap with selective value backup; it requires a paused executor and cannot justify mid-request GGML graph mutation. The external [Strata CUDA prefill-loan report](https://www.stratat.com/zh/pulls/1581) targets a serialized single NVIDIA GPU and excludes multi-GPU/layer split; its measurements do not transfer to AMD.


## Description

QFP17 owns QSA scratch reduction and the ub512/1024/2048 experiment; QFP45 owns reference-stack batching comparison; MSM02/03 own per-device Meta arenas/mirrored attention allocation; QFP24 owns pinned input staging. Their solutions should be exhausted before introducing a new allocator. Strata's temporary prefill VRAM borrowing suggests one residual mechanism: grant extra *phase-scoped* scratch from a demonstrably evictable, non-critical cache/pool and restore it before decode. It is not safe to evict persistent model weights, f16 KV, active MTP buffers or a graph-captured pointer.

This item owns only the lifecycle/fit proof for optional reclaimable memory. It does not create a generic allocator, modify GGML tensor strides, resize a live KV buffer, or supersede QFP17's QSA query tiling. Priority is conditional on a positive actual reclaimable-byte census.

## Steps

1. Gate 0 (updated above): on b11552 use 1340's integrated former-1339 report plus 1329/1331 and QFP17 controls at ub512/1024/2048, per physical rank. Identify the actual failing allocation and a named reclaimable buffer that is idle for all of prefill and needed again for decode. If none exists, close: do not invent free VRAM by trimming live weights or caches.
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
