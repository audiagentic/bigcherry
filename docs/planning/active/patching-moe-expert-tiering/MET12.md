---
id: MET12
order: 12
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-08T22:20:00+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Conditional expert-weight prefetch: overlap measured host-resident transfers with independent GPU work

## Description

Strata-style CPU expert staging is NOT assumed to help the resident three-card Flash-Next production layout. MET07 owns expert cache/profile qualification (1337/1338; prior hardware regressions must be respected); MET10 owns range-aware resident-plus-host-tail graph correctness; MET05 owns auxiliary ROCm transport; MET11 owns load-time placement; MET01/RPL01 own residency decisions. This new item owns only **asynchronous predicted hot-weight transfer and event-protected staging** if and when selected expert H2D copies are a demonstrated prefill critical-path cost on an over-VRAM lane. No duplicate cache policy, 1281 range logic, or Meta scheduler.

Fundamental limit: an exact same-layer expert list is unavailable until that layer's attention/router executes. The only traffic that can be sent before that dependency is speculative, immutable prefetched weight data (e.g. profile-hot slabs), not authoritative routed outputs. Incorrect predictions must cost only time/bandwidth, never model accuracy.

## Steps

1. Gate 0: use 1336 host-expert copy traces and rocprofv3 to measure per-chunk exact selected experts, unique slab bytes, pinned CPU copy, H2D enqueue/device copy, wait before expert MMQ, PCIe bandwidth, cache hits, and overlappable attention/other device interval. Separate prefill ub512/1024/2048 (large batches select many experts), short/deep prompts, and gfx1100/gfx1201. Mark production all-resident as a negative control; do not implement if uncovered H2D time is negligible.
2. Build a deterministic read-only prefetch chooser over an existing MET01/1338 profile; cap hot set per layer and total staging VRAM. Track predicted, delivered, exact selected, required misses, eviction/churn and all bytes including useless prefetch. Keep profile inferences optional, bounded and per-model; no token-content approximation.
3. Extend the existing 1336 upload owner with a pinned host/device staging ring (not a second scheduler). Enqueue only immutable slabs before their consuming MoE op. Each slot has request/chunk/layer/generation, rank/device owner, stream-ready and copy-done events. Reuse only after every consumer completes; fall back to current synchronous path on no event query, ring full, absent pins or graph-capture mismatch.
4. When authoritative router IDs become available, ensure every selected weight is resident or its exact miss has completed. No skipped experts, altered top-k, approximate weights or reordered F32 accumulation. Do not pre-stage host tail on all Meta ranks: only its designated owner gets it.
5. Test predicted-hit, all-miss, all-hot, empty routing, skew/Zipf, large ubatch high-coverage, rank cancellation, ring overflow, graph replay, context shift, two sessions, and restored decode placement. Record activation/fallback counters.
6. Hardware ABBA: default-off synchronous 1336/MET10 versus default-off-capable prefetch on, same full model, quant, placement, host residency, ubatch, context and profile. Measure prefill, exposed upload stalls, GPU copy/compute concurrency, peak VRAM, decode, acceptance, greedy IDs and CPU-f32 probes. Repeat on one genuine over-VRAM lane only; do not equate with Strata's published model/hardware.

## Detailed Solution & Technical Design

- Pre-stage can overlap *preceding independent compute or attention*, not a same-layer MoE computation whose router output is a prerequisite.
- Prefer a bounded transfer ahead of time for profile-hot candidates; after routing, issue exact-demand transfers for misses and gate the selected expert compute behind their completion event.
- Use MET10's id_base/global-id-to-local-id semantics for host tails and MET05 transport routing; no parallel reimplementation of either.
- Reserve GPU staging bytes explicitly; prefill scratch leasing in QFP47 must not evict a live slot. The 6900 XT PCH attachment is a separate negative/low-bandwidth lane, not an implicit fast prefetch worker.
- Disable prediction for requests without a matching validated profile, anomalous placement mapping, unknown quant block alignment, or adverse copy saturation.

## Files

Design owner: MET12. Future implementation only in patches/1336_sched_copy_callback and narrow Meta/backend helpers selected by MET10/MET05, plus a separate opt-in patch package if a complete independent feature is necessary. Synthetic miss/consumer oracle: tools/tests/prefill/test_prefill_contract_model.py. Hardware tooling: tools/lab/flash-next/moe-copy-ab.sh / queue-moe-copy.sh.

## Validation

Gate 0 must show measurable exposed expert-upload stalls, not merely many host-resident bytes. Synthetic tests prove only that misses cannot be used early; production tests must prove exact slab contents, event ordering, bounded memory, and unchanged model outputs. A/B and hardware timeline must show both useful copy overlap and lower prompt critical-path wall; otherwise reject without a second cache implementation.

## Acceptance Criteria

- The default-off path is unchanged and production fully resident workloads do not pay overhead.
- Every required expert waits on an actual completed staged copy, with correct owner/range, despite arbitrary predicted misses and cancellation.
- No extra collective or replicated upload across Meta ranks.
- Exposed H2D wait falls and end-to-end prefill improves repeatably in a declared over-VRAM lane, without fidelity/TTFT/decode regression. Otherwise close as negative evidence.

## Change Log

- 2026-10-08: Created as conditional transfer-overlap gap; separated from MET07 cache, MET10 graph and MET05 transport.
