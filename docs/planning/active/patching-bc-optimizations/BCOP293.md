---
id: BCOP293
order: 293
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-18T00:07:32Z'
created-by: agent
priority: P2
work: S
---

# QFP17: native HIP lightning-indexer two-head LDS-pass gate

**Discovery:** Pinned llama.cpp b11552 already contains native #29901; BigCherry 1335 backport is superseded. Upstream #30080 (merged 2026-10-07) introduced two-head-per-pass Q staging for MUSA<220, while HIP still stages four heads. In the native 64-key/8-token/8-wave32 tile, source-array LDS is 33,152 B (four heads) versus 24,960 B (two). Potential gfx1100/gfx1201 residency gain is **unmeasured**; two extra barriers and unchanged physical K/Q/output traffic may cancel it.

**Authoritative owner/consolidation:** QFP17 owns the exact source, algorithm, physical-work accounting, resource model, gates and fallback. 1335 remains superseded; QFP03/1294 TOP_K, QFP25/1334 sparse FA and QFP17/1332 QSA masks are separate. No new kernel owner, dispatcher, allocator, scheduler, runtime flag, patch or queued hardware work.

**12-hour and novelty:** selected independently active feat/rad06-fp16-wmma at f4625a5997c (2026-10-18 00:01:44 UTC), newer than main abaa59fe9d2. No independent QFP17/1335/indexer commit, PR or recorded hardware lane in the preceding 12 hours; off-repository queues not independently inspected. **Excluded** RAD06 helper replay, GEMV_LARGE_ROWS, grouped-MoE FP16 WMMA, long-row split, counters and their qualification. Unlike the older validated 1335 tile-vs-vector backport, this is a post-#30080 native **two-vs-four-head resource** discriminator.

**Measured boundary:** dev b11402 1335 tile-vs-vector +2.6% prefill at ~99K, +5.6% at ~202K and 192/192 backend-op passes per GPU. b11402 sparseprof indexer **plus TOP_K** 19.7 s / 272 s span at ~202K. Neither is current-pin isolated native indexer time or two-vs-four-head performance. Upstream MUSA tile-vs-vector gains are not AMD evidence.

**Cheapest check:** 5/5 disposable host source-shaped F32-bit 4/2-pass parity and unique-output/tail coverage cases passed (KV 1/63/64/65/129, batch 7/8/9/15/17); LDS formula checked. No repository tests, HIP compilation, GPU benchmark or patch application.

**Unresolved action:** Reuse existing QFP17 rocprof to prove native indexer exposed critical-path fraction and compiled LDS-limited occupancy. Reject if unknown, <3% optimistic E2E ceiling or no LDS resource limit. Only then compare disposable same-pin 4/2-head HIP builds, gfx1100/gfx1201 ISA/VGPR/spills/wave32, full backend-op F32 bits, multi-request/ubatch/graph/MTP and physical work. >=4 paired dev sessions/architecture, CI95-low >=3% E2E prefill and <=1% decode regression; otherwise retain native four-head pass. **Terminal:** NO_EXPOSED_INDEXER, NO_RESOURCE_LIMIT, BELOW_CEILING, NUMERIC_DRIFT, NO_GAIN or QUALIFIED_NATIVE_HEADPASS.

**Sources:** [llama.cpp #30080](https://github.com/ggml-org/llama.cpp/pull/30080), [#29901](https://github.com/ggml-org/llama.cpp/pull/29901), [ROCm GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html), [RDNA fork](https://github.com/stew675/llama-cpp-rdna-boosts/).
