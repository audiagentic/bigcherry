---
id: BCOP295
order: 295
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-18'
created-by: agent
priority: P2
work: S
---

# PRVP04: gate new Vulkan Q4_0 KV transpose shader by executed prefill route

**Discovery:** Open upstream llama.cpp [#30738](https://github.com/ggml-org/llama.cpp/pull/30738) (2026-10-17) adds a Q4_0 dequant+transpose SPIR-V variant and pipeline registration for the existing Vulkan FA scratch path. BigCherry b11552 only has Q8_0. The pinned path requires **Q rows >=64**, quantized dense K and V, both transpose pipelines, buffer-range capacity and !coopmat2. Thus the PR's GCN1 tg128 improvement is not proven to arise from this shader for ordinary n=1 decode. External pp512 gain is not BigCherry gfx1100/gfx1201 evidence.

**Owner/consolidation:** PRVP04 owns this one Vulkan FA shader admission; PKC04's completed lifecycle ownership is unchanged, PRVP02 retains CM1 safety, TRVP12/13/14/15 provide shared identity/route/replay, RRVP02 preserves the user-directed Vulkan implementation pause. HIP RNX02, PRBE54 and QFP04 are separate kernels. No new selector, allocator, cache, dispatch table, local patch or hardware job.

**Branch/exclusion:** Selected independently active `feat/rad06-profile2` at `a80aa61e45fcb7b87dc1e68306064169d18ead5e` (Audumla, 2026-10-18 04:10:39 UTC), later independent than main's `c06dd0a9` (04:08:34 UTC); main's later BCOP294 merge was scheduled-audit publication and excluded as branch-selection freshness. Recent independent RAD06 FP16 WMMA, AHALF, VECIO, GEMV/long-row, replay, profiles and queued qualification were excluded and untouched. No independent Vulkan Q4 transpose plan, patch, PR, GPU run or queued work found in the preceding 12 hours. This source mechanism is new versus prior BCOP/QFP kernel audits.

**Validation boundary:** 8/8 source checks, 5/5 deterministic Q4 layout/nibble host fixtures (zero F16-bit mismatches/coverage errors), 90 gate-model cases (0 eligible n=1) passed. No BigCherry repo tests, shader compile, Vulkan hardware or performance measurements. At D256/4 KV heads/80K, theoretical full K+V F16 scratch 312.5 MiB versus Q4 cache 87.9 MiB; no measured traffic or live VRAM impact.

**Unresolved action:** After RRVP02 pause is explicitly lifted, first recheck upstream merge/pin, compile+verify shader and Q4/Q8 mixed-type route, prove actual n>=64 activation and critical-path share, full F16/logit/graph/VRAM correctness; only then run per-ICD gfx1100/gfx1201 paired prefill ABBA. No decode claim without a separate executed route. Detailed implementation and thresholds in PRVP04.

**Terminal:** `UPSTREAM_ABSORBED`, `NO_ROUTE`, `NO_HEADROOM`, `NUMERIC_OR_MEMORY_FAIL`, `NO_RDNA_GAIN`, or `QUALIFIED_VK_Q4_TRANSPOSE`.
