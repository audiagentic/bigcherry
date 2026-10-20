---
id: BCOP337
order: 337
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-20'
created-by: agent
priority: P2
work: S
---

# TRVP12: RDNA3/4 Vulkan 128x128 GEMM split-K tail waves

**Discovery:** Open [llama.cpp #30280](https://github.com/ggml-org/llama.cpp/pull/30280) adds RDNA3/4 large-tile Vulkan MUL_MAT split-K 1/2/3 wave-aware geometry. External gfx1100/gfx1201 pp512 gains reach +9.8% on FP16 and +6.4% on INT8, not measured on BigCherry. Pinned b11552 already implements split-K and reduction; only existing launch geometry may be underoccupied. Upstream #30146 changes the same heuristic for small N. #30280 remains open with review/merge-conflict concerns.

**Owner:** TRVP12 native Vulkan MUL_MAT split/reduction. TRVP15/PRVP02 CM1 safety; TRVP14 route identity; RRVP02 Vulkan pause. No duplicate technical owner, shader, selector, scheduler, allocator, telemetry, configuration or hardware queue.

**Eligibility:** Selected feat/rad14-mtp-bisect from independent Audumla a7ee887ad04b (2026-10-19 23:48 UTC), newer than feat/rad14-mtp0 4afb74bd and main 09e3be7d. Repository-visible preceding 12h RAD14 FP8/MTP bisect, RAD13 FP8 expert GEMM, RCS09/RCS05 runtime helper work excluded; no matching Vulkan large-tile split-K independent work or queue found. Off-repository queues unverified. Novel versus recent BCOP328–336: large-N Vulkan GEMM tail-wave geometry, not HIP MMQ, Vulkan FA, GDN, FWHT or TOPK.

**Completed:** Inspected pinned ggml_vk_guess_split_k, ggml_vk_mul_mat_q_f16, ggml_vk_matmul and split scratch/reduction; upstream #30280/#30146 diffs/reviews, BigCherry owners and external engines. Host-only 3,360 geometry/tail/resource cases passed; Python compilation passed. Source-level tie-break may prefer split3 over lower modeled split2 cost (6272 vs 6400), not an observed regression. M4096/N512 split3 scratch=24 MiB and up to 48 MiB extra logical partial traffic. No Vulkan build, GPU test or BigCherry benchmark.

**Action/terminal:** Respect RRVP02 pause. Once released, prove exact 128x128 route and >=2.9126% removable exposed E2E time; otherwise NO_LARGE_TILE_ROUTE/BELOW_CEILING. Reconcile upstream first (WAIT_UPSTREAM/PIN_ABSORBED). If eligible, compare fixed 1/2/3 and heuristic with K-tail, full-work, alias/graph, LDS/VGPR/spill/traffic, logits/KLD/MTP and paired E2E CI95-low >=3%; else NUMERIC_FAIL/RESOURCE_REGRESSION/NO_E2E_GAIN/REJECT. Technical plan resides only in TRVP12.
