---
id: BCOP239
order: 239
plan: patching-bc-optimizations
state: done
created-at: '2026-10-15T19:12:52Z'
created-by: agent
priority: P3
work: S
---

# QFP48: retire unproven router split-K reduction fusion; reconcile 1357 status

**Discovery/change:** The old QFP48 candidate treats bc_moe_router_splitk_partial -> bc_moe_router_splitk_reduce as an adjacent-kernel fusion. Validated 1357 instead needs eight independently scheduled K partitions followed by a cross-workgroup fixed-order reduction; __syncthreads is block-local. Flash-Next K2560/M512/N512 produces 512 partial CTAs, 1024 reduce CTAs and 8 MiB scratch (16 MiB logical write/read). Historical 1.93 s partial time is not a removable reduce cost. 1357 README and SUMMARY incorrectly still claimed "Not promoted"/"Evidence None" despite 2026-10-10 promotion.

**Owner/consolidation:** QFP36/1357 owns the router, scratch, dispatch and correctness; QFP48 closes only this stale fusion candidate, remaining pending for BCOP206's batch-boundary trace join. BCOP191 already gated a different MMQ Q8_1 producer candidate. No new kernel, scheduler, allocator, dispatch table, telemetry, hardware queue or implementation. Corrected 1357 status documentation.

**Eligibility/novelty:** main independent HEAD 08441aa4a69e (2026-10-15 05:21 UTC). No BigCherry independent branch commit or PR update after the 2026-10-15 07:12 UTC cutoff in the checked GitHub data, nor new 1357/QFP48 submission. The last BCOP191/206 audits covered MMQ producer and graph/batch gaps, not router partial/reduce fusion. Separate queued QFP45 reference lane and open Radiance helper PR #231 remain untouched.

**New source comparison:** vLLM wvSplitKrc_ uses gfx950-only last-CTA counters and fixed-order scratch readback with per-stream ownership; its F16/BF16 skinny shape gate is not BigCherry F32 N512 on gfx1100/gfx1201. vLLM gfx950 FP32 router kernel is <=32-token/whitelisted-shape. Merged llama.cpp #25952 combines expert outputs after routing, not router split-K; open #30553 caps MMQ Stream-K work. Reject as direct ports.

**Terminal/unresolved:** ROUTER_REDUCTION_FUSION_NOT_ADMITTED. Reopen only after same-pin per-rank measured reduce+launch exposed E2E share >=3% and a proven graph-replay-safe last-completer or other deterministic protocol. QFP36 then owns one default-off variant with same-process, multi-ubatch, full-vocab/greedy/top-k/MTP, scratch and cancellation gates; require CI95-low >=3% E2E and <=1% decode regression or reject. No action or new experiment now.

**Validation:** 9/9 disposable host-only source-shaped invariants passed (grid/bytes, route fallback, F32 order, unsafe unfenced read, stale counter replay). Source/recipe/profile/patch and historical evidence inspected. No repo pytest, patch composition, HIP build, GPU run or new measured gain. Technical detail and source links reside in QFP48.
