---
id: BCOP300
order: 300
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-18'
created-by: agent
priority: P2
work: S
---

# QFP44: gate HIP MMQ Q8_1 Y-tile int2 copies on draft regressions

**Discovery:** Upstream [llama.cpp #30493](https://github.com/ggml-org/llama.cpp/pull/30493), updated 2026-10-18 07:27 UTC, is **draft** after same-GPU NVIDIA microbenchmark regressions (-8.08% Q5_K, -10.42% Q4_K, -11.09% IQ4_XS). It changes only `mmq.cuh::mul_mat_q_process_tile` activation Y-copy, not arithmetic, launch grid or required bytes. No BigCherry HIP gain or isolated copy bottleneck is measured.

**Authoritative owner/consolidation:** QFP44 owns the source-level copy/geometry interaction and technical gates. QFP17 owns prefill attribution; 0300 owns early J sizing, 1237/1265 compact MoE grids, QFP37/1350 Stream-K, PRBE110 Q6_K arithmetic. vLLM gfx1100 GPTQ W4A16 int4 loads are an alignment/ISA analogy, not a GGUF port. No new dispatcher, allocator, telemetry, patch or experiment queue.

**Branch/exclusion/novelty:** `main` at independent Audumla RAD06 commit `6b485c4c8f809abcfd0fa6868211d8c8187d0af6` (2026-10-18 09:57:42 UTC) is newer than other genuine development heads; BCOP-only branches were disregarded. No independent QFP44/MMQ Y-copy implementation, plan, PR, benchmark or queued lane in the preceding 12 hours on inspected branches/PRs; off-repository queues unverified. **Protected:** RAD06 helper FP16 WMMA, AHALF, VECIO, long-row/GEMV, balance, KLD, replay, profile and fused small-lane routing. The Radiance helper VECIO is a different engine/kernel path; it is not touched. Recent BCOP293/QFP17 concerns *lightning indexer LDS*, not this MMQ copy.

**Measured boundary:** Historical BigCherry 1350 prefill +1.8%/+1.2%/+1.9% at 8K/24K/98K; its 15% XTX summed-kernel-time HC class is **not** exposed MMQ Y-copy wall time. No new GPU measurement. Upstream NVIDIA positive pp and negative shape results cannot qualify AMD.

**Cheapest check:** Parsed pinned RDNA3/RDNA4 MMQ CASE tables and simulated scalar/int2 copies: 503/503 required-word fixtures match; 172 exact-divisible and 331 guarded-tail cases, with no source-offset alignment violation under aligned-base assumption. Padded scalar overreads summed 20,032 words **over fixture rows**, not runtime bytes. No repo tests, HIP build or benchmark.

**Unresolved action:** Wait for upstream draft resolution; then reuse existing per-rank prefill receipts to prove an executed, exposed MMQ copy bottleneck. If it survives, test one isolated int2-vs-stock build on gfx1100/gfx1201, ISA/VGPR/LDS/spills/traffic and full `MUL_MAT`/`MUL_MAT_ID`/MTP/graph correctness, four sessions/architecture and >=10 ABBA rounds. **Terminal:** `UPSTREAM_REGRESSION`, `NO_EXPOSED_COPY`, `NO_VECTOR_CODEGEN`, `RESOURCE_REGRESSION`, `NUMERIC_OR_WORK_DRIFT`, `NO_GAIN`, or `QUALIFIED_HIP_COPY` only with CI95-low >=3% E2E prefill and <=1% control regression. Stock scalar remains fallback.
