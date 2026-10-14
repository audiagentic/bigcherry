---
id: PRBE54
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:14.838615+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P3
---

# UP-KV-001: Dedicated F16 dequant path for Q5 KV

## Description

Materialized as `1271_prbe54_q5_kv_dequant_f16`. b11126 already has dedicated Q4_0/Q4_1 converter kernels; legacy Q5_0/Q5_1 still use the generic `dequantize_block_cont_cuda` scalar-callback path in `ggml_get_to_fp16_cuda()`. The patch adds a HIP half2 F16 Q5 path but exposes it only through a new FlashAttention-specific selector, so weight/general dequant callers remain unchanged. Direct vector-FA Q5 dequant helpers are already specialized; 1271 targets only the `need_f16_K`/`need_f16_V` staging fallback.

## Steps

1. Add one 32-thread Q5 F16 kernel: two Q5 blocks/workgroup, one thread per low/high nibble pair, Q5_0 scale and Q5_1 scale+minimum applied with half2 arithmetic.
2. Add Q5_0/Q5_1 launch wrappers and `ggml_get_to_fp16_fattn_cuda()`. On HIP it selects the new wrappers only for Q5_0/Q5_1; every other type delegates to `ggml_get_to_fp16_cuda()`.
3. Replace exactly the contiguous K and V F16-staging selector calls in `fattn-common.cuh`; no global selector table entry changes.
4. Add package-local activation/correctness/performance producer, contract, README/SUMMARY, and fail-closed mechanics tests.
5. Qualify on gfx1100 and gfx1201. Keep `untested` until >=4 sessions/architecture establish positive Q5 prefill effect and <=1% Q5 decode control regression.

## Detailed Solution & Technical Design

The existing generic Q5 path materializes a `float2` through `dequantize_q5_0`/`dequantize_q5_1`, then scalar-casts both values to F16. `1271` computes the same legacy-Q5 low/high values per thread and writes the two F16 outputs through half2 arithmetic. A 32-thread workgroup handles two 32-value Q5 blocks; an odd final block is bounds-checked.

The new selector is intentionally call-site scoped. `ggml_get_to_fp16_cuda()` stays byte-for-byte unchanged, preventing the experiment from changing non-FA weight/general conversion. The K and V selector replacements use `replace_all` with `expect_matches=1`, so upstream adding/removing a staging site fails closed.

Activation is a once-per-process `BIGCHERRY_PATCH_TRACE` marker emitted only when the FA staging selector receives Q5_0/Q5_1.

## Code Samples & Guidance

Primary source files:
- `ggml/src/ggml-cuda/convert.cu`: Q5 half2 kernel, wrappers, FA selector.
- `ggml/src/ggml-cuda/convert.cuh`: FA selector declaration.
- `ggml/src/ggml-cuda/fattn-common.cuh`: exactly one K and one V staging-selector substitution.

No Q4 changes. No edits to global `ggml_get_to_fp16_cuda()` cases. No change to direct vector-FA Q5 helpers.

## Files

- `patches/1271_prbe54_q5_kv_dequant_f16/{patch.toml,patch.py,README.md,SUMMARY.md,validation.toml}`
- `patches/1271_prbe54_q5_kv_dequant_f16/validation/{producer.toml,producer.py}`
- `tools/tests/patch/test_1271_prbe54_q5_kv_dequant_f16.py`
- `config/experiment-contracts.toml`
- upstream targets listed above.

## Validation

Offline: patch catalog load, contract registry load, mechanics apply/idempotence, missing-anchor/cardinality failures, patch lint/rebase check.

Hardware: gfx1100 + gfx1201; Q5_0 KV with FlashAttention for full-vocabulary backend-reference. Positive lane is pp512 prefill, which selects tile/MMA and therefore the F16 staging path; Q5_0 tg128 decode is the control because quantized decode selects the direct vector kernel and bypasses staging. Marker must be subject-only during the prompt/correctness path. Policy: `improvement_no_regression_v1`, 10 paired rounds/session, >=4 sessions/architecture.

## Effort & Risk

M / medium. Main risks are numerical drift from half2 arithmetic and accidentally broadening the conversion path. Full-vocabulary correctness gates the first; the FA-only selector and exact two call-site edits gate the second.

## Standards

Architecture-scoped evidence; no extrapolation. Small wins count only when established; no materiality bar.

## Acceptance Criteria

Q5 FA staging selector activates only in subject, full-vocabulary backend reference passes, session-bootstrap Q5 pp512 CI95 low > 0, Q5 tg128 control CI95 high <=1%, minimum four sessions on each declared architecture.

## Notes

Supersedes: RD64
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd64

2026-09-24 review corrected the original premise: Q4_0/Q4_1 are already specialized at b11126; Q5_0/Q5_1 are the remaining legacy generic converter cases. The global type selector has non-FA callers, so the implementation is FA-specific rather than globally replacing Q5 conversion.

2026-09-27: materialized as fresh `untested` package `1271_prbe54_q5_kv_dequant_f16` with contract `PRBE54-Q5-KV-DEQUANT-F16`; no hardware evidence is inherited. The positive lane is prefill because b11126's quantized decode path chooses the direct vector FA kernel and does not execute the F16 staging selector.


## 2026-10-09 audit: Q5_1 double-rounding and actual-path gate

**Status:** P3/pending; patch 1271 remains untested. Last independent plan triage was 2026-10-08 06:47 UTC, outside the 12-hour exclusion. No new Q5-specific GPU benchmark or queued qualification was found during that audit; archived GPU evidence is reconciled below. Evidence below is source inspection and deterministic host emulation, **not HIP execution**.

**Exact path:** Pinned llama.cpp b11474 `ggml/src/ggml-cuda/fattn.cu::ggml_cuda_flash_attn_ext_get_alloc_size` requests F16 K/V for TILE and MMA_F16; VEC generally uses direct Q5 handlers. `fattn-common.cuh::launch_fattn` calls `ggml_get_to_fp16_cuda` for **contiguously allocated** non-F16 K/V staging only. Noncontiguous conversion is separate; a V view of K may reuse K's buffer. Patch 1271 replaces only these two call-site selectors with `ggml_get_to_fp16_fattn_cuda`, returning `bigcherry_dequantize_block_q5_f16` for HIP Q5_0/Q5_1. Global conversion, FA allocation, direct vector Q5 and graph scheduling remain unchanged.

**Numerical finding:** Baseline `dequantize.cuh::dequantize_q5_1` computes `float(q)*float(d)+float(m)` then writes F16; candidate Q5_1 uses `__hmul2` then `__hadd2`, introducing an intermediate F16 rounding. A deterministic host fixture decoded 4,096 packed blocks (131,072 values/type): Q5_0 produced **0** F16-bit mismatches, Q5_1 **28,261 (21.56%)** against source float arithmetic; positive-scale/negative-offset Q5_1 control **40,635/131,072**. Output coverage passed for 1/2/3/7/8/9/511/4096 blocks. HIP FMA/denormal behavior was not emulated; this is not proof of GPU corruption or logits drift. The current `validation/producer.py` tests **Q5_0 only**; it cannot qualify Q5_1. The process-once trace marker is not a per-request/per-type kernel count.

**Cheapest discriminator and implementation decision:**
1. Add package-local packed-`qs`/`qh` high-bit, Q5_0 sign, odd-tail, F16 scale/offset and exact-output fixtures; compare candidate against the pinned source, including subnormal/finite extremes. Reuse existing mechanics tests. For Q5_1, either use baseline float multiply/add and one F16 conversion (verify HIP contraction) or gate the existing FA selector to **Q5_0 only**. Reject Q5_1 promotion if parity is not established; do not widen the global selector.
2. Prove actual TILE/MMA_F16 **contiguous** K and V staging, input/output byte counts, converter launches, scratch lifetime and graph capture/replay on isolated gfx1100 and gfx1201, including same-process repeated requests. Cover V-as-K-view and noncontiguous controls. Use Q5_0 and separate Q5_1, F16, Q8, and direct VEC tg128 controls. Do not share QFP17/1330 masked Flash-Next lanes or modify that recently active capability.
3. Profile converter share at pp512/pp2048, ub16/128/512, ctx8K/80K. If staging is <5% of prefill critical path or best-case Amdahl E2E ceiling <3%, **close without A/B**. Otherwise use the existing contract and >=4 sessions/architecture, >=10 paired ABBA rounds, full-vocab/logits/KLD/greedy parity, no missing work, CI95-low positive pp512 and <=1% decode/control regression. A >=3% E2E improvement is required before wider production adoption. No hardware improvement is claimed.

**Upstream/forks:** [llama.cpp #27140](https://github.com/ggml-org/llama.cpp/pull/27140) remains open; its Q5_1 float-then-F16 converter is a parity control, but reported Q5_0 1167 t/s and Q5_1 1164 t/s are on RTX 3090, not AMD. [#29827](https://github.com/ggml-org/llama.cpp/pull/29827) closed unmerged with a 64 MiB capped F16 scratch/chunked-KV alternative; [#29846](https://github.com/ggml-org/llama.cpp/pull/29846) closed unmerged with NVIDIA cp.async/IMMA fused Q4/Q8, not Q5. [vLLM ROCm](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backends/rocm_attn/) supports FP8 KV, not GGUF Q5. None establishes an RDNA gain. PRBE54 alone owns the converter; no new dispatch/cache/allocator/telemetry system.


## Change Log

- 2026-10-08 (triage): Kept pending at P3. Patch 1271_prbe54_q5_kv_dequant_f16 remains untested and only targets Q5 quantized KV F16 staging; production Flash-Next uses F16 KV. No matching measured Q5_KV optimization lane, so no in_progress status or promotion claim.

## 2026-10-14 audit: reconcile archived GPU correctness and cost gates

**Current disposition:** Patch 1271 remains registry-`untested` (**not validated**) and **NOT_ADOPTED**. The earlier 2026-10-09 audit said no hardware test; archived b11126 **Q5_0 GPU** records already existed in `engines/llamacpp/patches/1271_prbe54_q5_kv_dequant_f16/evidence/validation.json`. No new GPU work was done for this audit.

| Architecture | Q5_0 pp512 geometric effect (within-session CI95) | tg128 control | Full-vocab correctness | Contract |
|---|---|---|---|---|
| gfx1201 | +0.0143%, [-0.2110%, +0.2306%] | +0.1247%, [+0.0505%, +0.1962%] | PASS: 64 decode steps, 15,892,480 logprobs, KL=0, material diff=0 | INVALID: 1 of 4 sessions |
| gfx1100 | +0.7658%, [-0.0309%, +2.0003%] | +0.0341%, [-0.1487%, +0.2052%] | **FAIL:** material logprob diff 0.342 > 0.02 | INVALID: correctness + 1 of 4 sessions |

**Evidence interpretation:** These are real archived hardware runs, not validated gains or current-pin measurements. gfx1100 divergence is observed but **not causally attributed** to 1271: an A/A reproducibility control is missing. The process-once `BIGCHERRY_PATCH_HIT` marker proves selector entry, not each converter launch or critical-path work. The gfx1201 CI includes zero, and gfx1100 CI lower bound is negative; no architecture demonstrated positive pp512 CI95-low.

**b11552 source/data flow:** `ggml/src/ggml-cuda/fattn.cu::ggml_cuda_fattn_kv_type_supported` includes Q5_0 and Q5_1. `ggml_cuda_flash_attn_ext_get_alloc_size` sets `need_f16_K/V` for TILE/MMA_F16 and certain VEC fallbacks. `ggml/src/ggml-cuda/fattn-common.cuh::launch_fattn` invokes `ggml_get_to_fp16_cuda` only for *contiguously allocated* non-F16 K/V; noncontiguous conversion uses a separate selector, and a V view of K reuses K's F16 buffer. `ggml_cuda_flash_attn_ext_get_f16_extra_data` places aligned scratch after the graph output allocation; ownership/lifetime is the existing graph/output, not 1271. The 1271 patch swaps only those two FA call-site selectors for `ggml_get_to_fp16_fattn_cuda`, preserving the global converter and scheduler. Q5_0 candidate uses half multiply; Q5_1 uses `__hmul2` then `__hadd2`, unlike stock `dequantize.cuh::dequantize_q5_1` F32 multiply/add followed by one F16 conversion.

**Q5_1 correctness:** Prior 131,072-value host fixture found Q5_1 28,261 F16-bit mismatches; a fresh seeded host fixture found 28,727/131,072 (Q5_0 0/131,072). These are host arithmetic demonstrations, **not** HIP correctness or an explanation of the gfx1100 Q5_0 GPU failure. `validation/producer.py` exercises only Q5_0, so cannot qualify Q5_1.

**Implementation-ready bounded gates (existing owner only):**
1. **Correctness before optimisation:** On isolated gfx1100, replay the archived Q5_0 64-step full-vocab prompt with same-process A/A control/control and A/B control/1271, same pin/model/quant/flags, repeated request and graph reuse. Capture actual TILE/MMA_F16/VEC route, contiguous K/V shape, converter kernel counts, physical rank, event waits, F16 scratch bytes and graph lifetime. A/A mismatch -> `UNSTABLE_REFERENCE`; A/A pass and A/B mismatch -> `Q5_0_CORRECTNESS_FAIL` and reject 1271. Never infer GPU correctness from host half parity.
2. **Q5_1 is blocked:** Before any Q5_1 hardware lane, use the existing 1271-local selector to fall back to stock conversion for Q5_1 or implement F32 multiply/add with one final F16 rounding; validate packed high bits, odd blocks, F16 scales/minimums, subnormals, signed zero, contraction and exact output. Do not widen global dispatch or add a new kernel family.
3. **Cost gate:** Only if Q5_0 correctness passes, reuse existing rocprof and FA diagnostics at pp512/pp2048, ub16/128/512, 8K/80K; quantify physical K/V transfer/conversion bytes, scratch lifetime, converter time and *exposed* critical-path fraction. Include Q5_1 negative, F16/Q8 and direct VEC tg128 controls. If optimistic removable E2E time <3%, stop `BELOW_EXPOSED_GATE` **without** four-session hardware sweeps.
4. **Promotion gate:** Only if prior gates pass, reuse existing `PRBE54-Q5-KV-DEQUANT-F16` producer/contract: >=4 independent sessions per gfx1100/gfx1201, >=10 paired rounds, full-vocab/KLD/greedy, multi-request/ubatch/graph, no skipped work, CI95-low positive, >=3% E2E and <=1% unaffected-control regression. Otherwise `NO_VALIDATED_GAIN`.

**Consolidation:** PRBE54/1271 is the only FA Q5 converter owner; RD64 superseded, BCOP79 already covers Q5_1 rounding. QFP17 masked FA and PRBE115 Q8_1 MMVQ are distinct; no new scheduler, cache, allocator, dispatch table or telemetry. Active RAD02/RAD03 remote-MoE and RAD04 two-half prefill remain untouched.

**Upstream/forks:** [llama.cpp #27140](https://github.com/ggml-org/llama.cpp/pull/27140) remains open (checked 2026-10-14) and uses float-before-half Q5_1 conversion; RTX 3090 results do not transfer to AMD. [#29846](https://github.com/ggml-org/llama.cpp/pull/29846) closed unmerged (NVIDIA Q4/Q8 fused FA, not Q5). [SGLang quantized-KV docs](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/quantized_kv_cache.mdx) warn that unfused dequant can negate cache benefits, but use different FP8/FP4 formats. The old #28455 CPU fallback issue does not establish fallback at b11552, whose type-support switch explicitly includes Q5.

**Validation actually performed:** host arithmetic discriminator (131,072 values/type, Q5_0 0 differences, Q5_1 28,727), Python compilation, source/archived evidence inspection. No repository pytest, patch apply, HIP build, GPU run or new benchmark.
