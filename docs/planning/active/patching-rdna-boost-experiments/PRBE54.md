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

## 2026-10-12 evidence reconciliation — prior real Q5_0 GPU lanes are INVALID, not absent

**Disposition:** keep PRBE54 pending P3 and patch 1271 `untested`/default-off. **Do not queue the former four-session qualification matrix.** First adjudicate the already recorded real gfx1100/gfx1201 results and prove a new, materially different current-pin opportunity. This section supersedes earlier statements that no Q5-specific GPU measurement exists; the 2026-10-09 host-only Q5_1 arithmetic finding remains valid. No new hardware was run for this audit.

### Source-of-truth measured records (historical b11126; not current b11474 qualification)

`engines/llamacpp/patches/1271_prbe54_q5_kv_dequant_f16/evidence/validation.json` contains **two** hardware records, one per architecture, both `validation_disposition=incomplete` and contract `status=invalid`. Each contains ten paired rounds **in one session**, not four independent sessions.

| GPU | Q5_0 pp512 geometric effect (95% interval), percent | Q5_0 tg128 control effect (95% interval), percent | Full-vocabulary backend reference | Contract |
| --- | --- | --- | --- | --- |
| gfx1201 R9700 | +0.0143 [-0.2110, +0.2306] | +0.1247 [+0.0505, +0.1962] | PASS; 64 steps, 15,892,480 logprobs, max KL 0 | INVALID: 1/4 sessions |
| gfx1100 XTX | +0.7658 [-0.0309, +2.0003] | +0.0341 [-0.1487, +0.2052] | **FAIL**: material logprob diff 0.342 > 0.02 | INVALID: 1/4 sessions and failed correctness |

These are *measured within-session effects* but **not evidence of a positive reproducible speedup**. Both prefill intervals cross zero; gfx1100 is ineligible regardless of throughput because correctness failed. The gfx1201 result does not justify four more sessions without a new mechanism or critical-path measurement. Do not treat a failed contract as a negative benchmark with promotion authority.

### Implementation/producer audit and attribution

- `patch.py::_Q5_KERNEL` implements Q5_0 half2 multiply and Q5_1 half2 multiply **then** half2 add. Existing host emulation found 28,261/131,072 Q5_1 F16-bit mismatches versus the pinned float-multiply/add reference. No Q5_1 GPU correctness claim follows from the Q5_0-only records.
- `patch.py::_FATTN_SELECTOR` selects the Q5_0 or Q5_1 wrapper on HIP. Its process-once `BIGCHERRY_PATCH_HIT` marker is emitted **when the selector is queried**, before the wrapper launch; it cannot prove converter launch count, K/V bytes, critical-path occupancy, or repeated-request activation. Do not infer the actual executed kernel from marker alone.
- `validation/producer.py::_Q5_ARGS` uses `-ctk q5_0 -ctv q5_0 -fa on` for both arms; `run()` records a single marker Boolean across the server correctness phase and a separate paired `llama-bench` phase. It does not validate Q5_1, does not correlate the marker with pp512 runs, and does not account for converter launches in each benchmark cell. `test_1271_prbe54_q5_kv_dequant_f16.py` checks patch anchors/idempotence but not arithmetic parity or runtime dispatch.
- `ggml/src/ggml-cuda/fattn.cu::ggml_cuda_flash_attn_ext_get_alloc_size` chooses TILE/MMA F16 staging; `fattn-common.cuh::launch_fattn` calls the two contiguous K/V conversion selectors. Direct VEC Q5 and noncontiguous conversion remain outside 1271. F16 staging buffers are temporary graph/scratch resources, not persistent KV savings. Track allocated bytes, actual conversion launches and graph replay before attributing speed.
- Current Flash-Next Qwen4Exp production uses F16 KV; QFP17's active attention/prefill path is protected. This dormant legacy-Q5 conversion cannot claim that workload's gains. The mixed dual-XTX/R9700 topology, absent P2P and gfx1030 auxiliary lane do not create evidence of a Q5 converter benefit; qualify one device/architecture first if reopened.

### Bounded decision and ownership

1. **No new patch or telemetry system.** PRBE54 owns 1271; BCOP79 already owns the Q5_1 arithmetic gate. BCOP164 records this evidence-reconciliation disposition only. Do not modify QFP17 or active Radiance paths. Keep global `ggml_get_to_fp16_cuda`, vector FA Q5 handlers and existing contract framework untouched.
2. **Cheapest gate, without GPU:** add a repository test that parses both immutable `validation.json` records and asserts per-architecture contract INVALID, sample count, correctness and CI endpoints; ensure future reports cannot call them absent or validated. Extend existing package mechanics test with packed Q5_0/Q5_1 host reference, odd block, high-bit and F16 extremes. For Q5_1 either restore float arithmetic with one F16 rounding and verify HIP contraction, or gate selector to Q5_0 only. No speculative promotion.
3. **Only if a new production need arises:** first collect a *single* isolated current-pin (b11474 or newer) source/launch census for symmetric Q5_0, with `GGML_CUDA_FA_ALL_QUANTS` build identity, actual TILE/MMA dispatch, per-ubatch K/V converter launch counts, input/output bytes, conversion time and total prefill critical-path time. Include Q8_0, F16 and direct VEC decode negative controls, ctx 8K/80K, pp512 and pp2048, ub16/128/512; verify same-process repeated requests and graph replay. **If converter share <5% or even infinite-speedup Amdahl ceiling <3%, close/retire 1271 without more runs.** Missing attribution = INVALID, not PASS.
4. **If cost gate passes:** isolate gfx1100 full-vocab failure first (source/subject/stock, deterministic logits and KLD, K/V buffer bytes and graph identity). Failures block all promotion. Only then repeat paired ABBA with >=10 rounds and >=4 independent sessions per gfx1100/gfx1201; require source-parity Q5_0, Q5_1 separately if enabled, no missing work, CI95-low >=3% end-to-end prefill improvement, <=1% regression in decode/F16/Q8 controls and unchanged MTP/long-context behavior. Otherwise terminal REJECT. Preserve the default-off fallback.

### Upstream and other engines (checked 2026-10-12)

- [llama.cpp #27140](https://github.com/ggml-org/llama.cpp/pull/27140) remains **open** (last update 2026-09-26). Its Q5 converter uses float arithmetic and is a *numerical reference*, not an AMD speed claim; reported RTX 3090 gains do not transfer.
- [llama.cpp #29827](https://github.com/ggml-org/llama.cpp/pull/29827) and [#29846](https://github.com/ggml-org/llama.cpp/pull/29846) are closed **unmerged**. The former chunks a bounded F16 conversion scratch; the latter loads quantized K/V directly in MMA shared-memory tiles for Q4/Q8 with NVIDIA cp.async/IMMA. Both target different bottlenecks; neither qualifies a Q5 half2 converter on RDNA.
- [llama.cpp #29371](https://github.com/ggml-org/llama.cpp/issues/29371) documents the separate long-context scratch problem; [#28455](https://github.com/ggml-org/llama.cpp/issues/28455) documents build-time all-quants CPU fallback risk. An activated selector is not proof that all quantized FA workloads use the same GPU route.
- [ROCm AITER](https://github.com/ROCm/aiter) offers AMD attention/quant operators with experimental gfx1100/gfx1201 support, but no verified GGUF Q5_0/Q5_1 drop-in path; **do not port** or extrapolate MI300X throughput.

**Audit validation actually run:** 14/14 read-only source/evidence assertions in a standalone connector-side JavaScript fixture; no repository pytest, HIP build or GPU experiment. This is a documentation/qualification reconciliation, not a new performance measurement.

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

**Status:** P3/pending; patch 1271 remains untested. Last independent plan triage was 2026-10-08 06:47 UTC, outside the 12-hour exclusion. No Q5-specific GPU benchmark or queued qualification was found. Evidence below is source inspection and deterministic host emulation, **not HIP execution**.

**Exact path:** Pinned llama.cpp b11474 `ggml/src/ggml-cuda/fattn.cu::ggml_cuda_flash_attn_ext_get_alloc_size` requests F16 K/V for TILE and MMA_F16; VEC generally uses direct Q5 handlers. `fattn-common.cuh::launch_fattn` calls `ggml_get_to_fp16_cuda` for **contiguously allocated** non-F16 K/V staging only. Noncontiguous conversion is separate; a V view of K may reuse K's buffer. Patch 1271 replaces only these two call-site selectors with `ggml_get_to_fp16_fattn_cuda`, returning `bigcherry_dequantize_block_q5_f16` for HIP Q5_0/Q5_1. Global conversion, FA allocation, direct vector Q5 and graph scheduling remain unchanged.

**Numerical finding:** Baseline `dequantize.cuh::dequantize_q5_1` computes `float(q)*float(d)+float(m)` then writes F16; candidate Q5_1 uses `__hmul2` then `__hadd2`, introducing an intermediate F16 rounding. A deterministic host fixture decoded 4,096 packed blocks (131,072 values/type): Q5_0 produced **0** F16-bit mismatches, Q5_1 **28,261 (21.56%)** against source float arithmetic; positive-scale/negative-offset Q5_1 control **40,635/131,072**. Output coverage passed for 1/2/3/7/8/9/511/4096 blocks. HIP FMA/denormal behavior was not emulated; this is not proof of GPU corruption or logits drift. The current `validation/producer.py` tests **Q5_0 only**; it cannot qualify Q5_1. The process-once trace marker is not a per-request/per-type kernel count.

**Cheapest discriminator and implementation decision:**
1. Add package-local packed-`qs`/`qh` high-bit, Q5_0 sign, odd-tail, F16 scale/offset and exact-output fixtures; compare candidate against the pinned source, including subnormal/finite extremes. Reuse existing mechanics tests. For Q5_1, either use baseline float multiply/add and one F16 conversion (verify HIP contraction) or gate the existing FA selector to **Q5_0 only**. Reject Q5_1 promotion if parity is not established; do not widen the global selector.
2. Prove actual TILE/MMA_F16 **contiguous** K and V staging, input/output byte counts, converter launches, scratch lifetime and graph capture/replay on isolated gfx1100 and gfx1201, including same-process repeated requests. Cover V-as-K-view and noncontiguous controls. Use Q5_0 and separate Q5_1, F16, Q8, and direct VEC tg128 controls. Do not share QFP17/1330 masked Flash-Next lanes or modify that recently active capability.
3. Profile converter share at pp512/pp2048, ub16/128/512, ctx8K/80K. If staging is <5% of prefill critical path or best-case Amdahl E2E ceiling <3%, **close without A/B**. Otherwise use the existing contract and >=4 sessions/architecture, >=10 paired ABBA rounds, full-vocab/logits/KLD/greedy parity, no missing work, CI95-low positive pp512 and <=1% decode/control regression. A >=3% E2E improvement is required before wider production adoption. No hardware improvement is claimed.

**Upstream/forks:** [llama.cpp #27140](https://github.com/ggml-org/llama.cpp/pull/27140) remains open; its Q5_1 float-then-F16 converter is a parity control, but reported Q5_0 1167 t/s and Q5_1 1164 t/s are on RTX 3090, not AMD. [#29827](https://github.com/ggml-org/llama.cpp/pull/29827) closed unmerged with a 64 MiB capped F16 scratch/chunked-KV alternative; [#29846](https://github.com/ggml-org/llama.cpp/pull/29846) closed unmerged with NVIDIA cp.async/IMMA fused Q4/Q8, not Q5. [vLLM ROCm](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backends/rocm_attn/) supports FP8 KV, not GGUF Q5. None establishes an RDNA gain. PRBE54 alone owns the converter; no new dispatch/cache/allocator/telemetry system.


## Change Log

- 2026-10-08 (triage): Kept pending at P3. Patch 1271_prbe54_q5_kv_dequant_f16 remains untested and only targets Q5 quantized KV F16 staging; production Flash-Next uses F16 KV. No matching measured Q5_KV optimization lane, so no in_progress status or promotion claim.
