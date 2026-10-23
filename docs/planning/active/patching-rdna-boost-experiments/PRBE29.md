---
id: PRBE29
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:27.064093+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P3
---

# AMD-GEMM-002: Persistent F16 shadow of quantized dense weights

## 2026-10-09 PRBE28 ownership reconciliation

**PRBE29 alone owns quantized-weight F16 shadows.** PRBE28's original GGUF row padding is now float-only: never pad Q8_0/Q4_K/Q6_K packed source strides. AMD-Ecosystem/llama.cpp PR #57 implements an existing load-time shadow through `ggml_backend_cuda_buffer_set_tensor`, `tensor->extra`, and buffer-context ownership; inspect complete vs partial/async uploads, ownership, graph and split behaviour before choosing this hook over the plan's post-`load_all_data` alternative. The external gfx1151 performance evidence is not BigCherry qualification. PRBE30 owns any K-padding inside this shadow; no second shadow allocator belongs to PRBE28.


## 2026-10-10 BCOP126: load-complete shadow admission, immutable lifetime and bounded memory

**Disposition:** keep PRBE29 pending/default-off, but **do not port AMD-Ecosystem/llama.cpp #57's `set_tensor` hook or `tensor->extra` pointer as-is**. Pinned b11474 `llama_model_loader::load_all_data` has three distinct upload routes: mmap/host-backed allocation, synchronous `ggml_backend_tensor_set`, and chunked `ggml_backend_tensor_set_async` with pinned staging. The async path bypasses `ggml_backend_cuda_buffer_set_tensor`; a shadow created only there silently fails to cover eligible weights on that route. The loader synchronizes all upload events **before** `load_all_data` returns; only a successful, synchronized load of the relevant model-buffer group is a safe publication boundary. The external PR's `tensor->extra != nullptr` early return also leaves a previously built shadow stale if the source is later partially overwritten. These are source/contract hazards, not reproduced GPU failures. PR #57 remains open; its gfx1151 results are not gfx1100/gfx1201 evidence.

**Authoritative implementation sites (pinned b11474):** `src/llama-model.cpp::load_tensors` calls `llama_model_loader::load_all_data` for each context/buffer map; `src/llama-model-loader.cpp::load_all_data` drains async events before success; `ggml/src/ggml-cuda/ggml-cuda.cu::{ggml_backend_cuda_buffer_context,ggml_backend_cuda_buffer_set_tensor,ggml_backend_cuda_buffer_set_tensor_2d,ggml_cuda_mul_mat,ggml_cuda_mul_mat_cublas_impl}`; `ggml/src/ggml-cuda/convert.cuh::ggml_get_to_fp16_cuda`. Source model metadata and backend upload paths, not the generic pre-upload `buffer_type_alloc_buffer`, decide eligibility. Reconfirm these anchors on the **composed** production patch set.

**Bounded v1 algorithm / ownership (no second allocator/registry):**

1. PRBE70 supplies one **measured, typed, non-MoE dense `GGML_OP_MUL_MAT`** weight/ubatch/physical-GPU signature and exposed critical-path evidence. Reject `MUL_MAT_ID`, views, duplicated/tied/shared weights without unique ownership, host/offloaded/mmap-only weights, non-2D, unsupported quant types, unknown physical device or missing typed route. No candidate if the trace cannot prove a >=2.9126% E2E removable ceiling.
2. After the owning `load_all_data` returns success **and** its async events are drained, traverse only selected resident HIP model tensors. Use a sidecar entry owned by their existing `ggml_backend_cuda_buffer_context` (key: tensor identity + backing buffer + physical device + source generation; fields: original type/shape/stride, F16 pointer/lda/bytes, upload-complete event and READY state). Do not repurpose `tensor->extra` without proving all other consumers; do not create a second global cache/allocator. Keep the source quantized tensor intact.
3. Reserve the **real per-device** shadow bytes `2*ne[0]*ne[1]` plus alignment, optional row pad and conversion scratch. Start with <=256 MiB **per physical device** and require a separately verified KV/graph/MTP/context safety reserve; production VRAM is often 92-97% occupied. If allocation, conversion, coherence or budget admission fails, discard any partial shadow and retain stock MMQ/MMVQ; never abort model loading to gain a speculative optimisation. A billion shadow scalars cost ~1.863 GiB before padding; mirroring 27B scalars costs ~50.3 GiB and is ineligible.
4. Populate once on the **same physical GPU** as the source with existing `ggml_get_to_fp16_cuda(type)`, synchronize its completion event, validate against the reference conversion, then atomically publish READY. No P2P/RCCL or cross-card borrowing; gfx1030 is an explicit negative control. Any subsequent source write, copy, clear, buffer rebinding or generation change must invalidate READY **before** dispatch; either rebuild after complete re-upload or fail closed. Do not reuse a graph executable that captured an invalidated shadow pointer; preserve model-buffer-before-shadow-free and graph teardown ordering.
5. `ggml_cuda_mul_mat` may select the existing F16 `ggml_cuda_mul_mat_cublas_impl` **only** when a READY same-device sidecar exists, `ne11` (actual **u-batch** M, not prompt length) meets a measured crossover, tensor route is supported, and graph identity remains valid. Otherwise fall through unchanged to native MMVF/MMVQ/MMQ/BLAS. Preserve native decode and MTP verification paths unless numerical/acceptance parity is separately proven. No new hipBLASLt path belongs to PRBE29.

**Cheap discriminator executed in this audit:** eight disposable Python host-state assertions passed: incomplete chunk, undrained event, insufficient budget, complete post-sync publication, foreign-device rejection, partial-overwrite invalidation, rebuild after overwrite and per-device ownership. This models lifecycle invariants; **no actual HIP upload, conversion, graph or GPU correctness test ran**.

**Qualification / terminal gate:** first obtain PRBE70's typed hot dense GEMM signature and capacity receipt; if absent, record `BLOCKED_ON_TYPED_DENSE_GEMM` and do not queue a duplicate experiment. Next add a host/backend fixture for sync/full vs async/chunked vs mmap/host and overwrite/reload, source/shadow dequant comparison, memory-OOM fallback, graph recapture and same-process multi-request. Only after those pass compare pristine b11474+production versus one default-off selected shadow on gfx1100 and gfx1201: M=64/128/256/512/1024/2048 (actual `ne11`), contexts 8K/24K/98K, prefill/decode/MTP, load time, peak VRAM, correct kernel route and numerical/logits/KLD/greedy acceptance. Four independent sessions and >=10 paired ABBA rounds each; promote only CI95-low >=3% E2E, <=1% control regression and no correctness/VRAM violation. Reject/retire when no eligible trace, no safe resident budget, no stable shadow lifetime or no measured crossover. PRBE30 owns conditional shadow-only padding; PRBE31 owns BLAS/Lt crossover; PRBE28 owns original float GGUF padding. No overlapping active QFP prefill, 1360/1361 transfer or Radiance lane is to be edited/queued.

**External mechanism decision:** [AMD-Ecosystem PR #57](https://github.com/AMD-Ecosystem/llama.cpp/pull/57) supplies reusable conversion/lda ideas but is **not** a drop-in loader lifecycle. [ggml-org #26621](https://github.com/ggml-org/llama.cpp/pull/26621) covers float original-weight padding, not quantized shadows. vLLM's [RDNA3 quantized GEMM](https://github.com/vllm-project/vllm/blob/main/csrc/rocm/moe_q_gemm_rdna3.cu) demonstrates an alternative on-the-fly quantized path, not a portable shadow-speedup. External #57 measured gfx1151 Qwen2.5-3B pp1024 +22.0%, Llama-3.1-8B pp1024 +32.2%, and pp128 regressions; **no BigCherry PRBE29 gain is measured**.

## Description

TODO, first-principles investigation, foundational for PRBE30/PRBE31. Persistent F16 device-buffer 'shadow' of selected quantized dense (non-MoE-expert) weights, built once at model load (not per-GEMM), opt-in only. Real b11126 confirmed this batch: ggml-cuda.cu has cuBLAS integration (`ggml_cuda_mul_mat_cublas_impl` at line 1433, `cublasHandle_t`/`CUBLAS_CHECK` throughout) but NO hipBLASLt-specific path exists yet -- so this project's HIP build currently runs dense F16/F32 GEMM through hipBLAS's cuBLAS-compatibility shim, not a tuned hipBLASLt path; introducing the latter (needed for PRBE31's crossover measurement) is itself new integration work, not merely a dispatch flag.

## Steps

1. CORRECTED per GPT review: `ggml_backend_cuda_buffer_type_alloc_buffer(buft, size)` (ggml-cuda.cu:883) is the WRONG shadow-creation hook -- it has no tensor/name/type argument and runs BEFORE weight upload, so it cannot know which tensor it is allocating for or hold post-upload data. Create shadows only AFTER the load_all_data loop completes (verified: src/llama-model.cpp:1871 calls `ml.load_all_data(...)`, and src/llama-model-loader.cpp:1493 defines `llama_model_loader::load_all_data`) -- shadow population must happen once real weight bytes are resident.
2. Add CUDA-backend shadow ownership keyed by the original `ggml_tensor *`, stored in the existing `struct ggml_backend_cuda_buffer_context` (verified real struct at ggml-cuda.cu:726) rather than inventing a new ownership structure.
3. Populate each shadow using `ggml_get_to_fp16_cuda(src->type)` (verified real function, declared convert.cuh:13, defined convert.cu:547, already used at ggml-cuda.cu:1404 and by conv2d.cu/fattn-common.cuh) -- reuse this exact API rather than writing new dequant math.
4. Define eligibility explicitly: dense (non-MoE-expert) weight tensors only, an explicit opt-in allowlist/pattern with explicit MoE-expert exclusion, never a global default.
5. Wire the DENSE MUL_MAT dispatch to look up a shadow BEFORE the existing `ggml_cuda_should_use_mmq(src0->type, cc, ne11/ne12, n_experts)` decision (verified real calls at ggml-cuda.cu:1872/1899/1940) -- a shadow hit bypasses native MMQ only for eligible dense tensors at/above the configured M threshold; a miss/ineligible tensor falls through to the existing should_use_mmq decision unchanged.
6. Add explicit shadow-buffer cleanup in `ggml_backend_cuda_buffer_context`'s destructor (currently unspecified) so shadow memory is freed with its owning buffer.
7. Validate shadow contents against `ggml_get_to_fp16_cuda`'s own reference output (bit/tolerance match).
8. Measure model-load overhead and VRAM delta for Qwen3.6-27B Q8_0 primary, Q4/Q6 economics controls.
9. Measure PP across M/ubatch 64..4096 and confirm TG/decode-only neutrality (shadow must not regress small-batch decode).
10. Selected-tensor-shadow vs all-eligible-shadow A/B; expose explicit opt-in flag; publish durable candidate identity so PRBE30/PRBE31 can depend on it.

## Detailed Solution & Technical Design

This is new, first-principles work spanning model-load code and the dense GEMM dispatch point. The dequant-to-F16 kernel itself can likely reuse existing per-type dequantize_* device functions already in the codebase (used by MMVQ's own dequant paths) rather than writing new dequant math -- implementer should grep ggml-cuda/dequantize.cu or per-type convert.cu (convert.cu confirmed to exist at b11126) before writing a new dequant kernel. The dispatch-threshold logic (native MMQ vs shadow+GEMM) is new decision code at the ggml_cuda_mul_mat top-level dispatcher.

## Code Samples & Guidance

Real b11126 anchors (verified): ggml-cuda.cu:883 buffer allocation; ggml-cuda.cu:1433 `ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ...)` existing cuBLAS GEMM path (the shadow's eventual consumer, via cuBLAS for now, hipBLASLt in PRBE31); ggml/src/ggml-cuda/convert.cu/.cuh (existing dequant kernels, likely reusable for the shadow-population kernel -- exact function names not read this batch, must be verified). No hipBLASLt symbols found anywhere in ggml-cuda.cu at b11126 (grepped this batch) -- flag explicitly: PRBE29 itself only needs cuBLAS (already present) since its own PP measurement can run through the existing cuBLAS path; hipBLASLt integration is exclusively PRBE31's new-work scope.

## Files

ggml/src/ggml-cuda/ggml-cuda.cu (buffer alloc, dense MUL_MAT dispatch threshold, ~883 and ~1433); ggml/src/ggml-cuda/convert.cu/.cuh (reuse for shadow dequant); model-load path (llama.cpp weight-loading, opt-in flag); patches/12xx_rd36_f16_shadow_dense/ (new, base patch for the chain -- see notes on PRBE30/31 layering via `requires`).

## Validation

Shadow contents vs dequant reference; model output/PPL parity; load-time overhead; VRAM delta; PP across M/ubatch 64..4096; TG/decode-only neutrality control; selected-vs-all-eligible-tensor arms; VRAM-constrained control. dev hardware run not performed here.

## Effort & Risk

L: spans model-load code, a new dequant-at-load kernel invocation, and dense-dispatch threshold logic; foundational for PRBE30/31 so correctness bugs here propagate downstream.

## Standards

Selective shadow; explicit resource accounting; no global default; dependency-aware promotion.

## Acceptance Criteria

A declared target workload shows repeatable PP gain that justifies VRAM/load cost; no global default and no MoE shadow explosion; otherwise retain negative evidence and leave dependents blocked.

## Notes

Supersedes: RD36
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd36

2026-09-24 relevance at b11126: TODO, foundational. Confirmed NO hipBLASLt integration exists at b11126 (only cuBLAS/hipBLAS-compat) -- PRBE29 itself can and should measure through the existing cuBLAS path; PRBE31 alone needs new hipBLASLt integration. GPT design request for PRBE29+30+31 hit a queue-saturated gateway and was not obtained in-session; plan authored directly from verified source, dequant-kernel-reuse anchor flagged as unverified.

2026-09-24 GPT review req_2b717df095b44703 applied: corrected the shadow-creation hook -- ggml_backend_cuda_buffer_type_alloc_buffer has no tensor identity and runs pre-upload, so it cannot be the creation site. Moved shadow creation to after llama_model_loader::load_all_data (verified real call sites in llama-model.cpp:1871/llama-model-loader.cpp:1493), specified ownership in the existing ggml_backend_cuda_buffer_context struct (ggml-cuda.cu:726), specified the exact reusable conversion API ggml_get_to_fp16_cuda (convert.cuh:13/convert.cu:547, verified in use elsewhere), the exact lookup point relative to ggml_cuda_should_use_mmq (verified real calls at ggml-cuda.cu:1872/1899/1940), and added explicit dense-weight allowlist/expert-exclusion/cleanup requirements that were previously unspecified.

## 2026-10-08 b11474 composed-source experiment plan

**Rank 3, conditional on PRBE70 hot dense GEMM signatures and spare VRAM.** For 27B Q8_0 non-expert dense `MUL_MAT` weights, allocate a persistent F16 shadow **after** `llama_model_loader::load_all_data` completes, owned/freed by existing `ggml_backend_cuda_buffer_context`; reuse `ggml_get_to_fp16_cuda`, not a new dequant implementation. At `ggml_cuda_mul_mat` check a per-tensor opt-in shadow only before `ggml_cuda_should_use_mmq` for a measured M crossover; default-off `BIGCHERRY_DENSE_F16_SHADOW=off|allowlist`. VRAM cost approximately **2 bytes × dense weight scalar count plus alignment** (1 billion shadowed scalars ~1.86 GiB, before metadata; a whole 27B shadow is infeasible on 24 GiB cards). With production at 92-97% per-GPU VRAM, select <=256 MiB/device in initial trial or do not run; no tensor duplicated on multiple GPUs unless placement demands it. Hypothesized prefill gain only for repeated M>=512 where GEMM overtakes quantized MMQ; decode/small M uses native unchanged. A/B via `queue-env-ab.sh` with same process-separated baseline, 8K/24K/98K, exact shadow dequant parity, prefill/TG, load-time and per-device VRAM; no claimed gain without measurement.

Composed-source anchor text must be reverified after applying production patches at b11474 before writing any `Edit()`; the historic b11126 offsets in earlier sections are not authoritative.

## Change Log

- 2026-10-08 (triage): Experiment kept pending, priority P3; ranked and scoped b11474 mechanism, VRAM, env switch and queue-env-ab.sh evidence gates; no patch implemented.

- 2026-09-09T10:55:27.064093+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:39.409528+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.258047+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.995120+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:56:51.645822+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025719_dense-gemm-successors-prbe293_6872
- 2026-09-10T02:57:19.678291+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:33:57.527150+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:46:40.269201+00:00 (updated-by): Updated: section:steps, section:notes
