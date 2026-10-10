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

## 2026-10-10 authoritative source audit: load-completion and compute-type fence

**Disposition: pending P3, default-off; no implementation or performance qualification.** This audit follows independent Radiance development at `6d4560f15b7c` on `feat/rad11-gfx1100-bootstrap`. PRBE29 and its immediate shadow/dequant path have no independent commit, PR, queued hardware lane or material plan update in the preceding 12 hours. Active Radiance gfx1100 MXFP4, Flash-Next router/QSA, QFP35/41 Meta, QFP43 DFlash and MTP work are protected and untouched. Prior PRBE28 float-row-padding review is a different storage mechanism, not evidence that a quantized shadow works.

### Exact b11474 implementation path and uncovered lifecycle

- `src/llama-model-loader.cpp::load_all_data` (~1600-1760) uses **three different weight paths**: borrowed mmap/host-pointer `ggml_backend_tensor_alloc`, synchronous `ggml_backend_tensor_set`, or pinned-memory chunked `ggml_backend_tensor_set_async` with events. The async path synchronizes its events **after** all tensor chunks; its chunks may be smaller than a tensor. A synchronous-only shadow hook is not an all-loader-path completion callback.
- `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_backend_cuda_buffer_set_tensor` (~780) synchronizes a host-to-device copy; `ggml_backend_cuda_set_tensor_async` (~2531) instead calls `cudaMemcpyAsync` directly on the backend stream and **never calls** the buffer `set_tensor` hook. Borrowed mmap allocations likewise do not invoke it. Therefore AMD-Ecosystem PR #57's `ggml_cuda_maybe_build_f16_shadow` hook inside buffer `set_tensor` cannot guarantee coverage of normal asynchronous loading. It is a **source-proven coverage gap**, not a demonstrated BigCherry runtime failure.
- `ggml_backend_cuda_buffer_context` (~721-735) currently owns only the base device allocation; the fork adds shadow pointers and frees them in this context's destructor. The pinned CUDA implementation has no existing shadow registry or `tensor->extra` consumer. Do not assume `tensor->extra` is globally available or that borrowed/view/shared tensor identities own an independent GPU allocation. Require one immutable (tensor, device, owning buffer) association, no duplicate shadow, and explicit teardown before original weight storage is invalidated.
- `ggml_cuda_mul_mat` (~1910-1958) selects MMVF/MMVQ/MMQ before BLAS. `ggml_cuda_mul_mat_cublas` (~1622-1696) can select **F32 or BF16**, including `GGML_PREC_F32`, `GGML_PREC_BF16`, and `GGML_CUDA_CUBLAS_COMPUTE_TYPE` overrides. An F16 shadow may bypass MMQ **only if the effective BLAS compute type is F16**, not merely because `src0->extra` exists. A skipped MMQ that falls into F32/BF16 conversion does not consume the F16 shadow and invalidates activation/performance attribution.
- The fork's environment parsing treats `GGML_PREFILL_DEQUANT=0` or `off` as **enabled**: non-null `getenv` builds shadows and `atoi <= 1` becomes a 32-token threshold. Do not port that admission unchanged. Parse `off/0` as disabled, reject malformed/negative thresholds, freeze a validated value at model load, and use **actual per-ubatch `ne11`**, not total prompt length. No per-request toggle or mutable policy cache.
- AMD PR #57 allocates a full F16 shadow per eligible 2D quantized dense weight and, when K-padded, an additional temporary unpadded F16 allocation before a 2D copy. Count **peak** original + shadow + temporary + BLAS/graph scratch; use a bounded/tiled temporary or reject the candidate if it does not fit. Its Q4_K_M gfx1151 wins are external, not transferable to gfx1100/gfx1201.

### Smallest safe implementation and ownership contract

**Gate 0 — prove a real hot, fit-safe tensor without code changes.** Reuse PRBE70's typed BLAS/kernel attribution and existing per-device rocprof receipts. Identify one immutable GGUF dense `MUL_MAT` tensor (not `MUL_MAT_ID`, expert stack, view, tied borrowed tensor, offloaded CPU tensor or non-quantized weight), its original device/buffer, quant type, K/N, observed ubatch M, actual MMQ/BLAS route, `GGML_PREC`, measured serialized critical-path contribution, current per-device free/peak VRAM and graph status. Require a baseline where F16 BLAS is numerically admissible. If no eligible hot tensor or optimistic E2E prefill ceiling is below 3%, close `NO_HOT_DENSE`; no patch or GPU A/B.

**Gate 1 — prove exactly-once post-upload finalization, then choose one hook.** Source-backed preferred ordering: allocate original model buffers -> load all chunks/rows -> synchronize **every** upload event -> finalize only selected owned CUDA tensors on their physical device -> dequantize on a controlled stream -> record completion -> publish immutable shadow metadata -> begin graph capture/first eval. Do not allocate or publish a shadow during graph capture or after the first graph references the weight. If an existing loader completion seam cannot safely call an existing CUDA buffer-owned helper, stop at `NO_SAFE_FINALIZE`; do not add a generic backend API, second cache or lazy first-use allocation merely to make the experiment run. The fork's `set_tensor` hook alone fails this gate for async/borrowed modes.

**Gate 2 — one bounded opt-in candidate.** At most **256 MiB per physical GPU** of F16 shadow plus all padding/metadata/temporary/BLAS scratch in the initial trial, with a separate peak-free-memory safety reserve. `2 * n_elements` bytes is the unpadded shadow lower bound: 134,217,728 scalars already consume 256 MiB, while one billion scalars consume 1.863 GiB. Avoid dual-XTX/R9700 cross-device shadow sharing: P2P is unavailable. No gfx1030 promotion without independent qualification. Keep original quantized weight resident for decode and rollback. On allocation failure, preserve native MMQ with no half-built metadata; clean up on model-load cancellation, error and buffer teardown. PRBE30 may later add K-padding **within the same owner**; PRBE31 may compare existing hipBLAS/rocBLAS and `ROCBLAS_USE_HIPBLASLT=1` without first creating a new direct hipBLASLt backend.

Pseudocode (contract, not a compiled patch):

    if !model_load_complete || !all_upload_events_complete: reject
    if !owned_cuda_dense_quant_weight || view || split_alias || unsupported_type: reject
    if !verified_F16_compute || !observed_hot_prefill_M || !safe_peak_budget: reject
    shadow = allocate_in_existing_buffer_owner_once(tensor, physical_device)
    convert_on_owned_stream_after_upload(shadow, original_quantized_bytes)
    synchronize_or_record_completion_before_graph_capture()
    publish_immutable_tensor_device_shadow_only_after_success()
    dispatch_shadow_only_if_current_M >= validated_min_M && effective_compute == F16
    else: preserve_stock_MMVQ_MMQ_BLAS_path()

### Cheapest discriminator and correctness gates

Source inspection found two non-equivalent upload functions and the external `0/off` admission error. A disposable deterministic Python host model passed **12/12** ownership/phase/type/VRAM cases, **4/4** fork-env cases and **2/2** sync-vs-async loader distinctions. These are source-aligned logic fixtures, not a build, patch composition, model inference or GPU benchmark. The host model is disposable and is not a production selector.

Before hardware: fixtures for complete vs partial/async/mmap load, cancelled load, repeated load/unload, duplicate tensor aliases, K-padding boundaries, exact 2D shadow row contents, one- vs multi-ubatch M, malformed env, F16/F32/BF16 effective compute, no-patch fallback and graph allocation/replay. Compare shadow contents against the pinned `ggml_get_to_fp16_cuda` reference (including Q8_0, Q4_K, Q6_K) and logits/greedy/KLD on full-vocabulary deterministic inputs. Verify that decode, MTP acceptance, multi-request same-process and long context do not silently drop or duplicate work.

Only after all gates: single XTX gfx1100 and single R9700 gfx1201; native MMQ baseline, stock transient-conversion BLAS control, one-tensor F16-shadow candidate; M=128/256/512/1024, pp512/2048, ub128/256/512, ctx8K/80K, tg128 and no-shadow MoE/F32/BF16/gfx1030 controls. Record actual converter and GEMM launches, original/shadow/temporary allocation, per-device VRAM peak, load latency, graph nodes, copy bytes, work parity and kernel critical-path time. Require four independent sessions/architecture, >=10 paired ABBA rounds/session, CI95-low >=3% **end-to-end prefill throughput** improvement, <=1% decode/control regression, and numerical/graph correctness. Otherwise `NO_GAIN`, `NUMERIC_FAIL` or `OOM`; preserve negative evidence, do not promote.

### Consolidation and external source disposition

PRBE29 owns **one** quantized dense F16 shadow lifecycle; PRBE30 owns conditional K-padding inside it; PRBE31 owns the measured MMQ/BLAS crossover. PRBE28 owns **float original-weight row padding**, PRBE70 owns offline typed GEMM attribution. No new dispatch table, allocator family, shadow cache, runtime tuner or experiment queue. The external PRs remain **open** as checked 2026-10-10: [AMD-Ecosystem/llama.cpp #57](https://github.com/AMD-Ecosystem/llama.cpp/pull/57) (actual load-time shadow/dispatch diff; gfx1151 Q4_K_M pp1024 +1.7% to +32.2%, pp128 regressions up to -10%; not BigCherry data) and [ggml-org/llama.cpp #26621](https://github.com/ggml-org/llama.cpp/pull/26621) (float-only padding, not a quantized shadow). [AMD ROCm documentation](https://rocm.docs.amd.com/projects/llama-cpp/en/docs-25.09/install/llama-cpp-install.html) confirms `ROCBLAS_USE_HIPBLASLT` is an existing library-selection route; it is not proof of a crossover. [vLLM Zen weight-prepack RFC #35089](https://github.com/vllm-project/vllm/issues/35089) illustrates eager model-load preparation before graph execution but is CPU-only and supplies no RDNA performance evidence. **Measured BigCherry PRBE29 speedup: none.** Earlier broad plans below are historical wherever they conflict with these gates.

## 2026-10-09 PRBE28 ownership reconciliation

**PRBE29 alone owns quantized-weight F16 shadows.** PRBE28's original GGUF row padding is now float-only: never pad Q8_0/Q4_K/Q6_K packed source strides. AMD-Ecosystem/llama.cpp PR #57 implements an existing load-time shadow through `ggml_backend_cuda_buffer_set_tensor`, `tensor->extra`, and buffer-context ownership; inspect complete vs partial/async uploads, ownership, graph and split behaviour before choosing this hook over the plan's post-`load_all_data` alternative. The external gfx1151 performance evidence is not BigCherry qualification. PRBE30 owns any K-padding inside this shadow; no second shadow allocator belongs to PRBE28.


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

Shadow contents vs dequant reference; model output/PPL parity; load-time overhead; VRAM delta; PP across M/ubatch 64..4096; TG/decode-only neutrality control; selected-vs-all-eligible-tensor arms; VRAM-constrained control. Brutus hardware run not performed here.

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
