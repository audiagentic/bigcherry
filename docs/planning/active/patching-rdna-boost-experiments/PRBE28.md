---
id: PRBE28
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:23.381054+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P2
---

# AMD-GEMM-001: float-only GEMM weight row-stride alias qualification


## 2026-10-18 kernel-dispatch composition gate — 1347 thin-F32 versus row padding (BCOP294)

**Decision: no float-row-padding implementation or hardware lane until it preserves the validated 1347 dispatch.** This is a newly traced **conditional composition hazard**, not an observed BigCherry regression. Upstream [llama.cpp #26621](https://github.com/ggml-org/llama.cpp/pull/26621) remains **open** (last updated 2026-08-08) and defaults row padding only on gfx1151, not gfx1100/gfx1201; [AMD-Ecosystem #57](https://github.com/AMD-Ecosystem/llama.cpp/pull/57) remains **open** (last updated 2026-07-18). BigCherry has **not** enabled either proposal. This audit does not change completed QFP34 or its default-on 1347 kernel.

### Exact production seam and source proof

1. Pinned llama.cpp b11552 `src/models/qwen4exp.cpp` creates `hc_attn_inject` and `hc_ffn_inject` with `"weight"` suffix and dimensions `{hc_dim,hc}`. `src/llama-arch.cpp::llm_tensor_info_for` maps both tensor kinds to `GGML_OP_MUL_MAT`. The validated 1347 activation receipt is **F32 `k=10240 rows=4 cols=512`**, so each packed weight row is `10240*4=40960 B`, exactly `20*2048 B`.
2. The actual [#26621 diff](https://github.com/ggml-org/llama.cpp/pull/26621/files) tags `"weight"` tensors with `GGML_OP_MUL_MAT` or `MUL_MAT_ID` as `GGML_TENSOR_FLAG_PAD_ROWS`; its device padding predicate uses `packed_row % alias_stride == 0` and adds `128 B` per row when enabled. Thus a naive gfx1100/gfx1201 enablement would pad each HC-inject row to **41088 B**, adding **512 B per four-row weight** (logical storage delta only; not measured physical traffic or allocator cost).
3. `ggml/src/ggml.c::ggml_is_contiguous_m_n` rejects a four-row F32 tensor whose `nb[1] != ne[0]*sizeof(float)`. The validated `engines/llamacpp/patches/1347_f32_thin_transposed_mmvf/patch.py::_N_THIN` requires `ggml_is_contiguous(src0)`, `ggml_is_contiguous(src1)`, `ggml_is_contiguous(dst)`, `2 <= ne01 <= MMVF_MAX_BATCH_SIZE(8)`, `ne11>8`, `ne2=ne3=1`, and a valid `ggml_cuda_should_use_mmvf(src1,...,1)` before its role-swapped `ggml_cuda_mul_mat_vec_f` plus **256-thread-block transpose**. Padding `src0` alone disables that route, sending eligible work toward native MMF/BLAS dispatch. Native `ggml_cuda_mul_mat_cublas_impl` can use `nb01/src0_ts` as BLAS leading dimension; **stride-safe BLAS is not equivalent to retaining the faster 1347 path**.
4. AMD fork #57 adds `ggml_cuda_is_contiguous_rows` and a stride-aware native MMVF helper, but that does **not** relax BigCherry 1347's explicit `ggml_is_contiguous(src0)` gate or establish parity for its role-swapped input. Do not copy the helper or widen 1347's gate without proving source and destination indexing, graph/lifetime, numerical and timing equivalence.
5. The #26621 loader's generic `MUL_MAT` weight tag is broader than PRBE28's **actual BLAS-only** performance eligibility. Source-loader metadata cannot decide whether 1347 wins at the physical dispatch site. Per-rank Meta slicing and source upload stride must be accounted for independently; neither a model-level `PAD_ROWS` flag nor a GPU family code alone is a valid promotion receipt.

### Measured versus hypothetical effect

The original b11402 `1347/README.md` Brutus ABBA measured **+3.1–4.6% prefill with MTP**, **+6.8–7.1% without MTP** from 1347 ON versus OFF, at 8K/24K/98K controls; the F32 SGEMM family was 12.7% of one XTX's kernel time before that optimization (including router work). These are historical **1347** results, not a current-pin row-padding test. Loss of the 1347 route is a plausible performance regression, **not a quantified row-padding loss**. The patch uses two launches (MMVF + transpose) versus SGEMM; launch count alone is not a performance discriminator. No current compiled ISA, LDS/VGPR, spills, cache counters or exposed critical-path share for a padded candidate were measured.

### Bounded implementation and test gate (single owner, no new selector)

- **Gate 0 / no patch:** On the current composed b11552 recipe, enumerate **actual physical-device** F16/BF16/F32 `GGML_OP_MUL_MAT` weight names, `ne/nb`, buffer owner, `ggml_is_contiguous`, graph op, chosen post-fusion/post-Meta kernel, architecture, `M`/ubatch, calls and exposed prefill critical-path wall. Record both `BIGCHERRY_F32_THIN_MMVF=1/0` controls. Separate HC-inject `[10240,4]` (and all other 1347-eligible weights) from genuinely BLAS-bound float weights. Reject quantized weights, views, `MUL_MAT_ID`, shared/tied aliases, offloaded tensors and unknown routing. **Do not authorize row padding if it would remove a qualified 1347 hit.** Do not infer exposure from summed GPU-kernel time across concurrent ranks.
- **Cheapest fail-closed selector before a GPU build:** model the exact `packed_row`, `PAD_ROWS`, `nb[1..3]`, `ggml_is_contiguous` and 1347 eligibility predicates for `k=10240,rows=4,cols=512` with pad 0/128; include rows 1/8/9/512, `cols=8/9`, F16/BF16 and quantized negative controls, plus a nonalias F32 row. Require `would_hit_1347_packed => !pad` until a separately verified stride-safe 1347 implementation exists. Validate packed-source-size and 2D-upload bounds, no partial/async load corruption and no graph pointer invalidation. A default-off scoped experiment may be authored **only** after this passes and a real BLAS-bound alias candidate remains.
- **Proposed local admission, not a second dispatch system:** `if (actual_float_weight && verified_BLAS_route && measured_alias_class && !would_hit_1347 && per_device_headroom) { reserve_padded_rows(); upload_packed_rows_2d(); preserve_strided_BLAS(); } else { preserve_packed_source_and_1347_dispatch(); }`. Reuse the upstream loader/buffer metadata and existing `ggml_cuda_mul_mat` dispatch; do not add a new allocator, GEMM selector, scheduler, cache or runtime profile registry. Keep PRBE29/30's quantized F16 shadow independent.
- **Performance admission:** for a genuine remaining float-BLAS alias class, establish a serialized critical-path fraction `f > 1-1/1.03 = 0.029126` (necessary even for infinite kernel speedup), positive L2/cache-set evidence where counters are available, and per-rank VRAM headroom including padded allocation and upload cost. Otherwise terminal `NO_BLAS_FLOAT_CANDIDATE`, `NO_ALIAS_EVIDENCE`, `BELOW_CEILING` or `1347_ROUTE_CONFLICT`, without a patch/hardware queue.
- **Only after admission:** gfx1100 and gfx1201 separately, matched baseline/row-pad controls, 8K/24K/98K ub512 prefill plus tg128/MTP decode; preserve exact 1347 kernel-hit counts, MMVF/transpose work, physical bytes and GPU identities. Verify packed-to-padded row equality, gap canaries, loader mmap/async/direct paths, backend-op F32 tolerance, full-vocabulary logits/KLD/greedy, same-process multi-request, multi-ubatch, graph capture/replay/reallocation and no missing work. At least four independent sessions per architecture and ten paired ABBA rounds/session; promote only CI95-low >=3% **end-to-end** prefill gain, <=1% decode/negative-control regression, no numerical or memory failure. Fallback is unchanged packed 1347/BLAS dispatch.

**Consolidation:** PRBE28 alone owns original float GGUF row padding; completed QFP34/1347 owns the thin-F32 kernel and its existing on/off control; PRBE29/30/31 own quantized-to-F16 shadow, optional padding and BLAS crossover; PRBE70 supplies typed BLAS evidence. No new patch number, duplicate kernel, runtime configuration or queued lane. vLLM's `csrc/rocm/q_gemm_rdna3.cu` uses packed GPTQ W4A16 and deterministic split-K, **not** a compatible F32 GGUF row-padding mechanism; no transferable AMD gain is claimed.

**Validation of this audit:** disposable host predicate/byte model **10/10 dispatch and 4/4 alias/byte assertions passed**. Source inspected: current 1347 patch/README/SUMMARY, pinned b11552 Qwen4Exp/llama-arch/ggml contiguity/BLAS, actual upstream #26621 and AMD #57 diffs, vLLM RDNA3 source. No repository pytest, HIP compile, rocprof, hardware run, or new performance measurement.

## Authoritative audit / disposition (2026-10-09, b11474)

**Supersedes the 2026-10-08 Q8_0-first proposal.** Do not pad packed Q8_0/Q4_K/Q6_K GGUF source weights or assume a 128-byte row pad is a valid quant block stride. PRBE28 is now a conditional **F16/BF16/F32 original-weight** experiment. No BigCherry patch, hardware measurement, or promotion exists.

### Implementation evidence

Pinned llama.cpp b11474 (`b9acf138a1e28ce1fc23`): `src/llama-model-loader.cpp::create_tensor` copies GGUF tensor metadata using `ggml_dup_tensor`; `llama_model_loader::load_all_data` computes `n_size=ggml_nbytes(cur)` and reads/uploads exactly that many packed bytes (mmap, host read or asynchronous staging). `ggml/src/ggml.c::ggml_nbytes` uses `nb[1..3]`. Changing `nb[1]` without separately tracking packed source bytes can overread GGUF, misplace rows, and break mmap/async/progress accounting. `ggml_backend_cuda_buffer_type_get_alloc_size` already reserves quantized **end-of-tensor** MATRIX_ROW_PADDING, which is not per-row padding.

`ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_mul_mat` dispatches MMVF/MMVQ/MMQ before final cuBLAS. `ggml_cuda_mul_mat_cublas_impl` uses `s01=nb01/src0_ts` for direct F16/BF16/F32 GEMM; non-contiguously allocated quantized tensors instead take `traits::convert_nc` into scratch. Thus `nb01` support in cuBLAS does **not** make packed quantized MMQ/MMVQ stride-safe. Do not edit generic `ggml_new_tensor_impl`.

**New concrete upstream baseline:** AMD-Ecosystem/llama.cpp PR #57 (open; updated 2026-07-18) and ggml-org/llama.cpp PR #26621 (open; updated 2026-08-08) implement float-weight row padding through `GGML_TENSOR_FLAG_PAD_ROWS`, backend allocation and `nb[1..3]` updates, packed-to-padded 2D uploads and MMF/MMVF/cuBLAS stride checks. Both explicitly exclude quantized weights. Upstream #26621 enables gfx1151 by default, **not** gfx1100 or gfx1201. AMD #57 also contains a **separate** quantized-to-F16 shadow mechanism owned in BigCherry by PRBE29/30/31; do not transplant that into PRBE28.

External gfx1151 PR #26621 results: Qwen3.5-2B pp2048 +22.6%, Qwen3.5-9B pp2048 +19.1%, Nemotron-3-Nano-4B pp2048 approximately 0%, with some model-buffer costs +100–205 MiB. These are not BigCherry RDNA3/RDNA4 results. A 2048-byte cache alias period and 128-byte line pad are unverified on BigCherry hardware.

### Bounded implementation-ready gate

**Stage 0: no patch, cheapest discriminator.** On a composed pinned b11474 production set, use existing loader/operator-timing evidence (PRBE70) to enumerate actual resident `MUL_MAT` float `src0` weights, type, row_bytes, device, tensor ownership, dispatch and critical-path time. Exclude views, embeddings/GET_ROWS, unsupported split/shard aliases, quantized source weights, and non-GEMM tensors. Begin with `row_bytes % 2048 == 0` as a *hypothesis class*; prove architecture-specific L2 conflict before assuming it is a useful gate. Include nonalias float and Q8_0/Q4_K negative controls.

Compute `f = eligible_float_GEMM_time / total_E2E_prefill_time`. Even infinite speedup of these GEMMs cannot save more than `f` of total prefill (Amdahl ceiling). **Close without implementation if no eligible float weight, no causal L2 conflict, or `f <= 0.03`.** Do not use external gfx1151 gains to clear this gate. Estimate `delta_bytes = sum(nrows*(padded_stride-packed_row_bytes))` plus allocation alignment *per GPU*. Cap initial experiment at 256 MiB extra per device and reject any OOM, graph reallocation or unsafe 92–97%-full VRAM configuration.

**Stage 1: only if Stage 0 passes.** Prefer a minimal adaptation of upstream #26621 rather than a new tensor flag/allocator/dispatch registry. Reuse its proposed `GGML_CUDA_ROW_PAD`, `GGML_CUDA_ROW_ALIAS_STRIDE` and `GGML_CUDA_NO_PAD_WEIGHTS` controls, after verifying the port; default off on gfx1100/gfx1201 pending proof. Exact flow:

1. `src/llama-model-loader.cpp::create_tensor`: after `ggml_dup_tensor`/name, tag only actual float GEMM weights, including duplicated output resolution; never change GGUF file metadata or generic `ggml_new_tensor_impl`.
2. `ggml_backend_cuda_buffer_type_get_alloc_size`: decide per physical GPU and reserve padded bytes **before** allocation. `ggml_backend_cuda_buffer_init_tensor`: set `nb[1]=stride`, `nb[2]=nb[1]*ne[1]`, `nb[3]=nb[2]*ne[2]`, zero gaps, and refuse views/compute buffers and unsupported host-mmap aliases.
3. `llama_model_loader::load_all_data`: keep `packed_row=ggml_row_size(type,ne[0])`, `source_bytes=packed_row*ggml_nrows(cur)` independent of destination `ggml_nbytes(cur)`. For padded GPU tensors upload packed rows with existing `ggml_backend_tensor_set_2d` or a checked equivalent; bypass linear async upload until it is stride-aware. Validate mmap lifetime, direct I/O, row boundary, `size_done/size_data`, and event completion. Never treat packed file mmap as a padded zero-copy tensor.
4. `ggml_cuda_mul_mat`, `ggml_cuda_mul_mat_cublas_impl`, `mmf.cu`, `mmvf.cu`: verify actual stride-aware float dispatch and leading dimensions, including higher-dimension strides and fallback. A padding-induced fast decode-kernel disable is a rejection condition. Keep quantized MMQ/MMVQ and gfx1030 auxiliary control unchanged. No P2P or RCCL assumption for dual XTX/R9700; each device owns its own allocation.

```text
if actual_float_gemm_weight && !view && !quantized &&
   proven_alias_class(device, packed_row) && safe_memory_headroom(device):
    stride = packed_row + qualified_pad
    reserve(stride * nrows); update_nb1_nb2_nb3(); zero_gaps()
    upload_rows_2d(packed_file, packed_row, device_tensor, stride, nrows)
    dispatch_only_to_verified_stride_aware_float_kernel()
else:
    preserve_original_packed_layout_and_dispatch()
```

**Correctness before timing:** host F16/BF16/F32 alias/nonalias, single-row and 3D layout, Q8_0/Q4_K negative controls; prove packed->padded->packed byte equality, zero gaps, exact source-size accounting and flat-copy failure. Then offline patch composition and backend-ops; GPU mmap/direct-I/O/async/duplicate-output/graph replay, repeated same-process and multi-ubatch requests, logits/KLD/PPL and greedy comparisons, work/transfer accounting. Float GEMM tiling may change rounding: require bounded numerical parity and no unexplained token divergence, not unsupported bit-identity claims.

**Performance gate:** isolated gfx1100 XTX and gfx1201 R9700; float positive + quant negative arms, M=128/512/1024, 8K/24K/98K prompt context and tg128 decode. Record actual activation, kernel microseconds, available rocprofv3 L2 counters (UNKNOWN if unavailable), loader cost, per-device peak VRAM, graph allocation and matched work. Four independent sessions per architecture, >=10 paired ABBA rounds each, CI95-low >=3% **E2E prefill** improvement and <=1% decode/control regression. Reject/retire if absent, unsafe, neutral or non-causal. No GPU/build run occurred in this audit.

### Ownership / external mechanisms

PRBE28 owns float GGUF row padding; PRBE29 owns persistent quantized-weight F16 shadows; PRBE30 owns K-padding **inside** those shadows; PRBE31 owns BLAS crossover. Do not create another allocator, scheduler, profile registry or experiment queue. vLLM `csrc/rocm/q_gemm_rdna3.cu` uses format-specific packed weight indexing; its packed-GPTQ layout does not validate arbitrary GGUF quantized row strides or supply AMD row-pad speedups.

Sources:
- https://github.com/ggml-org/llama.cpp/pull/26621 (open, actual loader/backend/MMF/MMVF diff and gfx1151 measurements)
- https://github.com/AMD-Ecosystem/llama.cpp/pull/57 (open, two distinct mechanisms)
- https://github.com/ggml-org/llama.cpp/discussions/26379 (AMD grouping; #34/#37 historical)
- https://github.com/vllm-project/vllm/blob/main/csrc/rocm/q_gemm_rdna3.cu (packed-format contrast)

## Historical design below (superseded where inconsistent)

### Description

TODO, first-principles investigation (no fork source to port), RESCOPED per GPT review: the exact ggml.c anchor exists and is verified, but the generic tensor-creation-site approach is INVALID for weight-only row padding. `ggml_new_tensor_impl` has no weight identity (it is called for every tensor, activations included) and changing only nb[1] there would make GGUF-loaded tensor storage/reads disagree with tightly packed on-disk data (mmap/load code assumes tight packing). Real relevance for gfx1100/gfx1201 (both have set-associative L2) is unchanged; the implementation site must move to the model-loading layer.

### Steps

1. Verified exact anchor: ggml/src/ggml.c::ggml_new_tensor_impl contains `result->nb[0] = ggml_type_size(type);` / `result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));` -- do NOT edit this generic site (it applies to every tensor, not just weights, and changing it breaks tight-packing assumptions for mmap/GGUF-loaded data).
2. Instead, make the weight-specific stride edit in src/llama-model-loader.cpp::create_tensor, immediately after `ggml_set_name(tensor, ggml_get_name(&t_meta));` -- this is the real per-weight creation site with tensor identity available.
3. In load_all_data, anchor `size_t n_size = ggml_nbytes(cur);` and distinguish the tight source bytes (`ggml_nbytes(weight->tensor)`, i.e. the packed GGUF on-disk size) from the padded destination bytes (the new nb[1]); implement a row-wise/2D upload (not a flat memcpy) so each row lands at its padded stride while reading from the tightly-packed source, and disable mmap aliasing for padded tensors (mmap cannot support a stride mismatch between file layout and in-memory layout).
4. Confirm cuBLAS already consumes nb01 as the leading dimension (`s01 = nb01/src0_ts`) -- this means padded strides are consumable by the existing dense MUL_MAT/cuBLAS path (ggml-cuda.cu:1433) without further change there, once the row-wise upload in step 3 is correct.
5. Build a padding-sweep experiment: 0/64/128/256 bytes, grouped by row_bytes/cache-geometry class; keep quantized-packed rows and non-alias-class rows as separate controls.
6. Verify nb[1] propagation, tensor-layout/model-output parity (PPL match) after padding.
7. Capture L2 cache-set-conflict counters via rocprofv3 (causal, not just wall-clock).
8. Measure VRAM delta from padding and reject neutral/costly padding; enable only through an architecture/alias-class selector after repeatable evidence.

### Detailed Solution & Technical Design

This is genuinely new design work with no existing fork/patch to anchor against. The critical open question the implementer must resolve first is WHERE `nb[1]` is actually set for a freshly-loaded weight tensor (ggml.c, not ggml-cuda.cu) -- ggml-cuda.cu:883's buffer allocator only reserves the byte range; padding needs to change per-tensor stride bookkeeping upstream of it. This makes the change touch the CPU-side tensor-creation/model-loading path (llama.cpp's own weight-loading code, not just the CUDA backend), which is a materially bigger blast radius than a pure kernel change and should be flagged to whoever implements it.

### Code Samples & Guidance

Real b11126 anchor (verified): ggml/src/ggml-cuda/ggml-cuda.cu:883 `static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size)`. The tensor-metadata (nb[1]) assignment site in ggml.c was NOT read this batch -- flagged as the first thing the implementer/GPT design pass must locate and verify before any Edit() is written; do not guess this anchor. Patch package sketch (structure only, anchors TBD): patches/12xx_rd35_row_padding/patch.toml (state=untested, plan-item=RD35, kind=enhancement) with an Edit() in ggml.c's tensor-creation path plus an architecture/alias-class selector gate in the model-load path.

### Files

ggml.c (tensor nb[] assignment -- anchor not yet located, must be found first); ggml/src/ggml-cuda/ggml-cuda.cu:883 (buffer allocation, consumer audit); ggml/src/ggml-cuda/ggml-cuda.cu:1433 (dense MUL_MAT/cuBLAS consumer audit); patches/12xx_rd35_row_padding/ (new).

### Validation

Float/BF16/F16 alias-shape identification; padding sweep 0/64/128/256; non-alias/quantized controls; tensor/layout/model-output PPL parity; L2 cache-set-conflict rocprofv3 counters (causal evidence, not just timing); memory overhead accounting; per-op and E2E PP on Brutus (not run here).

### Effort & Risk

M-L: touches CPU-side tensor metadata/model-loading code (larger blast radius than a CUDA-kernel-only change), and the key anchor (nb[1] assignment site) is not yet located -- flag this as the first concrete unknown for whoever picks this item up.

### Standards

Layout-safe; architecture/alias scoped; causal isolation; resource accounting.

### Acceptance Criteria

Padding is selected only for proven alias classes/architectures with model/layout correctness and repeatable benefit outweighing memory cost; no unconditional padding.

### Notes

Supersedes: RD35
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd35

2026-09-24 relevance at b11126: TODO. Confirmed ggml_backend_cuda_buffer_type_alloc_buffer exists at ggml-cuda.cu:883 as the buffer-allocation entry point, but the actual nb[1]-setting tensor-metadata code (ggml.c) was not located this batch -- this is the load-bearing anchor still needing verification before implementation starts. GPT design request submission hit a queue-saturated gateway (8 queued/2 running gateway-wide) and was not obtained in-session; plan authored directly from partial verified source, flagged as needing a follow-up GPT/human pass specifically to locate the ggml.c anchor.

2026-09-24 GPT review req_2b717df095b44703 applied: corrected the implementation site -- the exact ggml.c::ggml_new_tensor_impl anchor exists but editing it is invalid (no weight identity, breaks GGUF/mmap tight-packing assumptions for every tensor including activations). Moved the weight-specific stride edit to src/llama-model-loader.cpp::create_tensor and load_all_data, requiring a row-wise/2D upload that reads tightly-packed source bytes into padded destination stride and disables mmap for padded tensors; confirmed cuBLAS's existing nb01-as-leading-dimension usage needs no further change.

### 2026-10-08 b11474 composed-source experiment plan

**Rank 2 / second slice: bounded row-stride padding experiment.** Target dense `GGML_OP_MUL_MAT`/cuBLAS GEMM with row_bytes/cache-set aliases, beginning with 27B Q8_0 hot projection signatures identified by PRBE70. In composed b11474, re-confirm `src/llama-model-loader.cpp::create_tensor`, `llama_model_loader::load_all_data`, and `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_mul_mat_cublas_impl`; never edit `ggml_new_tensor_impl`. Proposed `BIGCHERRY_DENSE_ROW_PAD_BYTES=0|64|128|256` default 0 and explicit per-tensor allowlist: upload tightly packed GGUF rows to separately allocated padded destinations, update `nb[1]`, disable mmap alias for padded tensors; preserve quant block size/alignment and every unfused read. Incremental VRAM = sum over opted-in tensors of `nrows * (padded_stride - packed_stride)`, plus alignment; record bytes/device before enabling. Production 92-97% VRAM utilization requires hard cap (initial <=256 MiB/device, never OOM/reallocation during graph capture). Hypothesis: measurable L2 conflict reduction and >1% *eligible dense prefill*, but likely null if bandwidth dominates. One-build ABBA with `tools/lab/flash-next/queue-env-ab.sh`: 8K/24K/98K, exact greedy/PPL parity, rocprofv3 L2 miss/conflict and GEMM us, matched decode and VRAM controls. Stop on >1% decode cost or unresolved layout mismatch.

Composed-source anchor text must be reverified after applying production patches at b11474 before writing any `Edit()`; the historic b11126 offsets in earlier sections are not authoritative.


## Change Log

- 2026-10-08 (triage): Experiment kept pending, priority P2; ranked and scoped b11474 mechanism, VRAM, env switch and queue-env-ab.sh evidence gates; no patch implemented.

- 2026-09-09T10:55:23.381054+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:35.177419+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.253885+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.988119+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:55:29.344874+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025542_rdna-successors-prbe2628-now_5552
- 2026-09-10T02:55:42.907095+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:33:31.976968+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:45:45.909867+00:00 (updated-by): Updated: section:description, section:steps, section:notes
