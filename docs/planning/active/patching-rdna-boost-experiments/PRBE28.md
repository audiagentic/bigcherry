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
priority: null
---

# AMD-GEMM-001: 128-byte row padding for cache-set aliasing

## Description

TODO, first-principles investigation (no fork source to port), RESCOPED per GPT review: the exact ggml.c anchor exists and is verified, but the generic tensor-creation-site approach is INVALID for weight-only row padding. `ggml_new_tensor_impl` has no weight identity (it is called for every tensor, activations included) and changing only nb[1] there would make GGUF-loaded tensor storage/reads disagree with tightly packed on-disk data (mmap/load code assumes tight packing). Real relevance for gfx1100/gfx1201 (both have set-associative L2) is unchanged; the implementation site must move to the model-loading layer.

## Steps

1. Verified exact anchor: ggml/src/ggml.c::ggml_new_tensor_impl contains `result->nb[0] = ggml_type_size(type);` / `result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));` -- do NOT edit this generic site (it applies to every tensor, not just weights, and changing it breaks tight-packing assumptions for mmap/GGUF-loaded data).
2. Instead, make the weight-specific stride edit in src/llama-model-loader.cpp::create_tensor, immediately after `ggml_set_name(tensor, ggml_get_name(&t_meta));` -- this is the real per-weight creation site with tensor identity available.
3. In load_all_data, anchor `size_t n_size = ggml_nbytes(cur);` and distinguish the tight source bytes (`ggml_nbytes(weight->tensor)`, i.e. the packed GGUF on-disk size) from the padded destination bytes (the new nb[1]); implement a row-wise/2D upload (not a flat memcpy) so each row lands at its padded stride while reading from the tightly-packed source, and disable mmap aliasing for padded tensors (mmap cannot support a stride mismatch between file layout and in-memory layout).
4. Confirm cuBLAS already consumes nb01 as the leading dimension (`s01 = nb01/src0_ts`) -- this means padded strides are consumable by the existing dense MUL_MAT/cuBLAS path (ggml-cuda.cu:1433) without further change there, once the row-wise upload in step 3 is correct.
5. Build a padding-sweep experiment: 0/64/128/256 bytes, grouped by row_bytes/cache-geometry class; keep quantized-packed rows and non-alias-class rows as separate controls.
6. Verify nb[1] propagation, tensor-layout/model-output parity (PPL match) after padding.
7. Capture L2 cache-set-conflict counters via rocprofv3 (causal, not just wall-clock).
8. Measure VRAM delta from padding and reject neutral/costly padding; enable only through an architecture/alias-class selector after repeatable evidence.

## Detailed Solution & Technical Design

This is genuinely new design work with no existing fork/patch to anchor against. The critical open question the implementer must resolve first is WHERE `nb[1]` is actually set for a freshly-loaded weight tensor (ggml.c, not ggml-cuda.cu) -- ggml-cuda.cu:883's buffer allocator only reserves the byte range; padding needs to change per-tensor stride bookkeeping upstream of it. This makes the change touch the CPU-side tensor-creation/model-loading path (llama.cpp's own weight-loading code, not just the CUDA backend), which is a materially bigger blast radius than a pure kernel change and should be flagged to whoever implements it.

## Code Samples & Guidance

Real b11126 anchor (verified): ggml/src/ggml-cuda/ggml-cuda.cu:883 `static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size)`. The tensor-metadata (nb[1]) assignment site in ggml.c was NOT read this batch -- flagged as the first thing the implementer/GPT design pass must locate and verify before any Edit() is written; do not guess this anchor. Patch package sketch (structure only, anchors TBD): patches/12xx_rd35_row_padding/patch.toml (state=untested, plan-item=RD35, kind=enhancement) with an Edit() in ggml.c's tensor-creation path plus an architecture/alias-class selector gate in the model-load path.

## Files

ggml.c (tensor nb[] assignment -- anchor not yet located, must be found first); ggml/src/ggml-cuda/ggml-cuda.cu:883 (buffer allocation, consumer audit); ggml/src/ggml-cuda/ggml-cuda.cu:1433 (dense MUL_MAT/cuBLAS consumer audit); patches/12xx_rd35_row_padding/ (new).

## Validation

Float/BF16/F16 alias-shape identification; padding sweep 0/64/128/256; non-alias/quantized controls; tensor/layout/model-output PPL parity; L2 cache-set-conflict rocprofv3 counters (causal evidence, not just timing); memory overhead accounting; per-op and E2E PP on Brutus (not run here).

## Effort & Risk

M-L: touches CPU-side tensor metadata/model-loading code (larger blast radius than a CUDA-kernel-only change), and the key anchor (nb[1] assignment site) is not yet located -- flag this as the first concrete unknown for whoever picks this item up.

## Standards

Layout-safe; architecture/alias scoped; causal isolation; resource accounting.

## Acceptance Criteria

Padding is selected only for proven alias classes/architectures with model/layout correctness and repeatable benefit outweighing memory cost; no unconditional padding.

## Notes

Supersedes: RD35
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd35

2026-09-24 relevance at b11126: TODO. Confirmed ggml_backend_cuda_buffer_type_alloc_buffer exists at ggml-cuda.cu:883 as the buffer-allocation entry point, but the actual nb[1]-setting tensor-metadata code (ggml.c) was not located this batch -- this is the load-bearing anchor still needing verification before implementation starts. GPT design request submission hit a queue-saturated gateway (8 queued/2 running gateway-wide) and was not obtained in-session; plan authored directly from partial verified source, flagged as needing a follow-up GPT/human pass specifically to locate the ggml.c anchor.

2026-09-24 GPT review req_2b717df095b44703 applied: corrected the implementation site -- the exact ggml.c::ggml_new_tensor_impl anchor exists but editing it is invalid (no weight identity, breaks GGUF/mmap tight-packing assumptions for every tensor including activations). Moved the weight-specific stride edit to src/llama-model-loader.cpp::create_tensor and load_all_data, requiring a row-wise/2D upload that reads tightly-packed source bytes into padded destination stride and disables mmap for padded tensors; confirmed cuBLAS's existing nb01-as-leading-dimension usage needs no further change.

## Change Log

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
