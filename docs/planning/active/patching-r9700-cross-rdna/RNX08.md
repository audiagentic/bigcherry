---
id: RNX08
order: 8
plan: patching-r9700-cross-rdna
state: pending
created-at: '2026-10-03T01:33:25.968209+00:00'
breadth: ''
skill: advanced
created-by: codex
work: M
priority: P2
---

# R9X08 — PLE, norm+RoPE, and transpose microfusion portfolio

## Description

Harvest independently gated PLE gather/dequant, Qwen-specific norm+RoPE, and transpose/short-conv layout wins without creating a monolithic or duplicate generic fusion.

## Steps

- Reject generic norm+RoPE re-port because 1004 and pinned llama.cpp already cover it; proceed only for a measured Qwen/Gemma-specific dataflow/layout specialization.
- Map PLE rows through Qwen4Exp GET_ROWS/dequant/loading and verify actual GGUF storage before proposing int6.
- Trace short-conv/attention layout round trips and prefer producer/consumer layout fusion over a standalone transpose.
- Give every surviving subfeature its own trace marker, disable path, and acceptance gate; split patch allocation if multiple unrelated features survive.
- Check row IDs before gather and test PLE host/VRAM placement.

## Detailed Solution & Technical Design

Perform tensor/callsite H2D attribution before implementing any PLE optimization. The relevant chain is src/models/qwen4exp.cpp::{llm_graph_input_qwen4exp_ple::set_input,build_inp_ple} -> host hash -> optional llama_prefetch_rows() in src/llama-impl.cpp -> ggml_backend_tensor_set(rows,...) -> ggml_get_rows(per_layer_tok_embd,rows). Instrument bytes, tensor identity, and caller, separating tiny PLE I32 row-index uploads, host-backed per-layer-token-embedding row movement, meta-backend placement/split copies, and unrelated H2D.

## Code Samples & Guidance



## Files

- kernels/r9k_ple.hip
- kernels/r9k_norm_rope.hip
- kernels/r9k_transpose.hip
- r9700_vllm/kernels/ple.py
- r9700_vllm/ple/int6.py
- r9700_vllm/ple/short_conv.py
- src/models/qwen4exp.cpp
- ggml/src/ggml-cuda/rope.cu
- ggml/src/ggml-cuda/rope.cuh

## Validation

Do not infer PLE ownership from aggregate 4.7 s H2D time. Produce a per-tensor/per-callsite attribution and only then compare direct mapped-host/device gather, unique-row packing, or device-side hash/index generation. Keep gfx103x fallback unchanged until gfx1100/gfx1201 evidence proves a portable route.

## Effort & Risk



## Standards

Original source alias is R9X08. Existing owner 1004 remains authoritative for generic RMS_NORM+MUL+ROPE.

## Acceptance Criteria

No PLE cache or gather implementation is added from aggregate traffic alone. The measured H2D attribution identifies the dominant tensor/callsite and distinguishes indices from row payloads before a code proposal is selected; unrelated norm/RoPE traffic is not folded into this item.

## Notes

Proposed slot 1309, split if needed. Current Flash-Next evidence flags PLE H2D traffic (~23/token) for targeted mapping.

Verbatim legacy source retained during R9X→RNX migration:

# R9X08 — PLE, norm+RoPE, and transpose microfusion portfolio

Status: planned
Proposed patch: `1309_r9x_decode_microfusions` (split if more than one subfeature survives)
Depends on: R9X01
External source: `kernels/r9k_ple.hip`, `kernels/r9k_norm_rope.hip`, `kernels/r9k_transpose.hip`, `r9700_vllm/kernels/ple.py`, `r9700_vllm/ple/int6.py`, `r9700_vllm/ple/short_conv.py`
Existing owner: `1004_rms_norm_mul_rope_fusion` plus Qwen4Exp model path



GPT design pass (req_83e7cdc000be4b9e): PLE row indices are host-generated and uploaded, proving some H2D but not proving that PLE dominates the observed traffic. This item is attribution-gated and remains analysis-only.

## Goal

Harvest small memory/launch wins that do not justify their own campaign: Qwen4Exp per-layer embedding (PLE) gather/dequant, model-specific norm+RoPE specialization, and transpose elimination/fusion around short conv/attention. Each subfeature has an independent go/no-go gate.

## Important non-duplication rule

`patches/1004_rms_norm_mul_rope_fusion/patch.py` targets `ggml/src/ggml-cuda/rope.cuh`, `ggml/src/ggml-cuda/rope.cu`, and `ggml/src/ggml-cuda/ggml-cuda.cu`, but is rejected because the generic RMS_NORM+MUL+ROPE fusion already existed in the pinned llama.cpp base. Therefore **do not** re-port r9k norm+RoPE generically. Only proceed if r9k contains a Qwen/Gemma-specific dataflow or output-layout specialization not present upstream and it is a measured hotspot.

## PLE mapping

r9k PLE is an int6 row gather+dequant, including a pinned-host/UVA use case. Map Qwen4Exp PLE/per-layer token embeddings in `src/models/qwen4exp.cpp` to current GGML GET_ROWS/dequant/backend path and model tensor loading. Determine actual GGUF storage format before proposing int6. If BigCherry's deployed model stores PLE in another quant type, port the fused gather+dequant concept to that representation rather than forcing checkpoint conversion.

Security/correctness note from source review: validate all row IDs before gather; do not reproduce a source path that assumes IDs are always in bounds.

## Transpose/short-conv mapping

Use `r9700_vllm/ple/short_conv.py` and `r9k_transpose.hip` only to identify avoidable layout round trips. In llama.cpp, trace the PLE/conv graph in `src/models/qwen4exp.cpp` and backend transpose/contiguous ops. Prefer changing producer/consumer layout or fusing a transpose into an existing kernel over adding a standalone faster transpose.

## Packaging

Do not create a monolithic 1309 if multiple unrelated subfeatures survive. Reserve 1309 for the first validated microfusion; allocate later free patch numbers for additional survivors and update this plan. Every subfeature must have its own trace marker and disable path during evaluation.

## RDNA adaptation

PLE gather/dequant and transpose reduction are strongly portable (`common`) with per-arch vector width/coalescing tuning. Norm+RoPE specialization is also primarily memory-bound, but any source ISA assumptions must be isolated. RDNA2 is a high-value test for these changes because they should not require gfx12 matrix hardware; a failure to port is a warning that the design is over-specialized.

## Change Log

- 2026-10-03T01:33:25.968209+00:00 (created-by): Created by codex
- 2026-10-03T01:39:08.818575+00:00 (updated-by): Updated: section:notes
- 2026-10-03T02:18:35.135369+00:00 (updated-by): Updated: section:detailed_solution, section:validation, section:acceptance_criteria, section:notes

## Reviews

- RV4207
