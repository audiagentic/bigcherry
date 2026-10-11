---
id: RNX05
order: 5
plan: patching-r9700-cross-rdna
state: pending
created-at: '2026-10-03T01:33:08.311145+00:00'
breadth: ''
skill: advanced
created-by: codex
work: L
priority: P3
---

# RNX05 — MXFP4/FP8 grouped MoE: reuse native GGUF path before any RDNA4 WMMA port

## 2026-10-11 independent MXFP4 implementation audit (authoritative; supersedes legacy 1305/1306 steps)

**Disposition:** pending/P3, *no new patch, GGML quant type, loader, dispatch registry or hardware queue*. The original question "can GGUF represent MXFP4?" is answered **yes at pinned llama.cpp b11474**. The remaining question is whether a distinct gfx1201 FP8-activation WMMA path has measurable, removable cost on a real **MXFP4-weight** model versus the existing Q8_1-activation MMVQ/MMQ paths. No BigCherry MXFP4 performance gain is measured. Do not repurpose the production UD-IQ4_XS model or claim that a new type is required.

### Exact implementation/data-flow facts (pinned b11474, b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea)

1. `ggml/src/ggml-common.h::block_mxfp4` is **32 E2M1 values, 16 packed nibble bytes plus one E8M0 scale byte = 17 bytes (4.25 bits/weight)**. `src/llama-quant.cpp` already implements `LLAMA_FTYPE_MOSTLY_MXFP4_MOE`: MoE expert matrices become `GGML_TYPE_MXFP4`, other tensors predominantly Q8_0. The format, type traits and GGUF quantization are not missing.
2. `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_mul_mat_id` chooses `ggml_cuda_mul_mat_vec_q` for eligible small `ne2`, `ggml_cuda_mul_mat_q` for eligible MMQ, then fallback. `mmvq.cu::get_vec_dot_q_cuda`, `get_vdr_mmvq`, `mul_mat_vec_q_moe` already include MXFP4/Q8_1. `mmq.cu::ggml_cuda_mul_mat_q` and `ggml_cuda_mul_mat_q_switch_type` already include MXFP4. The MMQ `ggml_prec` selector falls back to **Q8 activation on non-Blackwell**; the native FP4 Q4-activation path is Blackwell-specific, *not* an RDNA4 path.
3. `mmq-config-{rdna2,rdna3,rdna4}.cuh` already contain MXFP4 MMQ entries using `GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1`. Existing RDNA kernels are a real baseline on gfx1030/gfx1100/gfx1201; do not invent an MXFP4 fallback.
4. Reference `bkvargyas/r9700-stack@8dfda41f` uses `kernels/r9k_moe_mxfp4a8.hip` and `r9700_vllm/kernels/moe.py`: **load-time fragment-permuted weights**, E8M0 scales in a separate array, per-row E4M3 activation and scale, routing-sorted `sorted_ids/expert_ids` from `moe_align_block_size(16)`, split-K LDS reduction, gfx1201 `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`. The dense `r9k_gemm_fp8.hip` is a **different FP8-weight** format/operation, not a drop-in MXFP4 dense path. Native GGUF interleaves [E8M0,16 packed bytes] per 32 values; the reference kernel's fragment-order weights and separate scales **cannot alias GGUF bytes**. A deterministic two-block host fixture (34 bytes) demonstrated this representation mismatch; this is not a GPU correctness failure.
5. The reference's grouped expert addressing and padded routed-token blocks are not identical to the pinned `ggml_cuda_mul_mat_id` small-MMVQ path, which consumes original IDs. Any port must account for sorting/padding/unsorting and graph-safe scratch before claiming kernel-time gains. A new MoE routing system is explicitly disallowed.

### Consolidation / existing ownership

- RNX05 owns only the *adopt-versus-reject* decision for **gfx1201 MXFP4-weight × FP8-activation** and an independently proven skinny FP8-weight opportunity. RNX01 owns source-intake and capability classification; RNX09 owns cross-architecture integration. `1237/1265` already own compact expert MMQ grid geometry; `1273` owns IQ4_XS/IQ3_XXS VDR/nwarps; `1274` owns Q6_K F32-activation decode. Do not modify these selectors or patch anchors under RNX05.
- Historical proposed patch IDs **1305 and 1306 are occupied** by `1305_ffn_tensor_split` and `1306_qwen4exp_shexp_split`; they are not reserved for RNX05. No patch number is allocated until a discriminator passes.
- A Qwen/Flash-Next **IQ4_XS** GGUF is not an MXFP4 test subject: changing its quantization changes model accuracy and cache/residency economics. First qualify an existing MXFP4-MoE GGUF (e.g. GPT-OSS-20B) with the **same model bytes** in both arms. Any later model-format conversion is a separately accounted quality/VRAM decision.

### Upstream/fork mechanism and disposition

- Merged upstream llama.cpp **#29809** (2026-10-10 02:56 UTC) accelerates **SYCL** MXFP4 MoE using arithmetic nibble decoding plus lazy expert-local [packed-nibbles][E8M0] reordering, with a **32 MiB temporary cap** and generic fallback. Its Intel Arc Pro B70 55.84→106.64 tok/s GPT-OSS-20B decode is **external SYCL evidence, not AMD**. Use its per-expert alignment/temporary-lifetime pattern as a *correctness reference* only; CUDA/HIP already has `vec_dot_mxfp4_q8_1` table lookups and architecture-specific MMQ, so prove redundant cost before porting.
- Open upstream **#28709** is a CDNA N-tiles heuristic; it is not an RDNA4 WMMA qualification or reason to change existing BigCherry geometry. The source fork's reported throughput is on 2×R9700 with different vLLM routing, cache, model and compilation state; do not transfer it to llama.cpp.
- The format's **theoretical weight-only** footprint is 17/32=0.53125 B/weight, half of Q8_0's 34/32=1.0625 B/weight, before model-mix, scale/activation traffic and compute. That is not a 2× E2E ceiling versus IQ4_XS or a speedup measurement. FP8 activation quantization, fragment conversion and sorted routing can erase the arithmetic benefit.

### Implementation-ready bounded gates

**Gate 0 — model and route (host/static first).** Confirm real GGUF expert tensor type `GGML_TYPE_MXFP4`, dimensions/strides, `GGML_TYPE_F32` activation and output, exact `ggml_cuda_mul_mat_id` MMVQ/MMQ/fallback route at M=1/2/4/8/16/32/64/128. Record routed expert count, per-expert token histogram, device, architecture and graph state. Reuse `tools/bigcherry/profiling/rocprof.py` and existing lab kernel traces, not a new telemetry framework. If no real MXFP4 workload or candidate route, close as `NO_DEPLOYABLE_LANE`.

**Gate 1 — cheap format/shape fixture.** Decode 32-value E2M1/E8M0 reference blocks; test E8M0 0/1/254/255 and signed-zero/subnormal boundaries, K%32, expert slice stride, odd expert counts, sorting inverse, repeated IDs and inactive sentinel. Verify a deterministic GGUF AoS→reference fragment-order + separate-scales permutation and exact inverse **without changing source weights**. Host tests must reject wrong expert, stale mapping, non-contiguous views, partial tail and allocation failure. Use the existing native GGML dequant as numerical oracle. Reject on any mismatch.

**Gate 2 — measured exposed opportunity.** For each gfx1201/gfx1100 control collect per-rank critical-path contribution of MXFP4 MMVQ/MMQ, activation Q8_1 quantization, expert routing, host/PCIe copies and synchronization at decode/MTP-verify/prefill. Exclude time hidden by overlap; use same pinned build, GGUF, quant, split, context and MTP settings. If *removable exposed* end-to-end share <2.9126% or existing native route is faster, close `NATIVE_PATH_SUFFICIENT`. No GPU experiment may displace queued QFP41/QFP49 work.

**Gate 3 — only if Gate 2 passes.** Implement a **default-off, gfx1201-only** prototype at the existing `ggml_cuda_mul_mat_id` dispatch boundary. Preserve original native MMVQ/MMQ as fallback. Quantize F32 activations once per call into E4M3+row scales, prepare/load fragment-ordered weights only once per model/device after completed uploads, and reuse existing expert IDs or a proven graph-safe permutation with inverse; never create a second cache, scheduler, allocation policy or expert map. Physical-GPU-local ownership, bounded VRAM budget, explicit upload-completion event, immutable source generation, capture-safe buffers, graph recapture and same-process repeated requests are mandatory. gfx1100/gfx1030 remain native controls unless separately justified. Abort before implementation if host-format parity fails.

**Gate 4 — correctness/performance.** Compare original native MXFP4 path and candidate against independent dequantized F32 reference: per-expert outputs, full-vocabulary logits, KLD, greedy text, MTP acceptance, routing and inactive experts; cover multiple ubatches, long context, graph replay, 1/2/3 GPU no-P2P layouts, tail/rank skew and bytes/transfers/work accounting. Require >=4 independent hardware sessions per architecture under qualification, >=10 paired ABBA rounds/session, **CI95-low >=3% end-to-end improvement**, <=1% regression in native controls, and no quality/correctness regression. Any faster run with missing expert work is invalid. Rejection leaves the native path unchanged.

**Terminal outcome:** close RNX05 as `ALREADY_SUPPORTED_NATIVE` if no distinct FP8-activation win survives, or hand a *single* evidence-backed default-off patch candidate to the existing CUDA MoE owner. Do not keep both the 1305/1306 legacy proposal and a new owner.

## Historical provenance

Originally created 2026-10-03 as R9X05 with proposed 1305/1306 and a speculative new GGUF type. This section supersedes that implementation proposal; upstream format support and the occupied IDs were confirmed at b11474. External reference: https://github.com/bkvargyas/r9700-stack/tree/8dfda41f25f8ff6837cce48bd52b7fd249648f0e ; upstream: https://github.com/ggml-org/llama.cpp/pull/29809 and https://github.com/ggml-org/llama.cpp/pull/28709 .
