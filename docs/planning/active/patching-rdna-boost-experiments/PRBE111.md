---
id: PRBE111
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-10-04T04:04:36+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# IQ MMVQ decode: collapse duplicate type dispatch and specialize RDNA3/4 inner loops

## Description

**Disposition (2026-10-10):** Keep PRBE111 pending as a *profile-first IQ4_XS/IQ3_XXS decode decision*, not an automatic metadata-hoist or new dispatch implementation. Pinned b11474 already computes IQ4_XS block scale after its unrolled dot loop and IQ3_XXS `aux32` before its loop; two `constexpr` selectors already resolve vec-dot and VDR at template-instantiation time. No measured IQ-specific hot-loop gain exists. The remaining opportunity is conditional on actual gfx1100/gfx1201 kernel/ISA evidence. Do not duplicate 1273's VDR/nwarps experiment or PRBE47's Q6_K reduction work.

## Steps

1. **Source discriminator — completed:** Verify the two `constexpr` selectors, the two IQ vec-dot bodies, three VDR call sites, and existing 1273 selector/variant hooks against pinned b11474. Document implicit IQ1_S/IQ1_M VDR=1 fallback.
2. **No new queue while adjacent experiments run:** PRBE113's Q8_0 width census can supply profiling procedure, but **cannot** establish an IQ4_XS/IQ3_XXS time fraction. Check existing IQ evidence and queued/running experiments first.
3. **IQ-specific admission:** Only if a real eligible IQ4_XS model and idle hardware lane exist, capture one stock single-token `mul_mat_vec_q` profile per gfx1100/gfx1201 (and IQ3_XXS only if warranted): physical kernel count, per-call duration, `SQ_INSTS_VALU`, VMEM/TCC misses, VGPR/SGPR, spills, ISA, and E2E tg128/tg512. Record model hash, quant mix, `ncols_dst`, stream, context, MTP setting and exact patch composition. No inferred IQ timings from Q8_0.
4. **Reject cheaply:** If removable *exposed* IQ MMVQ time is <2.9126% of E2E latency, or if ISA shows no redundant per-iteration metadata operations, close the specialization without patch. For a 3% throughput gain, at least 2.9126% latency must be removed even with zero overhead.
5. **If surviving:** Prototype only the proven instruction-level subexpression, preserving the per-`j` IQ4 table lookups and per-`l0` IQ3 sign shifts. Reuse 1273's existing architecture/launch tuning ownership and evaluate pristine, 1273-only, candidate-only, and composed variants. Do not create another selector/env gate without an existing owner accepting it.
6. **Qualify before promotion:** Compare bit/logit/KLD and greedy/MTP acceptance, multi-request and multi-ubatch, graph replay, gfx1100/gfx1201, and Q4_K/Q6_K/Q8_0 negative controls. Four independent sessions per architecture with >=10 paired ABBA rounds; CI95-low >=3% E2E and no control regression >1%.

## Detailed Solution & Technical Design

### Current implementation and bounded decision (b11474)

- `ggml/src/ggml-cuda/mmvq.cu:40-94`: `get_vec_dot_q_cuda(type)` has **23** explicit type cases; `get_vdr_mmvq(type)` has **21**. Only `IQ1_S` and `IQ1_M` use the intentional `default: return 1`. Both are `constexpr`, so two source switches do **not** imply a runtime double dispatch. A descriptor rewrite is maintenance-only until generated HIP ISA demonstrates otherwise. If ever attempted, preserve all 23 mappings, the implicit VDR1 cases, and compile-time direct calls; reject function-pointer indirection or new runtime lookup.
- `mmvq.cu::mul_mat_vec_q` (~599-850), `mul_mat_vec_q_moe` (~857-990), and `mul_mat_vec_q_switch_ncols_dst` (~1088+) use `get_vdr_mmvq(type)`. The two kernels also use `get_vec_dot_q_cuda(type)`; the MoE multi-token path is a separate dispatch/lifetime/correctness surface. Do not assume single-token results cover it.
- `ggml/src/ggml-cuda/vecdotq.cuh::vec_dot_iq4_xs_q8_1` (~1357-1380) unrolls four *distinct* quant-table loads and eight DP4A calls; its block scale (`ls`, `d`) is already calculated **after** the loop. No repeated scale calculation is available to hoist. The table lookup varies with `j` and cannot be declared invariant without ISA/data proof.
- `vecdotq.cuh::vec_dot_iq3_xxs_q8_1` (~1172-1205) loads `aux32` once **before** its four-iteration loop; each iteration selects distinct grid bytes, sign shift, and q8 words. Merged upstream SWAR intrinsics already replace HIP's bytewise compare/subtract implementation; do not duplicate them.
- `engines/llamacpp/patches/1273_iq_mmvq_rdna_tuning/patch.py` inserts `bigcherry_iq_vec_dot_q_cuda<type,vdr>`, IQ4_XS VDR2/IQ3_XXS VDR1 variants, and two default-off VDR/nwarps environment gates. Its evaluated status is **not** promotion or a measured improvement. A new descriptor that removes/replaces `get_vec_dot_q_cuda` or `get_vdr_mmvq` must first prove 1273's patch anchors, compile-time variants, and default-off behavior still compose. Do not modify 1273 or schedule its four-arm campaign from PRBE111.
- Open upstream [#29025](https://github.com/ggml-org/llama.cpp/pull/29025) identifies out-of-range tail-row loads for MMVQ configurations with rows-per-block >1 and proposes clamping both ordinary and MoE paths. This is a **conditional correctness baseline**, not a reproduced BigCherry crash or evidence that gfx1100/gfx1201 single-token IQ is affected (their ordinary rows-per-block selection may be 1). Before changing geometry, assert actual rows-per-block and tail-row read bounds. Do not treat disappearing invalid work as speedup.

### Upstream and alternative engines

- Merged [llama.cpp #27962](https://github.com/ggml-org/llama.cpp/pull/27962) (2026-09-22) replaces HIP `__vsub4`/`__vcmpne4` with SWAR; the exact implementation is **already present** in b11474 `ggml/src/ggml-cuda/vendors/hip.h`. Its reported ~20% IQ3_S gfx1201 improvement is upstream-author evidence for a *different quant* and is not incremental BigCherry IQ3_XXS data. **Adopt: already included; no port.**
- [SGLang GGUF dispatch](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/gguf.py) uses IQ MMVQ for small batches (threshold 8 or 16 by output rows), and falls back to dequantized matmul for larger IQ batches. It supplies a bounded crossover test concept, not an AMD-qualified kernel or a reason to alter 1273's single-token scope.
- [vLLM GGUF plugin MMVQ](https://github.com/vllm-project/vllm-gguf-plugin/blob/main/vllm_gguf_plugin/csrc/gguf/mmvq.cuh) binds type/VDR/vec-dot at template instantiation using an older b2899-derived kernel. Its IQ4_XS VDR1 differs from b11474 VDR4 and cannot be transplanted without reconciling different quant-index mapping, fusion and launch geometry. **Reject direct port.**

### Cheap discriminator and evidence status

On 2026-10-10, **9/9 pinned-source assertions** passed (selectors, IQ loop invariants, implicit VDR1 cases, 1273 composition touchpoints, existing HIP SWAR). A disposable host arithmetic fixture checked **20,000** deterministic 32-bit input pairs for SWAR `__vsub4` and `__vcmpne4` against four independent byte lanes with **0 mismatches**; a separate **51-case** row-tail fixture demonstrated the conditional rows-per-block bounds issue and clamp behavior. These are host/static checks, not HIP codegen, backend-op tests or GPU benchmarks. **No IQ-specific BigCherry measured gain was found or claimed.**

### Implementation and terminal gate

If the IQ-specific profile proves exposed headroom and an actual redundant instruction sequence, use a default-off architecture-specialized helper at the existing `vecdotq.cuh` call site; retain existing `mmvq.cu` dispatch, quant block layouts, Q8_1 activations, scale placement, fusion/graph behavior, and 1273 controls. Compare identical `ncols_dst=1` baselines before expanding to MoE multi-token or MTP verification. If profiling is unavailable or shows insufficient headroom, mark **BLOCKED_ON_IQ_TRACE** (not a measured failure); if disproved, retire the hot-loop hypothesis. Descriptor-only source cleanup is not a performance optimization and must be independently justified by compile/ISA parity and patch composition.

## Reduction ownership boundary (2026-10-09 / BCOP84)

PRBE111 owns IQ4_XS/IQ3_XXS vec-dot/VDR descriptor consolidation and metadata hoisting only. PRBE47 owns the separate Q6_K wave32 DPP reduction-lowering hypothesis. Both share generic mmvq.cu call sites, including the dedicated multi-token MoE kernel: preserve the stock warp_reduce_sum for IQ and every unqualified type; do not combine the candidates, add a new dispatch table, or claim DPP gains from IQ A/B results. No PRBE111 implementation or queued lane is changed by this documentation cross-link.

## Files

- current pinned `vendor/llama.cpp/ggml/src/ggml-cuda/mmvq.cu`
- MMVQ declarations/helpers and IQ vec-dot implementation
- existing 1273 IQ MMVQ tuning package/table where ownership overlaps
- new patch package only after isolated A/B proves value
- focused mechanics/backend-op test and experiment contract/producer

## Validation

Build HIP for gfx1100+gfx1201. Run backend-op correctness before performance. Benchmark single-GPU decode with identical model/context/settings and paired rounds:

- gfx1201 R9700: IQ4_XS, IQ3_XXS; Q4_K/Q6_K/Q8_0 controls
- gfx1100 RX 7900 XTX: same lanes
- `tg128`, `tg512`; `pp512` as no-regression control only
- at least 10 paired rounds across 4 sessions per architecture

Capture kernel duration, VGPR/SGPR/spills, occupancy and generated ISA/code size where available. Compare baseline, descriptor-only cleanup, and descriptor+hot-loop specialization separately.

## Effort & Risk

Medium. Descriptor consolidation is low source-level risk but must preserve direct-call codegen. IQ metadata hoisting is architecture-sensitive; excess VGPRs are the main performance risk. Keep the experiment isolated from unrelated AllReduce/Q6_K changes.

## Standards

- One source of truth for MMVQ vec-dot/VDR metadata.
- No new runtime type branch in inner accumulation.
- No second tuning/dispatch registry.
- Generic fallback remains unchanged for unproven architectures/types.
- Generated-code/resource evidence accompanies performance claims.

## Acceptance Criteria

- No unconditional descriptor or metadata-hoist patch; current source already hoists the cited scale/aux values.
- Existing 1273 VDR/nwarps tuning and PRBE47 Q6_K reduction remain separate owners; no duplicate runtime registry, new kernel family, or unqualified geometry changes.
- A model- and patch-bound IQ-specific trace proves both an exposed E2E opportunity and actual redundant instructions before any kernel edit; otherwise explicit reject/defer disposition.
- If implemented: exact type/VDR mapping including IQ1 defaults, 1273 composition, full backend/logit/greedy/MTP correctness, tail-row bounds where applicable, multi-request/graph replay, and paired AMD hardware promotion threshold.
- Do not infer gain from upstream SWAR already in b11474 or from SGLang/vLLM/NVIDIA data.

## Notes

Transplanted from `agent-1272-ar-host-wire` before stale-branch cleanup. Branch-specific vendoring/history details were intentionally removed; implementation must rebase against the current `patch-refactor` llama.cpp pin and existing 1273 ownership before coding.

## Change Log

- 2026-10-10 (BCOP125 audit): source/static and host checks reject assumed IQ4_XS scale/IQ3_XXS aux hoisting; preserve 1273 composition; require IQ-specific trace before implementation. No HIP build or GPU measurement.

- 2026-10-08 (triage): Kept pending at P1. Decode shortlist rank #2 after PRBE113 Gate 0. IQ4_XS then IQ3_XXS MMVQ inner-loop metadata hoist, existing vector dispatch descriptor compile-time only, gfx1100/gfx1201; no new registry. Compare kernel fraction and resource use vs PRBE113 baseline; don't start without hot-loop evidence.

- 2026-10-04T04:04:36+11:00 (agent): Original side-branch plan created.
- 2026-10-05: Transplanted to `patch-refactor` as free sequence slot PRBE111; removed stale branch-history assumptions and aligned ownership with current IQ/HIP tuning plans.
