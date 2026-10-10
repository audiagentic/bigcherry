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

Reduce duplicated IQ MMVQ type metadata and qualify a narrow RDNA3/4 hot-loop specialization for IQ4_XS / IQ3_XXS decode on gfx1100/gfx1201. The relevant MMVQ path maintains parallel vec-dot/VDR selection and generic per-type plumbing; first consolidate that metadata without changing codegen, then separately test whether block-invariant IQ metadata/scale decode can be hoisted out of the innermost accumulation loop.

Primary target is token-generation MMVQ. Do not broaden into MMQ/prefill or other architectures until hardware evidence exists. This item complements existing IQ tuning work; it must not create a second runtime tuning registry or duplicate generic HIP-autotune ownership.

## Steps

1. Replace duplicated vec-dot/VDR switches with one compile-time descriptor returning both properties. Preserve direct compile-time resolution at hot call sites; no runtime map/table allocation.
2. Verify generated gfx1100/gfx1201 code before treating descriptor consolidation as free. If a function-pointer form prevents devirtualization/direct calls, use a templated descriptor or stop.
3. Add an RDNA3/4-only IQ MMVQ experiment that hoists block-invariant IQ metadata/scale decode outside repeated inner accumulation where current vec-dot code recomputes it. Keep the generic implementation as fallback.
4. Prefer template specialization / `if constexpr` over a new runtime branch inside the per-element loop. No new allocation, host/device transfer or barrier in the decode hot path.
5. Scope specialization to IQ4_XS first. Add IQ3_XXS only if the same helper reduces code rather than duplicating a second kernel body.
6. Benchmark gfx1100/gfx1201 with IQ4_XS/IQ3_XXS plus Q4_K/Q6_K/Q8_0 controls. Gate on correctness, generated-code/resource evidence and end-to-end decode improvement/no-regression.

## Detailed Solution & Technical Design

### Collapse duplicated type metadata

The MMVQ implementation asks independent switches for properties of the same `ggml_type`. Make the pair one compile-time source of truth so function and VDR values cannot drift and the compiler sees a single selection point.

```cpp
struct mmvq_vecdot_desc {
    vec_dot_q_cuda_t fn;
    int vdr;
};

template <ggml_type type>
static constexpr mmvq_vecdot_desc mmvq_vecdot_desc_for();
```

Specializations return the existing vec-dot function and VDR for each supported type. Bind the descriptor once per templated instantiation. A templated form is preferred over a runtime function pointer when required to guarantee direct-call codegen under HIP.

### RDNA IQ hot-loop specialization

For IQ4_XS, inspect the current pinned `vec_dot_iq4_xs_q8_1` implementation and identify metadata derived solely from the quant block and lane/sub-block. Decode/load it once per VDR group and reuse across accumulators. Select only for proven RDNA3/RDNA4 architecture lanes initially.

Avoid shared-memory staging unless register-only hoisting proves insufficient. Decode is latency/bandwidth sensitive; extra barriers can erase the benefit. Reject variants that increase VGPR pressure enough to reduce useful occupancy on gfx1100 or gfx1201.

Keep architecture/path choice in existing tuning/dispatch ownership. PRBE111 owns the experiment and evidence, not a new environment-variable or model-specific selector.

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

- Duplicated MMVQ vec-dot/VDR type metadata is consolidated without losing direct-call codegen.
- IQ4_XS backend correctness passes on gfx1100/gfx1201; IQ3_XXS passes if enabled.
- Descriptor cleanup reduces duplicate source/maintenance surface with no material performance regression.
- Any RDNA specialization improves `tg128/tg512` on at least one target without material regression on the other; Q4_K/Q6_K/Q8_0 controls remain within noise.
- Any specialization is gated to proven architectures and generic fallback remains unchanged.

## Notes

Transplanted from `agent-1272-ar-host-wire` before stale-branch cleanup. Branch-specific vendoring/history details were intentionally removed; implementation must rebase against the current `patch-refactor` llama.cpp pin and existing 1273 ownership before coding.

## Change Log

- 2026-10-08 (triage): Kept pending at P1. Decode shortlist rank #2 after PRBE113 Gate 0. IQ4_XS then IQ3_XXS MMVQ inner-loop metadata hoist, existing vector dispatch descriptor compile-time only, gfx1100/gfx1201; no new registry. Compare kernel fraction and resource use vs PRBE113 baseline; don't start without hot-loop evidence.

- 2026-10-04T04:04:36+11:00 (agent): Original side-branch plan created.
- 2026-10-05: Transplanted to `patch-refactor` as free sequence slot PRBE111; removed stale branch-history assumptions and aligned ownership with current IQ/HIP tuning plans.

## 2026-10-10 authoritative audit — BCOP111: IQ3 VDR rounding gate, no speculative metadata hoist

**Decision:** PRBE111 remains pending only for a bounded numerical/critical-path discriminator. No new IQ MMVQ selector, cache, scheduler, tuning registry or hot-loop patch is authorised. 1273 is the existing `evaluated` (not validated) owner of IQ4_XS/IQ3_XXS VDR/nwarps tuning. Last independent 1273 change found 2026-10-08 15:31 UTC; PRBE111's plan commit was 2026-10-05. Neither received independent work in the last 12 hours; active Radiance, Flash-Next/MTP, QFP35/36/41/43 and collective lanes remain protected.

### Current pinned implementation (llama.cpp b11474)

- `ggml/src/ggml-cuda/mmvq.cu::get_vec_dot_q_cuda` and `get_vdr_mmvq` use separate type switches, but `mul_mat_vec_q<type,...>` and `mul_mat_vec_q_moe<type,...>` bind both at **compile time**. There is no per-token runtime dispatch table to optimise. Consolidation is a maintenance-only candidate; prove identical direct-call ISA, binary size and occupancy before replacing it. Existing 1273 `bigcherry_iq_vec_dot_q_cuda<type,vdr>` already owns its IQ override, so no second descriptor.
- `ggml/src/ggml-cuda/vecdotq.cuh::vec_dot_iq4_xs_q8_1` loads varying quant words/LUT values inside the unrolled four-iteration loop, while block `ls` and `d` are applied **after** it. `vec_dot_iq3_xxs_q8_1` loads `aux32` **before** its unrolled loop and applies `ls`/`d` **after** it. The proposed generic hoist of block-invariant metadata/scale is already present at these obvious sites. No further hoist without emitted-ISA proof of redundant invariant work; moving per-iteration sign/LUT decoding without preserving `iqs` semantics is invalid.
- 1273's opt-in IQ3_XXS VDR1 helper computes `(ls*sumi + sumi/2)/2` for **each half**, whereas native VDR2 accumulates both halves in integer `sumi` and rounds **once**. Signed integer division truncates toward zero; `f(a)+f(b)` is not generally `f(a+b)`. A disposable compiled C++17 `g++ -O2` fixture of 100,000 synthetic half-sum/scale tuples yielded **37,433 mismatches** (max absolute integer delta 1). This is a **source arithmetic counterexample**, not a packed-IQ3, HIP, logits or greedy failure. Do not call the VDR1 quant math unchanged. IQ4_XS VDR2 can also alter FP32 summation order; qualify independently.
- `1273/patch.py` selects only `ncols_dst==1` IQ4_XS/IQ3_XXS on gfx1100/gfx1201. The dedicated multi-token MoE kernel, other types and gfx1030 remain controls. It composes with 0600 geometry and 1241 F32 activation. Its once-per-process `BIGCHERRY_PATCH_TRACE` marker cannot establish per-request/per-type launch counts; use one process per arm/type and existing rocprofv3 trace. BPB01 records no numeric 1273 hardware result; no speedup is established.

### Cheapest-first gate and terminal outcomes

1. **Packed oracle before hardware:** generate valid packed IQ3_XXS GGUF blocks, `qs`, `aux32`, sign/scale and Q8_1 activations; cover all 16 scales, positive/negative/zero half sums, block boundaries and odd tails. Reproduce pristine VDR2 and 1273 VDR1 lane work, C++ signed truncation and float conversion. Compare final float bits and CPU-dequant reference. If different, **reject IQ3 VDR1** without a performance campaign; a successor must combine integer partials before the single rounding, or retain VDR2. Test IQ4 VDR2 packed blocks and floating-point reduction order separately. Nwarps-only remains an independent arm.
2. **Conditional current-pin execution:** only for a numerically admissible arm, use existing 1273 `iq-mmvq` experiment and package mechanics. Confirm actual kernel activation and expected work on isolated gfx1100 XTX and gfx1201 R9700, `tg128/tg512` at 8K/98K, with Q4_K/Q6_K/Q8_0, gfx1030, ncols>1 MoE, fused-gate and pp512 negative/control lanes. Require full-vocab/logits/KLD/greedy/MTP acceptance, graph replay, and multi-request same-process correctness; test flags unset and explicitly 0.
3. **Read-only profiler/ISA census:** record per-type kernel count, duration, VMEM/VALU (available counters only), VGPR/SGPR/spills, code size and direct-call ISA. Compare baseline, VDR, nwarps, both; preserve existing geometry owner. If non-overlapped IQ MMVQ is <**2.913%** of E2E decode wall, even eliminating it cannot reach a 3% speedup: close. If generated ISA already hoists invariants, close the metadata-hoist subfeature without code.
4. **Promotion only after correctness and causal opportunity:** >=4 independent sessions/architecture, >=10 paired ABBA rounds/session, CI95-low >=3% E2E decode, <=1% prefill/other-type controls, no acceptance or graph regression. Otherwise retire the relevant arm. No overlapping experiment queue.

**Ownership:** PRBE111 owns descriptor/hoist disposition, 1273 owns VDR/nwarps implementation, 0600/0650 geometry, 1241 F32 activation, PRBE47 Q6_K DPP, QFP38 other MMVQ dispatch. Do not conflate Q6_K reduction with IQ vec-dot or current Flash-Next MTP work.

**Upstream/forks:** llama.cpp [#27828](https://github.com/ggml-org/llama.cpp/pull/27828) changed sm_60 MUL_MAT_ID MMVQ on 2026-10-10, not RDNA IQ. No upstream `vecdotq.cuh` commit since 2026-10-01 was found. [vLLM GGUF](https://docs.vllm.ai/en/v0.16.0/api/vllm/model_executor/layers/quantization/gguf/) enumerates IQ MMVQ support but provides no qualified AMD drop-in. [SGLang #35019](https://github.com/sgl-project/sglang/issues/35019) is an IQ **prefill/MMQ** gap, not this decode path. No external speedup transferred.

**Actually run:** 11/11 pinned-source structural assertions and compiled C++17 100,000-case signed-integer discriminator (37,433 differing). **Not run:** packed-GGUF oracle, repository pytest, HIP build, model inference, rocprof or GPU benchmark.
