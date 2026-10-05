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

- 2026-10-04T04:04:36+11:00 (agent): Original side-branch plan created.
- 2026-10-05: Transplanted to `patch-refactor` as free sequence slot PRBE111; removed stale branch-history assumptions and aligned ownership with current IQ/HIP tuning plans.
