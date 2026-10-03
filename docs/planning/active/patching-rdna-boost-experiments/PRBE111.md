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

The latest substantive branch `agent-1272-ar-host-wire` ends at `cd35a501` and deliberately vendors pristine b11233 `mmvq.cu`, `mmvq.cuh`, and `vecdotq.cuh` for IQ4_XS / IQ3_XXS decode work on gfx1100/gfx1201. The imported MMVQ path still maintains two parallel type switches (`get_vec_dot_q_cuda()` and `get_vdr_mmvq()`) and routes IQ decode through generic per-type function-pointer/VDR plumbing. Use that lab anchor to reduce dispatch code and test a narrow RDNA3/4 IQ specialization that removes invariant work from the hot dot-product loop.

Primary targets are IQ4_XS and IQ3_XXS token generation on gfx1100/gfx1201. Do not broaden into MMQ/prefill or alter other architectures until hardware evidence exists.

## Steps

1. Replace the duplicated vec-dot/VDR switches with one constexpr descriptor lookup returning both values. Preserve compile-time resolution at every existing call site; no runtime map/table allocation.
2. Add an RDNA3/4-only IQ MMVQ experiment that hoists block-invariant IQ metadata/scale decode outside the innermost `iqs` accumulation where the current vecdot implementation repeats it. Keep the generic implementation as fallback.
3. Prefer template specialization / `if constexpr` over a new runtime branch inside the per-element loop. The generated gfx1100/gfx1201 hot path must have no extra type dispatch, allocation, or host/device copy.
4. Keep package scope to IQ4_XS first; enable IQ3_XXS only if the same helper reduces code rather than duplicating a second kernel body.
5. Add contract lanes for gfx1100 and gfx1201 with IQ4_XS/IQ3_XXS plus Q4_K/Q6_K/Q8_0 controls. Gate acceptance on both generated-code reduction and decode improvement/no-regression.

## Detailed Solution

### Collapse duplicated type metadata

`mmvq.cu` currently asks two independent switches for properties of the same `ggml_type`. Make the pair a single compile-time object so adding/changing an IQ type cannot desynchronise the function and VDR definitions and the compiler sees one selection point.

```cpp
struct mmvq_vecdot_desc {
    vec_dot_q_cuda_t fn;
    int vdr;
};

static constexpr __device__ mmvq_vecdot_desc get_mmvq_vecdot_desc(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ4_XS:
            return { vec_dot_iq4_xs_q8_1, VDR_IQ4_XS_Q8_1_MMVQ };
        case GGML_TYPE_IQ3_XXS:
            return { vec_dot_iq3_xxs_q8_1, VDR_IQ3_XXS_Q8_1_MMVQ };
        // existing supported types: one row each
        default:
            return { nullptr, 1 };
    }
}
```

Callers bind `constexpr/const auto desc` once and use `desc.fn` / `desc.vdr`. If HIP fails to devirtualize the function pointer for a hot instantiation, stop: code-size cleanup must not trade away direct-call codegen. A templated `mmvq_vecdot_desc<type>()` is preferred if needed to guarantee direct calls.

### RDNA IQ hot-loop specialization

For IQ4_XS, inspect `vec_dot_iq4_xs_q8_1` in the vendored b11233 `vecdotq.cuh` and identify metadata derived solely from the quant block and lane/sub-block. Decode/load it once per VDR group, then reuse it across the accumulator operations. The specialization should be selected only for `RDNA3_0`/`RDNA4` initially. Follow the existing 1204 architecture-selection convention already referenced by the branch lab README.

Do not introduce shared-memory staging unless register-only hoisting is insufficient: decode is bandwidth/latency sensitive and extra barriers would likely erase the gain. Reject any variant that increases VGPR pressure enough to reduce occupancy on either gfx1100 or gfx1201.

## Files

- `lab/iq-mmvq/mmvq.cu`
- `lab/iq-mmvq/mmvq.cuh`
- `lab/iq-mmvq/vecdotq.cuh`
- new patch package after lab A/B proves value, using the next free package ID
- focused mechanics test and experiment contract/producer

## Validation

Build HIP fat binary for gfx1100+gfx1201. Run backend-op correctness before performance. Benchmark single-GPU decode with identical model/context/settings and paired rounds:

- gfx1201 R9700: IQ4_XS, IQ3_XXS; Q4_K/Q6_K/Q8_0 controls
- gfx1100 RX 7900 XTX: same lanes
- `tg128`, `tg512`; add `pp512` only as a no-regression control
- at least 10 paired rounds across 4 sessions per architecture

Capture kernel duration plus VGPR/occupancy where available. Acceptance: bit-identical/correct within existing contract, net source LOC reduction from descriptor consolidation, IQ decode improvement on at least one target with no material regression on the other, and controls within noise. If only descriptor consolidation survives, land it separately as cleanup and leave the kernel experiment untested/rejected as appropriate.

## Effort & Risk

Medium. Descriptor consolidation is low risk but must preserve direct-call codegen under HIP. IQ metadata hoisting is architecture-sensitive; excess VGPRs are the main performance risk. Keep the experiment isolated from the active AllReduce work and from 1204 Q6_K changes.

## Acceptance Criteria

- One source of truth for MMVQ vec-dot function + VDR metadata; duplicated type switches removed.
- No new runtime type branch in the inner MMVQ accumulation loop.
- IQ4_XS correctness passes on gfx1100 and gfx1201.
- Net source LOC reduction for dispatch cleanup.
- `tg128/tg512` improves on at least one RDNA3/4 target and does not materially regress the other; Q4_K/Q6_K/Q8_0 controls remain within noise.
- Any specialization is gated to proven architectures and generic fallback remains unchanged.

## Notes

Chosen because the branch tip explicitly introduced pristine MMVQ/vecdot sources for IQ decode optimization, making this the freshest substantive optimization area rather than extending the unrelated AllReduce planning branch. The proposal combines a guaranteed maintainability/code-reduction target with a separately gated hot-kernel optimization so a failed performance experiment does not block the cleanup.

## Change Log

- 2026-10-04T04:04:36+11:00 (agent): Created from latest substantive branch scan; target IQ MMVQ dispatch deduplication and RDNA3/4 invariant-hoist experiment.
