---
id: BCOP27
order: 27
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T09:06:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# Isolate the #29612 AMD Flash-Attention swizzle residual

## Description

Action ledger for the 2026-10-05 QFP22 optimisation audit. Technical authority remains QFP22/patch 1333. Existing b11402 bisect evidence places the remaining 27B 32K prefill regression (~0.3%) between 0eb6d9a81 and 2ca15f540, where #29612 is the only GPU-path change relevant to this HIP workload. The upstream #29612 diff is a swizzle/load-address refactor in `ggml/src/ggml-cuda/fattn-mma-f16.cuh` plus shared MMA helpers; it replaces explicit `swz` template dispatch with stride-derived `swizzle<stride_tile>` / `load_ldmatrix_*_swizzled` helpers. This is a hypothesis boundary, not yet an isolated #29612 A/B.

## Actions

1. Do not create a broad revert. Build one b11402+1333 diagnostic variant that restores only the pre-#29612 AMD WMMA/MFMA load/address helper path under `AMD_WMMA_AVAILABLE || AMD_MFMA_AVAILABLE`; NVIDIA/Turing swizzle behavior must remain current upstream.
2. Keep tile geometry, `nbatch_K2/V2`, stage count and non-AMD code identical. The diagnostic must change only shared-memory address/load formulation so the A/B identifies the mechanism rather than the whole PR.
3. Add compile-time/static checks for the tested DKQ/DV/ncols configurations proving old and new helpers address the same logical K/V tile elements and stay within `stride_tile_{K,V}`. Remove the diagnostic if this equivalence cannot be stated cleanly.
4. Build HIP for gfx1100 and gfx1201. Run existing correctness first: `test-backend-ops` FA coverage plus greedy 27B/Flash-Next text/logit control. No performance claim is valid after any correctness difference.
5. ABBA on dual-XTX 27B at 10K and 32K prefill, >=4 samples/arm, same ubatch/context/patch set. Add one long-context Flash-Next prefill lane because QFP22 predicts an FA load-path cost should scale there. Record FA kernel time with rocprof where available.
6. Promote a narrow AMD helper compatibility patch only if 32K whole-prefill improves >=0.25% with complete sample separation, 10K does not regress >0.2%, FA kernel timing moves in the same direction, and correctness is unchanged. Otherwise classify #29612 as accepted upstream/no local patch and close this residual.
7. If the diagnostic recovers the residual, upstream the AMD-specific result rather than maintaining a second swizzle framework. Any permanent implementation must reuse the existing `fattn-mma-f16.cuh` helper ownership; no BigCherry dispatch table is permitted.

## Code-level target

- `ggml/src/ggml-cuda/fattn-mma-f16.cuh`: `flash_attn_ext_f16_load_tile`, `flash_attn_ext_f16_iter`, K/V `load_ldmatrix*_swizzled` sites and stride/swizzle selection.
- Shared MMA helper touched by #29612: inspect only to support the AMD-gated compatibility helper; do not fork generic swizzle policy.
- `docs/planning/active/patching-qwen-flash-next/QFP22.md`: authoritative measurements/disposition.

Minimal diagnostic shape:

```cpp
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    // qualification only: use the pre-#29612 row-padded AMD load/address expression
    load_tile_amd_legacy_addressing(...);
#else
    // current upstream #29612 path
    load_tile_current_swizzled(...);
#endif
```

This is intentionally a disposable discriminator. Do not retain both helpers after qualification.

## Related / consolidation

QFP22 owns the b11402/1333 pin-regression evidence and final disposition. PHA08 owns D=72 FA correctness; do not mix that aperture investigation into this performance A/B. HIP-autotune owns any architecture/shape crossover if one is ultimately demonstrated. BCOP27 owns only follow-through on the #29612 residual.

Fresh upstream release b11403 (`9d3aba6`, 2026-10-05) is a CUDA thin-f16/bf16 MMVF change and does not supersede this HIP FA investigation. Open #29963/#29958/#29953 remain orthogonal.

## Acceptance Criteria

- #29612 is isolated with an AMD-only helper-level A/B rather than a whole-PR revert.
- gfx1100/gfx1201 HIP builds and FA correctness pass before benchmarking.
- 10K, 32K and one long-context Flash-Next lane have controlled ABBA evidence; FA kernel timing is captured where available.
- Outcome is one of: narrow AMD compatibility patch justified, upstream accepted/no local patch, or residual attributed elsewhere with evidence.
- Diagnostic code is removed if rejected; no duplicate swizzle/FA dispatch mechanism remains.
