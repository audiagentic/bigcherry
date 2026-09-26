---
id: PRBE55
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:18.934230+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# UP-VK-004: Small-N speculative execution avoids inappropriate MMVQ route

## Description

Qualify a narrowly-scoped Vulkan routing experiment for AMD RDNA: when the existing `ggml_vk_mul_mat_vec_q_f16()` selector would quantize the F32 activation and use MMVQ, optionally force N=2..8 onto its existing non-MMVQ DMMV fallback. No speculative semantic bit reaches this selector at b11126, so the experiment is explicitly shape+device+environment gated rather than pretending to distinguish verify graphs in backend code.

## Steps

1. Add `bigcherry_prbe55_force_dmmv_smalln(device,n)` before `ggml_vk_should_use_mmvq()`.
2. Return true only for `BIGCHERRY_VK_SMALLN_DMMV=1`, AMD vendor, RDNA1/2/3 classifier, and N in [2,8].
3. Leave the baseline `quantize_y = ... && ggml_vk_should_use_mmvq(...)` expression intact, then force it false only when it was already true and the override matches.
4. Emit a once-per-process activation marker after the override fires.
5. Add Q4_K/F32 MUL_MAT backend-op boundary cases for N={1,2,3,4,5,6,7,8,128}.
6. Validate MTP/speculative server throughput with draft max 8 and deterministic greedy token identity; use ordinary decode as the control lane.

## Detailed Solution & Technical Design

Patch `1269_prbe55_vk_smalln_dmmv`, order 1269, no prerequisites. The hook is inside `ggml_vk_mul_mat_vec_q_f16()` immediately after the real b11126 `quantize_y` expression. It does not change `ggml_vk_should_use_mmvq()` and does not touch generic matrix-matrix routing. Setting `quantize_y=false` reuses the function's existing DMMV fallback and avoids the Q8_1 activation quantization for the qualified width. Requiring baseline `quantize_y=true` prevents the experiment from perturbing shapes/types the existing selector already routes away from MMVQ.

The backend lacks a verify-intent signal at this call site; therefore the environment opt-in means "all otherwise-eligible AMD/RDNA N=2..8 vector matmuls" during qualification. Promotion must be based on an MTP lane that proves activation, not an inferred semantic label.

## Code Samples & Guidance

```cpp
bool quantize_y = baseline_predicate && ggml_vk_should_use_mmvq(...);
if (quantize_y && bigcherry_prbe55_force_dmmv_smalln(ctx->device, ne11)) {
    quantize_y = false; // existing fallback selects DMMV
}
```

Do not add a new kernel, loop N independent N=1 launches, or modify N=1 decode behavior.

## Files

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- `tests/test-backend-ops.cpp`
- `patches/1269_prbe55_vk_smalln_dmmv/*`
- `tools/tests/patch/test_1269_prbe55_vk_smalln_dmmv.py`
- `config/experiment-contracts.toml`

## Validation

Offline: patch lint/rebase-check; mechanics tests for default-off gating, N boundaries, idempotence, and anchor failure. Hardware: AMD Vulkan on gfx1100/gfx1201; MTP model `tierM-qwen35b-a3b-moe-mtp`, draft max 8, four independent sessions and ten paired rounds. Positive metric is batched MTP wall TPS (`server_requests_per_start=5`); greedy 64-token identity is required because accepted MTP draft tokens do not expose full-vocabulary rows. Control lane is combined-invocation llama-bench decode. Require subject marker and no control marker. Promotion uses `improvement_no_regression_v1`: target CI95 low > 0 and control CI95 high <=1%.

## Effort & Risk

M / medium. The route is hot but blast radius is bounded by explicit opt-in, vendor/architecture checks, N=2..8, and the existing baseline selector. Primary risk is a workload where MMVQ remains faster at a qualified N; hardware evidence decides promotion.

## Notes

b11126 source audit corrected the earlier placeholder: the real selector is `quantize_y = ... && ggml_vk_should_use_mmvq(...)` in `ggml_vk_mul_mat_vec_q_f16()`, and its vector path supports batched N. N=6/7 are included. No verify-intent parameter is invented. Keep state `pending` until hardware evidence exists.
