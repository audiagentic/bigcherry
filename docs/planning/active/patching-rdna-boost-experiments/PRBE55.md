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


## 2026-10-08 optimisation audit

### Repository and upstream evidence

Patch 1269 already implements the bounded opt-in route described above and remains `untested`; do not create a second routing patch. The remaining work is qualification and disposition.

Current upstream evidence materially strengthens the discriminator but broadens the suspected problem beyond speculative semantics: llama.cpp issue #21151 reports Q4_K/Q5_K/Q2_K Vulkan MMVQ on gfx1101/RADV running 6.8x-15.3x slower than the existing F32-dequant path for measured hot shapes. Treat those numbers as external evidence only, not expected BigCherry gain. Upstream also disabled MMVQ for an Intel Windows driver class (#20672), establishing precedent for route eligibility being device/driver sensitive rather than universally optimal.

The current BigCherry fleet therefore needs to answer a narrower question: does 1269 remove a material Q8_1-activation/MMVQ penalty for production N=2..8 MTP verification on gfx1100/gfx1201, without harming shapes where MMVQ wins?

### Ownership and implementation boundary

- `1269_prbe55_vk_smalln_dmmv` is the sole implementation owner for this experiment.
- Do not add a second MMVQ selector, shape table, runtime autotuner, or speculative semantic flag.
- Preserve upstream `ggml_vk_should_use_mmvq()` as the baseline owner. 1269 may only override an already-MMVQ-eligible call while its explicit experiment flag is enabled.
- PRBE61 owns Vulkan row-count/rm_kq tuning; do not fold that mechanism into 1269. PRBE68 owns submission batching; it is orthogonal.
- Any future default routing change must be expressed at the existing Vulkan MMVQ eligibility seam and justified by first-party device+driver+type+shape evidence.

### Cheapest discriminator and telemetry

Before a full server campaign, capture the actual production MTP operator mix with the existing Vulkan performance logger plus the 1269 activation marker. Record, per relevant `MUL_MAT`: weight type, N, K/M shape, baseline route, invocation count, and summed wall time. Reject further work if baseline-MMVQ N=2..8 calls are absent or contribute <5% of MTP verify wall time.

For the top two hot qualifying signatures per architecture, run same-binary flag-off/flag-on micro or backend-op controls. The flag-on arm must prove the DMMV route and must not change operation count or tensor shape. A speedup caused by missing work is a correctness failure.

### Hardware matrix and gates

Primary: gfx1100 RADV and gfx1201 RADV using the production MTP model/draft depth. Secondary compatibility only: gfx1030. Keep driver version and shader-cache state recorded.

1. Route proof: N={1,2,8,9} boundary controls plus observed production N values; marker only for 2..8 that baseline would send to MMVQ.
2. Correctness: backend-op reference, fixed-seed greedy output, repeated same-process requests, multi-ubatch prompt, and at least one long-context request. Require identical expected work counts and no NaN/Inf.
3. Performance: paired same-binary flag-off/on, >=10 pairs after warmup. Report kernel/operator wall share and E2E MTP TPS separately.
4. Promotion: CI95 low >=3% E2E MTP TPS, <=1% ordinary-decode regression, and no correctness/work-accounting failure on both gfx1100 and gfx1201.
5. Rejection: <5% eligible wall share, CI crossing zero after the bounded campaign, architecture disagreement requiring a new policy surface, or upstream current master already routes the qualified signatures equivalently.

Do not generalize the external gfx1101 6.8x-15.3x result into a BigCherry default. If BigCherry reproduces a broader K-quant MMVQ defect, open/reassign that broader routing issue to the existing Vulkan eligibility owner rather than expanding PRBE55 beyond small-N qualification.
