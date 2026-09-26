---
id: PRBE110
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-23T09:19:09.142442+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Extract RD05 and RD07 from rejected 1203 as fresh untested packages

## Description

Split rejected `1203_rd050607_rdna4_wmma_fa_q6k_mmq` into two independently identifiable packages. `1266_rd05_wmma_fa_tileq_sync` contains only RD05's WMMA flash-attention `tile_Q` reuse synchronization. `1267_rd07_q6k_mmq_scale_fold` contains only RD07's Q6_K MMQ scale hoist/fold plus its bounded tuning/test support. RD06 configuration expansion remains excluded. Neither package inherits 1203's receipt; each has a fresh contract and must collect fresh hardware evidence.

Source prior art is fork commit `1d525bd45f9e8f844856ecbc5dd8ae33c8d34eff`. The extraction uses current b11126-compatible anchors, including the corrected Q6_K dispatch call with no `forced_J` argument.

## Steps

1. Package 1266: move only `rd05-k00-sync` and `rd05-kbc-sync`; add the required `<atomic>` include and a standalone RD05 activation marker at `BEST_FATTN_KERNEL_MMA_F16` dispatch. Exclude RD06 table/gating/softcap edits.
2. Package 1267: move `rd07-hoist-base-scale`, `rd07-fold-subscale`, `rd07-sum-line` in that order; add RD07-only Q6_K dispatch marker, `GGML_CUDA_MMQ_J_MAX`, and RD07-only MUL_MAT perf cases. Exclude FA perf cases and per-op timing diagnostics.
3. Give each package `state="untested"`, fresh experiment-contract identity, explicit `expect_matches=1`, distinctive guards, and conflict with rejected 1203 because source edits overlap.
4. Add package-local producers using contract-declared paired rounds and measurement mode. Llama-bench lanes use one combined invocation per paired round.
5. Add focused mechanics tests for apply, idempotence, missing-anchor failure, and ambiguous-anchor failure.
6. Keep this plan `pending` until owner hardware validation completes.

## Detailed Solution

### 1266 / RD05

`fattn-mma-f16.cuh` receives exactly two barrier changes. The first extends the end-of-`k00` synchronization condition so writers cannot overwrite `tile_Q` while another warp still consumes it. The second inserts an unconditional block synchronization before the next `process_tile` iteration reuses the same scratch. No kernel configuration, head-size envelope, or softcap behavior changes.

`fattn.cu` receives only `<atomic>` plus a `BIGCHERRY_PATCH_TRACE` once-marker immediately before the actual WMMA-F16 launch. The marker proves the fixed implementation was dispatched; it does not claim that the race manifested.

### 1267 / RD07

`mmq-vec-dot.cuh` hoists row base scales out of inner loops, loads per-`k01` subscales once, folds them to F32, and uses the folded scale in the hot accumulation line. The three edits are intentionally ordered because later anchors depend on text inserted by earlier edits. The sum anchor accepts the two real base forms (plain and the independent 1006 cast form) but still requires exactly one match.

`mmq.cu` adds only the Q6_K MMQ dispatch marker. At b11126 the call is `mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);`; no `forced_J` parameter exists and the rejected package's older anchor must not be copied. `mmq.cuh` keeps the env-bounded J-max sweep knob. `tests/test-backend-ops.cpp` receives only Q6_K MUL_MAT performance shapes plus Q8_0/F16/F32 controls; RD05/RD06 flash-attention cases are excluded.

## Code Samples

1266 synchronization guard:
```cpp
if (np > 1 || k00 + nbatch_combine < DV/2) {
    __syncthreads();
}
```

1267 corrected dispatch shape:
```cpp
case GGML_TYPE_Q6_K: {
    // trace marker
    mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
    break;
}
```

Fresh contracts:
```toml
[contract.PRBE110-RD05-WMMA-TILEQ-SYNC.measurement]
bench_invocation = "combined"

[contract.PRBE110-RD07-Q6K-MMQ-SCALE-FOLD.measurement]
bench_invocation = "combined"
```
Both use `improvement_no_regression_v1`, `min_sessions=4`, `min_paired_rounds=10`, and zero materiality threshold per owner policy.

## Files

- `patches/1266_rd05_wmma_fa_tileq_sync/{patch.toml,patch.py,README.md,SUMMARY.md,validation.toml,validation/producer.py}`
- `patches/1267_rd07_q6k_mmq_scale_fold/{patch.toml,patch.py,README.md,SUMMARY.md,validation.toml,validation/producer.py}`
- `tools/tests/patch/test_1266_rd05_wmma_fa_tileq_sync.py`
- `tools/tests/patch/test_1267_rd07_q6k_mmq_scale_fold.py`
- `config/experiment-contracts.toml`
- source targets: `ggml/src/ggml-cuda/fattn-mma-f16.cuh`, `ggml/src/ggml-cuda/fattn.cu`, `ggml/src/ggml-cuda/mmq-vec-dot.cuh`, `ggml/src/ggml-cuda/mmq.cu`, `ggml/src/ggml-cuda/mmq.cuh`, `tests/test-backend-ops.cpp`

## Validation

Offline owner run: patch lint; mechanics tests; standalone apply/idempotence; missing/duplicate-anchor negative tests; rebase check against b11126 and production composition; build control/subject.

Hardware: 1266 on gfx1201 must show subject-only WMMA marker, full-vocabulary backend-reference correctness, positive decode and prefill control measurements over 10 paired rounds for each of at least 4 sessions. 1267 must do the same on each declared gfx1100/gfx1201/gfx1030 architecture, with Q6_K MMQ prefill as positive and decode as control. Historical 1203 results are provenance only and cannot satisfy either fresh contract.

## Effort & Risk

M. Code risk is low-to-medium because source logic is already reviewed, but split/lifecycle risk is material. Primary hazards: accidentally carrying RD06 edits; losing RD07 edit ordering; copying 1203's stale `forced_J` dispatch anchor; accepting an ambiguous scale-fold anchor; or reusing rejected evidence. All are fail-closed by package scope, exact match counts, mechanics tests, and fresh contracts.

## Notes

- 1203 remains rejected and is not a compatibility fallback.
- RD06 remains parked; neither new package broadens WMMA head coverage.
- Fresh contracts are intentional even though logical RD05/RD07 contract names already exist: rejected-bundle evidence must not bind the extracted patch identities.
- Source: https://github.com/stew675/llama.cpp/commit/1d525bd45f9e8f844856ecbc5dd8ae33c8d34eff
