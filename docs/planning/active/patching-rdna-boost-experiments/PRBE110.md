---
id: PRBE110
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-23T09:19:09.142442+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# RD07 Q6_K MMQ scale-fold — gfx1201-only qualification; RD05 retired

## Description

Fresh package evidence resolves the two extracted slices differently.

- **1266 / RD05 WMMA tile_Q synchronization: reject/retire.** Four gfx1201 sessions prove activation and full-vocabulary correctness but do not establish a gain. The completed contract fails: aggregate target point estimate -0.0499%, CI95 low -0.1609%. No stock correctness failure was reproduced, so more performance sessions are not justified without new race evidence.
- **1267 / RD07 Q6_K MMQ scale fold: continue only on gfx1201.** Four gfx1201 sessions pass with pp512 +2.4675%, CI95 [2.2776%, 2.6508%], while tg128 control is -0.0331%, CI95 [-0.0592%, -0.0071%]. Completed gfx1100 and gfx1030 evidence fails to establish a gain.

This item now owns one bounded decision: prove that 1267 still applies and retains its gfx1201 benefit on b11474, then promote through existing architecture dispatch or retire it. It owns no new MMQ selector, tuning table, or generic Q6_K implementation.

## Steps

1. Run current-pin patch lint/rebase, 1267 mechanics tests, and standalone composition with 0300/1006.
2. Compare materialized control/subject source. If upstream already performs equivalent scale folding, mark 1267 upstream-absorbed and stop.
3. Otherwise run exactly one current-pin gfx1201 campaign with the existing 1267 producer. Do not rerun gfx1100/gfx1030.
4. Require subject-only activation, full-vocabulary correctness, >=4 independent sessions and >=10 paired rounds/session.
5. Promote only if pp512 CI95-low >=2.0% and tg128 regression <=1.0%; otherwise retire unchanged.

## Repository / code path

- `ggml/src/ggml-cuda/mmq-vec-dot.cuh`: Q6_K row-scale hoist/fold.
- `ggml/src/ggml-cuda/mmq.cu`: existing Q6_K dispatch activation marker.
- `ggml/src/ggml-cuda/mmq.cuh`: existing bounded forced-J support; not owned by this decision.
- `patches/1267_rd07_q6k_mmq_scale_fold/validation/producer.py`: authoritative qualification producer.

The b11474 bump does not list 1267 as broken. Do not reimplement it unless current rebase/apply evidence fails.

## Ownership / consolidation

1267 is the sole owner for this scale-fold mechanism. Existing MMQ architecture/geometry dispatch remains authoritative; any promotion gate must reuse it rather than create another table. 1006 retains Q6_K codegen/cast ownership and 0300 retains forced-J tuning. 1203 remains rejected. 1266 is terminal here unless a reproducible stock correctness race appears.

## Current evidence

First-party archived evidence is promotion-relevant but pin-bound:
- 1266 gfx1201: four-session FAIL, target -0.0499% with CI95 low -0.1609%; correctness/activation pass.
- 1267 gfx1201: four-session PASS, pp512 +2.4675%, CI95 [2.2776%, 2.6508%]; tg128 -0.0331%, CI95 [-0.0592%, -0.0071%].
- 1267 gfx1100 and gfx1030: completed evidence fails the gain gate and converges near zero.

Current upstream and the rdna-boosts fork both continue architecture-sensitive AMD kernel selection; the fork still carries Q6_K MMQ prefill tuning. This supports architecture-specific qualification, not cross-RDNA extrapolation.

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

## Change Log

- 2026-10-08 (triage): pending; 1267_rd07_q6k_mmq_scale_fold patch.toml state=untested, SUMMARY.md requires new validation. Historical gfx1201 pp512 +2.4675% is not current-pin promotion; b11474 requalification remains; 1203 rejected.
