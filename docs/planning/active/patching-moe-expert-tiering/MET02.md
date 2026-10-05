---
id: MET02
order: 2
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-02T04:44:50.160922+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# 1281 moe_mul_mat_id_range: range-aware MUL_MAT_ID primitive (CPU + HIP)

## Description

Generic ggml primitive: `ggml_mul_mat_id_range(ctx, weights[K,M,n_local], act, ids, id_base)`. For each selected global expert id `g`, compute `local = g - id_base`; if `0 <= local < ne02`, execute the local expert, otherwise emit an exact zero lane and skip GEMM/GEMV. Implement this as an op-param variant of `GGML_OP_MUL_MAT_ID`; plain `ggml_mul_mat_id` retains strict `0 <= id < ne02` semantics.

This item owns range semantics only. MET01 owns placement/profile policy, MET03 owns tier graph/CPU-tail execution, MET04 owns cache-vs-EP policy, MET05 owns auxiliary-device execution, and MET06 owns compact expert materialisation. Do not create a second router, cache, placement table, or expert store here.

## Optimisation slice: compact active-lane dispatch

The zero-then-skip design is correct but can still launch/iterate work for lanes whose selected experts are outside the local range. Add a profile-gated compact path for sparse ownership: translate global IDs once, build a compact `(output_lane, local_expert)` list, zero inactive lanes once, dispatch only active lanes, and scatter to original output ordering. Reuse existing expert grouping metadata where available. Bypass compaction when setup/scatter costs exceed skipped compute; selection belongs in existing HIP autotune, not a new registry.

## Upstream compatibility gates

- Preserve current MMQ tail/scratch sizing rules for `MUL_MAT_ID`; never fork an older padding formula that can under-allocate the final expert tile.
- Preserve `GGML_PREC_F32` selection for quantized weights where supported upstream.
- Keep ordinary `MUL_MAT_ID` dispatch unchanged unless the range variant is explicitly requested.

## Steps

1. `ggml.h` / `ggml.c`: add `ggml_mul_mat_id_range` plus an op param carrying `id_base`; preserve ordinary op ABI/behaviour.
2. `ggml-cpu`: range-base handling and exact inactive-lane zeroing; optionally compact active rows only when local density is low.
3. HIP/CUDA: apply the same semantics across Q6_K, IQ4_XS, Q8_0 and F32 MMVQ/MMQ paths. Share existing expert grouping/sort metadata instead of allocating another router structure.
4. Audit Q8_1 scratch/tail sizing against the pinned/upstream MoE MMQ fixes before performance qualification.
5. Preserve `GGML_PREC_F32` dispatch for quantized weights.
6. Feed density/ubatch decisions into existing HIP-autotune keys: architecture, quant type, ubatch bucket, local/selected density, MMVQ vs MMQ.
7. Add backend-op and end-to-end tiered-MoE tests.

## Detailed Solution & Technical Design

Range translation happens before kernel expert indexing:

```text
global_id = ids[row, slot]
local_id  = global_id - id_base
active    = unsigned(local_id) < unsigned(n_local)
```

Inactive outputs are exact zero by contract. For compact dispatch, reserve scratch from existing backend workspace; no per-token allocation or host round-trip. A compact entry needs only original output lane plus local expert ID if token/row is already recoverable from existing sorted-ID metadata.

Profitability gate:

`T_compact = T_build + T_group + T_active_compute + T_scatter`

versus

`T_dense = T_existing_dispatch + T_inactive_checks + T_active_compute`.

Promote only on repeated end-to-end improvement; eliminating nominal kernel work is insufficient if list construction dominates small ubatches.

## Code Samples & Guidance

```cpp
const int32_t g = ids[i];
const int32_t l = g - id_base;
if ((uint32_t) l < (uint32_t) n_local) {
    active.push(i, l);
} else {
    zero_output_lane(i);
}
```

Keep the ordinary fast path dispatch-equivalent when the range variant is not requested. Avoid device-wide synchronization between list construction and matmul; use stream ordering/events. Check VGPR/spill impact from added range arithmetic and specialize dense/sparse paths if a universal check hurts dense ownership.

## Files

- `patches/1281_moe_mul_mat_id_range/`
- `ggml/include/ggml.h`
- `ggml/src/ggml.c`
- `ggml/src/ggml-cpu/ops.cpp`
- `ggml/src/ggml-cuda/mmid*`, `mmq*`, `mmvq*`
- existing HIP-autotune dispatch/key files for sparse/dense threshold only
- backend-op tests plus MoE tier integration tests

## Validation

Correctness: ordinary `MUL_MAT_ID` unchanged; full-range equals baseline; all IDs outside range produce exact zero; mixed adjacent/non-overlapping ranges; first/last local expert; negative translated ID; `id_base + n_local`; partial final MMQ tile; default and `GGML_PREC_F32`; F32/Q8_0/Q6_K/IQ4_XS on gfx1100/gfx1201/gfx1030 and CPU.

Performance: gfx1100/gfx1201 primary; ub1/4/16/32/64/128/256/512; local ownership 0/10/25/50/75/100%; MMVQ/MMQ; pp512/pp2048 and tg128/tg512. Record kernel count, routed/active rows, compact-list build/scatter time, kernel time, end-to-end tok/s, workspace, VGPRs and spills. Compare baseline, range-dense, range+compact.

## Effort & Risk

Touches hot HIP MoE kernels. Main risks: register-pressure regression, compact-list overhead at small batches, incorrect scatter, stale MMQ tail sizing, and duplicate routing infrastructure. Correctness-only range semantics must work without compaction.

## Standards

- Fail closed to correctness-only range path when compact dispatch is unsupported.
- No per-token allocation or global device synchronization.
- No duplicate placement/router/cache ownership.
- Preserve ordinary `MUL_MAT_ID` behaviour and precision semantics.

## Acceptance Criteria

- Range semantics pass the full correctness matrix on CPU, gfx1100 and gfx1201; gfx1030 has reference/correctness coverage.
- Ordinary `MUL_MAT_ID` has no material pp/tg regression.
- No OOB/guard failure on partial MMQ tiles.
- Sparse compaction promotes only if a representative <=25% local-density lane improves >=5% end-to-end or >=10% `MUL_MAT_ID` kernel time with no required dense lane >2% slower.
- HIP autotune owns sparse/dense selection; no second dispatch registry.

## Notes

2026-10-05, folded in from BCOP15 (audit backfill) - compact active-lane dispatch: profile 0/10/25/50/75/100% local ownership across ub1-512 on gfx1100/gfx1201; prototype global->local translation plus compact output-lane/local-expert list in existing workspace; keep current precision selection and upstream MMQ safety; gate through existing HIP autotune.

References: https://github.com/ggml-org/llama.cpp/issues/21948 ; https://github.com/ggml-org/llama.cpp/issues/27792 ; https://github.com/ggml-org/llama.cpp/pull/29911

## Change Log

- 2026-10-02T04:44:50.160922+00:00 (created-by): Created by agent
- 2026-10-05: BCOP15 audit backfill added compact-dispatch gate.
- 2026-10-05: Transplanted structured sparse-range dispatch, MMQ safety, precision, validation and ownership details from `automation-qfp-indexer-20261004`.
