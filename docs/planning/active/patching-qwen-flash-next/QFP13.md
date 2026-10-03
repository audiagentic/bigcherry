---
id: QFP13
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T17:26:47.771591+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Decode kernel-count reduction for Flash-Next (launch-gap bound: ~3.5 us gap per ~1.7 us kernel)

## Description

Inter-kernel gap census on production profile v2 decode (`flashnext-v2-profile/d8192`, rocprofv3): median gap between consecutive in-graph kernels is 3.1 us on XTX and 3.7 us on R9700, p99 5-6.5 us; median kernel duration is only 1.6-1.8 us. Decode executes ~1,300 kernels/token per tensor-split GPU, so launch gaps cost ~4.7-5.0 ms/token per GPU versus ~7.6-7.9 ms/token of kernel time. This is the bulk of the measured ~58-61% GPU idle. HIP graphs are already enabled, so the immediate lever is fewer graph nodes / launches rather than another graph-cache layer.

Per-token counts from `rank-census.py`: elementwise ~370-400, quantize ~180-200, other ~180, MMVQ ~150-166, norm/rope ~80-95, get/set_rows/cpy ~55-65, qsa/topk ~60, AllReduce produce/consume ~60.

First implementation target: PRBE38's literal `GEMV -> UNARY(SILU/SIGMOID) -> elementwise MUL` topology. b11126 already has the machinery needed to avoid a new kernel family: `ggml_cuda_mm_fusion_args_host/_device`, `ggml_cuda_should_fuse_mul_mat(...)`, and fused `mul_mat_vec_q/f` epilogues. The remaining literal-UNARY topology is handled by the separate `ggml_cuda_op_unary_mul` pointwise path, so eligible decode graphs still write the GEMV result, launch a second pointwise kernel, reread it, then write the final result. Fold this topology into the existing GEMV fusion descriptor and delete the eligible standalone launch/plumbing rather than adding another fusion implementation.

This item owns the Flash-Next prioritisation and measured launch-budget gate; PRBE38 remains the generic backend/correctness owner for the specific fusion semantics.

## Steps

1. Extend the trace tooling with an adjacent-kernel n-gram census for the decode window. Count exact `MMVQ/MMVF -> unary_mul` (and equivalent named) pairs per generated token on gfx1100 and gfx1201. Do not implement until the trace proves the pair exists in production Flash-Next.
2. Map the traced graph nodes back to `ggml_cuda_op_unary_mul` and the producing `GGML_OP_MUL_MAT[/ID]`. Prove the activation result has a single consumer and the MUL broadcast/alias/layout satisfies the existing fused epilogue constraints.
3. Extend the existing mul-mat fusion matcher/descriptor to accept the literal `MUL_MAT -> UNARY -> MUL` topology when the activation is SILU/SIGMOID and all safety guards pass. Reuse `ggml_cuda_mm_fusion_args_host/_device`; do not add a parallel descriptor or a new kernel body.
4. Dispatch the existing `mul_mat_vec_q/f(..., fusion_data)` path and suppress the matched UNARY+MUL node/kernel. Unsupported activation, alias, broadcast, dtype, multi-consumer, GEMM/prefill, or non-vector shapes must fall through unchanged.
5. Remove duplicated special-case plumbing made unreachable by the fused decode path where doing so reduces net production LOC. Do not delete `ggml_cuda_op_unary_mul` globally because it remains the fallback for non-GEMV and unsupported layouts.
6. Re-run kernel n-gram census and `rank-census.py`; record launches/token, gap ms/token, busy share, and end-to-end ms/step. If the first target removes fewer than 8 launches/token or saves <0.5% end-to-end, move to the next QFP13-ranked fusion instead of broadening this matcher.
7. Next targets, in evidence order only: PRBE39 GEMV->view->residual, PRBE40 paired K/V, PRBE05/1235 Q8_1 activation reuse, RNX08 norm+rope/transpose, RNX10 shared expert. No speculative fusion without a trace-proven recurring adjacency.

## Detailed Solution & Technical Design

### Keep one fusion descriptor

The code-size requirement is as important as launch reduction. The implementation must extend the existing fusion-data path, not create a Flash-Next-specific kernel family.

Pseudo-shape for the matcher extension:

```cpp
// ggml-cuda.cu: existing fusion matcher path
static bool try_literal_unary_mul_fusion(
        const ggml_tensor * mul,
        ggml_cuda_mm_fusion_args_host & fusion) {
    const ggml_tensor * unary = mul->src[0];
    const ggml_tensor * rhs   = mul->src[1];

    if (!unary || unary->op != GGML_OP_UNARY || unary->n_consumers != 1) {
        return false;
    }

    const ggml_tensor * mm = unary->src[0];
    if (!mm || (mm->op != GGML_OP_MUL_MAT && mm->op != GGML_OP_MUL_MAT_ID)) {
        return false;
    }

    const auto op = ggml_get_unary_op(unary);
    if (op != GGML_UNARY_OP_SILU && op != GGML_UNARY_OP_SIGMOID) {
        return false;
    }

    if (!broadcast_and_alias_safe(mm, rhs, mul)) {
        return false;
    }

    fusion.post_mul = rhs;        // use/extend existing descriptor fields where possible
    fusion.post_unary = op;
    return true;
}
```

The exact field names must follow current upstream structures; the important constraint is one descriptor and one vector-kernel epilogue. If the existing descriptor can express this as gate/glu data without semantic ambiguity, reuse those fields rather than adding `post_*` fields.

### Epilogue shape

For an eligible vector result, perform activation and multiply before the final global-memory store:

```cpp
float value = accumulated_value;
if constexpr (has_literal_postop) {
    value = apply_unary(value, fusion.post_unary);
    value *= load_broadcast(fusion.post_mul, row, col);
}
dst[idx] = value;
```

No intermediate GEMV output tensor should be written/read for the fused arm. Accumulation precision, quantization semantics and output dtype remain unchanged.

### Fail closed

Do not fuse when:
- UNARY result has more than one consumer;
- MUL aliases the producer/output in a way the epilogue cannot reproduce;
- unsupported broadcast/stride layout;
- unsupported activation/dtype;
- shape selects GEMM/MMQ instead of the vector path;
- graph/scheduler cannot legally elide the intermediate node.

The native `ggml_cuda_op_unary_mul` remains the fallback.

## Code Samples & Guidance

Verified planning anchors from PRBE38 at b11126:
- `ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_cuda_should_fuse_mul_mat(...)`, `ggml_cuda_mm_fusion_args_host`, and `ggml_cuda_op_unary_mul` dispatch/matcher sites.
- `ggml/src/ggml-cuda/mmvq.cu`: fused quantized GEMV epilogue / `fusion_data` consumer.
- `ggml/src/ggml-cuda/mmvf.cu`: fused floating-point GEMV epilogue / `fusion_data` consumer.

Prefer deleting or generalising an existing matcher branch over layering an additional Flash-Next matcher. Any implementation that increases net production LOC without removing launches must justify why the existing fusion descriptor cannot express the topology.

## Files

Planning/instrumentation:
- `tools/lab/flash-next/rank-census.py`
- new narrow adjacent-kernel census helper under `tools/lab/flash-next/` only if it cannot stay a small extension of `rank-census.py`

Expected upstream patch anchors:
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cuda/mmvq.cu`
- `ggml/src/ggml-cuda/mmvf.cu`
- focused backend-op tests for the literal topology and negative controls

No new `.cu/.cuh` fusion family for the first target.

## Validation

Profile-v2 ABBA on the existing XTX+XTX+R9700 tensor-split topology at shallow (~8-10K) and deep (~65-80K) context.

Required measurements:
- exact matching adjacency count/token before implementation;
- total kernels/token and elementwise kernels/token;
- inter-kernel gap ms/token and GPU busy share;
- target verify ms/step and effective TG;
- greedy/token identity against unfused control;
- source LOC delta for touched production fusion code.

Correctness matrix: SILU and SIGMOID positive cases; alternate broadcast, non-contiguous output, alias, unsupported activation, multiple consumers and GEMM/prefill negative cases. `test-backend-ops` plus production deterministic Flash-Next decode must pass.

Performance acceptance for the first target: remove the traced standalone pointwise launches with no compensating new launch, no material VGPR/occupancy regression in the producing GEMV, and >=0.5% end-to-end TG/ms-step improvement or a measured >=0.10 ms/token launch-gap reduction. If trace frequency is too low to meet that bound, close/park the target rather than expanding scope.

## Effort & Risk

M for the first fusion target. The kernel arithmetic already exists; risk is graph-topology/alias/broadcast proof, not novel math. Main performance risk is increasing register pressure in MMVQ/MMVF enough to offset the removed launch. Main correctness risk is eliding an intermediate tensor that has another consumer or different broadcasting semantics. Both are contained by fail-closed matching and negative controls.

## Standards

- Evidence-first: trace adjacency before coding.
- Reuse native fusion infrastructure; no duplicate kernel family.
- Net production LOC should decrease or remain effectively flat while launches decrease.
- Preserve fallback and exact accumulation/output semantics.
- No change to AllReduce placement, tensor split, MTP depth or model graph outside the matched local topology.
- One optimization at a time; re-profile after each accepted fusion.

## Acceptance Criteria

- Production trace proves the selected adjacency exists and records its frequency/token on gfx1100 and gfx1201.
- Eligible literal GEMV->UNARY->MUL executes as one GEMV kernel with the activation/MUL in its epilogue.
- Unsupported cases continue through the native unfused path.
- No new standalone fusion kernel family or duplicate descriptor is introduced.
- Net production LOC for the matched path decreases or stays flat; dead/special-case plumbing is removed where safe.
- Kernel launches/token and launch-gap time/token decrease by the predicted amount.
- Greedy output/correctness gates pass on gfx1100 and gfx1201.
- End-to-end profile-v2 decode improves by >=0.5%, or the target is parked and QFP13 advances to the next trace-ranked fusion.

## Notes

This re-ranks the fusion backlog because individually the historical fusion items looked like 1-3% opportunities, but together they attack the single largest measured cost: ~4.7-5.0 ms/token of launch gaps. QFP11 showed AllReduce split boundaries themselves are only ~25 us and the larger AR loss is rank-arrival skew, so generic graph-boundary work is not the first lever. QFP06 also showed capping graph instances below the live working set causes recapture churn. The immediate low-risk path is therefore to remove recurring local decode nodes while keeping the current graph and collective structure.

First concrete owner: PRBE38. Related: PRBE37 (existing native GEMV fusion descriptor), PRBE39/40, PRBE05/1235, RNX08, RNX10, QFP06, QFP09, QFP11.

2026-10-04 fusion census (XTX0, ~10K, per generated token): top kernels quantize_q8_1 183, mul_mat_vec_q 154, unary_op 102, k_bin_bcast 98, scale_f32 98, mul_mat_vec_f 82, rms_norm 79, __amd_rocclr_copyBufferRectAligned 55 + __amd_rocclr_copyBuffer 53 (HIP runtime copy kernels inside the graph: 108/token, origin unknown - find via apitrace), unary_gated 30, dsv4_hc_pre/post 30 each, cpu_root produce/consume 30 each, mmvq_moe 30. Top adjacent pairs: quantize_q8_1->mul_mat_vec_q 154 (dedupe/reuse q8_1 across consumers of the same activation, PRBE05/1235, or F32-act), scale_f32->unary_op 60, copyBufferRect<->copyBuffer 80, rms_norm->quantize 41 (PRBE06), mmvq->scale 30 / mmvf->scale 30 (PRBE37/38 epilogues), scale->dsv4_hc_post 30, mmvq->dsv4_hc_pre 30 (RNX04), unary_gated->quantize 30. Next: apitrace to attribute the 108 runtime copies; then quantize dedupe.

## Change Log

- 2026-10-03T17:26:47.771591+00:00 (created-by): Created by agent
- 2026-10-03T17:27:37.409987+00:00 (updated-by): Updated: section:notes
- 2026-10-03T19:38:00+00:00 (agent): Corrected branch selection to `patch-refactor`; made PRBE38 literal GEMV->UNARY->MUL the first trace-gated implementation target; added code-reuse, LOC-reduction, validation and performance gates.
