---
id: PRBE06
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:53:51.286520+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P0
---

# Eliminate RMSNorm -> Q8_1 quantize launches for MMVQ consumers

## Description

PRBE06 is the next QFP13 quantization target after the 1307/PRBE05 reuse experiment. 1307 reduced `quantize_q8_1` from ~183 to ~149/token on each XTX (-19%) and total kernels from ~1307 to ~1270/token, but decode was only neutral to ~1-2% better in drift-limited screens. The remaining trace contains ~41 `rms_norm -> quantize_q8_1` adjacencies/token, large enough to justify a producer-side launch-elimination experiment.

This item does **not** extend the Q8_1 cache. It removes a producer->quantizer boundary when the exact RMSNorm result is immediately consumed as Q8_1 by MMVQ. Prefer a narrow extension of existing RMSNorm/MMVQ helpers over a new cache subsystem or kernel family. Native F32 RMSNorm output remains available whenever the graph requires it.

## Stage 0 — Prove The Exact 41/token Topology

Before editing production code:
1. Extend/use QFP13 n-gram attribution to map the ~41/token `rms_norm -> quantize_q8_1` transitions to graph tensors and `ggml_cuda_mul_mat_vec_q` calls.
2. For each class record whether the RMSNorm output has one consumer or multiple consumers, whether the F32 output must remain materialized, dimensions/strides, and which device executes it.
3. Separate plain RMSNorm from `ggml_cuda_op_rms_norm_fused` (RMSNorm+MUL). For the fused form the candidate value is the **post-MUL output** actually consumed by MMVQ, not the pre-MUL RMS intermediate.
4. Predict the maximum removable launches/token. Continue only if >=20 launches/token are legally removable on the production path; otherwise return the item to QFP13 for re-ranking.

## Implementation Plan

1. Add one graph-level eligibility decision at the existing CUDA/HIP dispatch layer, where both producer and consumer relationships are visible. Do not make `ggml_cuda_op_rms_norm()` guess downstream use from shape.
2. Eligible fast path requires: vector/decode MMVQ consumer; exact supported contiguous/stride layout; F32 producer semantics; Q8_1 dimensions compatible with the native quantizer; no alias/lifetime ambiguity; and a graph topology that permits the standalone quantize node to be elided.
3. Reuse the existing RMSNorm kernel/template and the existing Q8_1 block math. Factor the smallest device-side Q8_1 block primitive out of the standalone quantizer if necessary so direct production and `quantize_row_q8_1_cuda` share one implementation. **Do not copy Q8_1 scale/round/packing arithmetic into `norm.cu`.** The code-size gate is no duplicated quantizer implementation and no new `.cu/.cuh` kernel family.
4. Add an optional Q8_1 destination to the qualifying RMSNorm launch. Each qualifying producer writes the normal F32 destination when required and Q8_1 blocks in the same launch. If source mapping proves the F32 value is dead after MMVQ consumption, allow a later follow-up to suppress the F32 store; do not combine that semantic change with the first experiment.
5. Feed the direct Q8_1 pointer into the existing `mul_mat_vec_q_switch_type(...)` path and suppress only the corresponding standalone `quantize_row_q8_1_cuda` launch. PRBE05 cache lookup remains for other activations/repeated consumers; direct producer output must not require a cache lookup to be useful.
6. For multi-consumer RMSNorm outputs, either publish the direct Q8_1 value through PRBE05's already-proven stable storage/lifetime mechanism or fall back. Do not add a second ownership/lifetime system.
7. Unsupported graph, layout, capture, allocation, or identity cases execute the current RMSNorm + standalone quantize path unchanged.

## Code / Reuse Guidance

Verified b11126 planning anchors:
- `ggml/src/ggml-cuda/norm.cu`: `rms_norm_f32_cuda(...)` around 304 and `rms_norm_f32<block, fused, ...>` launches around 311-401.
- `ggml/src/ggml-cuda/norm.cu`: `ggml_cuda_op_rms_norm(...)` around 478.
- `ggml/src/ggml-cuda/norm.cu`: `ggml_cuda_op_rms_norm_fused(...)` around 502.
- `ggml/src/ggml-cuda/mmvq.cu`: `ggml_cuda_mul_mat_vec_q(...)` around 1421, `quantize_row_q8_1_cuda(...)` around 1506, and `mul_mat_vec_q_switch_type(...)` around 1531.

Preferred source shape is one shared Q8_1 block conversion primitive used by both the native standalone quantizer and the RMSNorm direct-output specialization. If current quantizer structure cannot be shared without a broad refactor, keep the experiment narrow and document the blocker rather than cloning arithmetic.

Pseudo-interface only; exact signature follows current source:

```cpp
// shared device primitive; native quantizer and producer path both call it
q8_1_block make_q8_1_block(float values[QK8_1]);

// existing RMSNorm template gains an optional direct-Q8 destination
rms_norm_f32<block, fused, emit_q81>(..., float * dst_f32, block_q8_1 * dst_q81);
```

The direct path must preserve the standalone quantizer's exact scale, rounding, sum/metadata, packing, and block ordering. If byte identity cannot be preserved, reject the path.

## Files

Expected production anchors only:
- `ggml/src/ggml-cuda/norm.cu` — producer specialization/direct Q8_1 output.
- existing Q8_1 quantizer implementation/header used by `mmvq.cu` — factor/reuse block conversion only if required.
- `ggml/src/ggml-cuda/mmvq.cu` — consume pre-produced Q8_1 and skip native quantize for the eligible arm.
- `ggml/src/ggml-cuda/ggml-cuda.cu` only if graph-level producer/consumer eligibility cannot be expressed at the existing dispatch seam.
- one BigCherry patch package plus focused backend-op/campaign evidence.

No new cache subsystem and no new CUDA/HIP source file for this experiment.

## Validation

Correctness first:
- independently run the native standalone `quantize_row_q8_1_cuda` over the same RMSNorm F32 result and byte-compare every Q8_1 block;
- plain RMSNorm and RMSNorm+MUL positive cases;
- multi-consumer, non-contiguous/strided, alias, unsupported dimension/dtype, graph/non-graph, capture/replay, and fallback controls;
- deterministic/greedy output equality on gfx1100 and gfx1201.

Performance/profile-v2 ABBA, shallow (~8-10K) and deep (~65-80K):
- exact `rms_norm -> quantize_q8_1` count before/after;
- `quantize_q8_1` and total kernels/token;
- RMSNorm kernel duration/register/occupancy change;
- serial launch-gap ms/token and GPU busy share;
- verify ms/step and effective TG;
- source LOC delta for touched production code.

Compare against the current PRBE05/1307 arm, not the pre-cache baseline, so the marginal benefit is causal.

## Acceptance Criteria

- Stage 0 proves >=20 removable standalone quantization launches/token on the production path.
- Direct Q8_1 blocks are byte-identical to the existing standalone quantizer.
- The predicted standalone launches disappear with no compensating kernel launch.
- Existing Q8_1 arithmetic is shared, not copied; no new kernel family/cache subsystem is introduced.
- Net production LOC should remain small; any >~100 net-line increase requires evidence that sharing/factoring cannot express the path.
- Unsupported cases fall back unchanged.
- No material RMSNorm occupancy/register regression that erases the launch saving.
- Meet QFP13 gate: >=0.5% end-to-end improvement or >=0.10 ms/token measured serial launch-gap reduction without end-to-end regression. Otherwise park/supersede.

## Effort & Risk

M/advanced. Main risk is register/shared-memory pressure from combining normalization reduction with Q8_1 block reduction/packing. Main correctness risk is subtle non-identity with the native Q8_1 quantizer. The shared-primitive and byte-compare gates contain both risks.

## Notes

Supersedes RD10. PRBE05 remains the canonical Q8_1 reuse owner; PRBE06 owns only producer-side RMSNorm quantization launch elimination. QFP13 measured ~41 RMSNorm->quantize adjacencies/token after identifying ~150 remaining quantizations/token that 1307 reuse cannot remove.

2026-10-04: promoted to P0 after 1307 showed exact activation reuse removes only ~19% of Q8_1 quantizations. Re-scoped from cache-dependent direct publication to a narrower producer/consumer launch-elimination experiment, with shared quantizer arithmetic and a code-size gate.
