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
priority: null
---

# Evaluate RMS norm direct Q8_1 production

## Description

TODO, hard-dependent on PRBE05. Evaluate direct Q8_1 production in RMS-norm's own kernel, publishing straight into PRBE05's cache instead of relying on a later separate quantize kernel. Must not be evaluated until PRBE05's cache correctness and capture lifecycle are proven.

## Steps

1. Do not start until PRBE05's adversarial correctness matrix and capture-lifecycle gates pass -- this item compares B+PRBE05 against B+PRBE05+PRBE06, never against raw baseline B.
2. Add an exact graph-level eligibility selector BEFORE enabling direct production: ggml_cuda_op_rms_norm (norm.cu:478) has no visibility into its downstream consumer, so gate direct-Q8_1 production on an explicit check that the RMS-norm (or RMS-norm+MUL fused, norm.cu:502) output tensor is consumed by an eligible MMVQ dispatch (matching shape/dtype/stride the MMVQ seam in PRBE05 requires) -- do not quantize eagerly based on shape alone, since ggml_cuda_op_rms_norm's output may feed unrelated non-MMVQ consumers.
3. "Quantize immediately after the RMS kernel enqueue" is still a second, separate quantization kernel launch, not direct in-kernel Q8_1 production -- treat it as such (a scheduling/fusion optimization, not literal single-launch production) unless implementing a genuine fused kernel per step 4.
4. If pursuing true direct production: extend rms_norm_f32_cuda (norm.cu:304) and its templated rms_norm_f32<block,fused,...> launches (norm.cu:311-401) with a new template variant that emits Q8_1 blocks using the exact same semantics as quantize_row_q8_1_cuda, writing into PRBE05's cache via its reserve/publish API in the same kernel launch. If instead supporting ggml_cuda_op_rms_norm_fused (norm.cu:502), cache the actual mul_tensor->data (post-MUL) output, not the pre-MUL RMS intermediate -- the two are different tensors and only the actually-consumed one may be cached.
5. Restrict to qualifying patterns only (explicit output shape/stride/dtype checks, proven downstream MMVQ consumer per step 2); any nonqualifying pattern uses the existing native path unchanged.
6. Verify direct-output equivalence against the standalone quantizer/reference path via independent re-quantize + byte-compare.
7. Cover graph/non-graph execution and native fallback; measure quantization launch counts, memory, and graph effects.
8. Compare B+PRBE05 vs B+PRBE05+PRBE06 with balanced repeats; record rejection explicitly if the marginal effect is not a statistically supported positive.

## Detailed Solution & Technical Design

This is a dependent producer optimization, not a cache redesign: RMS-norm's kernel template already exists at multiple specializations (rms_norm_f32<256,false>, rms_norm_f32<256,true> for the fused-with-mul variant, at norm.cu:311/319/366/374/393/401) -- the direct-Q8_1 output path should be a new templated variant or a post-kernel step that reuses the already-computed F32 values while they're still warm, publishing via PRBE05's cache API rather than requiring a second kernel launch to re-read from global memory.

## Code Samples & Guidance

Real anchors verified in b11126 (re-copy exact literal text before authoring Edit() anchors):
- ggml/src/ggml-cuda/norm.cu:478 `void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst)`.
- ggml/src/ggml-cuda/norm.cu:502 `void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor)` -- the fused RMS-norm+MUL producer, the more likely integration point since its output already feeds a downstream consumer.
- ggml/src/ggml-cuda/norm.cu:304 `static void rms_norm_f32_cuda(...)` and the templated `rms_norm_f32<block, fused, ...>` kernel launches at norm.cu:311-401.
patch.toml sketch: schema=1, id="12xx_rd10_rms_norm_direct_q81", state="untested", kind="enhancement", backend="hip", experiment-contract="RD10-RMSNORM-DIRECT-Q81", requires=["12xx_rd09_q81_activation_cache_foundation"], validation-architectures=["gfx1100","gfx1201","gfx1030"].

## Files

ggml/src/ggml-cuda/norm.cu (RMS-norm producer seam); PRBE05's hip-q81-cache API (consumed, not modified); Q8 block/reference fixtures; patches/12xx_rd10_rms_norm_direct_q81/{patch.toml,patch.py}; graph/non-graph campaign artifacts.

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check --focal-overlay 12xx_rd10_rms_norm_direct_q81 --source bigcherry-tuning` (must declare `requires` on PRBE05's patch id). Hardware (Brutus, not run here): numerical equality + exact Q8 block identity vs standalone quantizer; graph/non-graph; fallback-pattern coverage; launch/memory accounting; causal B+PRBE05 vs B+PRBE05+PRBE06 balanced-repeat performance on gfx1100/gfx1201/gfx1030.

## Effort & Risk

M effort -- straightforward producer addition once PRBE05 exists; risk is entirely inherited from PRBE05's cache correctness, hence the hard prerequisite ordering.

## Standards

Dependent causal arm; fallback preservation; exact output evidence; fail closed on cache or numerical uncertainty.

## Acceptance Criteria

No direct producer is enabled before PRBE05 passes; all qualifying outputs match reference and unsupported patterns fall back; only a statistically supported marginal improvement without quality or memory regression can be promoted.

## Notes

Supersedes: RD10
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd10

2026-09-24 relevance at b11126: TODO, blocked on PRBE05. Real anchors verified (norm.cu:478/502/304 + kernel templates 311-401). GPT design request submitted (req_5e0e57d9e29f44b0, batched with PRBE05/PRBE10); gateway congested at submission -- authored directly against verified anchors as a fallback.

2026-09-24 GPT review req_7f4dea253b7247f0 applied: added an explicit graph-level producer/consumer eligibility selector (ggml_cuda_op_rms_norm has no downstream-consumer visibility, so shape-only gating would eagerly quantize unrelated outputs); clarified that post-enqueue quantization is a second kernel launch, not true direct production, unless a genuine fused template variant is built; corrected the fused-variant caching target to mul_tensor->data (post-MUL), not the RMS intermediate.

## Change Log

- 2026-09-09T10:53:51.286520+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:10:56.565121+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.152218+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.829930+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:32:14.424353+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023232_the-next-three-rdna-successors_5807
- 2026-09-10T02:32:32.993441+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:33:16.031005+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:36:43.619023+00:00 (updated-by): Updated: section:steps, section:notes
