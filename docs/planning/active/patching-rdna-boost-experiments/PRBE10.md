---
id: PRBE10
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:10.807250+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Evaluate gating MUL direct Q8_1 production

## Description

TODO, hard-dependent on PRBE05 (item text also names PRBE09 as a prerequisite, but PRBE09 is a Vulkan-AllReduce tracking item with no Q8/cache relationship -- that reference appears stale/mistaken; treat PRBE05 as the real and only hard prerequisite unless a source audit finds an actual PRBE09 coupling). Evaluate gating-MUL (SwiGLU-style elementwise MUL) direct Q8_1 production, publishing into PRBE05's cache at the exact native MUL seam.

## Steps

1. Do not start until PRBE05's cache correctness, generation lifecycle and capture gates pass.
2. In ggml_cuda_op_mul (ggml/src/ggml-cuda/binbcast.cu:445), add a direct Q8_1 quantization of the MUL output, publishing into PRBE05's cache via reserve/publish immediately after kernel enqueue, at the exact native seam only -- do not alter unrelated dispatch or existing GLU/MUL_MAT_ID fusion logic (ggml-cuda.cu:1691's is_mul_mat_id GLU-fusion path must remain untouched).
3. Restrict to qualifying shapes (explicit dtype/shape checks matching what a downstream MMVQ consumer expects); nonqualifying or cache-exhausted paths use the existing native quantize kernel unchanged.
4. Verify direct-output equivalence via independent re-quantize + byte-compare against the standalone reference.
5. Cover graph/non-graph and native fallback; measure launch/memory/latency effects.
6. Compare B+PRBE05 vs B+PRBE05+PRBE10 with balanced repeats; promote only with statistically supported end-to-end benefit and no numerical, memory, or fallback regression.

## Detailed Solution & Technical Design

Same dependent-producer pattern as PRBE06, applied to the elementwise MUL gating op instead of RMS-norm. binbcast.cu's ggml_cuda_op_mul is the exact native seam for this class of op; the direct-Q8_1 output must not interfere with the existing GLU/MUL_MAT_ID fusion checks already present in ggml-cuda.cu around line 1691 and the mul_mat_id_bias_glu_ops fusion pattern at line 3195 -- this item stays strictly downstream of MUL's own output, not a rewrite of the fusion matcher itself.

## Code Samples & Guidance

Real anchor verified in b11126 (re-copy exact literal text before authoring Edit() anchors):
- ggml/src/ggml-cuda/binbcast.cu:445 `void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst)`.
- Cross-check (do not modify): ggml/src/ggml-cuda/ggml-cuda.cu:1691 `const bool is_mul_mat_id = ffn_up->op == GGML_OP_MUL_MAT_ID && ffn_gate->op == GGML_OP_MUL_MAT_ID && glu->op == GGML_OP_GLU;` and line 3195's `mul_mat_id_bias_glu_ops` pattern -- existing GLU fusion precedence that must remain intact.
patch.toml sketch: schema=1, id="12xx_rd11_gating_mul_direct_q81", state="untested", kind="enhancement", backend="hip", experiment-contract="RD11-GATING-MUL-DIRECT-Q81", requires=["12xx_rd09_q81_activation_cache_foundation"], validation-architectures=["gfx1100","gfx1201","gfx1030"].

## Files

ggml/src/ggml-cuda/binbcast.cu (gating-MUL producer seam); PRBE05's hip-q81-cache API (consumed, not modified); Q8 reference fixtures; patches/12xx_rd11_gating_mul_direct_q81/{patch.toml,patch.py}; graph/non-graph and causal campaign artifacts.

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check --focal-overlay 12xx_rd11_gating_mul_direct_q81 --source bigcherry-tuning` (declares `requires` on PRBE05's patch id). Hardware (Brutus, not run here): exact Q8 block identity; numerical output; shape/fallback matrix; graph capture; launch/memory counters; causal B+PRBE05 vs B+PRBE05+PRBE10 performance on gfx1100/gfx1201/gfx1030.

## Effort & Risk

M effort -- same class of work as PRBE06, independent sibling; risk inherited from PRBE05.

## Standards

Dependent causal arm; exact output evidence; fail closed on cache and numerical uncertainty.

## Acceptance Criteria

No implementation is accepted before PRBE05 passes; qualifying outputs match reference and unsupported paths fall back; only a proven marginal improvement without regressions can promote.

## Notes

Supersedes: RD11
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd11

2026-09-24 relevance at b11126: TODO, blocked on PRBE05. Flagged and corrected an apparent stale cross-reference to PRBE09 in this item's own original text (PRBE09 is Vulkan AllReduce tracking, unrelated to Q8_1 caching) -- treated PRBE05 as the sole real prerequisite pending a source audit proving otherwise. Real anchor verified (binbcast.cu:445; cross-checked against ggml-cuda.cu:1691/3195 GLU fusion to avoid collision). GPT design request submitted (req_5e0e57d9e29f44b0, batched with PRBE05/PRBE06); gateway congested at submission -- authored directly against verified anchors as a fallback.

## Change Log

- 2026-09-09T10:54:10.807250+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:18.912368+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.173928+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.867375+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:35:17.696577+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023529_three-more-rdna-successors-now_3176
- 2026-09-10T02:35:29.343335+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:33:37.368559+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
