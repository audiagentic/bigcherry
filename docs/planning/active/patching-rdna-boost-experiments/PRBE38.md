---
id: PRBE38
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:05.203236+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-FUS-002: Fuse GEMV activation + elementwise MUL

## Description

UPSTREAM-ABSORBED. This item's target (GEMV -> SiLU/activation -> elementwise MUL, e.g. SwiGLU gate*up) is the exact pattern b11126's native mul_mat fusion already covers -- same evidence as PRBE37 (ggml_cuda_mm_fusion_args_host/_device, ggml_cuda_should_fuse_mul_mat, fusion_data.gate/glu_op wired into mmvq.cu/mmvf.cu's has_fusion GEMV kernels computing `value *= silu(gate_value)`). PRBE38 explicitly described itself as depending on PRBE37's 'root identity' for the same matcher family, and that root identity is now confirmed native upstream, not a gap to fill. No separate implementation is needed beyond the same verification already scoped under PRBE37.

TODO (reopened, GPT WRONG-confirmed). b11126 epilogue fusion only recognizes the canonical {MUL_MAT[/ID], MUL_MAT[/ID], GGML_OP_GLU} graph shape (gate_proj/up_proj feeding a single GGML_OP_GLU node with glu_op in {SWIGLU,GEGLU,SWIGLU_OAI,SWIGLU_CLAMP}, ggml-cuda.cu ~1673-1766, ~3736-3965). A literal GEMV -> UNARY(SILU) -> elementwise MUL graph (two separate ops, not one fused GGML_OP_GLU node) is NOT folded into the GEMV epilogue by this mechanism. b11126 does separately fuse UNARY(SILU|SIGMOID|SOFTPLUS) -> MUL via ggml_cuda_op_unary_mul (a distinct, post-GEMV pointwise fusion, not part of the mul_mat_vec_* epilogue), which runs after the GEMV result tensor already exists in memory -- it saves one elementwise pass, not the GEMV epilogue write this item asked to eliminate. This item is not upstream-absorbed for the literal-UNARY+MUL topology; it survives as its own successor with a correctly scoped target.

## Steps

1. This item's verification is the same as PRBE37's step 1 (ggml_cuda_should_fuse_mul_mat coverage check) -- do not duplicate that work; treat PRBE37's verification outcome as authoritative for this item too, since it is the same matcher/fusion_data mechanism applied to the same gate*up MUL pattern.
2. If PRBE37's verification finds a genuine gap (SIGMOID unsupported, or a broadcast/alias shape this item's acceptance criteria cared about that upstream's matcher rejects), scope that as a narrow follow-up extending the existing native matcher -- not a new from-scratch fusion mechanism.
3. No new patch package needed.

1. Grep b11126 ggml-cuda.cu for ggml_cuda_op_unary_mul call sites and confirm its dispatch conditions (topology it matches, which UNARY ops, contiguity/broadcast requirements) -- this is the actual native fusion covering literal GEMV->UNARY->MUL graphs, and it is a post-hoc pointwise fusion, not a GEMV-epilogue fusion.
2. Restrict this item's acceptance criteria to: literal (non-GLU-node) GEMV -> UNARY(activation) -> elementwise-MUL graphs where the activation+MUL currently execute as two separate kernel launches AFTER the GEMV (i.e. cases ggml_cuda_op_unary_mul does NOT already cover, if any remain), or explicitly fold ggml_cuda_op_unary_mul's coverage into the GEMV epilogue itself (a genuine new optimization: skip writing the pre-activation GEMV output to global memory at all).
3. If ggml_cuda_op_unary_mul already covers every topology this project's target models produce, close as covered-by-a-different-native-mechanism-than-assumed (not the GEMV-epilogue one originally claimed) with that evidence; otherwise scope the GEMV-epilogue-skip optimization as new kernel work with correctness/alias/broadcast guards per the original acceptance_criteria.

## Detailed Solution & Technical Design

Fuse the post-activation multiply into the HIP GEMV epilogue so the intermediate activation is not written and reread. The matcher must prove the exact GEMV -> SiLU -> MUL topology, compatible broadcast semantics, contiguous/strided output constraints, and destination ownership. Keep a conservative fallback for every other MUL shape or aliasing arrangement. Preserve existing accumulation precision and backend guards; this is a dispatch-path optimization, not permission to change numerical semantics.

Two distinct native mechanisms exist in b11126: (1) canonical GLU-node epilogue fusion inside mul_mat_vec_* (covers gate*up SwiGLU-family graphs, see PRBE37), and (2) ggml_cuda_op_unary_mul, a separate post-GEMV pointwise pass for literal UNARY->MUL graphs. Neither eliminates a GEMV's own epilogue write-then-reread for the literal-UNARY+MUL case if ggml_cuda_op_unary_mul still runs as its own kernel launch after the GEMV. Confirm via profiling/launch-count evidence (not source alone) whether (2) is truly a separate launch before treating this as a real, unaddressed optimization opportunity distinct from PRBE37.

## Code Samples & Guidance

Trigger: GEMV -> SiLU -> MUL with the supported scalar/per-channel broadcast and output layout. Controls: alternate MUL broadcasting, non-contiguous or aliased outputs, unsupported activation/dtype, and graphs where the activation result is consumed elsewhere. Required negative result: no fusion and unchanged unfused dispatch.

## Files

Same as PRBE37: ggml/src/ggml-cuda/ggml-cuda.cu, mmvq.cu, mmvf.cu (evidence/verification only).

ggml/src/ggml-cuda/ggml-cuda.cu (ggml_cuda_op_unary_mul call sites and dispatch conditions; evidence and possible edit site).

## Validation

See PRBE37's validation -- shared verification, no hardware run needed for this disposition.

Offline: grep/read verification per steps 1-2. Launch-count evidence (kernel trace showing GEMV and unary_mul as separate launches) before any new fusion is designed. If a follow-up patch is written: test-backend-ops case for the literal UNARY+MUL topology, alias/broadcast negative controls, and hardware bit-identical + paired-perf evidence via validation_campaign (not run here).

## Effort & Risk

M; matcher and epilogue changes are localized but incorrect broadcast or alias assumptions can silently corrupt decode outputs. Fail-closed fallback and differential tests contain the risk.

S for this item as written (evidence + scoping); a real epilogue-skip fusion would be M (new kernel variant, alias-safety proof).

## Standards

Preserve native-BF16/F32 accumulation guards, Q8_0 behavior, unsupported-hardware fallback, and the repository patch qualification/evidence rules.

## Acceptance Criteria

Acceptance requires exact GEMV->activation->MUL layout proof, independent output parity across supported dtypes/shapes, no fusion for unsupported broadcast/alias forms, and repeatable launch/TG or memory improvement versus unfused control; otherwise retain fallback.

## Notes

Supersedes: RD46
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd46

Supersedes RD46. Depends on PRBE37 (AMD-FUS-001). Related fusion work must not broaden this matcher without a separate acceptance decision.

append

2026-09-24 relevance at b11126: UPSTREAM-ABSORBED, same real anchors as PRBE37 (ggml-cuda.cu ~3750-3770, mmvq.cu ~599-758, mmvf.cu ~58-382) -- PRBE38's target (GEMV->SiLU->elementwise-MUL, i.e. SwiGLU gate*up) is literally the pattern that fusion_data.gate/glu_op implements; `value *= ggml_cuda_op_silu_single(gate_value)` in the GEMV epilogue is exactly this item's acceptance criterion. GPT design gateway unavailable this session (see PRBE32 notes); disposition written directly from verified source.

2026-09-24 GPT review req_c18183e0a9034c94 applied: WRONG disproven -- b11126 only fuses canonical GGML_OP_GLU node graphs in the GEMV epilogue; literal UNARY->MUL is handled (if at all) by the separate ggml_cuda_op_unary_mul post-pass, not the GEMV epilogue; reopened to pending, narrowed accordingly.

## Change Log

- 2026-09-09T10:56:05.203236+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:18.726737+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.301853+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.057267+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:02:34.325623+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- chg_20260910_030318_repaired-three-patching-succes_9681
- 2026-09-10T03:03:18.963781+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:05:27.878265+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20260910_030619_removed-migration-placeholder_7703
- 2026-09-10T03:06:19.279192+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:31:39.283002+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:notes
- 2026-09-24T02:31:59.126993+00:00 (updated-by): Updated: section:notes
- 2026-09-24T02:32:19.601706+00:00 (state-transition): State: pending → superseded
- chg_20260924_023553_re-scoped-11-rdna-boost-planni_1625
- 2026-09-24T02:36:23.519042+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:35:22.931353+00:00 (state-transition): State: superseded → pending
- 2026-09-24T04:36:36.775413+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk
- 2026-09-24T04:36:55.897312+00:00 (updated-by): Updated: section:notes
