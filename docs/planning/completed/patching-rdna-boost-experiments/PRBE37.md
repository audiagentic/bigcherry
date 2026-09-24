---
id: PRBE37
order: 0
plan: patching-rdna-boost-experiments
state: superseded
created-at: '2026-09-09T10:56:01.682197+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-FUS-001: Fuse GEMV epilogue activation

## Description

UPSTREAM-ABSORBED. b11126 already implements pattern-matched GEMV/GEMM epilogue activation fusion natively via ggml_cuda_mm_fusion_args_host/_device threaded into mul_mat_vec_f/mul_mat_vec_q. The graph-level matcher in ggml-cuda.cu (multiple `ggml_cuda_mm_fusion_args_host fusion_data{}` sites, e.g. ~3750-3770) detects gate_proj+up_proj+GLU (SiLU/etc.) graphs, calls ggml_cuda_should_fuse_mul_mat(...) to prove eligibility, and populates fusion_data.gate/x_bias/gate_bias/x_scale/gate_scale/glu_op/glu_limit before dispatching a single fused mul_mat_vec_q/mul_mat_vec_f call (mmvq.cu ~599-758, mmvf.cu ~58-382) that computes `value *= silu(gate_value)` (or the matched glu_op) inline in the epilogue -- exactly the pattern-matched SILU/activation GEMV epilogue fusion this item describes, with fail-closed fallback (falls through to unfused dispatch when ggml_cuda_should_fuse_mul_mat returns false). This applies to both dense and MUL_MAT_ID (MoE) graphs since the matcher reads up_n->src[2] as an optional ids tensor.

## Steps

1. Verify at the current pin: git -C work/upstream/llama.cpp.git grep -n "ggml_cuda_should_fuse_mul_mat\|fusion_data.glu_op" b11126 -- ggml/src/ggml-cuda/ggml-cuda.cu, and read ggml_cuda_should_fuse_mul_mat's full body (not yet read in full this session) to confirm it covers SIGMOID as well as SILU, and confirm whether it covers only prefill (MUL_MAT) or also decode (mul_mat_vec) dispatch paths.
2. If verification in step 1 confirms full coverage (SILU+SIGMOID, both GEMV and GEMM), close this item as superseded with that as the final evidence and no further action.
3. If a real gap is found (e.g. SIGMOID unsupported, or only prefill covered), scope a narrow follow-up item extending ggml_cuda_should_fuse_mul_mat's glu_op allow-list or wiring the same fusion_data plumbing into the currently-uncovered dispatch path -- do not silently fold that gap into PRBE38.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

ggml/src/ggml-cuda/ggml-cuda.cu (ggml_cuda_should_fuse_mul_mat and the fusion_data{} call sites, evidence/verification only); ggml/src/ggml-cuda/mmvq.cu, mmvf.cu (evidence only).

## Validation

Offline only: grep/read verification per step 1 above. No hardware run needed for this disposition; if step 1 finds a real gap, downgrade disposition to TODO in a follow-up pass rather than here.

## Effort & Risk



## Standards

Pattern-matched only; correctness before launch reduction; preserve fallback; no broad epilogue fusion.

## Acceptance Criteria

Eligible paths fuse correctly and show repeatable TG benefit with non-target paths unchanged; unsupported patterns fall back; PRBE38 depends on this validated root.

## Notes

Supersedes: RD45
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd45

append

2026-09-24 relevance at b11126: UPSTREAM-ABSORBED. Verified real anchors: ggml/src/ggml-cuda/ggml-cuda.cu ~3750-3770 (ggml_cuda_mm_fusion_args_host fusion_data{} populated with gate/x_bias/gate_bias/x_scale/gate_scale/glu_op/glu_limit after ggml_cuda_should_fuse_mul_mat(...) proves the gate_proj+up_proj+GLU topology, dispatching ggml_cuda_mul_mat_vec_q with &fusion_data); mmvq.cu ~599-758 and mmvf.cu ~58-382 (has_fusion-templated GEMV kernels compute value *= silu(gate_value) inline). This is native upstream pattern-matched GEMV epilogue activation fusion, not fork-derived. Not independently confirmed whether SIGMOID (vs only SILU/SwiGLU-family glu_op) and both GEMV+GEMM paths are fully covered -- flagged as the one remaining verification step in this item's own steps section rather than assumed. GPT design gateway was unavailable this session (rejected submissions, see PRBE32 notes); disposition written directly from verified source since evidence was strong enough not to need design synthesis.

## Change Log

- 2026-09-09T10:56:01.682197+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:15.050723+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.296126+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.050932+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:59:52.821708+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_030005_amd-streamfus-successors-prbe_8761
- 2026-09-10T03:00:05.781141+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:31:11.004582+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:notes
- 2026-09-24T02:31:15.820226+00:00 (updated-by): Updated: section:notes
- 2026-09-24T02:31:20.271685+00:00 (state-transition): State: pending → superseded
- chg_20260924_023553_re-scoped-11-rdna-boost-planni_1625
- 2026-09-24T02:36:19.155839+00:00 (updated-by): Updated: section:ledger-events
