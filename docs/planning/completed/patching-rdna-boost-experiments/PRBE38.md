---
id: PRBE38
order: 0
plan: patching-rdna-boost-experiments
state: superseded
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

## Steps

1. This item's verification is the same as PRBE37's step 1 (ggml_cuda_should_fuse_mul_mat coverage check) -- do not duplicate that work; treat PRBE37's verification outcome as authoritative for this item too, since it is the same matcher/fusion_data mechanism applied to the same gate*up MUL pattern.
2. If PRBE37's verification finds a genuine gap (SIGMOID unsupported, or a broadcast/alias shape this item's acceptance criteria cared about that upstream's matcher rejects), scope that as a narrow follow-up extending the existing native matcher -- not a new from-scratch fusion mechanism.
3. No new patch package needed.

## Detailed Solution & Technical Design

Fuse the post-activation multiply into the HIP GEMV epilogue so the intermediate activation is not written and reread. The matcher must prove the exact GEMV -> SiLU -> MUL topology, compatible broadcast semantics, contiguous/strided output constraints, and destination ownership. Keep a conservative fallback for every other MUL shape or aliasing arrangement. Preserve existing accumulation precision and backend guards; this is a dispatch-path optimization, not permission to change numerical semantics.

## Code Samples & Guidance

Trigger: GEMV -> SiLU -> MUL with the supported scalar/per-channel broadcast and output layout. Controls: alternate MUL broadcasting, non-contiguous or aliased outputs, unsupported activation/dtype, and graphs where the activation result is consumed elsewhere. Required negative result: no fusion and unchanged unfused dispatch.

## Files

Same as PRBE37: ggml/src/ggml-cuda/ggml-cuda.cu, mmvq.cu, mmvf.cu (evidence/verification only).

## Validation

See PRBE37's validation -- shared verification, no hardware run needed for this disposition.

## Effort & Risk

M; matcher and epilogue changes are localized but incorrect broadcast or alias assumptions can silently corrupt decode outputs. Fail-closed fallback and differential tests contain the risk.

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
