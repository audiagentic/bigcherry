---
id: PRBE49
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:53.658140+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# UP-VK-002: Vulkan FA MMQ FP32 quant-scale calculation

## Description

TODO. GPT unavailable this batch -- design authored directly from verified b11126 source. Relevance: no existing patch (grep for RD59 = no hits). Not clearly upstream-absorbed either way: ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_mmq_funcs.glsl's get_k_scale() (read directly, lines 46-54) returns `FLOAT_TYPEV2` -- a spec-constant-selected type that is fp16 (`f16vec2`) when the device's `device->fp16` capability is enabled (confirmed by the file's own header comment: 'BF16 uses the fp32 shader (FLOAT_TYPE=float)', implying other quant K-types use FLOAT_TYPE=fp16 by default when available) -- so the quantization scale (d) IS computed/stored in FP16 on fp16-capable devices today, matching the exact corruption risk this item describes. Disposition: TODO -- needs the reproduction step (tiny-scale Q4/Q8 KV synthetic test) run before deciding whether this is a real bug here or already mitigated elsewhere (e.g. clamping); could not run hardware/shader tests in this planning pass.

## Steps

1. Recheck llama.cpp PR #27413 (upstream, not in our pinned tree per the grep evidence above) for its exact FP32 scale/reciprocal fix shape, if fetchable; if not fetchable, proceed from the local evidence alone. 2. Reproduce the failure case first: construct tiny-amplitude Q4_0/Q8_0 K/V synthetic backend-op test (test-backend-ops.cpp test_flash_attn_ext with deliberately tiny d/scale values engineered to underflow fp16) and run it on a Vulkan RDNA backend to CONFIRM NaN/Inf actually occurs at this pin before writing the fix -- do not assume the bug reproduces just because get_k_scale returns fp16; some call sites may already upconvert before further math. 3. If reproduced: change get_k_scale (and its V-side counterpart, presumably get_v_scale in the same file, name TBD by grep) to compute in FLOAT_TYPE regardless of storage precision -- i.e. read the raw fp16 stored `d` value but do the subsequent reciprocal/scale MATH in fp32 (GLSL `float` cast), only narrowing back to FLOAT_TYPE at the point of use if required by downstream shared-memory layout. 4. Scope the change to the MMQ FA path only (this file, flash_attn_mmq_funcs.glsl) -- do not touch the non-MMQ FA shaders (flash_attn.comp, flash_attn_cm1/cm2.comp). 5. Add synthetic tiny-amplitude + normal-range + non-MMQ control test-backend-ops cases. 6. Add a deep-context Qwen model-level regression test (PPL) to catch any subtle accuracy change from the fp32 intermediate math. 7. Measure any performance regression from the extra fp32 conversion against normal-scale controls.

## Detailed Solution & Technical Design

The risk this item describes -- FP16 denormal/overflow corrupting the quantization scale reciprocal at long context / extreme value ranges -- is architecturally plausible given get_k_scale's real return type (FLOAT_TYPEV2, fp16 on fp16-capable devices per this file's own documented convention), but this planning pass could not run the shader to confirm the corruption actually manifests (no hardware access, per the brief's constraints) nor definitively prove the current pin lacks protection (there may be clamping/normalization elsewhere in the dequant path this excerpt doesn't show). The step-2 reproduction gate in this plan is therefore load-bearing: implementation must NOT proceed straight to a fix without first confirming the bug is real at this pin, per the item's own acceptance criteria ('carry only if ... the corruption case reproduces').

## Code Samples & Guidance

Real b11126 anchor (ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_mmq_funcs.glsl:46-54, exact):\n```glsl\n// Per-block scale/min, packed as (d, m). Single-scale types (Q4_0, Q5_0, Q8_0)\n// return (d, 0) so call sites always see the same shape.\nFLOAT_TYPEV2 get_k_scale(uint ib, uint a_offset) {\n    switch (FaTypeK) {\n        case GGML_TYPE_Q4_0: return FLOAT_TYPEV2(FLOAT_TYPE(k_packed_q4_0.data[a_offset + ib].d), 0.0);\n        ...\n        case GGML_TYPE_Q8_0: return FLOAT_TYPEV2(FLOAT_TYPE(k_packed_q8_0.data[a_offset + ib].d), 0.0);\n        default: return FLOAT_TYPEV2(0);\n    }\n}\n```\nFix sketch (compute in float regardless of FLOAT_TYPE, narrow only at the end if a caller truly needs FLOAT_TYPEV2 shared-mem layout):\n```glsl\nvec2 get_k_scale_f32(uint ib, uint a_offset) {\n    switch (FaTypeK) {\n        case GGML_TYPE_Q4_0: return vec2(float(k_packed_q4_0.data[a_offset + ib].d), 0.0);\n        ...\n    }\n}\n// call sites doing d_scale * something: do the multiply/reciprocal in float, cast down only for storage\n```\npatch.toml: id="<order>_rd59_vk_fa_mmq_fp32_scale", state="untested", backend="vulkan", plan-item="PRBE49", experiment-contracts=["UP-VK-002-FA-MMQ-FP32-SCALE"].

## Files

ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_mmq_funcs.glsl (get_k_scale and V-side counterpart), tests/test-backend-ops.cpp (new tiny-amplitude FLASH_ATTN_EXT synthetic cases), new package patches/<order>_rd59_vk_fa_mmq_fp32_scale/ (only to be authored if step 2's reproduction confirms the bug).

## Validation

Reproduction gate (must run before any fix is written): tiny-scale Q4/Q8 synthetic test-backend-ops case shows NaN/Inf at current pin. Correctness: reference FA output, no NaN/Inf post-fix, model PPL for tiny-scale synthetic and deep-context real cases. Performance: quantify regression vs normal-scale controls. This is one of the few items in this batch where hardware/shader execution is a genuine PREREQUISITE to writing the fix, not just to validating it -- flag this clearly to whoever picks up implementation.

## Effort & Risk

Unscored by the item itself (effort_risk was empty); set to S-M: the fix itself (if needed) is a small, localized shader change, but the reproduction-first requirement means real implementation time is dominated by hardware verification, not code.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance; correctness-first (do not carry a perf-motivated change that risks correctness without proof).

## Acceptance Criteria

Require exact/reference-quality FA output with no NaN/Inf on tiny-scale Q4/Q8 KV cases and no material normal-workload regression; otherwise retain or document existing equivalent protection.

## Notes

Supersedes: RD59
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd59

2026-09-24 relevance at b11126: no existing patch (RD59 grep = no hits). Evidence for TODO (not confirmed upstream-absorbed or confirmed necessary): flash_attn_mmq_funcs.glsl read directly at b11126, get_k_scale returns fp16-capable FLOAT_TYPEV2, matching the item's described risk shape, but no local reproduction was possible in this planning pass. GPT design requests this batch (req_6a45d0917b114112, req_3710dbae73fb49be) were unusable; plan authored directly from verified source with an explicit reproduction-first gate.

## Change Log

- 2026-09-09T10:56:53.658140+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:07.160220+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.349689+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.129637+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:10:31.384387+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031049_repaired-three-vulkanbackend_9010
- 2026-09-10T03:10:49.493059+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:47:36.272555+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
