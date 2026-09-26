---
id: PRBE50
order: 0
plan: patching-rdna-boost-experiments
state: deprecated
created-at: '2026-09-09T10:56:57.545335+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: null
---

# UP-VK-003: Quantized GET_ROWS view-offset correctness

## Description

Already-safe / no bug (re-dispositioned 2026-09-24 per GPT review req_e17e0bf5a68c48d5). The suspected bug is refuted: get_aoffset() is NOT a raw byte/element offset for GET_ROWS. Verified directly: host code computes `a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type)` (confirmed at multiple sites in ggml-vulkan.cpp, e.g. line 11342 `sp.misalign_offsets = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type)`, and generic_binary_head.glsl:42 `uint get_aoffset() { return p.misalign_offsets >> 16; }`). For quantized src0, ggml_type_size() returns the quant-block STORAGE size, so the host-side division already yields a block-index offset. The shader's `ib = a_offset + i00/QUANT_K` is therefore block-index + block-index -- dimensionally correct, not a unit mismatch. No shader patch needed.

## Steps

1. No implementation action -- host-side offset computation already divides by ggml_type_size (quant-block storage size for quantized types), making a_offset a block-index, matching the shader's ib addressing. 2. Optionally add nonzero block-aligned quantized-view GET_ROWS regression test cases (Q4_0/Q8_0, offset crossing block boundaries) to test-backend-ops.cpp to guard against future regression, if desired.

## Detailed Solution & Technical Design

This is the one item in the batch where direct source reading surfaced a plausible live correctness bug rather than just a feature gap -- but the plan correctly stops short of asserting it IS a bug, because get_aoffset()'s exact unit convention (bytes vs elements vs blocks) for quantized types was not confirmed in this pass (generic_binary_head.glsl was not read). Step 1 is the load-bearing first step: implementation must resolve this ambiguity before writing any shader change, since if get_aoffset() already accounts for quantization block sizing (which is plausible -- Vulkan backend code elsewhere in ggml-vulkan.cpp is disciplined about block-size-aware offset computation, e.g. the get_rows_quant.comp file's own d_offset/y_offset computations show careful QUANT_K/QUANT_R handling), then this item is actually already-safe (UPSTREAM-ABSORBED-equivalent) and should be re-dispositioned rather than patched.

## Code Samples & Guidance

Real b11126 anchor (ggml/src/ggml-vulkan/vulkan-shaders/get_rows_quant.comp, full file read, key lines):\n```glsl\nconst uint a_offset = get_aoffset() + i01*p.nb01 + i11*p.nb02 + i12*p.nb03;\nconst uint d_offset = get_doffset() + i10*p.nb21 + i11*p.nb22 + i12*p.nb23;\n\nconst uint ib = a_offset + i00/QUANT_K; // block index\nconst uint iqs = (i00%QUANT_K)/QUANT_R; // quant index\n```\nTo verify first (not yet read in this pass): `ggml/src/ggml-vulkan/vulkan-shaders/generic_binary_head.glsl`, function `get_aoffset()`. If it returns a raw byte/element offset (not pre-divided by QUANT_K), the fix is:\n```glsl\nconst uint ib = a_offset/QUANT_K + i00/QUANT_K; // a_offset assumed block-aligned by ggml view invariants\n```\npatch.toml: id="<order>_rd60_vk_get_rows_quant_offset", state="untested", backend="vulkan", plan-item="PRBE50", experiment-contracts=["UP-VK-003-GET-ROWS-QUANT-OFFSET"] (only to be authored if step 1 confirms a real bug).

## Files

ggml/src/ggml-vulkan/vulkan-shaders/get_rows_quant.comp, ggml/src/ggml-vulkan/vulkan-shaders/generic_binary_head.glsl (get_aoffset(), must read first), tests/test-backend-ops.cpp (GET_ROWS offset cases), new package patches/<order>_rd60_vk_get_rows_quant_offset/ (conditional on step 1).

## Validation

Step 1 verification gate (read generic_binary_head.glsl) must resolve before any code change. Correctness: exact reference rows for offset-zero and non-zero offsets across QUANT_K block boundaries for Q4_0/Q8_0 (and other quant types this shader handles), F16/F32/I32 GET_ROWS controls unaffected, no crash, no CPU fallback. Hardware (Brutus, not run here): GPU-residency confirmation and Qwen VL/TTS view-based gather smoke test if reachable.

## Effort & Risk

Unscored by the item itself; set to S: the shader fix (if needed) is a one-line addressing change, but correctness verification (block-alignment invariant, cross-type coverage) needs care since a wrong fix silently returns wrong rows rather than crashing.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance; fail-safe over silent wrong-data (this class of bug is worse than a crash).

## Acceptance Criteria

Require exact rows and stable GPU-resident execution for non-zero Q4/Q8 view offsets, including block-boundary cases, with no crash or material regression; otherwise retain fallback and keep item open.

## Notes

Supersedes: RD60
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd60

2026-09-24 relevance at b11126: no existing patch (RD60 grep = no hits). get_rows_quant.comp read in full and shows a plausible unit-mismatch (a_offset added directly to a block index) but generic_binary_head.glsl's get_aoffset() was NOT read in this pass, so whether this is a live bug or already-correct is UNRESOLVED -- flagged as the mandatory first implementation step rather than guessed at. GPT design requests this batch were unusable; plan authored directly from verified source.

2026-09-24 follow-up: generic_binary_head.glsl:42 `uint get_aoffset() { return p.misalign_offsets >> 16; }` -- offset is packed into the top 16 bits of a host-supplied push-constant `misalign_offsets`, NOT independently derivable from this shader file. Whether the host side (ggml-vulkan.cpp, the code that populates `misalign_offsets` for a GET_ROWS dispatch) already pre-divides by QUANT_K for quantized src tensors is still unresolved -- this needs a host-side grep (`misalign_offsets` in ggml-vulkan.cpp, specifically the vk_op_binary_push_constants / get_rows dispatch setup) as the concrete next step, not a shader-only read. Left as the mandatory step-1 verification per this plan's steps.

2026-09-24 GPT review req_e17e0bf5a68c48d5 applied: re-disposed from TODO/suspected-bug to already-safe -- verified host-side get_misalign_bytes(ctx,src0)/ggml_type_size(src0->type) at ggml-vulkan.cpp already produces a block-index offset for quantized types, matching the shader's block-index addressing; no shader patch needed.

## Change Log

- 2026-09-09T10:56:57.545335+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:11.442219+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.355299+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.136955+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:10:37.505870+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031049_repaired-three-vulkanbackend_9010
- 2026-09-10T03:10:49.504719+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:48:04.983117+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-24T04:48:34.124216+00:00 (updated-by): Updated: section:notes
- 2026-09-24T05:08:53.489292+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- 2026-09-24T05:08:56.364044+00:00 (state-transition): State: pending → deprecated
