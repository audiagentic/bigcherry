---
id: PRBE55
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:18.934230+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# UP-VK-004: Small-N speculative execution avoids inappropriate MMVQ route

## Description

TODO. Route small-N (2..8) MTP/speculative verification widths through a decode-like Vulkan path instead of standard MMVQ when MMVQ harms throughput/acceptance. Relevance at b11126: no existing shape/verify-intent selector found; mul_mat_vec pipelines exist (ggml-vulkan.cpp ~2739-2751) and rm_kq-style row-count constants exist (see PRBE61) but there is no N-aware verify-path branch yet -- genuinely new dispatch logic, gated opt-in only.

TODO, corrected real selector. GPT identified the real, already-existing selector (superseding the prior 'no selector found' TODO): `bool quantize_y = ... && ggml_vk_should_use_mmvq(ctx->device, ne01, ne11, ne10, src0->type);` inside `ggml_vk_mul_mat_vec_q_f16()` (verified via `git -C work/upstream/llama.cpp.git grep -n quantize_y b11126 -- ggml/src/ggml-vulkan/ggml-vulkan.cpp` -- real hits at lines 6109, 6442, with `ggml_vk_should_use_mmvq` defined at line 6321). The existing vector path already supports `mul_mat_vec_max_cols = 8` (confirmed, line 2735/5301) -- this item's scope must be N=2..8 (not excluding N=6/7 as the prior draft did). The experimental route is concretely: force `quantize_y = false` for qualified small-N (2..8) matmuls, selecting the F32/F16 DMMV path instead of the Q8_1 MMVQ path -- not an invented 'verify_intent' semantic signal.

## Steps

1. Run the two mandatory anchor-discovery greps before writing any Edit: `git -C work/upstream/llama.cpp.git grep -n -E 'ne\[1\].*(mul_mat_vec|MMQ|mmvq|split_k)|(mul_mat_vec|MMQ|mmvq|split_k).*ne\[1\]' b11126 -- ggml/src/ggml-vulkan/ggml-vulkan.cpp` and `git -C work/upstream/llama.cpp.git grep -n -E 'pipeline_dequant_mul_mat_vec|mul_mat_vec_q|ggml_vk_mul_mat|ggml_backend_vk_mul_mat_id|split_k' b11126 -- ggml/src/ggml-vulkan/ggml-vulkan.cpp` to find the real matmul-route selector branch and paste it into the patch before finalizing anchors -- do not invent the selector block.
2. Also grep `git -C work/upstream/llama.cpp.git grep -n -Ei 'speculat|draft|verify|mtp|n_draft' b11126 -- src common ggml` to check whether a semantic speculative/verify-intent bit already reaches graph execution; if yes, thread that bit into the predicate instead of relying only on shape.
3. Implement `bigcherry_vk_use_smalln_verify_route(device, src0, src1, verify_intent)`: true only when BIGCHERRY_VK_SMALLN_VERIFY=1, verify_intent is true (semantic signal if found in step 2, else the env opt-in itself during qualification), device is AMD/RDNA via the existing device/vendor classifier, and src1->ne[1] is exactly one of {2,3,4,5,8}. N==1, N in {6,7}, and N>=128 must retain unchanged b11126 routing.
4. Hook the predicate immediately before the existing small-batch MMVQ/MMQ selection branch found in step 1; false -> unchanged existing code; true -> route through the same decode-like dequant/mul-mat-vector implementation the verified decode (N=1) path uses. First confirm that implementation accepts N>1; if it is structurally N==1-only, add a small-N adapter that loops the existing N=1 op over the N rows rather than feeding an unsupported shape in directly.
5. Add the activation marker (BIGCHERRY_PATCH_HIT patch=1262_prbe55 path=vk_smalln_verify n=<N>) immediately before the real selected launch/dispatch, gated by BIGCHERRY_PATCH_TRACE, once-per-process (atomic flag pattern, see patches/1204_rd08_q6k_mmvq_vdr2/patch.py's _ACTIVATION_MARKER for the established idiom).
6. Add MUL_MAT backend-op test cases at N={1,2,3,4,5,8,128} for the relevant quantized type; env off must reproduce baseline exactly; env on + verify_intent must only alter routing for N in {2,3,4,5,8}.
7. Add a temp-0 identity check: same model/prompt/context/seed, baseline vs subject at --temp 0, final token-ID sequence must match; record acceptance/verify-time separately and require no acceptance regression before promotion.

1. Read `ggml_vk_should_use_mmvq()` (ggml-vulkan.cpp:6321) in full to understand its existing N/shape/type decision logic before adding an override.
2. Implement the override as an explicit, narrowly-scoped experiment: `bigcherry_vk_force_dmmv_smalln(device, ne01, ne11, ne10, src0_type)` returning true only when BIGCHERRY_VK_SMALLN_DMMV=1 (env-gated, qualification-only), device is AMD, and ne11 (N) is in [2,8] inclusive (not excluding 6/7) -- AND the graph is confirmed to be within a speculative/MTP verification context if that signal is cheaply available at this call site (grep for any n_draft/speculative context reaching ggml_vk_mul_mat_vec_q_f16's caller chain); if no such signal reaches this point, explicitly scope the experiment as an env-gated AMD small-N matmul route for ALL eligible small-N matmuls, not just speculative verification -- do not invent a verify_intent parameter that does not exist in the real call chain.
3. Wire the override immediately before the existing `quantize_y = ... && ggml_vk_should_use_mmvq(...)` assignment at line 6442 (and the equivalent at 6109 if that call site also needs it -- confirm both by reading their surrounding functions): when the override predicate is true, force `quantize_y = false` regardless of `ggml_vk_should_use_mmvq`'s result.
4. Add the activation marker (BIGCHERRY_PATCH_HIT patch=<id> path=vk_smalln_dmmv n=<N>) immediately after the override forces quantize_y=false, gated by BIGCHERRY_PATCH_TRACE.
5. Add MUL_MAT backend-op test cases at N={1,2,3,4,5,6,7,8,128} for the relevant quantized type; env off must reproduce baseline exactly; env on must only alter routing for N in [2,8].
6. Add a temp-0 identity check: same model/prompt/context/seed, baseline vs subject at --temp 0, final token-ID sequence must match; record acceptance/verify-time separately (if a speculative context) and require no acceptance regression before promotion.

## Detailed Solution & Technical Design

Opt-in, shape+device+(semantic-or-env) gated dispatch insertion ahead of the existing Vulkan small-batch matmul selector. No change to default routing (N=1 decode, N>=128 prefill untouched). The real anchors (selector block, launch call) MUST be pasted from the step-1 grep output before the patch.py Edit() anchors are finalized -- placeholder anchors below are explicitly marked TODO-verify.

## Code Samples & Guidance

patches/1262_prbe55_vk_smalln_verify_route/patch.toml:
```toml
schema = 1
id = "1262_prbe55_vk_smalln_verify_route"
order = 1262
state = "untested"
kind = "enhancement"
origin = "local"
backend = "vulkan"
plan-ids = ["PRBE55"]
requires = []
conflicts = []
requires-options = []
forbids-options = []
subsystems = ["vulkan", "mul-mat", "speculative"]
hardware = ["amd"]
validation-architectures = []
backends = ["vulkan"]
```
patches/1262_prbe55_vk_smalln_verify_route/patch.py (skeleton -- anchors marked TODO-verify must be replaced with real pasted b11126 text from step 1 before this patch is written):
```python
from bigcherry.patcher import Edit, FilePatch

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-vulkan/ggml-vulkan.cpp",
        description="PRBE55 opt-in small-N verification routing",
        edits=(
            Edit(
                id="prbe55-smalln-predicate",
                anchor=r"<TODO-VERIFY: exact unique anchor immediately before the real matmul route selector, from step 1 grep>",
                rationale="add env-gated AMD/shape/verify-intent predicate without changing default routing",
                mode="insert_before",
                text=r"<TODO-VERIFY: bigcherry_vk_use_smalln_verify_route() helper using verified b11126 device/tensor field names>",
                guard=r"bigcherry_vk_use_smalln_verify_route",
            ),
            Edit(
                id="prbe55-selector-route",
                anchor=r"<TODO-VERIFY: exact selector block from step 1>",
                rationale="route verified N=2/3/4/5/8 through the decode-like path; preserve all other branches",
                mode="replace",
                text=r"<TODO-VERIFY: old block + smallN branch + activation marker>",
                guard=r"BIGCHERRY_PATCH_HIT patch=1262_prbe55",
            ),
        ),
    ),
    FilePatch(
        path="tests/test-backend-ops.cpp",
        description="PRBE55 small-N MUL_MAT correctness coverage",
        edits=(
            Edit(
                id="prbe55-smalln-cases",
                anchor=r"<TODO-VERIFY: real test-case insertion anchor>",
                rationale="cover N=1/2/3/4/5/8/128 routing boundaries",
                mode="insert_before",
                text=r"<TODO-VERIFY: test_mul_mat cases for N in 1,2,3,4,5,8,128>",
                guard=r"PRBE55.*small-N",
            ),
        ),
    ),
]
```

## Files

ggml/src/ggml-vulkan/ggml-vulkan.cpp; tests/test-backend-ops.cpp; patches/1262_prbe55_vk_smalln_verify_route/{patch.toml,patch.py,SUMMARY.md,validation/producer.py}

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check --focal-overlay 1262_prbe55_vk_smalln_verify_route --source bigcherry-tuning`. Unit: MUL_MAT backend-op cases N={1,2,3,4,5,8,128}, env-off byte-identical to baseline, env-on only alters {2,3,4,5,8}. Hardware (Brutus, not run here): R9700/XTX MTP replay, temp-0 identity, kernel path/verify-time/effective-TG/acceptance across widths and contexts via `python -m bigcherry.patch.validation_campaign`.

## Effort & Risk

M / medium -- new dispatch branch touching hot matmul path; risk contained by opt-in env gate and byte-identical-when-off requirement.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Require temp-0 identity, no unexpected acceptance loss, and repeatable verify/TG improvement before selecting the small-N route; use shape-based gating and retain baseline otherwise.

## Notes

Supersedes: RD65
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd65

2026-09-24 relevance at b11126: no existing small-N verify-path selector found (grep for ne[1]+mul_mat_vec/MMQ/mmvq/split_k patterns not yet run against real selector body -- flagged as mandatory step 1 in this plan, not yet executed by this planning pass). GPT design request req_63a12ebff6544a0a (batched with PRBE62).

2026-09-24 GPT review req_2b65d50ebe9547fd applied: NOT-READY -- replaced 'no selector found' premise with real anchor: quantize_y = ... ggml_vk_should_use_mmvq(...) at ggml-vulkan.cpp:6442/6109; corrected scope to N=2..8 (was wrongly excluding 6/7); removed invented verify_intent, scoped as env-gated AMD small-N DMMV-forcing experiment.

## Change Log

- 2026-09-09T10:57:18.934230+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:32.113186+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.376629+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.175315+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:12:04.636398+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031217_repaired-four-more-active-succ_7909
- 2026-09-10T03:12:17.975805+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:30:11.685849+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:43:09.291591+00:00 (updated-by): Updated: section:description, section:steps
- 2026-09-24T04:43:18.780208+00:00 (updated-by): Updated: section:notes
