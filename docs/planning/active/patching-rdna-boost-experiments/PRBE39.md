---
id: PRBE39
order: 0
plan: patching-rdna-boost-experiments
state: in_progress
created-at: '2026-09-09T10:56:09.423127+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-FUS-003: Fuse GEMV -> view -> residual ADD

## Description

NOT-READY per GPT implementation-readiness review (req_e17e0bf5a68c48d5): existing patches/1206_rd13_mul_mat_add_view_fusion/ already implements the core MUL_MAT->RESHAPE->ADD MMVF/MMVQ fusion using ggml_cuda_mm_fusion_args_host.x_bias, gated on ggml_cuda_should_fuse_mul_mat_vec_f/q -- the same GEMV-detection predicates this plan's own design and GPT's design independently arrived at. PRBE39's 'no existing patch' premise is false. Re-scoped: this is now an EXTENSION/qualification of 1206, not a new package. 1206 lacks: GGML_OP_VIEW support (only RESHAPE), an explicit ggml_cuda_check_fusion_memory_ranges call on the 3-node span, and uses GGML_LOG_INFO (not GGML_LOG_WARN) for its activation marker.

## Steps

1. Read patches/1206_rd13_mul_mat_add_view_fusion/patch.py IN FULL (not just the previously-sampled lines 127-148) and diff its matched pattern/gating against AMD-FUS-003's requirements. 2. Extend 1206's has_view check from RESHAPE-only to RESHAPE|VIEW: for the VIEW case require view_src==mm_node, view_offs==0, contiguous, identical byte layout/nbytes to the mul_mat output. 3. Add an explicit ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, 3, out_nodes, 1) call on the 3-node span before accepting (1206 currently lacks this). 4. Change 1206's activation marker from GGML_LOG_INFO to GGML_LOG_WARN (matching BIGCHERRY_PATCH_TRACE convention used by 1203/1204). 5. Add negative test cases: VIEW with nonzero offset, strided/non-contiguous VIEW, aliased residual -- must show NO fusion. 6. Re-run patch-lint/patch-rebase-check on the modified 1206 package. 7. Do NOT author a new patches/<order>_rd47_gemv_view_add_fuse/ package -- this work lands inside 1206.

## Detailed Solution & Technical Design

b11126's ggml_cuda_can_fuse(cgraph, node_idx, ops, unary_ops) matches a fixed initializer_list<ggml_op> against the next N graph nodes via is_equal(), then calls a should_fuse_* predicate for semantic/shape gating, then ggml_cuda_check_fusion_memory_ranges for alias safety, mirroring existing patterns e.g.:\n```cpp\nstd::initializer_list<enum ggml_op> mul_mat_glu_ops = { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU };\nif (is_equal(mul_mat_glu_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {\n    ... if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) { ... ggml_cuda_check_fusion_memory_ranges(...); }\n}\n```\nAdd a new branch of the same shape for {GGML_OP_MUL_MAT, GGML_OP_VIEW, GGML_OP_ADD} (decode GEMV: mul_mat->ne[1]==1). The should-fuse predicate must positively prove the view is an exact byte-identical reshape (same total bytes, same stride at dim0, no padding gap) and that the residual destination is not read by any node between the mul_mat and the add (ggml_cuda_check_fusion_memory_ranges covers general overlap; this needs an additional 'no other reader of the residual buffer in this span' check since aliasing safety here is about write-destination reuse, not just read/write overlap of the fused span itself). The fused kernel simply changes the GEMV kernel's output pointer/stride to write straight into the add's destination tensor's memory (offset by the view's byte offset) instead of allocating an intermediate result, then performs an in-place add of the residual on top (or, if the add can be folded as a post-GEMV epilogue add, do the add inline in the GEMV kernel's final write -- the simpler and lower-risk implementation is the latter: pass the residual pointer and destination pointer to the existing mul_mat_vec/mmvq kernel as an optional epilogue-add argument, analogous to how mmvq.cu's ggml_cuda_op_mul_mat_vec_q already threads extra per-launch args through mul_mat_vec_q_switch_type). Reject (fall back to the 3 separate nodes) whenever any of: view non-contiguous, dtype mismatch between GEMV output and add destination, ne1 != 1 (not decode/GEMV shape), or the residual tensor has another reader inside the matched span.

## Code Samples & Guidance

Target: ggml/src/ggml-cuda/ggml-cuda.cu, inside ggml_cuda_can_fuse (~line 3180-3270 at b11126) and ggml_cuda_try_fuse (~line 3432+). New matcher (structural sketch, anchors to be re-verified against live source at implementation time -- this file changes often):\n```cpp\nstatic bool ggml_cuda_should_fuse_gemv_view_add(const ggml_tensor * mul_mat, const ggml_tensor * view, const ggml_tensor * add) {\n    if (mul_mat->ne[1] != 1) return false; // GEMV only (decode)\n    if (!ggml_is_contiguous(view) || view->nb[0] != mul_mat->nb[0]) return false;\n    if (ggml_nbytes(view) != ggml_nbytes(mul_mat)) return false;\n    if (add->type != mul_mat->type || add->type != GGML_TYPE_F32) return false;\n    // residual operand must not be read again inside [mul_mat, add]\n    return true; // + ggml_cuda_check_fusion_memory_ranges at the call site\n}\n\n// in ggml_cuda_can_fuse():\nstd::initializer_list<enum ggml_op> gemv_view_add_ops = { GGML_OP_MUL_MAT, GGML_OP_VIEW, GGML_OP_ADD };\nif (is_equal(gemv_view_add_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {\n    const ggml_tensor * mul_mat = cgraph->nodes[node_idx];\n    const ggml_tensor * view    = cgraph->nodes[node_idx + 1];\n    const ggml_tensor * add     = cgraph->nodes[node_idx + 2];\n    if (ggml_cuda_should_fuse_gemv_view_add(mul_mat, view, add)) {\n        int out_nodes[] = { node_idx + 2 };\n        return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);\n    }\n}\n```\nActivation marker (mirrors 1203/1204 precedent):\n```cpp\nif (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {\n    static std::atomic_flag bigcherry_rd47_logged = ATOMIC_FLAG_INIT;\n    if (!bigcherry_rd47_logged.test_and_set(std::memory_order_relaxed)) {\n        GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=<order>_rd47 path=gemv_view_add_fuse contract=AMD-FUS-003\\n");\n    }\n}\n```\npatch.toml: schema=1, id="<order>_rd47_gemv_view_add_fuse", state="untested", kind="enhancement", backend="hip", plan-item="PRBE39", experiment-contracts=["AMD-FUS-003-GEMV-VIEW-ADD-FUSE"], requires=[].

## Files

patches/1206_rd13_mul_mat_add_view_fusion/{patch.py,patch.toml,SUMMARY.md} (extend, do not create new package); ggml/src/ggml-cuda/ggml-cuda.cu (reference only, read the live ggml_cuda_can_fuse/ggml_cuda_try_fuse and ggml_cuda_mm_fusion_args_host.x_bias mechanism 1206 hooks into); tests/test-backend-ops.cpp (new VIEW negative-test cases appended to 1206's existing fusion tests).

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check --focal-overlay <id> --source bigcherry-tuning`. Correctness: test-backend-ops MUL_MAT+VIEW+ADD graph-capture-replay parity vs the 3-node baseline for contiguous positive cases; explicit negative-control cases (non-contiguous view, residual reused, ne1>1) must show NO fusion (verify via the activation marker not firing under BIGCHERRY_PATCH_TRACE=1). Hardware (Brutus, not run here): decode-path hybrid/GDN model (e.g. a Mamba/GDN hybrid Qwen variant) launch-count and TG timing vs unfused control, graph-capture replay must not abort.

## Effort & Risk

M-L (downgraded from the item's own M given the design is now scoped to an epilogue-add kernel argument rather than a new standalone fused kernel) -- but real risk remains in aliasing proof correctness (silent wrong-residual bugs are the failure mode) and in confirming which concrete GEMV kernel path (mmvq vs mmv F16/F32) hybrid/GDN decode actually uses before wiring the epilogue add.

## Standards

Preserve graph capture correctness, fail-closed unsupported-layout fallback, activation-evidence convention (BIGCHERRY_PATCH_TRACE atomic_flag + GGML_LOG_WARN), package-only edits.

## Acceptance Criteria

Acceptance requires exact GEMV->view->residual mapping and output/alias parity, graph-capture replay success, rejection of unsupported strides or ambiguous aliases, and repeatable launch/TG improvement versus control; otherwise retain fallback.

## Notes

Supersedes: RD47
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd47

Supersedes RD47. Keep separate from PRBE12/RD13: that existing mul_mat+add view fusion is a different operation and SSM graph shape.

2026-09-24 relevance at b11126: no existing patch (patches/*/patch.toml grep for RD47 = no hits); not upstream-absorbed (ggml_cuda_can_fuse's full pattern list at b11126 read directly, no GEMV->view->add case exists). GPT design requests req_6a45d0917b114112 (failed, EXT-GPTAUTO-003/composer-operation-timeout) and req_3710dbae73fb49be (retry, exceeded reasonable wait under the single-in-flight cap, abandoned unresolved) both did not yield usable output; this plan was authored directly by the planning agent from verified b11126 source (ggml-cuda.cu:3180-3270,3432+) instead, per the brief's fallback guidance to resubmit once and proceed. A future session should re-run agent_task_response(req_3710dbae73fb49be) opportunistically in case it eventually completed, and cross-check this design against it.

2026-09-24 GPT design retry req_3710dbae73fb49be COMPLETED (succeeded after ~9 min, well past the earlier abandonment point -- session ses_f9fb6da0779240e3). Its design is a stronger, more concrete alternative/supplement to this plan's self-authored design above: matches {MUL_MAT, VIEW|RESHAPE, ADD} via a ggml_cuda_gemv_residual_match struct checking view_src==mm/view_offs==0/same-shape/F32/contiguous/equal-nbytes, gates on the EXISTING ggml_cuda_should_fuse_mul_mat_vec_q/f predicates (real functions this session confirmed at ggml-cuda.cu:1767/1794), and implements the fused write via ggml_cuda_mm_fusion_args_host.x_bias (an existing epilogue-bias mechanism) rather than a new kernel -- i.e. it reuses MMVQ/MMVF's existing bias-add epilogue to write the residual directly, which is a materially better (lower-risk, less new code) implementation than this plan's original 'thread a new epilogue arg' sketch. GPT's anchors are explicitly marked tentative (e.g. anchoring on 'static bool ggml_cuda_can_fuse(' as an insert point) and were NOT independently re-verified against live b11126 by this session due to time constraints -- whoever implements MUST re-grep every anchor before use, per this plan's standing rule. GPT's test design (test_mul_mat_view_residual, GGML_CUDA_DISABLE_FUSION=1 differential) and patch.toml sketch are usable as-is. Prefer GPT's ggml_cuda_mm_fusion_args_host.x_bias mechanism over this plan's original epilogue-threading sketch if implementation confirms that struct/field exists at b11126 (not yet confirmed).

2026-09-24 CRITICAL OVERLAP FINDING (found via functional keyword grep, per the lesson learned on PRBE40/1205): patches/1206_rd13_mul_mat_add_view_fusion/ (state=untested, plan-ids=["RD13"]) already extends ggml_cuda_try_fuse to fuse {MUL_MAT, GGML_OP_RESHAPE, add/bias} and its patch.py (read directly, lines ~127-148) explicitly gates on `ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)` / `ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)` -- the EXACT SAME GEMV-detection predicates this plan's own design (and GPT's) independently arrived at for AMD-FUS-003. This is either (a) 1206 already substantially implements PRBE39's functional goal (GEMV -> reshape -> add fusion), making PRBE39 IMPLEMENTED-AS-PATCH (qualify 1206, don't write a new package), or (b) there is a real, narrower distinction this planning pass did not fully resolve -- this item's own pre-existing notes state '1206_rd13/RD13's existing mul_mat+add view fusion is a different operation and SSM graph shape', implying a prior reviewer found a real difference (possibly: RESHAPE only, not VIEW; or 1206 fuses the reshape+add generically for ANY mul_mat shape including full GEMM, while AMD-FUS-003 is decode/GEMV + exact-byte-mapping specific with a stricter aliasing proof). 1206 is currently BLOCKED for promotion per its own SUMMARY.md (correctness/activation evidence exists, performance/controls lanes intentionally absent, PA36/PA40). MANDATORY next step before any new-package work on PRBE39: read patches/1206_rd13_mul_mat_add_view_fusion/patch.py IN FULL (only lines 127-148/254 were sampled here) and diff its exact matched pattern/gating against AMD-FUS-003's stated requirements (byte-strides, dtype, destination aliasing, residual read/write ordering) to determine whether PRBE39 should be re-dispositioned to 'extend/unblock 1206' rather than 'author a new patch' -- this planning pass ran out of time budget to complete that full comparison and is leaving TODO as the disposition (per the brief's own 'if unsure between TODO and absorbed, choose TODO and say what to check first' rule) rather than guessing. Whoever picks this up should very likely start from 1206, not from GPT req_3710dbae73fb49be's from-scratch matcher design.

append

2026-09-24 GPT review req_e17e0bf5a68c48d5 applied: re-scoped PRBE39 from new-package TODO to extend/qualify 1206_rd13_mul_mat_add_view_fusion (add VIEW support with strict byte-map checks, explicit ggml_cuda_check_fusion_memory_ranges call, GGML_LOG_WARN marker, negative tests); do not author a new package.

2026-09-25 implemented in patch 1206 (commit 'accept zero-offset contiguous VIEW...'): has_view now accepts GGML_OP_RESHAPE or a GGML_OP_VIEW with view_src==matmul, view_offs==0, both contiguous, equal nbytes; ops[1] uses the real mid op; ggml_cuda_check_fusion_memory_ranges(cgraph,i,3,&(i+2),1) guards the view form; markers WARN. Qualified together with PRBE12 (same package) -- see PRBE12 notes for gfx1100 result. Negative-fixture test-backend-ops cases (nonzero-offset/strided VIEW, aliased residual) not yet authored.

## Change Log

- 2026-09-09T10:56:09.423127+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:25.553774+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.306463+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.063889+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:02:57.572846+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- chg_20260910_030318_repaired-three-patching-succes_9681
- 2026-09-10T03:03:18.986477+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:05:36.573029+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20260910_030619_removed-migration-placeholder_7703
- 2026-09-10T03:06:19.301698+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:43:52.380644+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-24T04:49:01.179486+00:00 (updated-by): Updated: section:notes
- 2026-09-24T04:50:54.912778+00:00 (updated-by): Updated: section:notes
- 2026-09-24T05:07:15.352025+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:notes
- 2026-09-24T05:07:21.603909+00:00 (updated-by): Updated: section:notes
- 2026-09-24T14:09:43.348776+00:00 (state-transition): State: pending → in_progress
- 2026-09-24T14:09:46.252839+00:00 (updated-by): Updated: section:notes
- chg_20260924_141016_five-experimental-rdna-patches_5706
- 2026-09-24T14:10:21.943775+00:00 (updated-by): Updated: section:ledger-events
