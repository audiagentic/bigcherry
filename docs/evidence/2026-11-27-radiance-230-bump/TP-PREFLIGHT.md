# Tensor-parallel chain: additional executable prework

2026-11-27; BigCherry aaba6b2f6; exact target 3b13fd32663ef52252a7f0be255523480aa22ffe. [Reproducer](tp-preflight.py) reads exact Git blobs into temporary scratch, uses existing patch objects/patcher, and applies only the selected prefix with local anchor edits in memory. [Captured successful output](tp-preflight.txt). Run `python3 docs/evidence/2026-11-27-radiance-230-bump/tp-preflight.py --mirror <upstream-git> --scratch <existing-executable-scratch>`. No production patch modifications, no full engine build, no device runs. The first attempt used the wrong include spelling; corrected scratch include directory mirrors arch/CMakeLists.txt's installed `<arch/...>` header arrangement. Successful job 2 exit 0; its stdout was captured before job history reset. This is a selected prefix, not the full serving recipe.

## 1. rad0016/17/18: exact candidate patch application

All 43 + 12 + 17 edits apply and reapply with no failures (72 edits). rad0016 uses the two fetches_rows anchors recorded in PREFLIGHT.md; rad0017 unchanged. rad0018 ABI insertion follows dspark_walk in both struct and fingerprint. Its three BF16 declaration edits must insert the existing span block after the new weight-handle assignment rather than replace upstream's decl_linear_bf16 calls:

| Edit | Exact new insertion anchor (including existing indentation and newline) |
|---|---|
| n2-attn-spans | `    w_o = out.w;` |
| n2-gdn-spans | `    w_out = out.w;` |
| n2-mlp-spans | `    w_down    = dn.w;` |

Preserve the old enhancement body and guard, not the old declaration prefix. The new upstream declaration supports lossless BF16x12 via LinearBF16 and dtype; do not undo it or assume all BF16-path weights are plain BF16. The actual patched qwen35_bf16.cpp passes g++ -std=c++17 -fsyntax-only against target ABI and architecture headers. The existing TpOwnershipTests.test_spec_parser also passes (1 test, 0.087 s, executable scratch TMPDIR); it exercises the unchanged real ownership parser, not the new declarations. No link, graph/weight-loader or numerical proof yet. Retain existing BF16 graph and asymmetric span tests; if BF16x12 selected, verify the shard semantics of its packed encoding before claiming that new upstream combination is supported.

## 2. rad0019: targeted rename and semantic review

Audit on the successfully patched prefix: retarget arch/common/rad_block_attn_gated_fp8.h -> rad_block_attn_gated_q.h, rad_block_mlp_fp8.h -> rad_block_mlp_q.h and rad_fp8.h -> rad_linear.h. 12/15 edits match expected anchor count; 3 drift. Anchor count is not semantic compatibility and the package has **not** been applied. Exact missing anchors:

- n2f-dense-scale-view-field: DenseQ now uses g8 = RAD_SCALE_GROUP; w_rb follows it. Put the same w8h_s_full member between these fields; keep default false.
- n2f-dense-scale-view-probe: existing declare_w8h now owns the code; insert after its four-space-indented const bool iq8r = linear_is_iq8r(enc); and before w = decl_view. Set w8h_s_full = linear_w8h_rowf(enc) || linear_w8h_scale2(enc) || iq8r, matching actual branch shapes: full logical [N,K] versus compact [N,K/g8]. Test reset if object reused.
- n2f-drafter-geom: upstream wraps divide_or_fail(n_head_all, ...) on two lines; reanchor against target exact even block, and wrap that exact block as the else arm; retain native kv replication, vocabulary refusal and newer metadata handling outside the block.

Two exact anchors need substantive review, not just a file rename:

- n2f-dense-span: DenseQ still declares mx, ref, w8h, g8; old body references removed m2x and RAD_FP8_BLOCK. New name RAD_SCALE_GROUP is 128, while actual g8 is probed per encoding. Keep native three-argument span() for native callers; reconcile the four-argument row/column overload for local calls in rad_block_dflash2.h (w_q/k/v/o/gu/dn). Row spans on full or compact IQR scale views use row indices; column spans on full logical scale views use columns, compact views divide by g8; require actual group and 16-row alignment before issuing spans, and propagate failure. For MXFP4, codes use model coordinates, scale/ref use original rows or K/32 columns. Validate with a tiny host fixture invoking actual candidate method with recording rad_weight_shard_span fake: valid row/column, invalid boundaries, scale/ref IDs, exact first/last ranks, full-vs-compact. Keep native row-only overload unchanged until separately proven; do not assert equivalence simply because the old and new names resemble one another.
- n2f-span-parts-wref: target LinearQ has w/ws and ws_row_div but no wref. Old insertion would compile-fail. Target LinearQ refuses retired encodings before declaring them; the ref plane still exists on DenseQ, not LinearQ. Document the old capability and decide whether an explicit supported path exists. Do not mechanically insert wref, restore the old quant engine, or silently discard the active enhancement. An actual reduction of supported capability needs owner-approved disposition. Preserve the surviving DenseQ ref spans.

**Important upstream exception:** target arch/common/rad_linear.h DenseQ::declare checks `mx = known && linear_is_mxfp4(enc)` and explicitly exempts plain MXFP4 from linear_retired because published IQR-V4 containers use DFlash2 drafters in this format. General LinearQ retirement is not proof that all MXFP4 kernels are unreachable. Current four configured files did not expose retired plane sets, but optional supported drafter paths and local MXFP4 enhancements remain part of the preservation requirement. Check actual callers, layout/registry, and draft graph before any retirement; no changes to supported model files or speculative settings. This corrects the overly broad encoding-retirement interpretation.

Also flag rad0028 rcd-buf: target qwen35_fp8.cpp now computes nodes = dc.tree_nodes > 0 ? dc.tree_nodes : dc.steps() before declaring dflash_tokens; old patch anchors on dc.steps() alone. Preserve native tree proposal width. Pad/gather only the existing non-tree rank-consistent path with correct effective width; validate tree vs linear path with small shape and declaration test, no guessed change to draft counts. rad0020 and downstream chain are not yet qualified because rad0019 remains unresolved.

## 3. Next implementation checkpoint

Use this preflight to port the exact rad0016/17/18 anchors first; apply the separate measured ABI guide when composing all four ABI packages. For rad0019, resolve the three drift anchors plus the two semantic conflicts above, then run host syntax on actual qwen35_fp8 and BF16 translation units, small span oracle, and existing patch fixtures. Only then compose rad0020/25/26/27/28/29/30, preserving source class and feature behavior. No repeated engine builds. Actual worker/collective and integrated release checks remain required only at affected boundaries. Evidence here does not qualify any production patch or justify retiring a capability.
