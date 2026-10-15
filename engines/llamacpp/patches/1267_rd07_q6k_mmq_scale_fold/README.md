# 1267 RD07 Q6_K MMQ scale fold

**Status:** `untested`; b11552 `known_broken` / `FAILED_NEEDS_RECONCILIATION` in `dispositions/1267_rd07_q6k_mmq_scale_fold.json`. **Not production, not a successful current-pin application.** Owner: PRBE110 (code/rebase), PRBE04 (downstream PEF01/HI71 safety).

The b11126 extraction of RD07 from rejected 1203 hoists Q6_K row scales and folds per-k01 sub-scales before the j0 accumulation in `ggml/src/ggml-cuda/mmq-vec-dot.cuh`. The original and folded expressions differ in F32 association. Historical b11126 gfx1201 pp512 +2.4675%, CI95 [2.2776%,2.6508%], tg128 -0.0331%, CI95 [-0.0592%,-0.0071%]; contract verdict PASS but the record remains `validation_disposition=incomplete`, `eligible_for_validated_state=false` and has failed controls/performance checks. No b11552 or production gain is claimed.

## Exact b11552 rebase failures

- `patch.py::rd07-activation-marker` anchors on `mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream, forced_J)`, but b11552 plus validated 0300 dispatches `mul_mat_q_case<GGML_TYPE_Q6_K, GGML_PREC_Q8>(ctx, args, stream)`.
- `patch.py::rd07-jmax-env` expects `int ret = std::min(ne11, int64_t(512)); ret -= ret % 8;` in `mmq.cuh`; the old J-max selection is gone. Validated 0300 already chooses J before activation padding and owns `GGML_HIP_MMQ_FORCE_J`. Do **not** reintroduce a late J override.
- The MMA hoist/fold/sum source still has a corresponding unhoisted Q6_K path at pinned b11552 and current upstream master, but the whole package cannot be applied as-is.

**Bounded next action:** PRBE110 first proves an actual Q6_K model/kernel route and >=3% optimistic exposed E2E ceiling; otherwise close without code. If admitted, update only this existing package's marker, drop J-max edit, preserve the PEF01 gfx1100 J112 quarantine, gate transformed arithmetic to qualified gfx1201, and run composed source/host safety checks before any HIP benchmark. No duplicate J dispatcher or profiling framework.

**Provenance:** https://github.com/stew675/llama.cpp/commit/1d525bd45f9e8f844856ecbc5dd8ae33c8d34eff ; current RDNA-boosts block 07 carries the same fold and relocates its own J cap into native J search (not directly portable over BigCherry 0300). Source package `1203_rd050607_rdna4_wmma_fa_q6k_mmq` remains rejected.
