---
id: PRBE20
order: 0
plan: patching-rdna-boost-experiments
state: deprecated
created-at: '2026-09-09T10:54:49.394058+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P2
---

# Make decode and speculative-verify batches bit-identical (attention + mmvq + CPU cluster)

## Description

IMPLEMENTED-AS-PATCH, needs extension. Patch 1210_rd26_bitidentical_decode_verify_standalone (state=untested) already ports 2 of 5 fork commits. Real contract-gate hardware run (2026-09-13/14, gfx1100) confirms the FULL cross-batch bit-identity property still FAILS with just these 2 hunks (first_file_byte_mismatch=480) -- Wave 1 (flash-attn, commit 93510434f) and Wave 2 (RDNA4/RDNA3 MMVQ/SSM, commits 10b83d6b2/6cdf5aff9) are not yet ported. CORRECTED (source-verified): the Wave-1 verified anchor is `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1` in ggml/src/ggml-cuda/fattn.cu, whose specialization selection branches on `Q->ne[1] <= 8/ncols2`, `<= 16/ncols2`, `<= 32/ncols2` -- the exact invariant specialization boundary for n_q=1..8 must be pinned here, not left TBD. Wave-2's nwarps_explicit/rows_per_block_explicit HI09 template params are NOT b11126 source -- they are added by patch 0600_mmvq_geometry, so 1210 must declare `requires=["0600_mmvq_geometry"]` before anchoring on them (currently requires=[]). The "RD26-determinism build flag" referenced below is undefined; package activation itself is the gate (no separate compile/runtime flag exists or should be invented).

## Steps

1. Wave 1 (flash-attn, commit 93510434f): the verified anchor is `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1` in ggml/src/ggml-cuda/fattn.cu -- audit its Q->ne[1] <= 8/ncols2, <=16/ncols2, <=32/ncols2 specialization-selection branches and specify the exact invariant specialization to force for n_q=1..8 so decode (n_q=1) and speculative-verify (n_q up to 8) select the same code path. Force the same reduction order/specialization across both, gated to gfx1100/gfx1201.
2. Wave 2 (RDNA4/RDNA3 MMVQ, commits 10b83d6b2/6cdf5aff9): in ggml/src/ggml-cuda/mmvq.cu, calc_nwarps/calc_rows_per_block branch on ncols_dst (verified at b11126). Add a determinism-mode instantiation via the HI09 nwarps_explicit/rows_per_block_explicit template params -- CORRECTED: these params are introduced by patch 0600_mmvq_geometry, not raw b11126 source; add `requires=["0600_mmvq_geometry"]` to patch 1210's patch.toml before anchoring on them.
3. Apply Wave 1 and Wave 2 as additive Edit()s to patch 1210 (new anchors, gfx1100/gfx1201-gated), not a new patch package.
4. Define package activation itself as the gate -- CORRECTED: do not reference an undefined "RD26-determinism build flag"; the determinism behavior is active whenever patch 1210's Wave 1/Wave 2 edits are applied, full stop.
5. Build a real within-binary cross-batch raw-logit comparison harness extending run_rd26_decode_verify_bit_identity_check().
6. Re-run --run-rd26-contract on gfx1100 (then gfx1201/gfx1030) only after all 5 commits are composed.

## Detailed Solution & Technical Design

The acceptance property is full-cluster cross-batch bit identity, not PPL equality. Wave 1 and Wave 2 are additive Edit sets on the SAME patch 1210 package (not new packages), each gated to its own architecture scope. Wave 2's mechanism is now concretely identified: mmvq.cu's ncols_dst-keyed calc_nwarps/calc_rows_per_block dispatch (verified real source, not inferred) is the RDNA3/RDNA4 half of the determinism gap, and the project's own existing HI09 explicit-geometry template parameters (nwarps_explicit/rows_per_block_explicit, patch 0600_mmvq_geometry, already validated/in the core framework) are the natural mechanism to force matching geometry across ncols_dst values under an RD26-determinism flag -- this avoids inventing a new geometry-override path. Wave 1's fattn mechanism still needs direct source audit (not yet done this batch) before an anchor can be named with certainty.

## Code Samples & Guidance

Real b11126 anchor (verified), ggml/src/ggml-cuda/mmvq.cu, MMVQ_PARAMETERS_RDNA3_0 branch inside calc_nwarps:
```
if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
    if (ncols_dst == 1) {
        switch (type) {
            case GGML_TYPE_Q4_0: ... return 8;
            case GGML_TYPE_Q6_K: return 2;
            ...
        }
    }
    return 1;
}
```
and the existing HI09 template (mmvq.cu, ~line 616): `template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, int nwarps_explicit = 0, int rows_per_block_explicit = 0>`. RD26 Wave 2 Edit should add generated instantiations (via the tools/bigcherry/tuning/catalog.py enumerate_mmvq path, or a dedicated RD26-only Edit if catalog integration is out of scope) that set nwarps_explicit/rows_per_block_explicit to the SAME value for ncols_dst 1..8 for the whitelisted types, active only under an RD26-determinism build flag. Wave 1 fattn anchor: NOT YET VERIFIED -- flagged as open work, do not code against a guessed anchor.

## Files

ggml/src/ggml-cuda/fattn*.cu (Wave 1, anchor TBD); ggml/src/ggml-cuda/mmvq.cu (Wave 2, calc_nwarps/calc_rows_per_block + HI09 template, verified); patches/1210_rd26_bitidentical_decode_verify_standalone/patch.py (additive Edits); tools/bigcherry/tuning/catalog.py (if geometry variants are generated via the catalog rather than a standalone Edit); patches/1210.../validation/rd26_correctness.py (extend for within-binary cross-batch harness).

## Validation

1. patch-lint + rebase-check on the extended 1210 package. 2. Extend run_rd26_decode_verify_bit_identity_check() (or add a sibling) for a real within-binary decode-vs-verify raw-logit byte comparison, not just subject-vs-control. 3. Native/non-RD26 control unaffected (existing types/ncols_dst not in the determinism whitelist keep upstream's own nwarps/rows_per_block). 4. Hardware (Brutus, not run here): gfx1100 --run-rd26-contract must flip from FAIL to PASS once all 5 commits are composed; then gfx1201/gfx1030 per the existing blocked-until-complete decision.

## Effort & Risk

L: Wave 1 (fattn) anchor is unverified and is real open risk; Wave 2 (mmvq) is lower risk since it reuses the already-validated HI09 mechanism. Determinism-mode geometry forcing could itself regress throughput for the forced ncols_dst range -- must be gated to only apply when RD26 determinism is explicitly requested, never as a new default.

## Standards

Coordinated determinism change; no half-cluster acceptance; immutable reviewed post-fix source-state provenance for affected Wave-2 regions.

## Acceptance Criteria

Full cluster, not only standalone subset, produces byte-identical decode/verify outputs under declared paths; composition dependencies are satisfied; partial ports are not promoted.

## Notes

Supersedes: RD26
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd26



REAL HARDWARE CORRECTNESS CHECK RUN (2026-09-11/12), first-ever real execution of the materialized 2/5-commit subset (patch 1210_rd26_bitidentical_decode_verify_standalone): authored patches/1210_rd26_bitidentical_decode_verify_standalone/validation/rd26_correctness.py (reusing the shared tools/bigcherry/experiment/perplexity.py primitive, its fifth real caller) plus run_rd26_ppl_check() in validation_campaign.py.

HONEST SCOPE LIMIT, stated in the producer's own docstring and NOT to be conflated with PRBE20's real acceptance criteria: this check does NOT prove the actual cross-batch-size determinism claim (decode n_q=1 vs speculative-verify n_q up to 8 producing bit-identical logits against EACH OTHER) -- that needs a materially different test structure (a real within-binary cross-batch comparison), not implemented here. What it proves: the two ported kernel-routing hunks (MMVF batch threshold in ggml-cuda.cu, sgemm batch gate in llamafile/sgemm.cpp) do not regress ordinary single-token decode output.

Ran for real on Brutus: gfx1201, tierA-qwen4b-q6k, real wikitext2 corpus. Result: PASS, exact PPL match (10.4463 both subject and control, delta=0.0) -- no regression from the two ported hunks on ordinary decode.

PRBE20's full acceptance criteria (the real cross-batch bit-identity property across the FULL five-commit cluster, including the composition-gated flash-attn and RDNA4/RDNA3 hunks not yet ported) remain unmet -- this closes one real, narrow data point (no ordinary-decode regression from the standalone subset), not the item's actual scope. Patch state remains "untested".

Supersedes: RD26 (closed historical predecessor). Preserve all five immutable commit IDs as provenance. Existing 2/5 PPL/no-regression evidence remains narrow and does not satisfy PRBE20 acceptance. Do not split PRBE20; use internal Wave 1/Wave 2 sections.

REAL HARDWARE CONTRACT-GATE RUN (2026-09-13/14), first-ever real execution of --run-rd26-contract / run_rd26_decode_verify_bit_identity_check() (the actual raw-logit decode-vs-verify producer, distinct from the earlier PPL no-regression check already recorded above): ran on Brutus, gfx1100, patch 1210_rd26_bitidentical_decode_verify_standalone at current pin. All 5 base builds plus the dedicated rd26-bit-identity control/subject builds succeeded. Result: bit_identical check FAILS -- subject decode vs verify raw-logit artifacts differ, first_file_byte_mismatch=480. Contract correctness gate: not passed. Evidence committed (patches/1210_rd26_bitidentical_decode_verify_standalone/evidence/validation.json, commit 3ec5d033).

This FAIL is EXPECTED and already anticipated in run_rd26_decode_verify_bit_identity_check()'s own docstring: the materialized 1210 patch is only 2 of the full 5-commit cluster, so the complete cross-batch bit-identity property cannot hold yet. This confirms (does not newly discover) that the 2/5 standalone subset cannot satisfy PRBE20's real acceptance criteria -- consistent with, and now backed by direct contract-gate evidence rather than only the narrower PPL-based reasoning above. Patch 1210 state remains "untested" (never promoted); this is not a demotion.

Decision (per GPT design consultation, req_9b384a623ff24ab8): do not run gfx1201/gfx1030 --run-rd26-contract for the current subset -- more architectures would only produce more expected FAILs and add no new information. Block further 1210 contract-gate qualification runs until the remaining 3 commits (93510434f flash-attn, 10b83d6b2 RDNA4 MMVQ/fused SSM, 6cdf5aff9 RDNA3 MMVQ) are authored/composed into the full cluster per Wave 1/Wave 2 above. Resume all-three-architecture contract coverage only after that.

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH (1210, untested), needs Wave1/Wave2 extension. GPT design request req_bc106a2613754b6b (dev-gpt-agent, gpt-auto) submitted but queue-saturated/no response in-session for the fattn (Wave 1) anchor specifically -- that part of this plan is unverified and flagged; mmvq (Wave 2) part is grounded in real verified source. PRBE19 post-fix bake-in rule applies to the Wave 2 MMVQ region sourced from the historical fork commits.

2026-09-24 GPT review req_2b717df095b44703 applied: pinned the Wave-1 anchor to the verified real function ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1 (was TBD) and its exact ne[1] specialization thresholds; corrected the Wave-2 HI09 template params' provenance (introduced by patch 0600_mmvq_geometry, not raw b11126) and added requires=["0600_mmvq_geometry"] to patch 1210; removed the undefined 'RD26-determinism build flag' language in favor of package activation itself being the gate.

2026-09-25 (760f9d0f): the stew675 fork was squash-rebased -- 93510434f/10b83d6b2/6cdf5aff9 no longer exist; the determinism hunks live in 'block 08' (5efcd85f). Ported ONLY those into 1210 against b11126's own code (no longer gated on 1202/1203; no 0600 requirement -- calc_nwarps body is upstream code): wave 1 = fattn.cu WMMA gate adds && Q->ne[1] > 8, fattn-tile.cuh uses decode's cols_per_block=max(ncols2,2) for every n_q<=8 (an existing instantiation); wave 2 = RDNA3/RDNA4 calc_nwarps whitelist covers ncols_dst<=MMVQ_MAX_BATCH_SIZE. Fork's Q6_K nwarps 2->8 retune and fused SSM/prefill kernels deliberately NOT ported. Producer gained the contract's tg128 controls lane (controls could never pass before). Contract text updated; identity claimed for F16/BF16 KV only (quantized KV n_q 1..2 -> vec kernel, 3..8 -> tile). Hardware run pending (queue after current lanes).

placeholder-not-used-append-only

2026-09-28 REAL HARDWARE RUN (t-1210-gfx1100-s1, t-1210-gfx1201-s1, t-1210-gfx1030-s1): --run-rd26-contract re-run against the 2026-09-25 block-08 Wave1+Wave2 port (fork squash-rebase composition). Result: bit_identical check STILL FAILS on gfx1100 and gfx1201 (gfx1030 session ran too but result not separately distinguished in this note) -- subject decode/verify raw-logit artifacts differ, first_file_byte_mismatch=480, IDENTICAL byte offset to the earlier 2026-09-13/14 pre-block-08 run. Confirms the full Wave1+Wave2 port did not close the gap.

NEW ROOT-CAUSE CANDIDATE (source-verified 2026-09-28, not yet source-patched or hardware-confirmed): ggml/src/ggml-cuda/ggml-cuda.cu has a SECOND, separate call site into ggml_cuda_should_use_mmvf() that 1210's rd26a-mmvf-decode-verify edit does not cover. The patched call site is the plain dispatch path (`if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11))`). A distinct function, ggml_cuda_should_fuse_mul_mat_vec_f() (~line 1922, called from 3 op-fusion decision sites ~4093/4136/4299 -- FFN up/gate fusion), makes its OWN call: `ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1])`. For GGML_OP_MUL_MAT_ID (MoE) tensors this uses src1->ne[2] as the batch-size analog, completely bypassing 1210's ne11-clamp fix. The test model in every 1210/1268 hardware run this session was qwen3.6-35B-A3B, an MoE model -- so this unpatched fusion-decision gate is a strong candidate for the actual remaining divergence: FFN up/gate fusion can be selected differently for decode-batch (ne11=1) vs verify-batch (ne11 up to 8) on MoE tensors, producing different (each individually correct) floating-point reduction order/results.

This also reframes 1268 (PRBE52-ADAPTIVE-MTP-WIRING, requires=1210): its correctness-gate failure ("greedy MTP N tokens diverge", seen at token 19 in t-1268c-gfx1100-s1 and token 36 in earlier t-1268-gfx1100-s3/s4) is very likely INHERITED from this same unresolved 1210 gap, not a bug in 1268's own adaptive-depth wiring -- 1268's producer never actually passes --spec-draft-n-min-adaptive, so the adaptive controller (1255) was inert (fixed depth 4) in every failing session; the only thing that changed between control and subject in those runs was 1210's own kernel-routing patch set.

Next step (not yet done): add an Edit to 1210 for the ggml_cuda_should_fuse_mul_mat_vec_f() call site (same ne11-normalization treatment: clamp is_mul_mat_id ? src1->ne[2] : src1->ne[1] through the same <=MMVF_MAX_BATCH_SIZE logic before the mmvf-decision call), then re-run --run-rd26-contract on gfx1100 to confirm the byte-480 mismatch clears. If it does not, the fusion-gate hypothesis is falsified and further bisection (per the reviewer-gpt-agent's suggested fixed-N=1..4 diagnostic, req_c9780b77b9bc402b) is needed.

2026-09-28 (later same day) HARDWARE RE-VERIFICATION OF THE FUSION-GATE FIX: t-1210d-gfx1100-s1 (patch-refactor commit 1d3d7a9c, the rd26a-mmvf-fusion-decode-verify edit committed above). Result: bit_identical check STILL FAILS, IDENTICAL byte offset (first_file_byte_mismatch=480) as every prior run. The fusion-gate hypothesis is FALSIFIED as the (sole) cause -- patching ggml_cuda_should_fuse_mul_mat_vec_f()'s MMVF call site made no observable difference to the divergence.

Refined observation: byte offset 480 is suspiciously small and has now been IDENTICAL across three separate runs against two materially different source states (pre-block-08, post-block-08 Wave1+Wave2, and now post-fusion-gate-fix). Given results.cpp's file layout (GGUF header + one string KV + a `tokens` tensor descriptor+data (n_tokens * 4 bytes, int32) + a `logits` tensor descriptor+data), and the RD26 probe prompt tokenizing to roughly 100-120 tokens, byte 480 plausibly lands at or very near the START of the logits tensor's data section -- i.e. the divergence likely appears from essentially the FIRST decoded token's logit row, not partway through the sequence. This is consistent with the divergence being upstream of FFN/MoE routing entirely (e.g. attention or an even earlier op), which would explain why the FFN-fusion-gate fix had zero effect.

Not yet done: the reviewer-gpt-agent's originally-suggested bisection diagnostic (req_c9780b77b9bc402b) -- run fixed depths N=1,2,3,4 with adaptive fully disabled and compare pairwise, to determine whether ANY two different ubatch sizes diverge (implicating a structural batch-shape-dependent kernel somewhere in the model, not specific to RD26's targeted kernels) versus only decode(1) vs verify(>=2) diverging (implicating something specific to the ubatch=1 code path). Also not yet done: directly comparing the two output .gguf files' raw bytes at offset 480 (are these token IDs, or already logit floats -- would immediately confirm/refute the layout-offset reasoning above) rather than reasoning about expected layout.

Honest status: the actual root cause of the RD26 bit-identity gap is still NOT found. Two source-verified hypotheses (Wave1+Wave2 fork port, fusion-gate MMVF call site) have both been tried and both left the exact same byte-480 failure. This needs either a smaller, more surgical diagnostic (compare the two raw output files directly, don't just trust the pass/fail flag) or fresh investigation rather than another guess-and-check patch.

2026-09-28 DECISIVE DIAGNOSTIC RESULT (t-1210diag-gfx1100-s1, commit 00688dee, the new rd26-decode-verify-diagnostic.json raw-byte dump): the GGUF-structural hypothesis from the prior note is RULED OUT. Bytes 416-495 (the tail of the `tokens` tensor: int32 values 911,704,7538,26,635,4107,281,9318,893,463,13,0,0,0,0,0) are byte-identical between the decode and verify runs, confirming the mismatch is not in tokenization or file layout. The divergence begins exactly at the start of the `logits` tensor (offset ~496): the very FIRST float32 logit values already differ -- decode=[3.564, 4.229, 5.191, 4.570, 3.240, 6.146, 5.760, 6.775] vs verify=[3.684, 4.346, 5.143, 4.714, 3.280, 6.132, 5.701, 6.729] -- real ~2-4% relative differences, not rounding noise. This is a genuine numerical divergence present from the very first decoded token's logits (position 0), not something that accumulates over the sequence.

This rules OUT the FFN-fusion-gate hypothesis as well (that fix, already applied and re-verified to have zero effect on the byte-480 failure, targets MoE FFN routing -- a divergence present at token 0's FIRST logits is more consistent with something upstream, e.g. attention or embedding-adjacent, or a batch-width-dependent kernel whose accumulation order depends on total ubatch size regardless of position). Wave1+Wave2 (flash-attn WMMA/tile config, RDNA MMVQ nwarps) were supposed to cover attention already; either they're incomplete too, or the real cause is elsewhere entirely (RMSNorm, rope, or a kernel not yet audited).

Next step (not yet done, and the one still worth doing before another guess-and-check patch): the reviewer-gpt-agent's original bisection suggestion (req_c9780b77b9bc402b) -- run fixed depths N=1,2,3,4 (adaptive off) pairwise and see whether ANY two differ, or only ubatch=1 vs ubatch>=2 specifically. That would confirm whether this is a batch-width-general kernel issue or specific to the decode(1)-vs-verify(>1) boundary RD26 targets. The rd26-decode-verify-diagnostic.json artifact (now permanently captured on every future failing run, not just this one) removes the need to re-derive raw bytes each time -- future investigation can build on this evidence directly.

2026-09-28 UBATCH BISECTION RESULT (tools/lab/prbe20-rd26-bisect/bisect_ubatch.py, commit ae432965, run against the already-built subject llama-results binary from t-1210diag-gfx1100-s1, gfx1100, no new build): pairwise raw-logit comparison across ubatch sizes {1,2,3,4,5} on the RD26 probe prompt.

Result: EVERY pair diverges (no two ubatch sizes are bit-identical), but with a clear structure:
- ubatch {2,3,4} are mutually close: they match almost the entire file, first diverging from EACH OTHER only very late, at byte offset ~1,987,040 (near the end of the sequence, not the start).
- ubatch=1 diverges from every other size (2,3,4,5) starting at byte 480 (the very first logit row, as already established).
- ubatch=5 ALSO diverges from every other size (1,2,3,4) starting at byte 480 -- it behaves like an outlier, not like a member of the {2,3,4} group.

Ruled out: MMVQ_MAX_BATCH_SIZE and MMVF_MAX_BATCH_SIZE are both 8 (static_assert'd equal in ggml-cuda.cu), so ubatch=5 falling out of the {2,3,4} group is NOT explained by crossing either of RD26's own targeted kernel-selection thresholds.

Interpretation: this is NOT simply "decode(1) vs everything-else" as RD26's design assumes -- it's a genuinely more complex, still-unexplained batch-width sensitivity: (a) ubatch=1 is uniquely different from the very first token (consistent with prior findings), (b) ubatch=5 is ALSO uniquely different from the very first token despite being within RD26's claimed <=8 coverage scope and despite 2/3/4 being mutually consistent there, and (c) even within the {2,3,4} group that IS consistent early on, a SEPARATE, later divergence appears near the end of the sequence (~1.98MB in, likely deep into the ~256-token context, possibly attention-window or KV-position-dependent, not applicable to a single first-token logit explanation).

This means at least two distinct mechanisms are likely in play: an early (token-0) divergence affecting ubatch=1 and =5 specifically but not 2-4, and a separate late-sequence divergence affecting all of 2/3/4 (and presumably 1/5 too, just already diverged earlier so not separately visible). RD26's current kernel-routing scope (attention tile config, MMVQ/MMVF batch-size normalization, sgemm gate) does not fully explain either pattern. Further work needed: (1) why does ubatch=5 specifically fall outside the group RD26 successfully normalizes for 2-4 -- worth checking if 5 as verify_width=n_draft+1 hits some OTHER code path unrelated to the already-audited kernels (e.g. rope, KV cache layout, or a flash-attention config keyed differently); (2) what causes the late (~byte 1987040) divergence even among the mutually-early-consistent 2/3/4 group -- this is a second, separate root cause RD26 has not addressed at all.

Not yet done: source-level investigation of what specifically differs about ubatch=5's code path vs 2-4, and what happens deep in the sequence (near ctx_size=256) that causes the late divergence among 2/3/4. This is real, substantial follow-up work -- flagging as the next concrete step rather than guessing another fix.

## Change Log

- 2026-10-08 (triage): 1210_rd26_bitidentical_decode_verify_standalone is rejected on current main (patch.toml); PRBE20 measured cross-ubatch logits diverging at byte 480 (ubatch1/5) and late between 2/3/4. Existing strategy failed; new proposal needs a new identity, not unqualified 1210 rework.

- 2026-10-08 (triage): pending; 1210_rd26_bitidentical_decode_verify_standalone state=untested (patch.toml); PRBE20 ubatch={1,2,3,4,5} byte-bisection recorded cross-batch divergence including byte offset 480. No validated fix/in-flight patch; reset stale in_progress.

- 2026-09-09T10:54:49.394058+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:02.133012+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.219018+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.935699+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:51:52.755045+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025218_rdna-successors-prbe2022-now_5714
- 2026-09-10T02:52:18.396789+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-11T16:37:09.158298+00:00 (updated-by): Updated: section:notes
- chg_20260911_163714_real-hardware-test-confirms-no_2191
- 2026-09-11T16:37:14.220103+00:00 (updated-by): Updated: section:ledger-events
- chg_20260911_212645_documented-two-more-optimizati_8645
- 2026-09-11T21:26:45.055158+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T09:52:40.373896+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:notes
- chg_20260912_095506_cleaned-the-active-rdna-boost_4906
- 2026-09-12T09:55:06.254140+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T10:10:04.682892+00:00 (updated-by): Updated: section:standards
- chg_20260912_101015_fixed-the-remaining-plan-taxon_2574
- 2026-09-12T10:10:15.563853+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-13T17:29:17.167484+00:00 (updated-by): Updated: section:notes
- chg_20260913_172927_fixed-a-real-hardware-attestat_9172
- 2026-09-13T17:29:30.492149+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:29:43.023924+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes, section:effort_risk_2
- 2026-09-24T04:42:04.224768+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- 2026-09-24T15:40:35.936956+00:00 (state-transition): State: pending → in_progress
- 2026-09-24T15:40:39.804717+00:00 (updated-by): Updated: section:notes
- 2026-09-28T05:07:11.120532+00:00 (updated-by): Updated: section:notes
- 2026-09-28T05:07:36.641511+00:00 (updated-by): Updated: section:notes
- chg_20260928_050743_traced-the-root-cause-of-a-stu_6041
- 2026-09-28T05:07:46.299946+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-28T06:07:07.967232+00:00 (updated-by): Updated: section:notes
- 2026-09-28T06:46:25.005345+00:00 (updated-by): Updated: section:notes
- 2026-09-28T06:57:57.849652+00:00 (updated-by): Updated: section:notes
