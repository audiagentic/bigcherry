---
id: PRBE20
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:49.394058+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Make decode and speculative-verify batches bit-identical (attention + mmvq + CPU cluster)

## Description

IMPLEMENTED-AS-PATCH, needs extension. Patch 1210_rd26_bitidentical_decode_verify_standalone (state=untested) already ports 2 of 5 fork commits (MMVF ne11<=8 decision in ggml-cuda.cu; n<=8/n<2 rejection fix in llamafile/sgemm.cpp). Real contract-gate hardware run (2026-09-13/14, gfx1100) confirms the FULL cross-batch bit-identity property still FAILS with just these 2 hunks (first_file_byte_mismatch=480) -- expected, since Wave 1 (flash-attn, commit 93510434f) and Wave 2 (RDNA4/RDNA3 MMVQ/SSM, commits 10b83d6b2/6cdf5aff9) are not yet ported. GPT design request req_bc106a2613754b6b submitted for the remaining-commits design but the gpt-auto queue was saturated and did not return in-session; design below is a first-pass grounded in verified b11126 mmvq.cu source and must be reviewed by GPT/human before coding starts.

## Steps

1. Wave 1 (flash-attn, commit 93510434f): audit ggml/src/ggml-cuda/fattn*.cu kernel-selection-by-ncols_dst / tile-size logic for a code path that differs between n_q=1 (decode) and n_q up to 8 (speculative-verify) -- likely a different DKQ/tile-size or reduction-width chosen by batch size. Force the same reduction order/code path across both, gated to gfx1100/gfx1201.
2. Wave 2 (RDNA4/RDNA3 MMVQ, commits 10b83d6b2/6cdf5aff9): in ggml/src/ggml-cuda/mmvq.cu, calc_nwarps(type, ncols_dst, table_id) and calc_rows_per_block(ncols_dst, table_id) both branch on ncols_dst (confirmed at b11126: MMVQ_PARAMETERS_RDNA3_0 and MMVQ_PARAMETERS_RDNA4 table_id branches each special-case ncols_dst==1 with a wider nwarps=8 for whitelisted types, and return nwarps=1 for ncols_dst>1) -- this is the concrete mechanism that changes reduction-tree depth (warp_reduce_sum width) between n_q=1 and n_q>1, and is exactly why decode vs verify logits diverge. Add a determinism-mode instantiation (via the existing HI09 nwarps_explicit/rows_per_block_explicit template params already in mmvq.cu, patches/0600_mmvq_geometry) that forces the SAME nwarps/rows_per_block for ncols_dst in {1..8} when the RD26 contract gate is active, sourced per PRBE19's post-fix rule.
3. Apply this as additive Edit()s to patch 1210 (new anchors, gfx1100/gfx1201-gated), not a new patch package -- keep the five-commit cluster's identity in one place.
4. Build a real within-binary cross-batch raw-logit comparison harness: run decode (n_q=1) and speculative-verify (n_q=N<=8) on an identical prefix in the SAME process/build, extract raw logits for both, and byte-compare -- extending the existing run_rd26_decode_verify_bit_identity_check() producer (already used for the contract-gate run) rather than writing a new one, since it already does subject-vs-control artifact byte comparison; the gap is that the current control/subject pair is only 2/5 commits, not decode-vs-verify within one binary.
5. Re-run --run-rd26-contract on gfx1100 (then gfx1201/gfx1030 per the existing blocked-until-complete decision, GPT req_9b384a623ff24ab8) only after all 5 commits are composed.

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

## Change Log

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
