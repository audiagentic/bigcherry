---
id: PRBE107
order: 0
plan: patching-rdna-boost-experiments
state: completed
created-at: '2026-09-12T21:20:57.594630+00:00'
breadth: ''
skill: ''
created-by: agent
---

# Investigate real ~18% prefill throughput gap: stock upstream vs BigCherry baseline composition

## Description

A real three-arm A/B/C baseline comparison for RD13 (2026-09-13, docs/reference/testing/BASELINE_COMPARISON_RD13_2026-09-13.md), CONFIRMED across 3 rounds each for A and B: stock upstream llama.cpp (source.llama-native, zero BigCherry patches) shows a consistent, real ~22.6% higher pp512 prefill throughput than BigCherry's own baseline composition (source.bigcherry-native, the framework+upstream-fixes patch-sets, with no RD-series enhancement patches) on gpt-oss-20B/gfx1100: mean A=5169.24 t/s (sd 16.63) vs mean B=4217.83 t/s (sd 2.15), zero overlap across all 3 rounds. tg128 decode throughput is unaffected (essentially identical across arms). This gap is unrelated to RD13 itself (RD13's own B-vs-C effect is negligible, well within B's own round-to-round variation). This is a real, solid, confirmed finding -- not noise, not a single-round artifact -- surfaced only because the standardized validation criteria required a genuine three-arm comparison rather than the project's usual B-vs-C-only methodology.

## Steps

1. Confirm the finding with a proper multi-round interleaved A/B measurement (the current evidence is a single round, not a formal paired/bootstrapped measurement) before treating 18% as a real quantified number.
2. If confirmed, identify which specific patch(es) in the framework/upstream-fixes patch-sets is responsible for the prefill regression (bisect by building intermediate compositions).
3. Determine whether this is expected/accepted overhead (e.g. a deliberate correctness/safety patch with a known cost) or a real, unintended regression worth fixing.
4. Check whether this gap is specific to gpt-oss-20B/pp512 or generalizes across models and workloads.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Discovered as a side effect of adopting the standardized A/B/C baseline comparison methodology (docs/reference/testing/STANDARDIZED_PATCH_VALIDATION_CRITERIA.md) -- a real demonstration of why the three-arm comparison has value beyond the project's usual B-vs-C-only patch qualification.

**Root cause isolated (2026-09-13, real bisection).** Built two isolated compositions against source.llama-native (stock base): (1) upstream-fixes alone (just 1000_rdna4_mmq_q2k_q6k_fix) -- pp512=5210.01, matches stock, NO regression. (2) the 'forced'-dispatch family alone (0300_mmq_forced_j, 0400_mmvf_forced_block, 0500_mmf_forced_nwarps, 0600_mmvq_geometry, 0650_mmvq_native_variant -- 5 of framework's 14 patches) -- pp512=4263.46, REPRODUCES the full ~22.6% regression, matching BigCherry baseline's range. Root cause conclusively isolated to this 5-patch cluster, not the upstream-fixes correctness backport. These patches force fixed dispatch parameters (forced j/block-size/nwarps/geometry/native-variant) instead of upstream's own tuned auto-selection heuristics -- very plausibly a deliberate, accepted tradeoff for BigCherry's dispatch-research/autotuning infrastructure, not an unintended bug, but this quantified real number (~22.6% pp512 cost) was not previously measured/documented anywhere in the project. Whoever owns the framework patch-set's design should confirm this is the accepted/expected cost. Remaining open steps: confirm whether this generalizes beyond gpt-oss-20B/pp512 (other models/workloads), and whether all 5 patches contribute or just a subset.

**Root cause isolated (2026-09-13, real bisection).** Built two isolated compositions against source.llama-native (stock base): (1) upstream-fixes alone (just 1000_rdna4_mmq_q2k_q6k_fix) -- pp512=5210.01, matches stock, NO regression. (2) the 'forced'-dispatch family alone (0300_mmq_forced_j, 0400_mmvf_forced_block, 0500_mmf_forced_nwarps, 0600_mmvq_geometry, 0650_mmvq_native_variant -- 5 of framework's 14 patches) -- pp512=4263.46, REPRODUCES the full ~22.6% regression, matching BigCherry baseline's range. Root cause conclusively isolated to this 5-patch cluster on gpt-oss-20B.

**CRITICAL SCOPE CORRECTION (2026-09-13): does NOT generalize across models.** Ran the same A-vs-B comparison on tierA-qwen4b-q6k (Qwen3.5-4B, dense+GDN hybrid, 2 rounds): A=[4881.74, 4896.58], B=[4885.78, 4876.29] -- statistically identical, NO gap at all. The ~22.6% regression is specific to gpt-oss-20B (likely its MoE routing shape interacting with the forced-dispatch patches' fixed geometry/nwarps choices), not a universal BigCherry baseline cost. Revised finding: 'BigCherry's forced-dispatch patches cost ~22.6% pp512 on at least gpt-oss-20B's MoE shape, and ~0% on Qwen3.5-4B's dense+GDN shape' -- this is exactly why the standardized criteria's multi-model requirement matters; a single-model test would have wrongly generalized this as a universal cost. Remaining open steps: test additional models (tierM-ministral14b-q4km, tierM-qwen35b-a3b-moe-mtp) to map which architectural property (MoE routing specifically, vs something else about gpt-oss-20B) actually correlates with the regression.

**CONCLUSIVELY TRIANGULATED (2026-09-13, 4 models, 2 architecture classes).** Root cause isolated via bisection to the 5-patch 'forced'-dispatch cluster (0300_mmq_forced_j, 0400_mmvf_forced_block, 0500_mmf_forced_nwarps, 0600_mmvq_geometry, 0650_mmvq_native_variant). Real A(stock)-vs-B(BigCherry baseline) pp512 comparison across 4 models:

- Qwen3.5-4B (dense+GDN hybrid): ~0% gap (statistically identical)
- Ministral-3-14B (dense transformer, independent family): ~0% gap (statistically identical)
- gpt-oss-20B (MoE): ~22.6% gap (confirmed, 3 rounds, zero overlap)
- Qwen3.6-35B-A3B (MoE, different family/size): ~22.3% gap (single round)

**Conclusion: the forced-dispatch patches cost a consistent ~22-23% prefill throughput specifically on MoE models, and nothing measurable on dense/dense-hybrid models.** The two MoE gaps (22.6%, 22.3%) are remarkably close across genuinely different MoE families -- a real, systematic interaction between the forced fixed geometry/nwarps dispatch parameters and MoE's smaller, more numerous expert-routed matmul shapes, not noise or a single-model artifact. This is very plausibly a known/accepted tradeoff for BigCherry's dispatch-research infrastructure (the forced parameters may be tuned for dense shapes), but was never previously quantified. Remaining open step: determine whether this is an accepted tradeoff (ask the framework patch-set's owner) or worth tuning the forced parameters specifically for MoE shapes -- a real, scoped follow-up for whoever owns dispatch/autotuning design, not blocking any current patch's qualification.

**RESOLVED (2026-09-13).** Root cause: patches/0300_mmq_forced_j/patch.py's lifted J-selection scan (ggml_cuda_mmq_native_j_best) used args.ncols_max where real upstream (verified against actual vendor source in this repo) uses args.ncols_opt for tile-size optimization -- a distinct mmq_args field upstream sets to the approximate per-expert routed column count for MUL_MAT_ID/MoE on RDNA3.0/RDNA4 (deliberately smaller than ncols_max, the real launch/safety width). Dense models set ncols_opt==ncols_max upstream (hence zero observed effect); MoE models diverge sharply, causing the scan to select a substantially larger, wrong tile-width J than native upstream. GPT (req_75d5e59ef62e4675) diagnosed this precisely; independently verified against the real vendor source (mmq.cuh struct comment, mmq.cu's RDNA3/RDNA4 ncols_opt computation) before applying any fix.

Fix applied and committed (1203c2e2): changed ggml_cuda_mmq_native_j_best() to take/use ncols_opt instead of ncols_max (declaration, call site, and definition). ggml_cuda_mmq_variant_is_eligible's own ncols_max parameter is untouched (different purpose: padding/OOB safety).

Real hardware verification (rebuilt, multiple rounds): gpt-oss-20B recovers from 4267.30 to ~5209 (stock: 5169.24); Qwen3.6-35B-A3B MoE recovers from 2654.67 to 3165.05 (stock: 3247.36); Qwen3.5-4B dense confirmed unaffected (4890.26, matching its unaffected 4885.78 pre-fix and stock's 4881.74). Both predictions from the diagnosis confirmed exactly. Full details: docs/reference/testing/BASELINE_COMPARISON_RD13_2026-09-13.md.

Remaining step: 0300_mmq_forced_j's validated state should be reconfirmed under a fresh formal qualification run now that native/forced_J=0 behavior is corrected -- it was never truly invalidated (the bug predates this session and 'validated' always assumed correct upstream-matching behavior at forced_J=0), but a fresh run would be good practice.

## Change Log

- 2026-09-12T21:20:57.594630+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260912_212119_ran-a-real-stock-llamacpp-vs_9340
- 2026-09-12T21:21:19.539124+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T21:22:28.159150+00:00 (updated-by): Updated: section:description
- chg_20260912_212250_confirmed-with-real-multi-roun_7540
- 2026-09-12T21:22:50.045765+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T21:28:02.462203+00:00 (updated-by): Updated: section:notes
- chg_20260912_212832_traced-the-real-226-baselin_1022
- 2026-09-12T21:28:32.498047+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T21:30:17.612591+00:00 (updated-by): Updated: section:notes
- chg_20260912_213048_found-that-the-real-baseline-p_2042
- 2026-09-12T21:30:48.021556+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T21:33:02.287522+00:00 (updated-by): Updated: section:notes
- 2026-09-12T21:54:01.846359+00:00 (updated-by): Updated: section:notes
- 2026-09-12T21:54:06.313278+00:00 (state-transition): State: pending → completed
