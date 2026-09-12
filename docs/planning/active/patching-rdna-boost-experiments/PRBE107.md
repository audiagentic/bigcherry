---
id: PRBE107
order: 0
plan: patching-rdna-boost-experiments
state: pending
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
