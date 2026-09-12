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

A real three-arm A/B/C baseline comparison for RD13 (2026-09-13, docs/reference/testing/BASELINE_COMPARISON_RD13_2026-09-13.md) found stock upstream llama.cpp (source.llama-native, zero BigCherry patches) shows ~18% higher pp512 prefill throughput than BigCherry's own baseline composition (source.bigcherry-native, the framework+upstream-fixes patch-sets, with the focal patch RD13 absent) on gpt-oss-20B/gfx1100: A=5174.61 t/s vs B=4215.71 t/s. tg128 decode throughput is unaffected (177.76 vs 178.13, effectively identical). This gap is unrelated to RD13 itself (RD13's own B-vs-C effect is negligible). This is the first real evidence of this specific gap -- it was only surfaced because the standardized validation criteria required a genuine three-arm comparison rather than the project's usual B-vs-C-only methodology.

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

## Change Log

- 2026-09-12T21:20:57.594630+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260912_212119_ran-a-real-stock-llamacpp-vs_9340
- 2026-09-12T21:21:19.539124+00:00 (updated-by): Updated: section:ledger-events
