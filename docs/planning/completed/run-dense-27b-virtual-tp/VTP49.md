---
id: VTP49
order: 0
plan: run-dense-27b-virtual-tp
state: completed
created-at: '2026-11-23T20:42:29.814399+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P2
---

# Drafter vocabulary head: coarse low-bit candidate scoring plus exact rerank (measure the drafter logits share first)

## Description

Idea considered: score the drafter's vocabulary head from a coarse low-bit copy, take the top candidates, and re-rank only those against exact rows, so the drafter streams a fraction of the head's bytes. The drafter only proposes; the target still verifies, so output correctness is unchanged and only acceptance can move. It pays only if the drafter pays for a full-vocabulary head today.

## Steps

1. Measure first: from the step-phase trace, separate the DFlash2 drafter's head projection and select cost per pass and per step (static-analysis of the trace JSONL, no new run needed if the segments are labelled; otherwise one traced run under the lease).
2. If the share is below 1 ms a step, record a NO-GO and stop.
3. Otherwise design: coarse low-bit table built at load from the shipped head (a load-time format conversion in libr3/libr4d plugin config, not a model constant), candidate count and rerank width as manifest fields, acceptance A/B at fixed seeds.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- Draft acceptance (mean tokens a step) and temp-0 text compared against the current drafter at 6 and 8 streams, 3 seeds.
- Decode ms/step ABBA.

## Effort & Risk

Medium. Risk: acceptance loss from missing the true top candidates costs more than the bytes saved.

## Standards



## Acceptance Criteria

- Net decode ms/step gain above the seed spread with acceptance loss no larger than the gain's equivalent in tokens a step; NO-GO recorded otherwise, including at step 2.

## Notes

**2026-11-24 step 1 done: NO-GO.** Static read of the 2020.8.3 step-phase trace (rank 0, 104 drafter passes of 17-32 rows, one pass 6.09 ms wall): the drafter pass has no full-vocabulary logits op. Its candidate path is already a top-k plus rerank: row_topk 0.046 ms, row_topk_merge 0.014, logit_rerank 0.018, dflash_select 0.013, together about 0.09 ms a pass; the pass is gemm_nt_q 2.39, all_reduce 0.79, gemm_nt 0.46 (the drafter's own projections), attention 0.23. The verify pass's logits op (0.88 ms on rank 0, 0.66 on rank 2) is the target's exact head and cannot be approximated without changing the output. Coarse low-bit candidate scoring has nothing to replace on the drafter side. Closed; no hardware run.

**2026-11-24 step 1 done: NO-GO.** Static read of the 2020.8.3 step-phase trace (rank 0, 104 drafter passes of 17-32 rows, one pass 6.09 ms wall): the drafter pass has no full-vocabulary logits op. Its candidate path is already a top-k plus rerank: row_topk 0.046 ms, row_topk_merge 0.014, logit_rerank 0.018, dflash_select 0.013, together about 0.09 ms a pass; the pass is gemm_nt_q 2.39, all_reduce 0.79, gemm_nt 0.46 (the drafter's own projections), attention 0.23. The verify pass's logits op (0.88 ms on rank 0, 0.66 on rank 2) is the target's exact head and cannot be approximated without changing the output. Closed; no hardware run.

## Change Log

- 2026-11-23T20:42:29.814399+00:00 (created-by): Created by agent
- 2026-11-23T20:43:21.663083+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-23T20:48:17.510291+00:00 (updated-by): Updated: section:notes
- 2026-11-23T20:48:27.240696+00:00 (state-transition): State: pending → completed
- 2026-11-23T20:54:48.968405+00:00 (updated-by): Updated: section:description, section:notes
