---
id: VTP49
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-23T20:42:29.814399+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P2
---

# Drafter vocabulary head: coarse low-bit candidate scoring plus exact rerank (measure the drafter logits share first)

## Description

Source: a reference drafter head scores the whole vocabulary from a 2-bit copy of the head (per-group sums, coarse kernels at 1, 2 and 4 rows), reduces to block winners, runs a bitonic top-N (1024 or 4096 down to 128 candidates), re-ranks the 128 candidates against exact bf16 rows and selects the top 1 or 16. The drafter only proposes; the target still verifies, so output correctness is unchanged and only acceptance can move. Streams about a quarter of the int8 head bytes for the drafter's logits. Our share: the verify-pass logits op is 0.87-0.92 ms of a 41 ms traced verify pass on rank 0 (phase trace, 2020.8.3); the drafter pass's own logits cost is not separated in that trace. The lever exists only if the drafter's head projection plus select is more than about 1 ms of a step.

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



## Change Log

- 2026-11-23T20:42:29.814399+00:00 (created-by): Created by agent
- 2026-11-23T20:43:21.663083+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
