---
id: HTR06
order: 0
plan: hip-tune-recovery
state: pending
created-at: '2026-08-29T23:57:59.533598+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
---

# Investigate synthetic-to-E2E candidate ranking fidelity (bounded empirical study, not a ranking change)

## Description

User question, adversarially assessed with GPT (session ses_330ae3c055084f38, 2026-08-30): BigCherry's tuner picks the single fastest MEASURED candidate per signature from synthetic per-op tune-time timing (effective_us, median, p95, host median, MAD, sign-test statistics -- not just raw latency, so not naively blind to variance). Could a candidate discarded as 'not fastest in synthetic tuning' actually be BETTER under real end-to-end conditions the tune-time harness doesn't replicate -- contention/occupancy interactions, cache/state effects, real production batch composition, sustained thermal/clock behavior, tail latency, or interaction with surrounding kernels?

GPT's verdict: a real, technically plausible concern, but NOT currently demonstrated as an actual BigCherry failure. HI141 proved synthetic INPUT DATA can fail to predict real CORRECTNESS/behavioral outcomes (the whole reason HI143/HTR01 exist) -- it does NOT prove synthetic PERFORMANCE RANKING is wrong. Conflating the two would be a real category error: changing candidate-selection/ranking policy now, without evidence, would be pure speculation. This item is scoped as a bounded empirical INVESTIGATION to find out whether the concern is real, not a mandate to change ranking behavior.

## Steps

1. Select several high-cost and/or high-frequency signatures (real production dispatch shapes, not synthetic ones) as the study's targets.
2. For each target signature, take its tune-time winner plus its top-ranked already-measured alternatives (HTR01's own SignatureAssignment.alternatives is a ready-made, no-new-measurement source for this).
3. Construct multiple full-cache variants that differ by EXACTLY ONE signature's candidate against an otherwise-FIXED ensemble (holding everything else constant is essential -- HI141 already proved candidate-mix effects are real and non-monotonic, so results from this study must be scoped to 'this specific ensemble', never generalized to 'this candidate is universally better/worse').
4. Benchmark each variant with randomized/ABBA ordering (not simple A-then-B, to control for thermal/warm-up/ordering bias) measuring warmed E2E throughput plus p50/p95/variance over a SUSTAINED real run (not a single short measurement) -- directly addressing the sustained-thermal/tail-latency part of the concern.
5. Compare the tune-time effective_us ranking against the real E2E ranking for the same candidates in the same fixed ensemble. Quantify any rank inversions and their real effect size (not just 'did the order change', but 'does it matter and by how much').
6. Explicit non-goal, per GPT: do NOT infer or propose a universal candidate-ranking change from this study's results. Per HI141's own non-monotonicity finding, any observed inversion is evidence about THIS ensemble/signature combination only. A repeatable, material inversion pattern found across MULTIPLE independent ensembles would be the actual bar for considering a ranking-methodology change -- and that follow-up work is explicitly out of THIS item's scope if the investigation reaches that point.

## Detailed Solution & Technical Design

This item is investigation-only by design (GPT explicit): 'implement HTR03 now with the provenance additions... Create HTR06 as a bounded empirical investigation; do not change candidate selection until HTR06 demonstrates a real ranking-fidelity problem.' Do not let this item's scope creep into an implementation task for a new ranking algorithm -- that would be new, separately-scoped, evidence-gated work only if this investigation's own results warrant it.

Relationship to HTR03 (kept deliberately separate, per GPT): HTR03 answers 'is this exact cache behaviorally SAFE on representative workloads' (a correctness/regression-detection question). This item answers 'does tune-time candidate ranking predict real E2E candidate ranking' (a performance-selection-quality question). HTR03's corpus vectors MAY later be reused as this item's real workload shapes, but performance investigation results must never contaminate HTR03's correctness contract, and vice versa.

## Code Samples & Guidance



## Files

New: a benchmarking/comparison script under tools/bigcherry/tuning/ or a scratch investigation script (final home TBD based on whether this becomes durable tooling or a one-off study) -- do not build permanent infrastructure before the investigation itself justifies it.

## Validation

This item's own deliverable is the empirical findings and their write-up (quantified rank-inversion frequency/effect-size across the studied signatures/ensembles), not code. A follow-up implementation item should only be created if the findings show a repeatable, material problem across multiple independent ensembles.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Pending -- not started. Do not pick this up opportunistically alongside HTR03 work; it is a genuinely separate investigation with its own real GPU time cost (multiple sustained E2E benchmark runs across multiple candidate variants and ensembles) and should be scoped/scheduled deliberately.

## Change Log

- 2026-08-29T23:57:59.533598+00:00 (created-by): Created by agent
