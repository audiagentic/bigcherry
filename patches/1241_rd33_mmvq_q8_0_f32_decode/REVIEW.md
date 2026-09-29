# Independent review — 1241 RD33 MMVQ Q8_0 F32 decode

Reviewed against branch evidence at `tools/lab/native-vs-patched/runs/rd33-divergence/` and `rd33-ab1/`.

## Scoped verdict

I sign off on the narrow claim that, on the measured dual-gfx1100 Qwen3.8-27B-Q8_0 `-sm tensor` plain greedy decode workload, RD33 produced a real observed speed advantage: mean 33.06 -> 34.67 tok/s (+4.9%), with RD33 faster in all eight executions. I do **not** sign off on promotion, output equivalence, general quality equivalence, or an MTP throughput benefit.

## Performance evidence

The fixed-work plain-decode experiment is the right isolation for RD33's `ncols_dst == 1` path: no MTP, fixed 256-token requests, temperature 0/top-k 1, four prompts and two repetitions. The consistency of the per-run rates (control 32.95-33.13, RD33 34.59-34.72) makes a +4.9% measured effect credible and much larger than within-arm variation.

It is not a strong causal performance experiment yet because the arms were sequential and control always ran first. That leaves arm order, warm-up/cache state, clock/power/thermal state, and slow host/system drift confounded with the patch. Two repetitions of four prompts are repeated requests inside one arm session, not independent order-balanced A/B sessions. Before promotion, rerun fixed-work plain decode with alternating/order-balanced A/B pairs (preferably multiple independent sessions), identical warm-up, recorded clocks/temperature/power or equivalent host isolation, and the contract's statistical policy. The current result supports "observed +4.9% in this run", not yet a robust +4.9% population estimate.

## Numerics and quality

The existing stock-vs-RD33 max-absolute-logprob gate is scientifically mismatched to this patch. RD33 deliberately replaces Q8_1 activation quantisation/int32-style dot arithmetic with original-F32 activation/F32 accumulation, so stock is not a correctness oracle for numerical identity. The observed sampled-token logprob deltas (0.017-0.075) and greedy divergence in 2/4 prompts correctly disprove bit-close/greedy-equivalent claims; they do not alone establish a defect.

The batch-1 perplexity result (11.6132 control vs 11.5989 RD33, RD33 lower at all eight running checkpoints) is useful supporting evidence that this one 4096-token corpus did not regress aggregate likelihood. It is not adequate by itself for promotion: the sample is small, one corpus/model, running checkpoints are correlated rather than eight independent quality samples, and perplexity can hide localized large distribution errors or regressions on other content.

`RD33-MMVQ-Q8_0-F32-DECODE` should therefore replace stock-relative `max_abs_logprob_diff <= 5e-4` with a **reference-based numerical accuracy gate**. For eligible Q8_0 `ncols_dst==1` shapes, compare both stock and RD33 full output/logit vectors against an independently computed high-precision/F32-activation reference on deterministic test vectors and require RD33 to remain within a predeclared absolute/relative or normalized-error tolerance and not materially worsen error versus stock. Prefer the existing `backend_reference`/CPU-reference machinery where it represents the intended Q8_0-weight + F32-activation operation; add explicit exact production shapes. Separately retain an end-to-end quality guard using a substantially larger/multiple-corpus perplexity evaluation and, if feasible, full-vocabulary KL/cross-entropy delta versus the same reference. Thresholds must be declared before collecting promotion evidence rather than fitted to this result.

## MTP risk

The MTP A/B is correctly void for throughput attribution: acceptance changed from about 0.9010 to 0.9558, so the arms performed different speculative work. This is a material behavioral effect, not noise to normalize away. Because small target-logit changes can alter greedy verification and then acceptance, RD33 can change MTP workload composition even though its direct kernel gate is single-column decode. No MTP speed claim is supportable from the current run.

Before promotion into an MTP serving composition, require an order-balanced contract campaign that treats acceptance as a first-class behavioral metric: report acceptance distribution/rate, greedy divergence and quality/reference checks alongside throughput. A higher acceptance rate is potentially beneficial but cannot substitute for correctness; first establish that the changed target decisions satisfy the reference-based quality gate, then measure end-to-end MTP performance with the changed acceptance explicitly acknowledged rather than claiming a fixed-work kernel effect.

## Promotion conditions

Promotion remains blocked pending: (1) an amended, predeclared reference-based correctness contract appropriate to intentional numerical change; (2) contract-produced validated evidence for that gate, including production-shape backend/reference tests and broader end-to-end quality evidence; (3) order-balanced independent plain-decode performance sessions confirming a positive effect without fixed arm-order confounding; and (4) MTP-specific correctness/quality plus acceptance characterization before any MTP performance claim. The in-progress 8-pair MTP rerun is useful evidence, but cannot by itself resolve the correctness-contract issue or turn acceptance-changing throughput into a fixed-work attribution.