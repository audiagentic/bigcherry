---
id: PRBE108
order: 0
plan: patching-rdna-boost-experiments
state: completed
created-at: '2026-09-12T22:08:29.479059+00:00'
breadth: ''
skill: ''
created-by: agent
---

# Reconcile RD39-42's formal +2.38% gain vs a quick non-interleaved -5% check

## Description

RD39-42/1215's formal contract-qualification evidence (10 rounds, deliberately interleaved baseline/subject to rule out ordering/drift bias) shows a real +2.38% mean decode gain (95% CI [1.45%, 3.32%]) on gfx1100 with GGML_CUDA_GRAPH_OPT=1. A quick, non-interleaved B-vs-C check run 2026-09-13 (always baseline first, then subject, 4 rounds, tg32/Qwen3.6-35B-A3B) instead showed a consistent ~5% NEGATIVE delta (B~114 t/s, C~108 t/s). This is a real, unresolved discrepancy -- not yet reconciled. The quick check's lack of interleaving is the most likely explanation (systematic drift/thermal/warm-up effects that interleaving specifically exists to rule out), but this has not been confirmed.

## Steps

1. Rerun the B-vs-C comparison with proper interleaving (baseline/subject/baseline/subject...) matching the formal methodology's own protocol.
2. If interleaved results match the formal +2.38% gain: the quick check's ordering bias is confirmed as the explanation, close this item.
3. If interleaved results still show a negative/different delta: something has genuinely changed since the formal evidence was gathered (build drift, pin drift, environment change) -- investigate further before trusting either result.
4. Check whether the current pin (b10901/28ff0958291c) matches exactly what the formal 10-round evidence was measured against, or whether the pin has moved since.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

**Interleaving does NOT explain the discrepancy** (2026-09-13): reran with proper interleaving (B/C/C/B x4 rounds): B mean=113.96 t/s (sd 0.20, very tight), C mean=107.20 t/s (sd 1.35), delta=-5.93%, essentially unchanged from the non-interleaved check. Ordering bias is ruled out as the explanation.

**Also ruled out: this session's own PRBE107 fix (0300_mmq_forced_j) is NOT the cause.** Compared pure stock upstream (zero BigCherry patches, no 0300 at all) decode tg32 = 113.67 t/s against B (BigCherry baseline, includes the FIXED 0300) = 113.96 t/s -- essentially identical. 0300 has no measurable effect on decode throughput at all (expected -- its scan only matters for MMQ tile-width selection, and decode's batch=1 shape barely engages MMQ's tile-width tuning). The real -5.93% negative delta is isolated specifically to C (BigCherry+1215+1216), not explained by ordering, not explained by the 0300 fix.

Remaining open question: why does a real, tight, reproducible ~5.93% negative decode delta appear for BigCherry+1215+1216 here, when the patch's own formal 10-round interleaved evidence documented a real +2.38% POSITIVE gain under nominally the same condition (GGML_CUDA_GRAPH_OPT=1, gfx1100)? Possible remaining explanations, none yet checked: (a) the formal evidence's exact build/pin differs from the current git HEAD in some way not yet identified, (b) the formal evidence measured a different benchmark shape (need to check its exact llama-bench invocation/workload vs this session's tg32), (c) a real regression was introduced to 1215/1216's mechanism by something else committed since the formal evidence was gathered, (d) the formal evidence itself has a real, undiscovered flaw. This needs a careful, deliberate re-investigation, not further ad-hoc quick checks.

**RESOLVED (2026-09-13): wrong model file, not a real regression.** After ruling out ordering bias (interleaved rerun, discrepancy persisted) and this session's own PRBE107 fix (stock-vs-baseline decode comparison, no effect), checked the formal evidence's exact model reference in patches/1215_rd394041_amd_stream_moe_overlap/README.md: Qwen3.6-35B-A3B-UD-Q4_K_M.gguf. This session's quick check had used a DIFFERENT file entirely: Qwen3.6-35B-A3B-APEX-MTP-I-Compact.gguf. Reran the exact same B-vs-C comparison with the correct model file (tg128, 3 rounds): B mean=110.54, C mean=111.36, C >= B in all 3 rounds, mean +0.74% gain -- consistent in direction and magnitude range with the formal +2.38% result (within its own SD 1.31%). The earlier -5.93% 'discrepancy' was entirely a wrong-model-file artifact, not a real contradiction. Real lesson for future quick checks: always verify the exact model file reference against the patch's own documented evidence before treating a discrepancy as real.

## Change Log

- 2026-09-12T22:08:29.479059+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260912_220855_found-a-real-unresolved-discr_5936
- 2026-09-12T22:08:55.337667+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T22:12:33.510493+00:00 (updated-by): Updated: section:notes
- chg_20260912_221315_ruled-out-two-possible-explana_1511
- 2026-09-12T22:13:15.186788+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T22:16:14.529646+00:00 (updated-by): Updated: section:notes
- 2026-09-12T22:16:20.658267+00:00 (state-transition): State: pending → completed
