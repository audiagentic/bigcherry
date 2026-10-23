---
id: BCOP168
order: 168
plan: patching-bc-optimizations
state: deprecated
created-at: '2026-10-12T18:08:00+00:00'
created-by: agent
priority: P2
work: S
---

# PRBE115: retire Q8_0 raw-F32 verify-width follow-up

## Description

Patch 1301 is already rejected. Right-model dual-XTX MTP4 evidence reports widths 2/3 neutral and width 5 about 6–12% slower. Raw-F32 activation arithmetic differs from stock Q8_1 quantized-activation integer accumulation; hoisting weight dequantization does not restore numerical identity. PRBE115 owns the terminal disposition; QFP21 owns measured evidence, 1241 retains validated width-1 behaviour, and PRBE113 retains the profiling dependency. No new patch or hardware queue. Reopen only with an exact-numerics-preserving mechanism, causal >=3% theoretical E2E opportunity, complete logits/MTP acceptance, multi-request graph correctness, and four-session ABBA performance qualification. No new GPU runs or measured gain in this audit.

## Steps


## Detailed Solution & Technical Design


## Code Samples & Guidance


## Files


## Validation


## Effort & Risk


## Standards


## Acceptance Criteria


## Notes


## Change Log

- 2026-10-23: state 'done' (invalid) corrected to deprecated
